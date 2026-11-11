"""Sandboxed execution of commands and LLM-written code.

Layers, weakest to strongest:
  1. Static checks (AST for Python, patterns for bash). These catch honest
     mistakes and obvious overreach; they are NOT a security boundary.
  2. Process limits: scrubbed env, cwd jail, CPU/memory rlimits, timeout.
  3. Network isolation: `unshare -rn` on Linux when available.
  4. Container isolation: `docker run --network none` when use_containers.
Use (4) for any untrusted code on shared machines.
"""
from __future__ import annotations

import ast
import functools
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

BANNED_PY_MODULES = {"socket", "requests", "urllib", "http", "httpx", "ftplib", "smtplib", "paramiko", "ctypes"}
BANNED_PY_CALLS = {"os.system", "os.popen", "shutil.rmtree", "os.remove", "os.unlink", "os.rmdir", "eval", "exec",
                   "subprocess.Popen", "subprocess.run", "subprocess.call", "subprocess.check_output", "__import__"}
BASH_DENY = [
    (r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+(/|~|\$HOME)(\s|$)", "recursive delete of root/home"),
    (r"\bsudo\b", "sudo"),
    (r"\b(curl|wget|scp|ssh|nc|ncat|ftp|rsync)\b", "network client"),
    (r"\bmkfs|\bdd\s+if=", "disk-level command"),
    (r">\s*/(etc|usr|bin|boot|sys|proc)/", "write to system path"),
    (r":\(\)\s*\{", "fork bomb"),
    (r"\bchmod\s+-R\s+777\s+/", "global chmod"),
]


def static_check(language: str, code: str, network: bool = False) -> list[str]:
    issues: list[str] = []
    if language == "python":
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"syntax error: {e}"]
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for n in names:
                    root = n.split(".")[0]
                    if root in BANNED_PY_MODULES and not network:
                        issues.append(f"imports network/unsafe module '{n}'")
                    if root == "subprocess":
                        issues.append("imports subprocess (use registry tools instead)")
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in BANNED_PY_CALLS:
                    issues.append(f"calls {name}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if v.startswith(("/etc", "/usr", "/bin", "/boot", "/sys", "~/.ssh")) and len(v) < 200:
                    issues.append(f"references system path {v!r}")
    elif language == "bash":
        for pat, why in BASH_DENY:
            if why == "network client" and network:
                continue
            if re.search(pat, code):
                issues.append(f"bash: {why}")
    else:
        issues.append(f"unsupported language {language}")
    return sorted(set(issues))


def _call_name(f) -> str:
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f"{_call_name(f.value)}.{f.attr}"
    return ""


@functools.lru_cache(maxsize=1)
def unshare_available() -> bool:
    if not sys.platform.startswith("linux") or not shutil.which("unshare"):
        return False
    try:
        return subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


def _limits(cpu_s: int | None, mem_gb: float | None):
    def fn():
        try:
            import resource

            if cpu_s:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
            if mem_gb and sys.platform.startswith("linux"):  # RLIMIT_AS is unreliable on macOS
                b = int(mem_gb * 1024**3)
                resource.setrlimit(resource.RLIMIT_AS, (b, b))
        except Exception:
            pass
        os.setsid()

    return fn


SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "CONDA_PREFIX", "JAVA_HOME", "NXF_HOME",
                 "AWS_PROFILE", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
                 "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "AWS_REGION", "AWS_DEFAULT_REGION", "KUBECONFIG", "SNAKEMAKE_CONDA_PREFIX")


def scrubbed_env(extra: dict | None = None) -> dict:
    env = {k: os.environ[k] for k in SAFE_ENV_KEYS if k in os.environ}
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLBACKEND"] = "Agg"
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


class RunResult:
    def __init__(self, exit_code: int, log_path: Path, timed_out: bool = False):
        self.exit_code = exit_code
        self.log_path = log_path
        self.timed_out = timed_out

    def tail(self, n: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except FileNotFoundError:
            return ""


def run_command(
    cmd: list[str],
    cwd: str | Path,
    log_path: str | Path,
    timeout_s: int = 3600,
    cpu_s: int | None = None,
    mem_gb: float | None = None,
    isolate_network: bool = False,
    env: dict | None = None,
    container: str | None = None,
    cancel: threading.Event | None = None,
) -> RunResult:
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path)
    if container:
        net = ["--network", "none"] if isolate_network else []
        cmd = ["docker", "run", "--rm", *net, "-v", f"{cwd.resolve()}:{cwd.resolve()}", "-w", str(cwd.resolve()),
               "-u", f"{os.getuid()}:{os.getgid()}", container, *cmd]
    elif isolate_network and unshare_available():
        cmd = ["unshare", "-rn", *cmd]
    with log_path.open("ab") as log:
        log.write(("$ " + " ".join(cmd) + "\n").encode())
        log.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=scrubbed_env(env),
                                preexec_fn=_limits(cpu_s, mem_gb) if os.name == "posix" else None)
        timed_out = False
        try:
            waited = 0.0
            while True:
                try:
                    proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    waited += 0.5
                    if waited >= timeout_s or (cancel is not None and cancel.is_set()):
                        timed_out = waited >= timeout_s
                        _kill(proc)
                        break
        finally:
            if proc.poll() is None:
                _kill(proc)
    return RunResult(proc.returncode if proc.returncode is not None else -9, log_path, timed_out)


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, 9)
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except Exception:
        pass

"""The Compute Graph: turns a PipelineSpec into processes and tracks them.

The cognitive graph never runs a tool itself. It calls `JobManager.submit`,
records the job id, and suspends (LangGraph interrupt). The job runs in a
worker thread here; completion is signalled by the process exit, and for
Nextflow cloud runs also by `-with-weblog` events posted to the API webhook.
Either way the runner resumes the graph with the final JobRecord.

Backends:
  local      ordered subprocess steps (registry tools, sandboxed scripts)
  snakemake  `snakemake --cores N` on a registry workflow (local HPC/workstation)
  nextflow   `nextflow run -profile awsbatch|k8s` head process (cloud)
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from .config import Settings
from .provenance import EventBus, BUS, Provenance, sha256_file
from .registry import Registry
from .sandbox import run_command, static_check
from .state import JobRecord, PipelineSpec, TaskSpec, new_id, now


@dataclass
class Step:
    name: str
    cmd: list[str]
    outdir: str
    container: str | None = None
    isolate_network: bool = False
    timeout_s: int = 3600


@dataclass
class ExecutionPlan:
    backend: str
    steps: list[Step]
    outputs: dict[str, str]
    collect_dir: str | None = None  # for scripts: every file under it is an output
    meta: dict[str, Any] = field(default_factory=dict)


class MaterializeError(ValueError):
    pass


PLACEHOLDER = re.compile(r"\{(input|stage|prev):([A-Za-z0-9_.\-]+)\}")


class Materializer:
    def __init__(self, settings: Settings, registry: Registry):
        self.s = settings
        self.reg = registry

    def resolve(self, value: Any, task: TaskSpec, stage_outputs: dict, prev: dict) -> Any:
        if isinstance(value, list):
            return [self.resolve(v, task, stage_outputs, prev) for v in value]
        if not isinstance(value, str):
            return value

        def rep(m):
            kind, ref = m.group(1), m.group(2)
            if kind == "input":
                if ref not in task.inputs:
                    raise MaterializeError(f"unknown task input '{ref}'; have {sorted(task.inputs)}")
                return str(Path(task.inputs[ref]).expanduser().resolve())
            src, _, key = ref.partition(".")
            table = stage_outputs if kind == "stage" else prev
            if src not in table or key not in table[src]:
                raise MaterializeError(f"unresolved reference {{{kind}:{ref}}}; available: "
                                       f"{ {k: list(v) for k, v in table.items()} }")
            return table[src][key]

        return PLACEHOLDER.sub(rep, value)

    def build(self, spec: PipelineSpec, stage: str, job_dir: Path, task: TaskSpec,
              stage_outputs: dict[str, dict[str, str]]) -> ExecutionPlan:
        c = self.s.compute
        if spec.kind == "tools":
            if not spec.tool_calls:
                raise MaterializeError("pipeline kind=tools with no tool_calls")
            steps, prev, outputs = [], {}, {}
            for i, call in enumerate(spec.tool_calls):
                tool = self.reg.get(call.tool_id)
                ok, how = tool.available(c.use_containers)
                if not ok:
                    raise MaterializeError(f"tool {tool.id} unavailable: {how}")
                if not self.reg.network_allowed(tool):
                    raise MaterializeError(f"tool {tool.id} calls an external API; blocked in local mode")
                outdir = job_dir / f"{i:02d}_{tool.id.replace('.', '_')}"
                params = {k: self.resolve(v, task, stage_outputs, prev) for k, v in call.params.items()}
                if tool.kind == "openapi":
                    cmd = [sys.executable, "-m", "bioagent.openapi_call", "--method", tool.method or "GET",
                           "--url", tool.url or "", "--out", str(outdir / "response.json"),
                           "--params", json.dumps(params)]
                else:
                    cmd = self.reg.render_command(tool, params, str(outdir), c.cores)
                container = tool.container if (c.use_containers and tool.container and tool.kind == "cli") else None
                steps.append(Step(tool.id, cmd, str(outdir), container, False, c.job_timeout_s))
                outs = self.reg.render_outputs(tool.outputs, str(outdir))
                prev[tool.id] = outs
                outputs.update({f"{tool.id}.{k}": v for k, v in outs.items()})
                outputs.update(outs)  # short names; last tool wins on collision
            return ExecutionPlan("local", steps, outputs)

        if spec.kind == "script":
            if spec.script is None:
                raise MaterializeError("pipeline kind=script with no script")
            issues = static_check(spec.script.language, spec.script.code, c.sandbox_network)
            if issues:
                raise MaterializeError("script failed static checks: " + "; ".join(issues))
            job_dir.mkdir(parents=True, exist_ok=True)
            out = job_dir / "outputs"
            out.mkdir(exist_ok=True)
            ext = "py" if spec.script.language == "python" else "sh"
            path = job_dir / f"script.{ext}"
            code = spec.script.code
            for ref in set(m.group(0) for m in PLACEHOLDER.finditer(code)):
                code = code.replace(ref, str(self.resolve(ref, task, stage_outputs, {})))
            path.write_text(code)
            interp = [sys.executable] if ext == "py" else ["bash", "-euo", "pipefail"]
            step = Step("script", [*interp, str(path)], str(out), None, not c.sandbox_network, c.job_timeout_s)
            return ExecutionPlan("local", [step], {}, collect_dir=str(out))

        if spec.kind == "workflow":
            if not spec.workflow_id or spec.workflow_id not in self.reg.workflows:
                raise MaterializeError(f"unknown workflow '{spec.workflow_id}'")
            wf = self.reg.workflows[spec.workflow_id]
            params = {k: self.resolve(v, task, stage_outputs, {}) for k, v in spec.workflow_params.items()}
            for k, p in wf.params.items():
                if p.required and k not in params:
                    raise MaterializeError(f"workflow {wf.id}: missing required param '{k}'")
                params.setdefault(k, p.default)
            job_dir.mkdir(parents=True, exist_ok=True)
            engine = c.backend if c.backend in wf.engines else ("snakemake" if "snakemake" in wf.engines else None)
            if engine is None:
                raise MaterializeError(f"workflow {wf.id} has no engine for backend {c.backend}")
            entry = str(Path(wf.dir) / wf.engines[engine])
            params["outdir"] = str(job_dir / "results")
            if engine == "snakemake":
                if not shutil.which("snakemake"):
                    raise MaterializeError("snakemake not on PATH (pip install 'bioagent[workflows]')")
                cfgp = job_dir / "config.yaml"
                cfgp.write_text(yaml.safe_dump(params))
                cmd = ["snakemake", "-s", entry, "--configfile", str(cfgp), "--cores", str(c.cores),
                       "--directory", str(job_dir), "--sdm", "conda", "--rerun-incomplete", "--printshellcmds"]
            else:
                if not shutil.which("nextflow"):
                    raise MaterializeError("nextflow not on PATH")
                pf = job_dir / "params.json"
                pf.write_text(json.dumps(params, indent=2))
                cmd = ["nextflow", "run", entry, "-params-file", str(pf), "-profile", c.nextflow_profile,
                       "-ansi-log", "false"]
                if c.nextflow_work_dir:
                    cmd += ["-work-dir", c.nextflow_work_dir]
                if c.webhook_base_url:
                    cmd += ["-with-weblog", f"{c.webhook_base_url.rstrip('/')}/webhooks/nextflow/{{job_id}}"]
            outs = {k: v.replace("{outdir}", params["outdir"]) for k, v in wf.outputs.items()}
            return ExecutionPlan(engine, [Step(wf.id, cmd, str(job_dir), None, False, c.job_timeout_s)], outs)

        raise MaterializeError(f"unknown pipeline kind {spec.kind}")


class JobManager:
    def __init__(self, settings: Settings, registry: Registry, bus: EventBus = BUS):
        self.s = settings
        self.reg = registry
        self.bus = bus
        self.materializer = Materializer(settings, registry)
        self.dir = settings.resolve(settings.state_dir) / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._done: dict[str, threading.Event] = {}
        self._callbacks: list[Callable[[str, JobRecord], None]] = []
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- state
    def _path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def _save(self, rec: JobRecord, run_id: str) -> None:
        self._path(rec.job_id).write_text(json.dumps({"run_id": run_id, **rec.model_dump()}, indent=2))

    def get(self, job_id: str) -> JobRecord:
        d = json.loads(self._path(job_id).read_text())
        d.pop("run_id", None)
        rec = JobRecord.model_validate(d)
        if rec.status in ("submitted", "running") and job_id not in self._threads:
            # The process that owned this job is gone (restart). We cannot reattach
            # to a local subprocess, so report it as lost; the graph will retry.
            rec.status, rec.error = "lost", "job owner process exited before completion"
        return rec

    def on_complete(self, fn: Callable[[str, JobRecord], None]) -> None:
        self._callbacks.append(fn)

    def wait(self, job_id: str, timeout: float | None = None) -> JobRecord:
        ev = self._done.get(job_id)
        if ev is not None:
            ev.wait(timeout)
        return self.get(job_id)

    def cancel(self, job_id: str) -> None:
        if job_id in self._cancel:
            self._cancel[job_id].set()

    # --------------------------------------------------------------- submit
    def submit(self, run_id: str, stage: str, spec: PipelineSpec, task: TaskSpec,
               stage_outputs: dict, run_dir: Path, prov: Provenance | None = None) -> JobRecord:
        job_id = new_id("job")
        job_dir = Path(run_dir) / "jobs" / f"{stage}_{job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)
        plan = self.materializer.build(spec, stage, job_dir, task, stage_outputs)
        for st in plan.steps:
            st.cmd = [a.replace("{job_id}", job_id) for a in st.cmd]
        rec = JobRecord(job_id=job_id, stage=stage, backend=plan.backend, workdir=str(job_dir), status="submitted")
        (job_dir / "plan.json").write_text(json.dumps(
            {"backend": plan.backend, "steps": [s.__dict__ for s in plan.steps], "outputs": plan.outputs}, indent=2))
        self._save(rec, run_id)
        if prov:
            prov.log("job_submitted", job_id=job_id, stage=stage, backend=plan.backend,
                     commands=[s.cmd for s in plan.steps])
        self._done[job_id] = threading.Event()
        self._cancel[job_id] = threading.Event()
        t = threading.Thread(target=self._run, args=(run_id, rec, plan, prov), daemon=True, name=job_id)
        with self._lock:
            self._threads[job_id] = t
        t.start()
        return rec

    def _run(self, run_id: str, rec: JobRecord, plan: ExecutionPlan, prov: Provenance | None) -> None:
        rec.status = "running"
        self._save(rec, run_id)
        self.bus.publish(run_id, {"event": "job_running", "job_id": rec.job_id, "ts": now()})
        log = Path(rec.workdir) / "job.log"
        try:
            for st in plan.steps:
                Path(st.outdir).mkdir(parents=True, exist_ok=True)
                self.bus.publish(run_id, {"event": "job_step", "job_id": rec.job_id, "step": st.name, "ts": now()})
                res = run_command(st.cmd, st.outdir, log, timeout_s=st.timeout_s, mem_gb=self.s.compute.memory_gb,
                                  isolate_network=st.isolate_network, container=st.container,
                                  cancel=self._cancel[rec.job_id])
                rec.exit_code = res.exit_code
                if res.exit_code != 0:
                    rec.status = "failed"
                    rec.error = f"step '{st.name}' exited {res.exit_code}" + (" (timeout)" if res.timed_out else "")
                    rec.log_tail = res.tail()
                    break
            else:
                outputs = dict(plan.outputs)
                if plan.collect_dir:
                    for f in sorted(Path(plan.collect_dir).rglob("*")):
                        if f.is_file():
                            outputs[f.relative_to(plan.collect_dir).as_posix()] = str(f)
                missing = [k for k, v in outputs.items() if not Path(v).exists()]
                if missing:
                    rec.status, rec.error = "failed", f"declared outputs missing: {missing}"
                else:
                    rec.status = "succeeded"
                rec.outputs = outputs
                rec.log_tail = _tail(log)
        except Exception as e:
            rec.status, rec.error = "failed", f"{type(e).__name__}: {e}"
            rec.log_tail = _tail(log)
        rec.finished_at = now()
        self._save(rec, run_id)
        if prov:
            prov.log("job_finished", job_id=rec.job_id, status=rec.status, exit_code=rec.exit_code, error=rec.error,
                     outputs={k: {"path": v, "sha256": sha256_file(v)} for k, v in rec.outputs.items()})
        self._done[rec.job_id].set()
        with self._lock:
            self._threads.pop(rec.job_id, None)
        for cb in self._callbacks:
            try:
                cb(run_id, rec)
            except Exception:
                pass

    # -------------------------------------------------------------- webhook
    def handle_nextflow_event(self, job_id: str, payload: dict) -> None:
        """Nextflow -with-weblog posts {event: started|process_submitted|process_completed|completed|error, ...}.
        Progress is forwarded to the event bus; final status still comes from the head process exit."""
        try:
            run_id = json.loads(self._path(job_id).read_text()).get("run_id", "*")
        except FileNotFoundError:
            return
        trace = payload.get("trace") or {}
        self.bus.publish(run_id, {"event": "nextflow_" + str(payload.get("event")), "job_id": job_id,
                                  "process": trace.get("process"), "status": trace.get("status"), "ts": now()})


def _tail(p: Path, n: int = 40) -> str:
    try:
        return "\n".join(p.read_text(errors="replace").splitlines()[-n:])
    except FileNotFoundError:
        return ""

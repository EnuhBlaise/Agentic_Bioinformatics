"""Hyper-modular tool & workflow registry, loaded from disk at runtime.

Drop any of these into `tools/` and it is available on the next run:
  tools/<name>/tool.yaml      CLI tool (host binary or container)
  tools/python/<name>.py      Python script with a literal TOOL_SPEC dict
  tools/openapi/<name>.yaml   OpenAPI 3 spec; each operationId becomes a tool
and into `workflows/<id>/workflow.yaml` for Snakemake/Nextflow workflows.

Python TOOL_SPEC is read with `ast.literal_eval`, never imported, so a broken
or hostile script cannot execute code at discovery time.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from .config import is_local_url


class ParamSpec(BaseModel):
    type: Literal["path", "str", "int", "float", "bool", "list"] = "str"
    required: bool = False
    default: Any = None
    description: str = ""


class ToolSpec(BaseModel):
    id: str
    name: str = ""
    kind: Literal["cli", "python", "openapi"]
    description: str
    category: str = "general"
    binary: str | None = None
    container: str | None = None
    conda: str | None = None
    command: list[Any] = Field(default_factory=list)
    inputs: dict[str, ParamSpec] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    qc_parser: str | None = None
    # openapi
    method: str | None = None
    url: str | None = None
    source: str = ""

    def available(self, use_containers: bool = False) -> tuple[bool, str]:
        if self.kind == "python":
            return True, "python"
        if self.kind == "openapi":
            return True, "http"
        if use_containers and self.container and shutil.which("docker"):
            return True, f"docker:{self.container}"
        if self.binary and shutil.which(self.binary):
            return True, f"host:{shutil.which(self.binary)}"
        return False, f"binary '{self.binary}' not on PATH" + (" and no container" if not self.container else "")


class WorkflowSpec(BaseModel):
    id: str
    description: str
    engines: dict[str, str] = Field(default_factory=dict, description="engine -> relative path of entry file")
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    qc_parser: str | None = None
    dir: str = ""


class Registry:
    def __init__(self, tools_dir: str | Path, workflows_dir: str | Path | None = None, mode: str = "local"):
        self.tools_dir = Path(tools_dir)
        self.workflows_dir = Path(workflows_dir) if workflows_dir else None
        self.mode = mode
        self.tools: dict[str, ToolSpec] = {}
        self.workflows: dict[str, WorkflowSpec] = {}
        self.errors: list[str] = []
        self.reload()

    # ------------------------------------------------------------ discovery
    def reload(self) -> None:
        self.tools.clear()
        self.workflows.clear()
        self.errors.clear()
        if self.tools_dir.exists():
            for p in sorted(self.tools_dir.rglob("tool.yaml")):
                self._safe(self._load_cli, p)
            for p in sorted(self.tools_dir.rglob("*.py")):
                if p.name.startswith("_"):
                    continue
                self._safe(self._load_python, p)
            for p in sorted(list(self.tools_dir.rglob("*.openapi.yaml")) + list(self.tools_dir.rglob("*.openapi.json"))):
                self._safe(self._load_openapi, p)
        if self.workflows_dir and self.workflows_dir.exists():
            for p in sorted(self.workflows_dir.glob("*/workflow.yaml")):
                self._safe(self._load_workflow, p)

    def _safe(self, fn, p: Path) -> None:
        try:
            fn(p)
        except Exception as e:  # one bad manifest must not take down the registry
            self.errors.append(f"{p}: {e}")

    def _load_cli(self, p: Path) -> None:
        d = yaml.safe_load(p.read_text())
        d.setdefault("kind", "cli")
        d["source"] = str(p)
        t = ToolSpec.model_validate(d)
        self.tools[t.id] = t

    def _load_python(self, p: Path) -> None:
        tree = ast.parse(p.read_text(), filename=str(p))
        spec = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TOOL_SPEC" for t in node.targets):
                spec = ast.literal_eval(node.value)
        if spec is None:
            return  # helper module, not a tool
        spec = dict(spec)
        spec["kind"] = "python"
        spec["source"] = str(p.resolve())
        spec.setdefault("id", p.stem)
        t = ToolSpec.model_validate(spec)
        self.tools[t.id] = t

    def _load_openapi(self, p: Path) -> None:
        doc = json.loads(p.read_text()) if p.suffix == ".json" else yaml.safe_load(p.read_text())
        servers = doc.get("servers") or [{"url": ""}]
        base = servers[0]["url"].rstrip("/")
        prefix = re.sub(r"\.openapi$", "", p.stem)
        for path, ops in (doc.get("paths") or {}).items():
            for method, op in ops.items():
                if method.lower() not in ("get", "post") or "operationId" not in op:
                    continue
                inputs = {}
                for prm in op.get("parameters", []):
                    sch = prm.get("schema", {})
                    typ = {"integer": "int", "number": "float", "boolean": "bool"}.get(sch.get("type"), "str")
                    inputs[prm["name"]] = ParamSpec(
                        type=typ, required=prm.get("required", False), default=sch.get("default"),
                        description=prm.get("description", ""),
                    )
                tid = f"{prefix}.{op['operationId']}"
                self.tools[tid] = ToolSpec(
                    id=tid, name=op.get("summary", tid), kind="openapi",
                    description=op.get("description") or op.get("summary", ""), category="api",
                    method=method.upper(), url=base + path, inputs=inputs,
                    outputs={"response": "{outdir}/response.json"}, source=str(p),
                )

    def _load_workflow(self, p: Path) -> None:
        d = yaml.safe_load(p.read_text())
        d["dir"] = str(p.parent.resolve())
        w = WorkflowSpec.model_validate(d)
        self.workflows[w.id] = w

    # ------------------------------------------------------------- queries
    def get(self, tool_id: str) -> ToolSpec:
        if tool_id not in self.tools:
            raise KeyError(f"unknown tool '{tool_id}'. Known: {sorted(self.tools)}")
        return self.tools[tool_id]

    def network_allowed(self, tool: ToolSpec) -> bool:
        return tool.kind != "openapi" or self.mode != "local" or is_local_url(tool.url or "")

    def describe_for_llm(self, use_containers: bool = False, categories: list[str] | None = None) -> str:
        lines = ["## Tools (call by id)"]
        for t in sorted(self.tools.values(), key=lambda x: (x.category, x.id)):
            if categories and t.category not in categories:
                continue
            ok, how = t.available(use_containers)
            if t.kind == "openapi" and not self.network_allowed(t):
                ok, how = False, "blocked: external API in air-gapped mode"
            params = ", ".join(
                f"{k}:{v.type}{'*' if v.required else ''}" + (f"={v.default}" if v.default is not None else "")
                for k, v in t.inputs.items()
            )
            lines.append(f"- {t.id} [{t.category}] {'AVAILABLE' if ok else 'UNAVAILABLE (' + how + ')'}: "
                         f"{t.description.strip()} | params: {params} | outputs: {', '.join(t.outputs)}")
        if self.workflows:
            lines.append("\n## Workflows")
            for w in self.workflows.values():
                params = ", ".join(f"{k}:{v.type}{'*' if v.required else ''}" for k, v in w.params.items())
                lines.append(f"- {w.id} engines={list(w.engines)}: {w.description.strip()} | params: {params}")
        lines.append("\n(* = required. Reference task inputs as {input:<name>} and earlier stage outputs as "
                     "{stage:<stage_name>.<output>}.)")
        return "\n".join(lines)

    # -------------------------------------------------------------- render
    def render_command(self, tool: ToolSpec, params: dict[str, Any], outdir: str, threads: int) -> list[str]:
        values = {k: v.default for k, v in tool.inputs.items() if v.default is not None}
        values.update({k: v for k, v in params.items() if v is not None})
        missing = [k for k, v in tool.inputs.items() if v.required and k not in values]
        if missing:
            raise ValueError(f"{tool.id}: missing required params {missing}")
        values["outdir"] = outdir
        values["threads"] = threads
        values["tool_dir"] = str(Path(tool.source).parent.resolve()) if tool.source else ""
        values = {k: (_fmt(v, values) if isinstance(v, str) and "{tool_dir}" in v else v) for k, v in values.items()}
        if tool.kind == "python":
            cmd = [sys.executable, tool.source]
            for k, v in values.items():
                if k in ("threads",) and "threads" not in tool.inputs:
                    continue
                if k not in tool.inputs and k != "outdir" or k == "tool_dir":
                    continue
                if isinstance(v, bool):
                    if v:
                        cmd.append(f"--{k}")
                    continue
                if isinstance(v, (list, tuple)):
                    if not v:
                        continue
                    cmd += [f"--{k}", *[str(x) for x in v]]
                    continue
                cmd += [f"--{k}", str(v)]
            return cmd
        out: list[str] = []
        for item in tool.command:
            if isinstance(item, dict):
                cond = item.get("if")
                if cond and values.get(cond) in (None, "", False):
                    continue
                out += [_fmt(a, values) for a in item.get("args", [])]
            else:
                out.append(_fmt(item, values))
        return out

    def render_outputs(self, outputs: dict[str, str], outdir: str) -> dict[str, str]:
        return {k: v.replace("{outdir}", outdir) for k, v in outputs.items()}


def _fmt(template: str, values: dict) -> str:
    def rep(m):
        k = m.group(1)
        if k not in values:
            raise ValueError(f"template placeholder '{k}' has no value")
        return str(values[k])

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", rep, str(template))

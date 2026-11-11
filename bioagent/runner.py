"""Runtime: wires config, router, registry, compute, memory and the graph, and drives runs.

`drive()` is the suspend/resume loop that joins the two graphs:
  cognitive graph --interrupt(await_job)--> wait for compute job --> resume(JobRecord)
  cognitive graph --interrupt(approval)---> ask a human          --> resume(decision)
Because state is checkpointed in SQLite at every step, `drive()` can be
re-entered after a crash or reboot and continues from the last checkpoint.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .compute import JobManager
from .config import Settings
from .graph.builder import build_graph
from .llm.router import LLMRouter
from .memory import make_memory
from .observability import attach_exporters
from .provenance import BUS, Provenance
from .registry import Registry
from .state import Blackboard, JobRecord, Phase, TaskSpec

ApprovalHandler = Callable[[str, dict], dict]


def auto_approve(run_id: str, payload: dict) -> dict:
    return {"decision": "approved", "reviewer": "auto", "comment": "auto-approved (policy override)"}


def cli_approve(run_id: str, payload: dict) -> dict:
    print("\n=== APPROVAL REQUIRED ===")
    print(f"run {run_id} stage {payload['stage']}: {payload['reason']}")
    print(json.dumps(payload["pipeline"], indent=2)[:4000])
    ans = input("approve? [y/N/comment]: ").strip()
    if ans.lower() in ("y", "yes"):
        return {"decision": "approved", "reviewer": "cli"}
    return {"decision": "rejected", "reviewer": "cli", "comment": ans or "rejected at CLI"}


class Runtime:
    def __init__(self, settings: Settings, router: LLMRouter | None = None):
        self.s = settings
        self.bus = BUS
        self.workspace = settings.resolve(settings.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        state_dir = settings.resolve(settings.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self.registry = Registry(settings.resolve(settings.tools_dir), settings.resolve(settings.workflows_dir), settings.mode)
        self.router = router or LLMRouter(settings.llm)
        self.jobs = JobManager(settings, self.registry, self.bus)
        self.memory = make_memory(settings.memory, settings.resolve(settings.memory.path))
        db = state_dir / "checkpoints.sqlite"
        self._conn = sqlite3.connect(str(db), check_same_thread=False)
        self.checkpointer = SqliteSaver(self._conn)
        try:
            self.checkpointer.setup()
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                f"cannot open checkpoint database {db}: {e}. SQLite WAL mode does not work on many network, FUSE "
                "or cloud-synced folders. Point the state directory at a local disk, e.g. "
                "`export BIOAGENT_STATE_DIR=~/.bioagent/state`."
            ) from e
        self.graph = build_graph(self, self.checkpointer)
        self._provs: dict[str, Provenance] = {}
        self._lock = threading.Lock()
        attach_exporters(settings, self.bus)

    def run_dir(self, run_id: str) -> Path:
        d = self.workspace / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def provenance(self, run_id: str) -> Provenance:
        with self._lock:
            if run_id not in self._provs:
                self._provs[run_id] = Provenance(self.run_dir(run_id), run_id, self.bus)
            return self._provs[run_id]

    def _cfg(self, run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}, "recursion_limit": 10 * self.s.graph.max_cycles}

    # ------------------------------------------------------------------ API
    def start(self, task: TaskSpec | dict, run_id: str | None = None) -> str:
        task = task if isinstance(task, TaskSpec) else TaskSpec.model_validate(task)
        bb = Blackboard(task=task, **({"run_id": run_id} if run_id else {}))
        prov = self.provenance(bb.run_id)
        prov.log("run_started", task=task.model_dump(), mode=self.s.mode,
                 premium=self.router.providers["premium"].name, budget=self.router.providers["budget"].name,
                 compute=self.s.compute.backend, registry_errors=self.registry.errors)
        (self.run_dir(bb.run_id) / "task.json").write_text(task.model_dump_json(indent=2))
        self.graph.invoke({"bb": bb.model_dump(mode="json")}, self._cfg(bb.run_id))
        return bb.run_id

    def snapshot(self, run_id: str) -> Blackboard | None:
        st = self.graph.get_state(self._cfg(run_id))
        if not st or not st.values:
            return None
        return Blackboard.model_validate(st.values["bb"])

    def pending(self, run_id: str) -> list[dict]:
        st = self.graph.get_state(self._cfg(run_id))
        ints = getattr(st, "interrupts", None)
        if ints is None:
            ints = [i for t in st.tasks for i in t.interrupts]
        return [i.value for i in ints]

    def is_finished(self, run_id: str) -> bool:
        st = self.graph.get_state(self._cfg(run_id))
        return bool(st.values) and not st.next

    def resume(self, run_id: str, value) -> None:
        self.graph.invoke(Command(resume=value), self._cfg(run_id))

    def drive(self, run_id: str, approve: ApprovalHandler = cli_approve, job_timeout: float | None = None) -> Blackboard:
        while not self.is_finished(run_id):
            pend = self.pending(run_id)
            if not pend:
                # Crashed mid-node or never started: continue from last checkpoint.
                self.graph.invoke(None, self._cfg(run_id))
                continue
            p = pend[0]
            if p["type"] == "await_job":
                rec: JobRecord = self.jobs.wait(p["job_id"], timeout=job_timeout)
                self.resume(run_id, rec.model_dump(mode="json"))
            elif p["type"] == "approval":
                self.bus.publish(run_id, {"event": "approval_required", **p})
                self.resume(run_id, approve(run_id, p))
            else:
                raise RuntimeError(f"unknown interrupt {p}")
        bb = self.snapshot(run_id)
        (self.run_dir(run_id) / "blackboard.json").write_text(bb.model_dump_json(indent=2))
        self.provenance(run_id).log("run_finished", phase=bb.history[-1].dst.value if bb.history else None,
                                    cost=bb.cost.model_dump())
        self.bus.publish(run_id, {"event": "run_finished", "status": "done" if bb.phase == Phase.REPORT else "failed"})
        return bb

    def run(self, task: TaskSpec | dict, approve: ApprovalHandler = cli_approve) -> Blackboard:
        return self.drive(self.start(task), approve)

    def list_runs(self) -> list[dict]:
        out = []
        for d in sorted(self.workspace.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if (d / "task.json").exists():
                bbp = d / "blackboard.json"
                status = "unknown"
                if bbp.exists():
                    b = json.loads(bbp.read_text())
                    status = "done" if b.get("phase") == "report" else "failed"
                else:
                    status = "running/suspended"
                out.append({"run_id": d.name, "status": status,
                            "goal": json.loads((d / "task.json").read_text()).get("goal")})
        return out

    def close(self) -> None:
        self._conn.close()

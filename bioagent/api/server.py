"""Experience & API layer: REST, WebSocket event streaming, HITL approvals, compute webhooks.

  GET  /                          HITL dashboard
  GET  /api/runs                  list runs
  POST /api/runs                  start a run {task: {...}}  -> {run_id}
  GET  /api/runs/{id}             blackboard snapshot + pending interrupts
  GET  /api/runs/{id}/report      report markdown
  POST /api/runs/{id}/approve     {decision: approved|rejected, comment, reviewer}
  WS   /ws/runs/{id}              provenance backlog, then live events ("thought streaming")
  POST /webhooks/jobs/{job_id}    generic job-finished callback (external schedulers)
  POST /webhooks/nextflow/{job_id}  Nextflow -with-weblog events
  GET  /api/lessons               memory; POST /api/lessons/{id}/status {status}
  GET  /api/tools                 registry
"""
from __future__ import annotations

import asyncio
import threading
import traceback
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse

from ..provenance import Provenance
from ..runner import Runtime
from ..state import TaskSpec

STATIC = Path(__file__).parent / "static"


class ApprovalBroker:
    """Blocks a run's driver thread until a human answers in the dashboard."""

    def __init__(self) -> None:
        self._events: dict[str, threading.Event] = {}
        self._answers: dict[str, dict] = {}
        self.pending: dict[str, dict] = {}

    def __call__(self, run_id: str, payload: dict) -> dict:
        ev = threading.Event()
        self._events[run_id] = ev
        self.pending[run_id] = payload
        ev.wait()
        self.pending.pop(run_id, None)
        return self._answers.pop(run_id)

    def answer(self, run_id: str, decision: dict) -> bool:
        ev = self._events.pop(run_id, None)
        if not ev:
            return False
        self._answers[run_id] = decision
        ev.set()
        return True


def create_app(rt: Runtime) -> FastAPI:
    app = FastAPI(title="bioagent", version="0.1.0")
    broker = ApprovalBroker()
    threads: dict[str, threading.Thread] = {}

    def drive(run_id: str, task: TaskSpec | None) -> None:
        try:
            if task is not None:
                rt.start(task, run_id=run_id)
            rt.drive(run_id, approve=broker)
        except Exception as e:
            rt.provenance(run_id).log("driver_error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-3000:])
            rt.bus.publish(run_id, {"event": "run_finished", "status": "error", "error": str(e)})

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/api/runs")
    def runs():
        out = rt.list_runs()
        for r in out:
            r["live"] = r["run_id"] in threads and threads[r["run_id"]].is_alive()
            r["awaiting_approval"] = r["run_id"] in broker.pending
        return out

    @app.post("/api/runs")
    def start(body: dict = Body(...)):
        task = TaskSpec.model_validate(body.get("task", body))
        from ..state import new_id

        run_id = new_id("run")
        t = threading.Thread(target=drive, args=(run_id, task), daemon=True, name=run_id)
        threads[run_id] = t
        t.start()
        return {"run_id": run_id}

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str):
        if run_id in threads and threads[run_id].is_alive():
            return {"status": "already running"}
        t = threading.Thread(target=drive, args=(run_id, None), daemon=True)
        threads[run_id] = t
        t.start()
        return {"status": "resumed"}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        bb = rt.snapshot(run_id)
        if bb is None:
            raise HTTPException(404, "unknown run (or not yet checkpointed)")
        return {"blackboard": bb.model_dump(mode="json"), "pending": rt.pending(run_id),
                "awaiting_approval": broker.pending.get(run_id)}

    @app.get("/api/runs/{run_id}/report", response_class=PlainTextResponse)
    def get_report(run_id: str):
        p = rt.workspace / run_id / "report.md"
        if not p.exists():
            raise HTTPException(404, "no report yet")
        return p.read_text()

    @app.post("/api/runs/{run_id}/approve")
    def approve(run_id: str, body: dict = Body(...)):
        decision = body.get("decision")
        if decision not in ("approved", "rejected"):
            raise HTTPException(400, "decision must be approved|rejected")
        ok = broker.answer(run_id, {"decision": decision, "comment": body.get("comment"),
                                    "reviewer": body.get("reviewer", "dashboard")})
        if not ok:
            raise HTTPException(409, "run is not waiting for approval")
        return {"ok": True}

    @app.post("/webhooks/jobs/{job_id}")
    def job_webhook(job_id: str, body: dict = Body(default={})):
        # Local/Nextflow head processes report completion themselves; this endpoint lets an
        # external scheduler post status, and is recorded for the audit trail.
        rt.bus.publish("*", {"event": "job_webhook", "job_id": job_id, "payload": body})
        return {"ok": True}

    @app.post("/webhooks/nextflow/{job_id}")
    def nextflow_webhook(job_id: str, body: dict = Body(default={})):
        rt.jobs.handle_nextflow_event(job_id, body)
        return {"ok": True}

    @app.get("/api/lessons")
    def lessons():
        return [l.model_dump() for l in rt.memory.all()]

    @app.post("/api/lessons/{lesson_id}/status")
    def lesson_status(lesson_id: str, body: dict = Body(...)):
        if body.get("status") not in ("validated", "rejected", "candidate"):
            raise HTTPException(400, "status must be validated|rejected|candidate")
        try:
            return rt.memory.set_status(lesson_id, body["status"]).model_dump()
        except KeyError:
            raise HTTPException(404, "unknown lesson")

    @app.get("/api/tools")
    def tools():
        return {"tools": [t.model_dump(include={"id", "kind", "category", "description"}) | {"available": t.available()[0]}
                          for t in rt.registry.tools.values()],
                "workflows": list(rt.registry.workflows), "errors": rt.registry.errors}

    @app.websocket("/ws/runs/{run_id}")
    async def ws(websocket: WebSocket, run_id: str):
        await websocket.accept()
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        unsub = rt.bus.subscribe(run_id, lambda rec: loop.call_soon_threadsafe(q.put_nowait, rec))
        try:
            p = rt.workspace / run_id / "provenance.jsonl"
            if p.exists():
                for rec in Provenance.read(p):
                    await websocket.send_json({k: v for k, v in rec.items() if k not in ("prev", "hash")} | {"backlog": True})
            if run_id in broker.pending:
                await websocket.send_json({"event": "approval_required", **broker.pending[run_id]})
            while True:
                rec = await q.get()
                await websocket.send_json({k: v for k, v in rec.items() if k not in ("prev", "hash")})
        except WebSocketDisconnect:
            pass
        finally:
            unsub()

    return app

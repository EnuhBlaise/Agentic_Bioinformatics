import time

import yaml

from bioagent.memory import LocalMemory
from bioagent.config import MemoryConfig
from bioagent.provenance import Provenance
from bioagent.runner import Runtime, auto_approve
from bioagent.state import Lesson


def _task(d):
    return yaml.safe_load(open(d["task"]))


def test_e2e_batch_effect_is_detected_and_modelled(settings_factory, dataset):
    rt = Runtime(settings_factory())
    bb = rt.run(_task(dataset), approve=auto_approve)
    assert bb.phase.value == "report", bb.errors
    path = [t.dst.value for t in bb.history]
    assert "anomaly" in path and path[-1] == "done"
    assert bb.plan.stages[0].tool_ids == ["confounder_check"]  # ToT picked the QC-first plan
    assert bb.design_adjustments["covariates"] == ["batch"]
    de_call = bb.executed_pipelines["differential_expression"].tool_calls[0]
    assert de_call.params["covariates"] == ["batch"]  # enforced although the designer omitted it
    assert bb.evaluation["differential_expression"]["fdr"] <= 0.1
    assert Provenance.verify(rt.run_dir(bb.run_id))[0]
    assert (rt.run_dir(bb.run_id) / "report.md").exists()
    assert bb.lessons_written and bb.lessons_written[0].status == "candidate"


def test_e2e_fully_confounded_is_reported_not_adjusted(settings_factory, confounded_dataset):
    rt = Runtime(settings_factory())
    bb = rt.run(_task(confounded_dataset), approve=auto_approve)
    assert bb.phase.value == "report", bb.errors
    assert bb.anomalies[0].resolution.action == "accept_with_caveat"
    assert "adjust_design" not in bb.anomalies[0].admissible_actions
    assert not bb.design_adjustments.get("covariates")
    assert any("confounded" in c for c in bb.caveats)
    assert "UNRESOLVED CRITICAL" in (rt.run_dir(bb.run_id) / "report.md").read_text()


def test_inadmissible_resolution_is_rejected(settings_factory, dataset):
    rt = Runtime(settings_factory(llm={"premium": {"provider": "mock", "model": "mock-inadmissible"}}))
    bb = rt.run(_task(dataset), approve=auto_approve)
    assert bb.phase.value == "report", bb.errors
    events = [r["event"] for r in Provenance.read(rt.run_dir(bb.run_id))]
    assert "inadmissible_resolution" in events
    assert "not_a_column" not in bb.design_adjustments.get("covariates", [])


def test_hitl_rejection_then_approval(settings_factory, dataset):
    rt = Runtime(settings_factory(hitl={"approval_policy": "always"}))
    answers = iter([{"decision": "rejected", "comment": "use fewer PCs"}] + [{"decision": "approved"}] * 10)
    bb = rt.run(_task(dataset), approve=lambda run_id, p: next(answers))
    assert bb.phase.value == "report", bb.errors
    assert bb.approvals[0].decision == "rejected" and bb.approvals[1].decision == "approved"
    path = [t.dst.value for t in bb.history]
    assert path[:5] == ["plan", "design", "validate", "approval", "design"]


def test_resume_after_restart(settings_factory, dataset):
    s = settings_factory(hitl={"approval_policy": "always"})
    rt1 = Runtime(s)
    run_id = rt1.start(_task(dataset))           # runs until the first approval interrupt
    assert rt1.pending(run_id)[0]["type"] == "approval"
    rt1.close()                                   # "process exits"
    rt2 = Runtime(s)                              # new process, same SQLite checkpoint
    bb = rt2.drive(run_id, approve=auto_approve)
    assert bb.phase.value == "report", bb.errors
    assert Provenance.verify(rt2.run_dir(run_id))[0]


def test_memory_dedup_and_auto_validation(tmp_path):
    m = LocalMemory(MemoryConfig(), tmp_path)
    text = "Run a PCA covariate association check before differential expression testing."
    for run in ("r1", "r2", "r3"):
        les = m.add(Lesson(text=text, source_run=run))
    assert len(m.all()) == 1 and les.status == "validated"
    m.add(Lesson(text="Check strandedness before featureCounts.", source_run="r1"))
    hits = m.search("differential expression PCA batch", 1)
    assert hits and "PCA" in hits[0].text


def test_api_approval_flow(settings_factory, dataset):
    from fastapi.testclient import TestClient
    from bioagent.api.server import create_app

    rt = Runtime(settings_factory(hitl={"approval_policy": "always"}))
    c = TestClient(create_app(rt))
    run_id = c.post("/api/runs", json={"task": _task(dataset)}).json()["run_id"]
    for _ in range(600):
        runs = {r["run_id"]: r for r in c.get("/api/runs").json()}
        r = runs.get(run_id)
        if r and r["awaiting_approval"]:
            assert c.post(f"/api/runs/{run_id}/approve", json={"decision": "approved"}).status_code == 200
        if r and r["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert runs[run_id]["status"] == "done"
    assert "Analysis report" in c.get(f"/api/runs/{run_id}/report").text
    assert c.get("/").status_code == 200

import json

import pytest

from bioagent import transitions
from bioagent.config import Settings
from bioagent.llm.router import BudgetExceeded, LLMRouter, TaskClass, extract_json
from bioagent.provenance import Provenance
from bioagent.registry import Registry
from bioagent.sandbox import run_command, static_check
from bioagent.state import CostLedger, JudgmentVerdict, Phase, QCFinding

from conftest import ROOT


def test_transition_table():
    transitions.check(Phase.QC, Phase.ANOMALY)
    transitions.check(Phase.PLAN, Phase.FAILED)
    with pytest.raises(transitions.InadmissibleTransition):
        transitions.check(Phase.PLAN, Phase.EXECUTE)  # cannot skip design/validation
    with pytest.raises(transitions.InadmissibleTransition):
        transitions.check(Phase.DONE, Phase.PLAN)


def test_semantic_admissibility_blocks_adjusting_aliased_batch():
    f = [QCFinding(check="batch_effect", flagged=True, evidence="", metrics={"technical_covariates": ["batch"]}),
         QCFinding(check="condition_batch_confounding", flagged=True, evidence="",
                   metrics={"covariate": "batch", "fully_confounded": True})]
    allowed = transitions.admissible_resolutions(f)
    assert "adjust_design" not in allowed and "accept_with_caveat" in allowed and "exclude_samples" not in allowed


def test_provenance_chain_and_tamper(tmp_path):
    p = Provenance(tmp_path, "r1")
    for i in range(5):
        p.log("e", i=i)
    assert Provenance.verify(tmp_path) == (True, "ok")
    p2 = Provenance(tmp_path, "r1")  # resume continues the chain
    p2.log("after_restart")
    assert Provenance.verify(tmp_path)[0]
    lines = (tmp_path / "provenance.jsonl").read_text().splitlines()
    rec = json.loads(lines[2]); rec["i"] = 99; lines[2] = json.dumps(rec)
    (tmp_path / "provenance.jsonl").write_text("\n".join(lines) + "\n")
    ok, msg = Provenance.verify(tmp_path)
    assert not ok and "modified" in msg


def test_airgap_rejects_hosted_llm():
    with pytest.raises(ValueError, match="air-gapped"):
        Settings.model_validate({"mode": "local", "llm": {
            "premium": {"provider": "anthropic", "model": "x"}, "budget": {"provider": "ollama", "model": "y"}}})
    with pytest.raises(ValueError):
        Settings.model_validate({"mode": "local", "llm": {
            "premium": {"provider": "ollama", "model": "x", "base_url": "https://example.com"},
            "budget": {"provider": "mock", "model": "y"}}})
    Settings.model_validate({"mode": "cloud", "llm": {
        "premium": {"provider": "anthropic", "model": "x"}, "budget": {"provider": "ollama", "model": "y"}}})


def test_registry_discovery_and_rendering(tmp_path):
    reg = Registry(ROOT / "tools", ROOT / "workflows", mode="local")
    assert not reg.errors
    assert {"fastp", "spades", "confounder_check", "differential_expression", "ncbi_eutils.esearch"} <= set(reg.tools)
    assert {"rnaseq_fastq_to_counts", "bacterial_assembly"} <= set(reg.workflows)
    fp = reg.get("fastp")
    se = reg.render_command(fp, {"reads_r1": "a.fq.gz"}, "/o", 4)
    pe = reg.render_command(fp, {"reads_r1": "a.fq.gz", "reads_r2": "b.fq.gz"}, "/o", 4)
    assert "-I" not in se and pe[pe.index("-I") + 1] == "b.fq.gz"
    de = reg.render_command(reg.get("differential_expression"),
                            {"counts": "c", "metadata": "m", "condition": "x", "contrast": ["t", "c"], "covariates": ["batch"]}, "/o", 2)
    assert de[de.index("--contrast") + 1: de.index("--contrast") + 3] == ["t", "c"] and "--covariates" in de
    assert not reg.network_allowed(reg.get("ncbi_eutils.esearch"))  # external API blocked when air-gapped
    assert "UNAVAILABLE" in reg.describe_for_llm()  # bio binaries are not installed in CI
    # drop-in: a new python tool appears on reload
    (tmp_path / "x.py").write_text('TOOL_SPEC = {"id": "newtool", "description": "d", "inputs": {}}\nraise SystemExit')
    reg2 = Registry(tmp_path)
    assert "newtool" in reg2.tools  # discovered without executing the module


def test_sandbox_static_and_limits(tmp_path):
    assert any("socket" in m for m in static_check("python", "import socket"))
    assert any("subprocess" in m for m in static_check("python", "import subprocess"))
    assert static_check("python", "import pandas as pd\nprint(1)") == []
    assert static_check("bash", "rm -rf / ") and static_check("bash", "curl http://x")
    r = run_command(["python3", "-c", "import time; time.sleep(30)"], tmp_path, tmp_path / "log", timeout_s=1)
    assert r.timed_out and r.exit_code != 0


def test_extract_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}


def test_router_repair_and_budget(settings_factory, tmp_path):
    s = settings_factory(llm={"premium": {"provider": "mock", "model": "mock-bad-json"},
                              "budget": {"provider": "mock", "model": "mock"}})
    router = LLMRouter(s.llm)
    ledger = CostLedger()
    prov = Provenance(tmp_path, "r")
    v = router.call(TaskClass.JUDGMENT, "sys", "q", JudgmentVerdict, ledger, prov, ctx={"rule_answer": True})
    assert v.answer is True and ledger.schema_failures == 1  # malformed first, repaired second
    s2 = settings_factory(llm={"premium": {"provider": "mock", "model": "mock", "cost_in_per_mtok": 1e6},
                               "budget": {"provider": "mock", "model": "mock"}, "run_budget_usd": 0.01})
    r2 = LLMRouter(s2.llm)
    led = CostLedger()
    r2.call(TaskClass.JUDGMENT, "s", "q", JudgmentVerdict, led, ctx={"rule_answer": True})
    assert led.usd > 0.01
    assert r2.tier_for(TaskClass.PLANNING, led) == "budget"  # downgraded
    with pytest.raises(BudgetExceeded):
        r2.tier_for(TaskClass.JUDGMENT, led)  # judgment is never silently downgraded

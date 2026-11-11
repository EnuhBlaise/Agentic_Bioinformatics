import json
import subprocess
import sys

from bioagent.demo import evaluate_de, make_dataset
from bioagent.qc import evaluate

from conftest import ROOT

CC = str(ROOT / "tools" / "python" / "confounder_check.py")
DE = str(ROOT / "tools" / "python" / "differential_expression.py")


def _cc(d, out):
    subprocess.run([sys.executable, CC, "--counts", d["counts"], "--metadata", d["metadata"], "--condition", "condition",
                    "--outdir", str(out)], check=True, capture_output=True)
    return json.loads((out / "report.json").read_text())


def test_batch_detected_not_aliased(dataset, tmp_path):
    r = _cc(dataset, tmp_path / "cc")
    assert r["batch_effect"]["flagged"] and [h["covariate"] for h in r["batch_effect"]["hits"]] == ["batch"]
    assert not r["confounding"][0]["fully_confounded"]
    assert r["recommended_design"] == "~ batch + condition"
    f = {x.check: x for x in evaluate("confounder_check", {"report": str(tmp_path / "cc" / "report.json")})}
    assert f["batch_effect"].flagged and f["batch_effect"].severity == "critical"


def test_fully_confounded_detected(confounded_dataset, tmp_path):
    r = _cc(confounded_dataset, tmp_path / "cc")
    assert r["confounding"][0]["fully_confounded"]
    assert r["recommended_design"] == "~ condition"


def test_null_data_not_flagged(tmp_path):
    flagged = 0
    for seed in range(8):
        d = make_dataset(tmp_path / f"n{seed}", n_genes=800, batch_frac=0.0, seed=seed)
        flagged += _cc(d, tmp_path / f"n{seed}" / "cc")["batch_effect"]["flagged"]
    assert flagged == 0


def test_de_with_and_without_batch(dataset, tmp_path):
    res = {}
    for name, cov in (("naive", []), ("adjusted", ["batch"])):
        out = tmp_path / name
        subprocess.run([sys.executable, DE, "--counts", dataset["counts"], "--metadata", dataset["metadata"],
                        "--condition", "condition", "--contrast", "treated", "control", "--method", "ebayes",
                        "--outdir", str(out), *(["--covariates", *cov] if cov else [])], check=True, capture_output=True)
        res[name] = evaluate_de(str(out / "de_results.tsv"), dataset["truth"])
    assert res["adjusted"]["recall"] >= res["naive"]["recall"]
    assert res["adjusted"]["fdr"] <= 0.1


def test_de_refuses_aliased_design(confounded_dataset, tmp_path):
    p = subprocess.run([sys.executable, DE, "--counts", confounded_dataset["counts"], "--metadata",
                        confounded_dataset["metadata"], "--condition", "condition", "--contrast", "treated", "control",
                        "--covariates", "batch", "--outdir", str(tmp_path / "x")], capture_output=True, text=True)
    assert p.returncode != 0 and "rank deficient" in (p.stderr + p.stdout)

"""Deterministic QC evaluators: tool outputs -> QCFindings.

These are the "rule" half of each judgment. They are intentionally simple and
transparent; every threshold is in QC_THRESHOLDS so it can be varied in
experiments. A tool without an evaluator yields an explicit "no_qc_parser"
finding, so the judgment layer knows it has no deterministic evidence.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from .state import QCFinding

QC_THRESHOLDS = {
    "fastp_q30_min": 0.80,
    "fastp_pass_frac_min": 0.70,
    "fastp_dup_max": 0.50,
    "star_unique_min_pct": 60.0,
    "fc_assigned_min": 0.50,
    "quast_n50_min": 10_000,
    "quast_contigs_max": 500,
    "quast_size_tolerance": 0.20,
    "de_pi0_min": 0.30,
    "libsize_ratio_max": 5.0,
}

Evaluator = Callable[[dict[str, str], dict], list[QCFinding]]
EVALUATORS: dict[str, Evaluator] = {}


def evaluator(name: str):
    def deco(fn: Evaluator) -> Evaluator:
        EVALUATORS[name] = fn
        return fn

    return deco


def evaluate(parser: str | None, outputs: dict[str, str], params: dict | None = None) -> list[QCFinding]:
    if not parser or parser not in EVALUATORS:
        return [QCFinding(check="no_qc_parser", flagged=False, severity="info",
                          evidence=f"no deterministic QC evaluator for '{parser}'; judgment relies on LLM reading of outputs")]
    try:
        return EVALUATORS[parser](outputs, params or {})
    except Exception as e:
        return [QCFinding(check=f"{parser}_parse_error", flagged=True, severity="warning", evidence=f"{type(e).__name__}: {e}")]


@evaluator("confounder_check")
def _confounders(outputs, params):
    r = json.loads(Path(outputs["report"]).read_text())
    out: list[QCFinding] = []
    be = r["batch_effect"]
    cond_assoc = r["associations"][r["condition"]]["weighted_r2"]
    out.append(QCFinding(
        check="batch_effect", flagged=be["flagged"], severity="critical" if be["flagged"] else "info",
        evidence=("technical covariates associated with major PCs: " +
                  ", ".join(f"{h['covariate']} (weighted R2={h['weighted_r2']}, strong PCs={h['strong_pcs']})" for h in be["hits"])
                  if be["flagged"] else "no technical covariate strongly associated with top PCs")
                 + f"; condition weighted R2={cond_assoc}; PC variance={r['pca_variance_explained']}",
        metrics={"hits": be["hits"], "technical_covariates": [h["covariate"] for h in be["hits"]],
                 "condition_weighted_r2": cond_assoc, "recommended_design": r.get("recommended_design")},
    ))
    for c in r["confounding"]:
        flagged = c["fully_confounded"] or c["cramers_v"] >= 0.5
        out.append(QCFinding(
            check="condition_batch_confounding", flagged=flagged,
            severity="critical" if c["fully_confounded"] else ("warning" if flagged else "info"),
            evidence=f"{c['covariate']} vs condition: Cramer's V={c['cramers_v']}, fully_confounded={c['fully_confounded']}, "
                     f"table={c['table']}",
            metrics={"covariate": c["covariate"], "cramers_v": c["cramers_v"], "fully_confounded": c["fully_confounded"]},
        ))
    out.append(QCFinding(
        check="outlier_samples", flagged=bool(r["outliers"]), severity="warning" if r["outliers"] else "info",
        evidence=f"median inter-sample correlation robust z < -3.5: {r['outliers']}" if r["outliers"] else "no PCA outliers",
        metrics={"samples": [o["sample"] for o in r["outliers"]]},
    ))
    ratio = r["library_size"]["ratio"]
    out.append(QCFinding(
        check="library_size", flagged=ratio > QC_THRESHOLDS["libsize_ratio_max"],
        severity="warning" if ratio > QC_THRESHOLDS["libsize_ratio_max"] else "info",
        evidence=f"library size max/min ratio {ratio} (min {r['library_size']['min']}, max {r['library_size']['max']})",
        metrics=r["library_size"],
    ))
    return out


@evaluator("differential_expression")
def _de(outputs, params):
    s = json.loads(Path(outputs["summary"]).read_text())
    out = [QCFinding(check="de_summary", flagged=False, severity="info",
                     evidence=f"{s['n_significant']} of {s['n_tested']} genes padj<{s['alpha']} "
                              f"({s['n_up']} up, {s['n_down']} down), design {s['design']}, method {s['method']}",
                     metrics={k: s[k] for k in ("n_significant", "n_tested", "n_up", "n_down", "design", "method")})]
    pi0 = s.get("pi0")
    if pi0 is not None and pi0 < QC_THRESHOLDS["de_pi0_min"]:
        out.append(QCFinding(check="pvalue_distribution", flagged=True, severity="warning",
                             evidence=f"estimated null fraction pi0={pi0:.2f}: implausibly many genes affected; "
                                      "suspect unmodelled confounding or normalization failure",
                             metrics={"pi0": pi0, "hist": s.get("pvalue_hist")}))
    if s["n_significant"] == 0:
        out.append(QCFinding(check="no_de_genes", flagged=True, severity="warning",
                             evidence="no genes significant: underpowered, wrong contrast, or no effect",
                             metrics={"n_samples": s["n_samples"]}))
    return out


@evaluator("fastp")
def _fastp(outputs, params):
    d = json.loads(Path(outputs["json"]).read_text())
    after = d["summary"]["after_filtering"]
    before = d["summary"]["before_filtering"]
    q30 = after.get("q30_rate", 0)
    passed = after["total_reads"] / max(before["total_reads"], 1)
    dup = d.get("duplication", {}).get("rate", 0)
    t = QC_THRESHOLDS
    return [
        QCFinding(check="read_quality", flagged=q30 < t["fastp_q30_min"], severity="warning" if q30 < t["fastp_q30_min"] else "info",
                  evidence=f"Q30 rate after filtering {q30:.3f}", metrics={"q30_rate": q30}),
        QCFinding(check="reads_passing", flagged=passed < t["fastp_pass_frac_min"],
                  severity="critical" if passed < t["fastp_pass_frac_min"] else "info",
                  evidence=f"{passed:.1%} of reads passed filtering", metrics={"pass_fraction": passed}),
        QCFinding(check="duplication", flagged=dup > t["fastp_dup_max"], severity="warning" if dup > t["fastp_dup_max"] else "info",
                  evidence=f"duplication rate {dup:.3f}", metrics={"duplication": dup}),
    ]


@evaluator("star")
def _star(outputs, params):
    txt = Path(outputs["log"]).read_text()
    m = re.search(r"Uniquely mapped reads %\s*\|\s*([\d.]+)%", txt)
    pct = float(m.group(1)) if m else 0.0
    bad = pct < QC_THRESHOLDS["star_unique_min_pct"]
    return [QCFinding(check="mapping_rate", flagged=bad, severity="critical" if bad else "info",
                      evidence=f"uniquely mapped {pct}% (wrong genome/index, rRNA or adapter contamination if low)",
                      metrics={"unique_pct": pct})]


@evaluator("featurecounts")
def _fc(outputs, params):
    lines = Path(outputs["summary"]).read_text().splitlines()
    rows = {l.split("\t")[0]: [int(x) for x in l.split("\t")[1:]] for l in lines[1:] if l.strip()}
    assigned = sum(rows.get("Assigned", [0]))
    total = sum(sum(v) for v in rows.values()) or 1
    frac = assigned / total
    bad = frac < QC_THRESHOLDS["fc_assigned_min"]
    return [QCFinding(check="assignment_rate", flagged=bad, severity="critical" if bad else "info",
                      evidence=f"{frac:.1%} of reads assigned to genes (check strandedness / annotation if low)",
                      metrics={"assigned_fraction": frac})]


def _parse_quast(path: str) -> dict:
    d = {}
    for line in Path(path).read_text().splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            d[k.strip()] = v.strip()
    return d


@evaluator("quast")
def _quast(outputs, params):
    q = _parse_quast(outputs["report"])
    n50 = int(q.get("N50", 0))
    contigs = int(q.get("# contigs", q.get("# contigs (>= 0 bp)", 0)))
    total = int(q.get("Total length", 0))
    t = QC_THRESHOLDS
    out = [
        QCFinding(check="contiguity", flagged=n50 < t["quast_n50_min"], severity="warning" if n50 < t["quast_n50_min"] else "info",
                  evidence=f"N50={n50}", metrics={"N50": n50}),
        QCFinding(check="fragmentation", flagged=contigs > t["quast_contigs_max"],
                  severity="warning" if contigs > t["quast_contigs_max"] else "info",
                  evidence=f"{contigs} contigs", metrics={"contigs": contigs}),
    ]
    exp = params.get("expected_genome_size")
    if exp:
        dev = abs(total - int(exp)) / int(exp)
        out.append(QCFinding(check="genome_size", flagged=dev > t["quast_size_tolerance"],
                             severity="critical" if dev > t["quast_size_tolerance"] else "info",
                             evidence=f"assembly {total} bp vs expected {exp} bp ({dev:.0%} deviation; contamination if much larger)",
                             metrics={"total_length": total, "expected": int(exp), "deviation": dev}))
    return out


@evaluator("assembly_workflow")
def _assembly_wf(outputs, params):
    return _fastp({"json": outputs["fastp_json"]}, params) + _quast({"report": outputs["quast_report"]}, params)


@evaluator("rnaseq_workflow")
def _rnaseq_wf(outputs, params):
    qc = Path(outputs.get("multiqc_inputs", ""))
    out: list[QCFinding] = []
    for f in sorted(qc.glob("*.Log.final.out")):
        for x in _star({"log": str(f)}, params):
            x.check = f"{f.name.split('.')[0]}:{x.check}"
            out.append(x)
    for f in sorted(qc.glob("featurecounts.txt.summary")):
        out += _fc({"summary": str(f)}, params)
    return out or [QCFinding(check="no_qc_files", flagged=True, severity="warning", evidence=f"no QC files under {qc}")]

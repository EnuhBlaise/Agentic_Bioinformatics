"""Final report: Markdown (canonical), PDF (reportlab), and machine-readable JSON."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .provenance import Provenance
from .state import Blackboard


def _md(bb: Blackboard, run_dir: Path) -> str:
    L: list[str] = []
    status = "COMPLETED" if bb.next_phase and bb.next_phase.value == "done" else "FAILED"
    critical = [a for a in bb.anomalies if a.resolution and a.resolution.action == "accept_with_caveat"
                and any(f.severity == "critical" for f in a.findings)]
    if status == "COMPLETED" and critical:
        status = (f"COMPLETED WITH {len(critical)} UNRESOLVED CRITICAL ANOMAL{'Y' if len(critical) == 1 else 'IES'} "
                  "— results below may not support the stated goal; read Caveats first")
    L += [f"# Analysis report — {bb.run_id}", "", f"**Status:** {status}  ", f"**Goal:** {bb.task.goal}  ",
          f"**Analysis type:** {bb.task.analysis_type}  ",
          f"**LLM spend:** ${bb.cost.usd:.4f} ({bb.cost.tokens_in} in / {bb.cost.tokens_out} out tokens; "
          f"calls by tier {bb.cost.calls_by_tier}; escalations {bb.cost.escalations}; schema failures {bb.cost.schema_failures})", ""]
    if bb.errors:
        L += ["## Errors", *[f"- {e}" for e in bb.errors], ""]
    if bb.interpretation:
        it = bb.interpretation
        L += ["## Summary", it.summary, ""]
        if it.findings:
            L += ["## Findings", "| Claim | Support | Confidence |", "|---|---|---|",
                  *[f"| {f.claim} | {f.support} | {f.confidence} |" for f in it.findings], ""]
        if it.hypothesis_outcomes:
            L += ["## Hypothesis outcomes", *[f"- {h}" for h in it.hypothesis_outcomes], ""]
    caveats = list(dict.fromkeys(bb.caveats + (bb.interpretation.caveats if bb.interpretation else [])))
    if caveats:
        L += ["## Caveats and limitations", *[f"- {c}" for c in caveats], ""]
    if bb.evaluation:
        L += ["## Evaluation against ground truth (benchmark only)", "| Stage | Called | TP | FP | FDR | Recall |",
              "|---|---|---|---|---|---|",
              *[f"| {s} | {e['called']} | {e['tp']} | {e['fp']} | {e['fdr']} | {e['recall']} |" for s, e in bb.evaluation.items()], ""]
    if bb.hypotheses:
        L += ["## Hypotheses", *[f"- **{h.statement}** — {h.rationale} (test: {h.testable_by})" for h in bb.hypotheses], ""]
    if bb.candidate_plans:
        L += ["## Planning (Tree of Thoughts)", "| Branch | Plan | Score | Selected |", "|---|---|---|---|"]
        for i, p in enumerate(bb.candidate_plans):
            sel = "yes" if bb.plan and p.plan_id == bb.plan.plan_id else ""
            L.append(f"| {i + 1} | {p.title} | {p.score} | {sel} |")
        if bb.plan:
            L += ["", f"**Selected plan:** {bb.plan.title} — {bb.plan.rationale}", ""]
            L += [f"{i + 1}. **{s.name}** — {s.objective} (tools: {s.tool_ids or s.workflow_id})" for i, s in enumerate(bb.plan.stages)]
        L.append("")
    if bb.judgments:
        L += ["## Judgments", "| Question | Rule | LLM | Final | Agree |", "|---|---|---|---|---|"]
        for j in bb.judgments:
            L.append(f"| {j.question} | {j.rule_answer} | {j.llm_verdict.answer if j.llm_verdict else ''} | "
                     f"**{j.final_answer}** | {j.agreement} |")
        L.append("")
    if bb.anomalies:
        L += ["## Anomalies"]
        for a in bb.anomalies:
            r = a.resolution
            L += [f"- **{a.stage}**: {', '.join(f.check for f in a.findings)} → admissible {a.admissible_actions} → "
                  f"**{r.action if r else 'unresolved'}**" + (f" (covariates {r.covariates_to_add})" if r and r.covariates_to_add else ""),
                  *([f"  - {r.reasoning}"] if r else [])]
        L.append("")
    if bb.qc_findings:
        L += ["## QC findings"]
        for stage, fs in bb.qc_findings.items():
            L.append(f"**{stage}**")
            L += [f"- [{f.severity}{' ⚑' if f.flagged else ''}] {f.check}: {f.evidence}" for f in fs]
        L.append("")
    for stage, outs in bb.stage_outputs.items():
        res = outs.get("differential_expression.results") or outs.get("results")
        if res and Path(res).exists() and res.endswith(".tsv"):
            df = pd.read_csv(res, sep="\t").head(20)
            L += [f"## Top results — {stage}", df.to_markdown(index=False, floatfmt=".3g") if _has_tabulate() else df.to_string(index=False), ""]
    if bb.approvals:
        L += ["## Human approvals", *[f"- {a.decision} by {a.reviewer}: {a.reason} {a.comment or ''}" for a in bb.approvals], ""]
    if bb.jobs:
        L += ["## Compute jobs", "| Job | Stage | Backend | Status | Workdir |", "|---|---|---|---|---|",
              *[f"| {j.job_id} | {j.stage} | {j.backend} | {j.status} | `{j.workdir}` |" for j in bb.jobs], ""]
    if bb.retrieved_lessons or bb.lessons_written:
        L += ["## Memory", "Retrieved:", *([f"- [{l.status}] {l.text}" for l in bb.retrieved_lessons] or ["- none"]),
              "Written:", *([f"- [{l.status}] {l.text} (evidence: {l.evidence})" for l in bb.lessons_written] or ["- none"]), ""]
    if bb.interpretation and bb.interpretation.next_experiments:
        L += ["## Suggested next experiments", *[f"- {x}" for x in bb.interpretation.next_experiments], ""]
    ok, msg = Provenance.verify(run_dir)
    L += ["## Provenance", f"- Path through graph: {' → '.join([bb.history[0].src.value] + [t.dst.value for t in bb.history]) if bb.history else ''}",
          f"- Provenance log `provenance.jsonl` hash chain: **{msg}**", f"- Full LLM transcripts: `llm_calls.jsonl`", ""]
    return "\n".join(L)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def _pdf(md: str, path: Path) -> bool:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer
    except ImportError:
        return False
    from xml.sax.saxutils import escape
    import re

    ss = getSampleStyleSheet()
    flow = []
    table_buf: list[str] = []

    def flush():
        if table_buf:
            flow.append(Preformatted("\n".join(table_buf), ss["Code"], maxLineLength=110))
            table_buf.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            table_buf.append(line)
            continue
        flush()
        txt = escape(line)
        txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt)
        txt = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", txt)
        if line.startswith("# "):
            flow.append(Paragraph(txt[2:], ss["Title"]))
        elif line.startswith("## "):
            flow.append(Paragraph(txt[3:], ss["Heading2"]))
        elif line.strip():
            flow.append(Paragraph(txt, ss["BodyText"]))
        else:
            flow.append(Spacer(1, 4))
    flush()
    SimpleDocTemplate(str(path), pagesize=A4, title="bioagent report").build(flow)
    return True


def write_report(bb: Blackboard, run_dir: Path, registry=None) -> dict[str, str]:
    run_dir = Path(run_dir)
    md = _md(bb, run_dir)
    paths = {"markdown": str(run_dir / "report.md"), "json": str(run_dir / "blackboard.json")}
    Path(paths["markdown"]).write_text(md)
    Path(paths["json"]).write_text(bb.model_dump_json(indent=2))
    if _pdf(md, run_dir / "report.pdf"):
        paths["pdf"] = str(run_dir / "report.pdf")
    return paths

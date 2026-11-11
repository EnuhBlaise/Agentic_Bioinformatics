"""Cognitive-graph nodes.

Every node has the signature `fn(rt, bb, prov) -> None`: it mutates the typed
Blackboard and sets `bb.next_phase`. `wrap()` turns it into a LangGraph node
that validates state on entry/exit, enforces admissible transitions and the
cycle cap, and writes provenance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from .. import qc as qcmod
from .. import transitions
from ..compute import MaterializeError
from ..llm.router import TaskClass
from ..sandbox import static_check
from ..state import (
    Anomaly, AnomalyResolution, ApprovalRecord, Blackboard, HypothesisSet, Interpretation, JobRecord,
    JudgmentRecord, JudgmentVerdict, Lesson, LessonSet, Phase, PipelineSpec, Plan, PlanCandidate, PlanScore,
    Transition, ValidationIssue,
)
from . import prompts

if TYPE_CHECKING:
    from ..runner import Runtime

P = Phase
NodeFn = Callable[["Runtime", Blackboard, Any], None]


def wrap(rt: "Runtime", phase: Phase, fn: NodeFn):
    def node(state: dict) -> dict:
        bb = Blackboard.model_validate(state["bb"])
        prov = rt.provenance(bb.run_id)
        bb.phase = phase
        bb.next_phase = None
        bb.cycles += 1
        rt.bus.publish(bb.run_id, {"event": "node_enter", "node": phase.value, "cycle": bb.cycles})
        if bb.cycles > rt.s.graph.max_cycles:
            bb.errors.append(f"cycle cap {rt.s.graph.max_cycles} exceeded")
            bb.next_phase = P.FAILED
        else:
            try:
                fn(rt, bb, prov)
            except GraphBubbleUp:
                raise  # LangGraph interrupt: suspend, state is checkpointed
            except Exception as e:  # node crash -> recorded, run fails cleanly
                bb.errors.append(f"{phase.value}: {type(e).__name__}: {e}")
                prov.log("node_error", node=phase.value, error=f"{type(e).__name__}: {e}")
                bb.next_phase = P.FAILED
        if phase not in (P.FAILED,):
            nxt = bb.next_phase or P.FAILED
            try:
                transitions.check(phase, nxt)
            except transitions.InadmissibleTransition as e:
                bb.errors.append(str(e))
                prov.log("inadmissible_transition", src=phase.value, dst=nxt.value)
                nxt = P.FAILED
            bb.next_phase = nxt
            bb.history.append(Transition(src=phase, dst=nxt, node=phase.value))
            prov.log("transition", src=phase.value, dst=nxt.value, cycle=bb.cycles, stage_index=bb.stage_index,
                     cost_usd=bb.cost.usd)
        out = bb.model_dump(mode="json")
        Blackboard.model_validate(out)  # exit validation
        return {"bb": out}

    node.__name__ = phase.value
    return node


# ------------------------------------------------------------------ helpers
def _task_text(bb: Blackboard) -> str:
    t = bb.task.model_dump(exclude={"ground_truth"})
    return json.dumps(t, indent=2)


def _metadata_preview(bb: Blackboard) -> str:
    path = bb.task.inputs.get("metadata")
    if not path or not Path(path).exists():
        return "(no sample metadata provided)"
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
        lines = [f"columns: {list(df.columns)}; n_samples={len(df)}"]
        for c in df.columns[1:]:
            vc = df[c].value_counts()
            lines.append(f"  {c}: {vc.to_dict() if len(vc) <= 12 else f'{len(vc)} distinct values'}")
        return "\n".join(lines)
    except Exception as e:
        return f"(metadata unreadable: {e})"


def _lessons_text(bb: Blackboard) -> str:
    if not bb.retrieved_lessons:
        return "(none)"
    return "\n".join(f"- [{l.status}{', unverified' if l.status == 'candidate' else ''}] {l.text}"
                     for l in bb.retrieved_lessons)


def _stage_outputs_text(bb: Blackboard) -> str:
    if not bb.stage_outputs:
        return "(no previous stages)"
    return "\n".join(f"- {stage}: {sorted(outs)}" for stage, outs in bb.stage_outputs.items())


def _judge(rt, bb: Blackboard, prov, question: str, system: str, evidence: str, rule_answer: bool | None,
           rule_basis: str | None, ctx: dict) -> JudgmentRecord:
    prompt = f"QUESTION: {question}\n\nTASK:\n{_task_text(bb)}\n\nEVIDENCE:\n{evidence}\n\n" \
             f"DETERMINISTIC RULE ANSWER: {rule_answer} ({rule_basis})"
    verdict = rt.router.call(TaskClass.JUDGMENT, system, prompt, JudgmentVerdict, bb.cost, prov,
                             ctx={**ctx, "rule_answer": rule_answer}, node=bb.phase.value)
    mode = rt.s.graph.judgment_combination
    if rule_answer is None or mode == "llm":
        final = verdict.answer
    elif mode == "rule":
        final = rule_answer
    else:  # conservative: for QC 'yes' = problem; for validation 'yes' = OK -> handled by caller via ctx
        final = (verdict.answer or rule_answer) if ctx.get("yes_means_problem", True) else (verdict.answer and rule_answer)
    rec = JudgmentRecord(question=question, stage=bb.current_stage.name if bb.current_stage else None,
                         rule_answer=rule_answer, rule_basis=rule_basis, llm_verdict=verdict, final_answer=final,
                         combination=mode, agreement=None if rule_answer is None else verdict.answer == rule_answer,
                         model=rt.router.providers["premium"].name)
    bb.judgments.append(rec)
    prov.log("judgment", question=question, rule=rule_answer, llm=verdict.answer, final=final,
             agreement=rec.agreement, confidence=verdict.confidence)
    rt.bus.publish(bb.run_id, {"event": "thought", "node": bb.phase.value, "kind": "judgment",
                               "text": f"{question} -> {'YES' if final else 'NO'}: {verdict.reasoning}"})
    return rec


def _advance_stage(bb: Blackboard) -> None:
    bb.stage_index += 1
    bb.design_revisions = 0
    bb.job_retries = 0
    bb.design_feedback = []
    bb.pipeline = None
    bb.next_phase = P.DESIGN if bb.current_stage else P.INTERPRET


# ------------------------------------------------------------------- nodes
def retrieve(rt, bb, prov):
    q = f"{bb.task.analysis_type} {bb.task.goal} {' '.join(bb.task.constraints)}"
    bb.retrieved_lessons = rt.memory.search(q, rt.s.memory.top_k, include_candidates=rt.s.memory.inject_candidates)
    prov.log("memory_retrieved", lessons=[(l.lesson_id, l.status, l.score) for l in bb.retrieved_lessons])
    bb.next_phase = P.PLAN


TOT_STYLES = [
    "the most standard, community-accepted workflow for this analysis type",
    "the most rigorous approach to technical confounders and data quality",
    "the minimal-compute approach that still answers the question",
    "an approach that maximises interpretability of the final result",
]


def plan(rt, bb, prov):
    reg_text = rt.registry.describe_for_llm(rt.s.compute.use_containers)
    base = (f"TASK:\n{_task_text(bb)}\n\nSAMPLE METADATA:\n{_metadata_preview(bb)}\n\n"
            f"LESSONS FROM PREVIOUS RUNS:\n{_lessons_text(bb)}\n\nREGISTRY:\n{reg_text}")
    hs = rt.router.call(TaskClass.HYPOTHESIS, prompts.HYPOTHESIS, base, HypothesisSet, bb.cost, prov,
                        ctx={"task": bb.task.model_dump()}, node="plan")
    bb.hypotheses = hs.hypotheses
    hyp = "\n".join(f"- {h.statement}" for h in bb.hypotheses)

    # Tree of Thoughts: N diverse branches, each scored, deterministic feasibility check, argmax.
    cands: list[Plan] = []
    n = max(1, rt.s.graph.tot_branches)
    for i in range(n):
        style = TOT_STYLES[i % len(TOT_STYLES)]
        pc = rt.router.call(
            TaskClass.PLANNING, prompts.PLANNER,
            f"{base}\n\nHYPOTHESES:\n{hyp}\n\nBranch {i + 1} of {n}: optimise for {style}.",
            PlanCandidate, bb.cost, prov,
            ctx={"task": bb.task.model_dump(), "branch": i, "tools": list(rt.registry.tools),
                 "workflows": list(rt.registry.workflows)}, node="plan")
        p = Plan(**pc.model_dump())
        unknown = [t for s in p.stages for t in s.tool_ids if t not in rt.registry.tools] + \
                  [s.workflow_id for s in p.stages if s.workflow_id and s.workflow_id not in rt.registry.workflows]
        sc = rt.router.call(
            TaskClass.SCORING, prompts.SCORER,
            f"{base}\n\nCANDIDATE PLAN:\n{p.model_dump_json(indent=2, exclude={'plan_id', 'score', 'score_rationale'})}"
            + (f"\n\nNOTE: references unknown ids {unknown}" if unknown else ""),
            PlanScore, bb.cost, prov, ctx={"candidate": p.model_dump(), "unknown": unknown}, node="plan")
        p.score = 0.0 if unknown else sc.total
        p.score_rationale = sc.rationale + (f" [deterministic: unknown ids {unknown} -> score 0]" if unknown else "")
        cands.append(p)
        rt.bus.publish(bb.run_id, {"event": "thought", "node": "plan", "kind": "tot_branch",
                                   "text": f"branch {i + 1}: {p.title} -> score {p.score}"})
    bb.candidate_plans = cands
    best = max(cands, key=lambda c: c.score or 0)
    if not best.score:
        raise ValueError("no feasible plan: every candidate referenced unknown tools/workflows")
    bb.plan = best
    bb.stage_index = 0
    prov.log("plan_selected", plan_id=best.plan_id, title=best.title, score=best.score,
             branches=[(c.title, c.score) for c in cands])
    bb.next_phase = P.DESIGN


def _apply_adjustments(rt, bb, spec: PipelineSpec) -> list[str]:
    """Enforce anomaly decisions on downstream tools instead of trusting the LLM to remember them."""
    notes = []
    adj = bb.design_adjustments
    for call in spec.tool_calls:
        tool = rt.registry.tools.get(call.tool_id)
        if not tool:
            continue
        for key in ("covariates", "exclude_samples"):
            want = adj.get(key) or []
            # covariates mean "model these" only for analysis tools (for QC tools they mean "test these")
            if want and key in tool.inputs and (key == "exclude_samples" or tool.category == "analysis"):
                have = list(call.params.get(key) or [])
                merged = have + [w for w in want if w not in have]
                if merged != have:
                    call.params[key] = merged
                    notes.append(f"{call.tool_id}.{key} += {[w for w in want if w not in have]}")
    return notes


def design(rt, bb, prov):
    st = bb.current_stage
    if st is None:
        bb.next_phase = P.INTERPRET
        return
    prompt = (f"TASK:\n{_task_text(bb)}\n\nSAMPLE METADATA:\n{_metadata_preview(bb)}\n\n"
              f"PLAN: {bb.plan.title}\nCURRENT STAGE ({bb.stage_index + 1}/{len(bb.plan.stages)}):\n"
              f"{st.model_dump_json(indent=2)}\n\nPREVIOUS STAGE OUTPUTS:\n{_stage_outputs_text(bb)}\n\n"
              f"DESIGN ADJUSTMENTS IN FORCE: {json.dumps(bb.design_adjustments)}\n"
              f"REVIEWER / SYSTEM FEEDBACK TO ADDRESS:\n" + ("\n".join(f"- {f}" for f in bb.design_feedback) or "(none)") +
              f"\n\nLESSONS:\n{_lessons_text(bb)}\n\nREGISTRY:\n{rt.registry.describe_for_llm(rt.s.compute.use_containers)}")
    spec = rt.router.call(TaskClass.DESIGN, prompts.DESIGNER, prompt, PipelineSpec, bb.cost, prov,
                          ctx={"task": bb.task.model_dump(), "stage": st.model_dump(), "adjustments": bb.design_adjustments,
                               "stage_outputs": bb.stage_outputs, "feedback": bb.design_feedback}, node="design")
    notes = _apply_adjustments(rt, bb, spec)
    if notes:
        prov.log("adjustments_enforced", notes=notes)
    bb.pipeline = spec
    rt.bus.publish(bb.run_id, {"event": "thought", "node": "design", "kind": "pipeline",
                               "text": f"{st.name}: {spec.kind} {[c.tool_id for c in spec.tool_calls] or spec.workflow_id or 'script'}"
                                       f" — {spec.rationale}"})
    bb.next_phase = P.VALIDATE


def _static_validate(rt, bb) -> list[ValidationIssue]:
    spec, issues = bb.pipeline, []
    if spec is None:
        return [ValidationIssue(severity="error", message="no pipeline")]
    if spec.kind == "tools":
        if not spec.tool_calls:
            issues.append(ValidationIssue(severity="error", message="kind=tools but no tool_calls"))
        for call in spec.tool_calls:
            tool = rt.registry.tools.get(call.tool_id)
            if not tool:
                issues.append(ValidationIssue(severity="error", message=f"unknown tool '{call.tool_id}'"))
                continue
            ok, how = tool.available(rt.s.compute.use_containers)
            if not ok:
                issues.append(ValidationIssue(severity="error", message=f"{tool.id} unavailable: {how}"))
            if not rt.registry.network_allowed(tool):
                issues.append(ValidationIssue(severity="error", message=f"{tool.id} is an external API; blocked in local mode"))
            for k, p in tool.inputs.items():
                if p.required and k not in call.params and p.default is None:
                    issues.append(ValidationIssue(severity="error", message=f"{tool.id}: missing required param '{k}'"))
            for k in call.params:
                if k not in tool.inputs:
                    issues.append(ValidationIssue(severity="error", message=f"{tool.id}: unknown param '{k}'"))
            prev = {c.tool_id: {o: "x" for o in rt.registry.tools[c.tool_id].outputs}
                    for c in spec.tool_calls if c.tool_id in rt.registry.tools}
            for k, v in call.params.items():
                try:
                    r = rt.jobs.materializer.resolve(v, bb.task, bb.stage_outputs, prev)
                except MaterializeError as e:
                    issues.append(ValidationIssue(severity="error", message=f"{tool.id}.{k}: {e}"))
                    continue
                if tool.inputs.get(k) and tool.inputs[k].type == "path" and isinstance(r, str) and r != "x" \
                        and not Path(r).exists():
                    issues.append(ValidationIssue(severity="error", message=f"{tool.id}.{k}: path does not exist: {r}"))
    elif spec.kind == "workflow":
        if spec.workflow_id not in rt.registry.workflows:
            issues.append(ValidationIssue(severity="error", message=f"unknown workflow '{spec.workflow_id}'"))
    elif spec.kind == "script":
        if not spec.script:
            issues.append(ValidationIssue(severity="error", message="kind=script but no script"))
        else:
            for m in static_check(spec.script.language, spec.script.code, rt.s.compute.sandbox_network):
                issues.append(ValidationIssue(severity="error", message=f"script: {m}"))
            issues.append(ValidationIssue(severity="warning", message="LLM-written code: requires human approval under 'risky' policy"))
    return issues


def validate(rt, bb, prov):
    issues = _static_validate(rt, bb)
    bb.validation_issues = issues
    errors = [i for i in issues if i.severity == "error"]
    st = bb.current_stage
    rec = _judge(
        rt, bb, prov,
        question=f"Is this pipeline appropriate and correct for stage '{st.name}' (objective: {st.objective})?",
        system=prompts.VALIDATOR,
        evidence=f"PIPELINE:\n{bb.pipeline.model_dump_json(indent=2)}\n\nSTATIC CHECKS:\n" +
                 ("\n".join(f"- {i.severity}: {i.message}" for i in issues) or "- all passed") +
                 f"\n\nDESIGN ADJUSTMENTS IN FORCE: {json.dumps(bb.design_adjustments)}",
        rule_answer=not errors, rule_basis=f"{len(errors)} static errors",
        ctx={"kind": "validation", "yes_means_problem": False, "errors": [e.message for e in errors]},
    )
    if not rec.final_answer:
        bb.design_revisions += 1
        bb.design_feedback = [e.message for e in errors] + ([rec.llm_verdict.reasoning] if not rec.llm_verdict.answer else [])
        if bb.design_revisions > rt.s.graph.max_design_revisions:
            bb.errors.append(f"stage '{st.name}': design rejected {bb.design_revisions} times: {bb.design_feedback}")
            bb.next_phase = P.FAILED
        else:
            bb.next_phase = P.DESIGN
        return
    policy = rt.s.hitl.approval_policy
    risky = bb.pipeline.kind == "script" or any(i.severity == "warning" for i in issues)
    bb.next_phase = P.APPROVAL if policy == "always" or (policy == "risky" and risky) else P.EXECUTE


def approval(rt, bb, prov):
    reason = "policy=always" if rt.s.hitl.approval_policy == "always" else \
        "; ".join(i.message for i in bb.validation_issues if i.severity == "warning") or "flagged for review"
    # Must be the first effectful line: on resume LangGraph re-runs this node and interrupt() returns the decision.
    decision = interrupt({"type": "approval", "run_id": bb.run_id, "stage": bb.current_stage.name, "reason": reason,
                          "pipeline": bb.pipeline.model_dump(mode="json")})
    d = decision if isinstance(decision, dict) else {"decision": str(decision)}
    rec = ApprovalRecord(reason=reason, pipeline=bb.pipeline, decision=d.get("decision", "rejected"),
                         reviewer=d.get("reviewer"), comment=d.get("comment"))
    bb.approvals.append(rec)
    prov.log("approval", decision=rec.decision, reviewer=rec.reviewer, comment=rec.comment)
    if rec.decision == "approved":
        bb.next_phase = P.EXECUTE
    else:
        bb.design_revisions += 1
        bb.design_feedback = [f"human reviewer rejected the pipeline: {rec.comment or 'no comment'}"]
        bb.next_phase = P.DESIGN if bb.design_revisions <= rt.s.graph.max_design_revisions else P.FAILED


def execute(rt, bb, prov):
    st = bb.current_stage
    try:
        rec = rt.jobs.submit(bb.run_id, st.name, bb.pipeline, bb.task, bb.stage_outputs, rt.run_dir(bb.run_id), prov)
    except (MaterializeError, KeyError, ValueError) as e:
        bb.design_revisions += 1
        bb.design_feedback = [f"compute layer could not materialize the pipeline: {e}"]
        bb.next_phase = P.DESIGN if bb.design_revisions <= rt.s.graph.max_design_revisions else P.FAILED
        return
    bb.jobs.append(rec)
    bb.executed_pipelines[st.name] = bb.pipeline
    rt.bus.publish(bb.run_id, {"event": "thought", "node": "execute", "kind": "job",
                               "text": f"submitted {rec.job_id} ({rec.backend}) for stage {st.name}; cognitive graph suspended"})
    bb.next_phase = P.AWAIT_JOB


def await_job(rt, bb, prov):
    job = bb.jobs[-1]
    result = interrupt({"type": "await_job", "run_id": bb.run_id, "job_id": job.job_id})
    rec = JobRecord.model_validate(result)
    bb.jobs[-1] = rec
    st = bb.current_stage
    if rec.status == "succeeded":
        bb.stage_outputs[st.name] = rec.outputs
        bb.next_phase = P.QC
        return
    bb.job_retries += 1
    bb.design_feedback = [f"job {rec.job_id} {rec.status}: {rec.error}\nlog tail:\n{(rec.log_tail or '')[-2000:]}"]
    if bb.job_retries > rt.s.graph.max_job_retries:
        bb.errors.append(f"stage '{st.name}': job failed {bb.job_retries} times; last error: {rec.error}")
        bb.next_phase = P.FAILED
    else:
        bb.next_phase = P.DESIGN


def _stage_findings(rt, bb) -> list:
    st = bb.current_stage
    spec = bb.executed_pipelines.get(st.name, bb.pipeline)
    outs = bb.stage_outputs.get(st.name, {})
    findings = []
    if spec.kind == "tools":
        for call in spec.tool_calls:
            tool = rt.registry.tools.get(call.tool_id)
            mine = {k.split(".", 1)[1]: v for k, v in outs.items() if k.startswith(call.tool_id + ".")}
            findings += qcmod.evaluate(tool.qc_parser if tool else None, mine, call.params)
    elif spec.kind == "workflow":
        wf = rt.registry.workflows.get(spec.workflow_id)
        findings += qcmod.evaluate(wf.qc_parser if wf else None, outs, spec.workflow_params)
    else:
        findings += qcmod.evaluate(None, outs)
    return findings


def qc(rt, bb, prov):
    st = bb.current_stage
    findings = _stage_findings(rt, bb)
    bb.qc_findings[st.name] = findings
    prov.log("qc_findings", stage=st.name, findings=[f.model_dump() for f in findings])
    critical = [f for f in findings if f.flagged and f.severity == "critical"]
    question = st.qc_question or f"Do the QC results of stage '{st.name}' show a problem that must be resolved before proceeding?"
    prior = [a for a in bb.anomalies if a.stage == st.name]
    rec = _judge(
        rt, bb, prov, question=question, system=prompts.QC_JUDGE,
        evidence="QC FINDINGS:\n" + "\n".join(f"- [{f.severity}{' FLAGGED' if f.flagged else ''}] {f.check}: {f.evidence}"
                                               for f in findings) +
                 f"\n\nANOMALIES ALREADY RESOLVED FOR THIS STAGE: {[a.resolution.action for a in prior if a.resolution]}",
        rule_answer=bool(critical), rule_basis=f"{len(critical)} critical flagged findings",
        ctx={"kind": "qc", "yes_means_problem": True, "findings": [f.model_dump() for f in findings]},
    )
    if rec.final_answer and len(prior) < 2:
        bb.anomalies.append(Anomaly(stage=st.name, description=question,
                                    findings=[f for f in findings if f.flagged],
                                    admissible_actions=transitions.admissible_resolutions(findings)))
        bb.next_phase = P.ANOMALY
        return
    if rec.final_answer:
        bb.caveats.append(f"stage '{st.name}': QC problem persisted after {len(prior)} resolution attempts: {question}")
    _advance_stage(bb)


def anomaly(rt, bb, prov):
    an = bb.anomalies[-1]
    allowed = an.admissible_actions
    meta_cols = []
    if bb.task.inputs.get("metadata") and Path(bb.task.inputs["metadata"]).exists():
        meta_cols = list(pd.read_csv(bb.task.inputs["metadata"], sep="\t", nrows=1).columns)
    prompt = (f"TASK:\n{_task_text(bb)}\n\nANOMALY at stage '{an.stage}': {an.description}\n\nFLAGGED FINDINGS:\n" +
              "\n".join(f"- [{f.severity}] {f.check}: {f.evidence} | metrics={json.dumps(f.metrics, default=str)}"
                        for f in an.findings) +
              f"\n\nADMISSIBLE ACTIONS: {allowed}\nMETADATA COLUMNS: {meta_cols}\n"
              f"REMAINING STAGES: {[s.name for s in bb.plan.stages[bb.stage_index + 1:]]}")
    res: AnomalyResolution | None = None
    for attempt in range(2):
        res = rt.router.call(TaskClass.ANOMALY, prompts.ANOMALY, prompt, AnomalyResolution, bb.cost, prov,
                             ctx={"admissible": allowed, "findings": [f.model_dump() for f in an.findings]}, node="anomaly")
        bad_cov = [c for c in res.covariates_to_add if meta_cols and c not in meta_cols]
        if res.action in allowed and not bad_cov:
            break
        prov.log("inadmissible_resolution", action=res.action, bad_covariates=bad_cov, allowed=allowed)
        prompt += f"\n\nREJECTED: action '{res.action}'" + (f" with unknown covariates {bad_cov}" if bad_cov else "") + \
                  f" is not admissible. Choose from {allowed}."
    else:
        fallback = "accept_with_caveat" if "accept_with_caveat" in allowed else "abort"
        res = AnomalyResolution(action=fallback, reasoning="system fallback: model proposed inadmissible actions twice",
                                caveat=f"Unresolved anomaly at stage '{an.stage}': {an.description}")
    an.resolution = res
    prov.log("anomaly_resolution", stage=an.stage, action=res.action, covariates=res.covariates_to_add,
             exclude=res.samples_to_exclude, caveat=res.caveat)
    rt.bus.publish(bb.run_id, {"event": "thought", "node": "anomaly", "kind": "resolution",
                               "text": f"{res.action}: {res.reasoning}"})
    adj = bb.design_adjustments
    if res.caveat:
        bb.caveats.append(res.caveat)
    if res.action == "adjust_design":
        adj["covariates"] = sorted(set(adj.get("covariates", [])) | set(res.covariates_to_add))
        _advance_stage(bb)
    elif res.action == "exclude_samples":
        adj["exclude_samples"] = sorted(set(adj.get("exclude_samples", [])) | set(res.samples_to_exclude))
        bb.design_feedback = [f"re-run this stage excluding samples {res.samples_to_exclude}"]
        bb.next_phase = P.DESIGN
    elif res.action == "rerun_with_params":
        bb.design_feedback = [f"re-run this stage with parameter changes {res.param_changes}: {res.reasoning}"]
        bb.next_phase = P.DESIGN
    elif res.action == "accept_with_caveat":
        _advance_stage(bb)
    else:
        bb.errors.append(f"aborted at stage '{an.stage}': {res.reasoning}")
        bb.next_phase = P.FAILED


def _results_digest(bb: Blackboard) -> str:
    parts = []
    for stage, outs in bb.stage_outputs.items():
        parts.append(f"## stage {stage}")
        for f in bb.qc_findings.get(stage, []):
            parts.append(f"- QC {f.check}: {f.evidence}")
        for k, v in outs.items():
            if v.endswith(".json") and Path(v).exists() and Path(v).stat().st_size < 60_000:
                try:
                    d = json.loads(Path(v).read_text())
                    if isinstance(d, dict):
                        d = {kk: vv for kk, vv in d.items() if kk not in ("associations", "pvalue_hist")}
                    parts.append(f"- {k}: {json.dumps(d, default=str)[:3000]}")
                except Exception:
                    pass
    return "\n".join(parts)


def interpret(rt, bb, prov):
    prompt = (f"TASK:\n{_task_text(bb)}\n\nHYPOTHESES:\n" + "\n".join(f"- {h.statement}" for h in bb.hypotheses) +
              f"\n\nPLAN: {bb.plan.title}\n\nRESULTS:\n{_results_digest(bb)}\n\nDESIGN ADJUSTMENTS: "
              f"{json.dumps(bb.design_adjustments)}\nCAVEATS RECORDED:\n" + "\n".join(f"- {c}" for c in bb.caveats))
    bb.interpretation = rt.router.call(TaskClass.INTERPRETATION, prompts.INTERPRETER, prompt, Interpretation, bb.cost, prov,
                                       ctx={"task": bb.task.model_dump(), "digest": _results_digest(bb),
                                            "caveats": bb.caveats, "adjustments": bb.design_adjustments},
                                       node="interpret")
    # Ground-truth evaluation: deterministic, never shown to the LLM.
    truth = bb.task.ground_truth.get("de_truth")
    if truth:
        from ..demo import evaluate_de

        for stage, outs in bb.stage_outputs.items():
            res = outs.get("differential_expression.results") or outs.get("results")
            if res and Path(res).exists():
                bb.evaluation[stage] = evaluate_de(res, truth)
        prov.log("evaluation", evaluation=bb.evaluation)
    bb.next_phase = P.REFLECT


def reflect(rt, bb, prov):
    summary = {
        "goal": bb.task.goal, "analysis_type": bb.task.analysis_type, "plan": bb.plan.title if bb.plan else None,
        "validation_failures": [i.message for i in bb.validation_issues if i.severity == "error"],
        "judgments": [{"q": j.question, "rule": j.rule_answer, "llm": j.llm_verdict.answer if j.llm_verdict else None,
                       "final": j.final_answer} for j in bb.judgments],
        "anomalies": [{"stage": a.stage, "findings": [f.check for f in a.findings],
                       "resolution": a.resolution.action if a.resolution else None} for a in bb.anomalies],
        "job_failures": [j.error for j in bb.jobs if j.status != "succeeded"],
        "approvals": [{"decision": a.decision, "comment": a.comment} for a in bb.approvals],
        "caveats": bb.caveats, "design_adjustments": bb.design_adjustments,
    }
    ls = rt.router.call(TaskClass.REFLECTION, prompts.REFLECTOR, json.dumps(summary, indent=2), LessonSet, bb.cost, prov,
                        ctx={"summary": summary}, node="reflect")
    for d in ls.lessons:
        stored = rt.memory.add(Lesson(text=d.text, tags=[bb.task.analysis_type, *d.tags], evidence=d.evidence,
                                      source_run=bb.run_id))
        bb.lessons_written.append(stored)
    prov.log("lessons_written", lessons=[(l.lesson_id, l.status) for l in bb.lessons_written])
    bb.next_phase = P.REPORT


def report(rt, bb, prov):
    from ..report import write_report

    bb.next_phase = P.DONE
    bb.report_paths = write_report(bb, rt.run_dir(bb.run_id), rt.registry)
    prov.log("report_written", paths=bb.report_paths)


def failed(rt, bb, prov):
    from ..report import write_report

    try:
        bb.report_paths = write_report(bb, rt.run_dir(bb.run_id), rt.registry)
    except Exception as e:
        bb.errors.append(f"report: {e}")
    prov.log("run_failed", errors=bb.errors)


NODES: dict[Phase, NodeFn] = {
    P.RETRIEVE: retrieve, P.PLAN: plan, P.DESIGN: design, P.VALIDATE: validate, P.APPROVAL: approval,
    P.EXECUTE: execute, P.AWAIT_JOB: await_job, P.QC: qc, P.ANOMALY: anomaly, P.INTERPRET: interpret,
    P.REFLECT: reflect, P.REPORT: report, P.FAILED: failed,
}

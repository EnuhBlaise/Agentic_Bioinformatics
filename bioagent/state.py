"""Typed Blackboard state and every structured artifact a node may emit.

The LangGraph state is `{"bb": <json dict>}`. Every node validates the dict
into `Blackboard` on entry and dumps it back on exit (see graph/node.py), so a
node that produces malformed state fails at the boundary where it happened,
and checkpoints stay plain JSON that any later code version can read.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class Phase(str, Enum):
    INIT = "init"
    RETRIEVE = "retrieve"
    PLAN = "plan"
    DESIGN = "design"
    VALIDATE = "validate"
    APPROVAL = "approval"
    EXECUTE = "execute"
    AWAIT_JOB = "await_job"
    QC = "qc"
    ANOMALY = "anomaly"
    INTERPRET = "interpret"
    REFLECT = "reflect"
    REPORT = "report"
    DONE = "done"
    FAILED = "failed"


# ---------------------------------------------------------------- task input
class TaskSpec(BaseModel):
    goal: str
    analysis_type: Literal["rnaseq_de", "rnaseq_fastq", "genome_assembly", "custom"] = "custom"
    organism: str | None = None
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description="Named input paths, e.g. counts, metadata, reads_r1, reads_r2, reference",
    )
    condition_column: str | None = None
    contrast: list[str] | None = Field(default=None, description="[test_level, reference_level]")
    constraints: list[str] = Field(default_factory=list)
    ground_truth: dict[str, str] = Field(
        default_factory=dict, description="Optional paths used only for evaluation, never shown to the LLM"
    )


# ---------------------------------------------------------- LLM output schemas
class Hypothesis(BaseModel):
    statement: str
    rationale: str
    testable_by: str


class HypothesisSet(BaseModel):
    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=5)


class Stage(BaseModel):
    name: str
    objective: str
    tool_ids: list[str] = Field(default_factory=list, description="Registry tool ids to run, in order")
    workflow_id: str | None = Field(default=None, description="Registry workflow id (alternative to tool_ids)")
    qc_question: str | None = Field(
        default=None, description="Binary question the judgment layer must answer after this stage"
    )


class Plan(BaseModel):
    plan_id: str = Field(default_factory=lambda: new_id("plan"))
    title: str
    rationale: str
    stages: list[Stage] = Field(min_length=1)
    risks: list[str] = Field(default_factory=list)
    score: float | None = None
    score_rationale: str | None = None


class PlanCandidate(BaseModel):
    title: str
    rationale: str
    stages: list[Stage] = Field(min_length=1)
    risks: list[str] = Field(default_factory=list)


class PlanScore(BaseModel):
    scientific_validity: float = Field(ge=0, le=10)
    confounder_awareness: float = Field(ge=0, le=10)
    feasibility: float = Field(ge=0, le=10)
    rationale: str

    @property
    def total(self) -> float:
        return round((self.scientific_validity * 0.5 + self.confounder_awareness * 0.3 + self.feasibility * 0.2), 3)


class ToolCall(BaseModel):
    tool_id: str
    params: dict[str, Any] = Field(default_factory=dict)


class ScriptSpec(BaseModel):
    language: Literal["python", "bash"]
    code: str
    purpose: str


class PipelineSpec(BaseModel):
    """What the cognitive graph hands to the compute graph."""
    kind: Literal["tools", "workflow", "script"]
    tool_calls: list[ToolCall] = Field(default_factory=list)
    workflow_id: str | None = None
    workflow_params: dict[str, Any] = Field(default_factory=dict)
    script: ScriptSpec | None = None
    rationale: str = ""


class JudgmentVerdict(BaseModel):
    answer: bool
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    evidence_cited: list[str] = Field(default_factory=list)


class AnomalyResolution(BaseModel):
    action: Literal["adjust_design", "rerun_with_params", "exclude_samples", "accept_with_caveat", "abort"]
    covariates_to_add: list[str] = Field(default_factory=list)
    samples_to_exclude: list[str] = Field(default_factory=list)
    param_changes: dict[str, Any] = Field(default_factory=dict)
    caveat: str | None = None
    reasoning: str


class Finding(BaseModel):
    claim: str
    support: str
    confidence: Literal["low", "medium", "high"]


class Interpretation(BaseModel):
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    hypothesis_outcomes: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    next_experiments: list[str] = Field(default_factory=list)


class LessonDraft(BaseModel):
    text: str
    tags: list[str] = Field(default_factory=list)
    evidence: str


class LessonSet(BaseModel):
    lessons: list[LessonDraft] = Field(default_factory=list, max_length=5)


# --------------------------------------------------------------- run records
class Lesson(BaseModel):
    lesson_id: str = Field(default_factory=lambda: new_id("lesson"))
    text: str
    tags: list[str] = Field(default_factory=list)
    evidence: str = ""
    source_run: str | None = None
    status: Literal["candidate", "validated", "rejected"] = "candidate"
    created_at: str = Field(default_factory=now)
    score: float | None = None


class JudgmentRecord(BaseModel):
    judgment_id: str = Field(default_factory=lambda: new_id("jdg"))
    question: str
    stage: str | None = None
    rule_answer: bool | None = None
    rule_basis: str | None = None
    llm_verdict: JudgmentVerdict | None = None
    final_answer: bool
    combination: str
    agreement: bool | None = None
    model: str | None = None
    ts: str = Field(default_factory=now)


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    message: str


class ApprovalRecord(BaseModel):
    approval_id: str = Field(default_factory=lambda: new_id("appr"))
    reason: str
    pipeline: PipelineSpec
    decision: Literal["pending", "approved", "rejected"] = "pending"
    reviewer: str | None = None
    comment: str | None = None
    ts: str = Field(default_factory=now)


class JobRecord(BaseModel):
    job_id: str
    stage: str
    backend: str
    status: Literal["submitted", "running", "succeeded", "failed", "lost"] = "submitted"
    workdir: str
    outputs: dict[str, str] = Field(default_factory=dict)
    exit_code: int | None = None
    error: str | None = None
    log_tail: str | None = None
    submitted_at: str = Field(default_factory=now)
    finished_at: str | None = None


class QCFinding(BaseModel):
    check: str
    flagged: bool
    severity: Literal["info", "warning", "critical"] = "info"
    evidence: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class Anomaly(BaseModel):
    anomaly_id: str = Field(default_factory=lambda: new_id("anom"))
    stage: str
    description: str
    findings: list[QCFinding] = Field(default_factory=list)
    resolution: AnomalyResolution | None = None
    admissible_actions: list[str] = Field(default_factory=list)


class Transition(BaseModel):
    src: Phase
    dst: Phase
    node: str
    ts: str = Field(default_factory=now)
    note: str | None = None


class CostLedger(BaseModel):
    usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    calls_by_tier: dict[str, int] = Field(default_factory=dict)
    escalations: int = 0
    schema_failures: int = 0


# ------------------------------------------------------------------ blackboard
class Blackboard(BaseModel):
    run_id: str = Field(default_factory=lambda: new_id("run"))
    created_at: str = Field(default_factory=now)
    task: TaskSpec
    phase: Phase = Phase.INIT
    next_phase: Phase | None = None
    history: list[Transition] = Field(default_factory=list)
    cycles: int = 0

    retrieved_lessons: list[Lesson] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    candidate_plans: list[Plan] = Field(default_factory=list)
    plan: Plan | None = None
    stage_index: int = 0

    pipeline: PipelineSpec | None = None
    design_revisions: int = 0
    design_feedback: list[str] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    approvals: list[ApprovalRecord] = Field(default_factory=list)

    executed_pipelines: dict[str, PipelineSpec] = Field(default_factory=dict, description="last pipeline run per stage")
    jobs: list[JobRecord] = Field(default_factory=list)
    job_retries: int = 0
    stage_outputs: dict[str, dict[str, str]] = Field(default_factory=dict)

    qc_findings: dict[str, list[QCFinding]] = Field(default_factory=dict)
    judgments: list[JudgmentRecord] = Field(default_factory=list)
    anomalies: list[Anomaly] = Field(default_factory=list)
    design_adjustments: dict[str, Any] = Field(
        default_factory=dict, description="Accumulated covariates/exclusions applied to later stages"
    )
    caveats: list[str] = Field(default_factory=list)

    interpretation: Interpretation | None = None
    lessons_written: list[Lesson] = Field(default_factory=list)
    report_paths: dict[str, str] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)

    errors: list[str] = Field(default_factory=list)
    cost: CostLedger = Field(default_factory=CostLedger)

    @property
    def current_stage(self) -> Stage | None:
        if self.plan and self.stage_index < len(self.plan.stages):
            return self.plan.stages[self.stage_index]
        return None

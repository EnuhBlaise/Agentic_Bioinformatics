"""Admissible state transitions.

Two layers:
1. Phase-level: which phase may follow which. A node requests `next_phase`;
   the graph router refuses anything not in ADMISSIBLE and fails the run with
   the offending edge recorded, instead of letting the LLM wander.
2. Semantic-level: which anomaly resolutions are admissible given the
   evidence (e.g. "add batch as covariate" is inadmissible when batch is
   fully confounded with condition, because the design matrix is singular).
"""
from __future__ import annotations

from .state import Phase, QCFinding

P = Phase

ADMISSIBLE: dict[Phase, set[Phase]] = {
    P.INIT: {P.RETRIEVE},
    P.RETRIEVE: {P.PLAN},
    P.PLAN: {P.DESIGN},
    P.DESIGN: {P.VALIDATE},
    P.VALIDATE: {P.APPROVAL, P.EXECUTE, P.DESIGN},
    P.APPROVAL: {P.EXECUTE, P.DESIGN},
    P.EXECUTE: {P.AWAIT_JOB, P.DESIGN},  # DESIGN when materialization fails
    P.AWAIT_JOB: {P.QC, P.DESIGN},
    P.QC: {P.DESIGN, P.ANOMALY, P.INTERPRET},
    P.ANOMALY: {P.DESIGN, P.INTERPRET},
    P.INTERPRET: {P.REFLECT},
    P.REFLECT: {P.REPORT},
    P.REPORT: {P.DONE},
    P.DONE: set(),
    P.FAILED: set(),
}
# Any non-terminal phase may fail.
for _src in ADMISSIBLE:
    if _src not in (P.DONE, P.FAILED):
        ADMISSIBLE[_src].add(P.FAILED)


class InadmissibleTransition(RuntimeError):
    pass


def check(src: Phase, dst: Phase) -> None:
    if dst not in ADMISSIBLE.get(src, set()):
        raise InadmissibleTransition(f"{src.value} -> {dst.value} is not an admissible transition")


ALL_RESOLUTIONS = ["adjust_design", "rerun_with_params", "exclude_samples", "accept_with_caveat", "abort"]


def admissible_resolutions(findings: list[QCFinding]) -> list[str]:
    """Restrict anomaly resolutions using deterministic evidence."""
    allowed = set(ALL_RESOLUTIONS)
    for f in findings:
        if f.check == "condition_batch_confounding" and f.metrics.get("fully_confounded"):
            # Batch and condition are aliased: no covariate model can separate them.
            allowed.discard("adjust_design")
            allowed.discard("exclude_samples")
        if f.check == "batch_effect" and not f.metrics.get("technical_covariates"):
            allowed.discard("adjust_design")
    if not any(f.check == "outlier_samples" and f.flagged for f in findings):
        allowed.discard("exclude_samples")
    return [a for a in ALL_RESOLUTIONS if a in allowed]

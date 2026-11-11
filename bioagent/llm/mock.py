"""Deterministic offline provider.

Produces schema-valid, sensible outputs from the structured context each node
passes to the router. It exists so the whole system (graph, compute, QC,
memory, HITL, provenance) can be tested and demoed without a model or network.
It is a test double, not a reasoning engine: its "judgments" mirror the
deterministic rules, so it cannot be used to study judgment quality.
Behaviour can be perturbed for tests via ModelSpec.model:
  "mock"               well-behaved
  "mock-bad-json"      first response to each call is malformed (tests repair)
  "mock-inadmissible"  anomaly node first proposes an inadmissible action
"""
from __future__ import annotations

import json
from typing import Any

from .providers import LLMResponse, Provider


class MockProvider(Provider):
    def __init__(self, spec):
        super().__init__(spec)
        self._seen: dict[str, int] = {}

    def complete(self, system, messages, json_schema, ctx) -> LLMResponse:
        schema = ctx.get("schema")
        key = f"{schema}:{len(messages)}"
        if self.spec.model == "mock-bad-json" and len(messages) == 1:
            return LLMResponse("Sure! Here is the plan: {not json", self.spec.model, 50, 10)
        fn = getattr(self, f"_{schema}", None)
        out: Any = fn(ctx, messages) if fn else {"text": "ok"}
        text = json.dumps(out) if schema else str(out)
        tin = sum(len(m["content"]) for m in messages) // 4 + len(system) // 4
        return LLMResponse(text, self.spec.model, tin, len(text) // 4, 0.0)

    # ----------------------------------------------------------- handlers
    def _HypothesisSet(self, ctx, messages):
        t = ctx["task"]
        if t["analysis_type"] in ("rnaseq_de", "rnaseq_fastq"):
            cond = t.get("condition_column") or "condition"
            test, ref = (t.get("contrast") or ["treated", "control"])[:2]
            return {"hypotheses": [
                {"statement": f"A subset of genes is differentially expressed between {test} and {ref}.",
                 "rationale": "Stated goal of the experiment.", "testable_by": f"DE test on '{cond}' with FDR control"},
                {"statement": "Part of the apparent expression variation is explained by technical covariates (batch).",
                 "rationale": "Batch is recorded in metadata and often dominates PCA in bulk RNA-seq.",
                 "testable_by": "association of principal components with metadata covariates"},
            ]}
        return {"hypotheses": [{"statement": "The sequenced isolate assembles into a contiguous genome of expected size.",
                                "rationale": "Isolate sequencing.", "testable_by": "QUAST N50, contig count, total length"}]}

    def _PlanCandidate(self, ctx, messages):
        t, i = ctx["task"], ctx["branch"]
        if t["analysis_type"] == "rnaseq_de":
            de = {"name": "differential_expression", "objective": "Test condition effect with FDR control",
                  "tool_ids": ["differential_expression"],
                  "qc_question": "Does the p-value distribution or number of DE genes indicate a modelling problem?"}
            qc = {"name": "exploratory_qc", "objective": "PCA and confounder check before inference",
                  "tool_ids": ["confounder_check"],
                  "qc_question": "Did the PCA/confounder check indicate a batch effect or confounding that must be handled?"}
            if i == 2:
                return {"title": "Direct DE (minimal)", "rationale": "Fastest path to a gene list.", "stages": [de],
                        "risks": ["unmodelled batch effects"]}
            if i == 3:
                return {"title": "DE then confounder check", "rationale": "Interpret first.", "stages": [de, qc], "risks": []}
            return {"title": "QC-first DE with confounder control" + (" (rigorous)" if i else ""),
                    "rationale": "Check PCA/covariate structure, then model accordingly.", "stages": [qc, de],
                    "risks": ["small sample size limits power"]}
        if t["analysis_type"] == "genome_assembly":
            return {"title": "Bacterial isolate assembly", "rationale": "fastp -> SPAdes -> QUAST",
                    "stages": [{"name": "assembly", "objective": "assemble and assess", "workflow_id": "bacterial_assembly",
                                "qc_question": "Is the assembly too fragmented or of unexpected size?"}]}
        tools = ctx.get("tools") or ["confounder_check"]
        return {"title": "Generic", "rationale": "", "stages": [{"name": "stage1", "objective": t["goal"], "tool_ids": tools[:1]}]}

    def _PlanScore(self, ctx, messages):
        c = ctx["candidate"]
        stages = c["stages"]
        names = [t for s in stages for t in s.get("tool_ids", [])]
        qc_first = bool(names) and names[0] == "confounder_check"
        return {"scientific_validity": 8 if qc_first else 6, "confounder_awareness": 9 if qc_first else (5 if "confounder_check" in names else 2),
                "feasibility": 9 if not ctx.get("unknown") else 0,
                "rationale": "QC before inference" if qc_first else "skips or postpones confounder assessment"}

    def _PipelineSpec(self, ctx, messages):
        t, st, adj = ctx["task"], ctx["stage"], ctx.get("adjustments") or {}
        if st.get("workflow_id"):
            return {"kind": "workflow", "workflow_id": st["workflow_id"],
                    "workflow_params": {k: f"{{input:{k}}}" for k in t["inputs"]}, "rationale": "registry workflow"}
        calls = []
        for tid in st.get("tool_ids", []):
            p: dict[str, Any] = {}
            if tid in ("confounder_check", "differential_expression"):
                p = {"counts": "{input:counts}", "metadata": "{input:metadata}", "condition": t.get("condition_column") or "condition"}
                if adj.get("exclude_samples"):
                    p["exclude_samples"] = adj["exclude_samples"]
            if tid == "differential_expression":
                p["contrast"] = t.get("contrast") or ["treated", "control"]
                # Deliberately omit covariates: the system must enforce anomaly decisions itself.
            calls.append({"tool_id": tid, "params": p})
        return {"kind": "tools", "tool_calls": calls, "rationale": f"registry tools for {st['name']}"}

    def _JudgmentVerdict(self, ctx, messages):
        rule = ctx.get("rule_answer")
        ans = bool(rule) if rule is not None else (ctx.get("kind") != "qc")
        flagged = [f["check"] for f in ctx.get("findings", []) if f.get("flagged")]
        return {"answer": ans, "confidence": 0.8, "reasoning": f"mock judgment mirrors deterministic rule; flagged={flagged}",
                "evidence_cited": flagged or ctx.get("errors", [])}

    def _AnomalyResolution(self, ctx, messages):
        allowed = ctx["admissible"]
        if self.spec.model == "mock-inadmissible" and len(messages) == 1:
            return {"action": "adjust_design", "covariates_to_add": ["not_a_column"], "reasoning": "deliberately bad"}
        tech, aliased = [], set()
        for f in ctx["findings"]:
            if f["check"] == "batch_effect":
                tech = f["metrics"].get("technical_covariates", [])
            if f["check"] == "condition_batch_confounding" and f["metrics"].get("fully_confounded"):
                aliased.add(f["metrics"]["covariate"])
        usable = [c for c in tech if c not in aliased]
        if "adjust_design" in allowed and usable:
            return {"action": "adjust_design", "covariates_to_add": usable,
                    "reasoning": f"technical covariates {usable} explain major PCs and are not aliased with condition; "
                                 "model them in the design", "caveat": None}
        if "accept_with_caveat" in allowed:
            return {"action": "accept_with_caveat",
                    "reasoning": "covariate is aliased with condition; no model can separate them",
                    "caveat": f"Condition is fully confounded with {sorted(aliased) or 'a technical covariate'}: observed "
                              "differences cannot be attributed to the condition rather than batch."}
        return {"action": "abort", "reasoning": "no admissible resolution"}

    def _Interpretation(self, ctx, messages):
        return {"summary": "Mock interpretation (no model): see QC findings and results tables for the actual evidence.",
                "findings": [{"claim": "Results tables were produced", "support": "stage outputs", "confidence": "low"}],
                "hypothesis_outcomes": [], "caveats": list(ctx.get("caveats", [])),
                "next_experiments": ["Re-run with a real premium model for scientific interpretation."]}

    def _LessonSet(self, ctx, messages):
        s = ctx["summary"]
        out = []
        for a in s.get("anomalies", []):
            if a["resolution"] == "adjust_design" and "batch_effect" in a["findings"]:
                out.append({"text": "For bulk RNA-seq DE, run a PCA/covariate association check before testing and include "
                                    "technical covariates that drive major PCs in the design when they are not aliased with condition.",
                            "tags": ["batch", "design"], "evidence": f"anomaly at stage {a['stage']} resolved by adjust_design"})
            if a["resolution"] == "accept_with_caveat" and "condition_batch_confounding" in a["findings"]:
                out.append({"text": "If condition and batch are fully confounded, report that the effect is not identifiable "
                                    "rather than adjusting; recommend a balanced re-design.",
                            "tags": ["confounding"], "evidence": f"anomaly at stage {a['stage']}"})
        return {"lessons": out}

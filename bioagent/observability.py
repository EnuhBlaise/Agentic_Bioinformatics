"""Observability exporters and Process-Reward-Model (PRM) dataset export.

The provenance JSONL is always written and is the source of truth. Optional
exporters subscribe to the event bus:
  phoenix   OpenTelemetry spans to Arize Phoenix (local, localhost:6006)
  wandb     one W&B row per graph transition / judgment
LangSmith tracing of LangGraph is enabled by its own env vars
(LANGSMITH_TRACING=true, LANGSMITH_API_KEY) and is refused in local mode.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .config import Settings
from .provenance import EventBus, Provenance

_attached = False


def attach_exporters(s: Settings, bus: EventBus) -> list[str]:
    global _attached
    if _attached:
        return []
    _attached = True
    active = []
    o = s.observability
    if o.phoenix:
        try:
            from opentelemetry import trace
            from phoenix.otel import register

            provider = register(project_name="bioagent", endpoint=o.phoenix_endpoint)
            tracer = trace.get_tracer("bioagent", tracer_provider=provider)

            def to_span(rec: dict) -> None:
                with tracer.start_as_current_span(rec.get("event", "event")) as span:
                    for k, v in rec.items():
                        span.set_attribute(f"bioagent.{k}", v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str)[:4000])

            bus.subscribe("*", to_span)
            active.append("phoenix")
        except ImportError:
            print("[bioagent] phoenix requested but arize-phoenix-otel is not installed; skipping")
    if o.wandb:
        try:
            import wandb

            run = wandb.init(project=o.wandb_project, job_type="agent-trace", reinit=True)

            def to_wandb(rec: dict) -> None:
                if rec.get("event") in ("transition", "judgment", "llm_call", "job_finished"):
                    run.log({f"{rec['event']}/{k}": v for k, v in rec.items() if isinstance(v, (int, float, bool))})

            bus.subscribe("*", to_wandb)
            active.append("wandb")
        except ImportError:
            print("[bioagent] wandb requested but not installed; skipping")
    return active


def export_prm_dataset(run_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """One record per reasoning step (graph node visit) with its LLM outputs and
    heuristic outcome labels, for training/evaluating process reward models.

    Heuristic labels (field `auto_label`, -1/0/+1) are weak supervision only:
      -1  the step's output was rejected downstream (validation fail, human
          rejection, inadmissible transition/resolution, job failure, run failed here)
      +1  the step was on the final successful path and not later revised
       0  otherwise
    `human_label` is left null for annotation.
    """
    run_dir = Path(run_dir)
    prov = Provenance.read(run_dir)
    calls = [json.loads(l) for l in (run_dir / "llm_calls.jsonl").read_text().splitlines() if l.strip()] \
        if (run_dir / "llm_calls.jsonl").exists() else []
    bb = json.loads((run_dir / "blackboard.json").read_text()) if (run_dir / "blackboard.json").exists() else {}
    success = bb.get("phase") == "report"

    steps, cur, by_node = [], [], defaultdict(list)
    for c in calls:
        by_node[c["node"]].append(c)
    node_visit = defaultdict(int)
    events_since = []
    for rec in prov:
        if rec["event"] != "transition":
            events_since.append(rec)
            continue
        node = rec["src"]
        k = node_visit[node]
        node_visit[node] += 1
        llm = [c for c in events_since if c["event"] == "llm_call"]
        texts = by_node.get(node, [])[: len(llm)]
        by_node[node] = by_node.get(node, [])[len(llm):]
        steps.append({
            "run_id": rec["run_id"], "seq": rec["seq"], "node": node, "visit": k, "dst": rec["dst"],
            "llm_outputs": [{"task_class": t["task_class"], "model": t["model"], "response": t["response"]} for t in texts],
            "events": [{kk: vv for kk, vv in e.items() if kk not in ("prev", "hash")} for e in events_since
                       if e["event"] in ("judgment", "anomaly_resolution", "approval", "job_finished", "inadmissible_resolution",
                                         "plan_selected", "adjustments_enforced", "node_error")],
        })
        events_since = []
    for i, st in enumerate(steps):
        later = steps[i + 1:]
        bad = (
            (st["node"] == "design" and later and later[0]["node"] == "validate" and later[0]["dst"] == "design")
            or any(e["event"] in ("inadmissible_resolution", "node_error") for e in st["events"])
            or any(e["event"] == "approval" and e.get("decision") == "rejected" for e in st["events"])
            or any(e["event"] == "job_finished" and e.get("status") != "succeeded" for e in st["events"])
            or st["dst"] == "failed"
        )
        revisited = any(s["node"] == st["node"] for s in later) and st["node"] in ("design", "validate")
        st["auto_label"] = -1 if bad else (1 if success and not revisited else 0)
        st["human_label"] = None
    out = Path(out_path) if out_path else run_dir / "prm_steps.jsonl"
    out.write_text("".join(json.dumps(s, default=str) + "\n" for s in steps))
    return out

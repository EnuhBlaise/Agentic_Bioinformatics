"""Assemble the cognitive graph (LangGraph StateGraph) with SQLite checkpointing."""
from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, START, StateGraph

from ..state import Phase
from .nodes import NODES, wrap

if TYPE_CHECKING:
    from ..runner import Runtime


class GraphState(TypedDict):
    bb: dict  # JSON form of state.Blackboard, validated at every node boundary


def build_graph(rt: "Runtime", checkpointer):
    g = StateGraph(GraphState)
    for phase, fn in NODES.items():
        g.add_node(phase.value, wrap(rt, phase, fn))
    g.add_edge(START, Phase.RETRIEVE.value)

    def route(state: GraphState) -> str:
        nxt = state["bb"].get("next_phase") or Phase.FAILED.value
        return END if nxt == Phase.DONE.value else nxt

    targets = {p.value: p.value for p in NODES} | {END: END}
    for phase in NODES:
        if phase == Phase.FAILED:
            g.add_edge(phase.value, END)
        else:
            g.add_conditional_edges(phase.value, route, targets)
    return g.compile(checkpointer=checkpointer)


def mermaid() -> str:
    """Static diagram of admissible transitions (for docs and the dashboard)."""
    from ..transitions import ADMISSIBLE

    lines = ["stateDiagram-v2", "    [*] --> retrieve"]
    for src, dsts in ADMISSIBLE.items():
        for d in sorted(dsts, key=lambda x: x.value):
            if src in (Phase.INIT,) or (d == Phase.FAILED and src != Phase.PLAN):
                continue
            lines.append(f"    {src.value} --> {d.value}")
    lines += ["    done --> [*]", "    failed --> [*]", "    note right of failed: any phase may transition to failed"]
    return "\n".join(lines)

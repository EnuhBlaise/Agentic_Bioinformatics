"""Command line interface.

  bioagent demo [--fully-confounded] [--config configs/mock.yaml]
  bioagent run task.yaml [--config ...] [--auto-approve]
  bioagent resume RUN_ID [--auto-approve]
  bioagent serve [--host 127.0.0.1 --port 3000]
  bioagent tools
  bioagent lessons [--validate ID | --reject ID]
  bioagent verify RUN_ID          check provenance hash chain
  bioagent prm-export RUN_ID      PRM step dataset (JSONL)
  bioagent graph                  Mermaid diagram of admissible transitions
  bioagent check-config           validate config (incl. air-gap rules)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _settings(args):
    from .config import load_settings

    return load_settings(args.config)


def _runtime(args):
    from .runner import Runtime

    return Runtime(_settings(args))


def _approver(args):
    from .runner import auto_approve, cli_approve

    return auto_approve if args.auto_approve else cli_approve


def _summary(bb, rt) -> None:
    ok = bb.phase.value == "report"
    print(f"\nrun {bb.run_id}: {'DONE' if ok else 'FAILED'}")
    print("path: " + " -> ".join([bb.history[0].src.value] + [t.dst.value for t in bb.history]) if bb.history else "")
    if bb.errors:
        print("errors:\n  " + "\n  ".join(bb.errors))
    if bb.design_adjustments:
        print(f"design adjustments: {bb.design_adjustments}")
    for c in bb.caveats:
        print(f"CAVEAT: {c}")
    if bb.evaluation:
        print(f"ground-truth evaluation: {json.dumps(bb.evaluation)}")
    print(f"cost: ${bb.cost.usd:.4f}  calls: {bb.cost.calls_by_tier}")
    for k, v in bb.report_paths.items():
        print(f"{k}: {v}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bioagent")
    ap.add_argument("--config", default=None, help="YAML config (default $BIOAGENT_CONFIG or configs/local.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="synthetic RNA-seq with a planted batch effect, end to end")
    d.add_argument("--out", default="examples/synthetic")
    d.add_argument("--fully-confounded", action="store_true")
    d.add_argument("--auto-approve", action="store_true", default=True)

    r = sub.add_parser("run")
    r.add_argument("task")
    r.add_argument("--auto-approve", action="store_true")

    rs = sub.add_parser("resume")
    rs.add_argument("run_id")
    rs.add_argument("--auto-approve", action="store_true")

    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=3000)

    sub.add_parser("tools")
    l = sub.add_parser("lessons")
    l.add_argument("--validate")
    l.add_argument("--reject")
    v = sub.add_parser("verify")
    v.add_argument("run_id")
    pe = sub.add_parser("prm-export")
    pe.add_argument("run_id")
    sub.add_parser("graph")
    sub.add_parser("check-config")

    args = ap.parse_args(argv)

    if args.cmd == "graph":
        from .graph.builder import mermaid

        print(mermaid())
        return 0
    if args.cmd == "check-config":
        st = _settings(args)
        print(f"OK mode={st.mode} premium={st.llm.premium.provider}:{st.llm.premium.model} "
              f"budget={st.llm.budget.provider}:{st.llm.budget.model} compute={st.compute.backend}")
        return 0
    if args.cmd == "demo":
        from .demo import make_dataset

        if args.config is None:
            args.config = "configs/mock.yaml"
        paths = make_dataset(args.out, fully_confounded=args.fully_confounded)
        print(f"synthetic data -> {args.out} (fully_confounded={args.fully_confounded})")
        rt = _runtime(args)
        bb = rt.run(yaml.safe_load(Path(paths["task"]).read_text()), approve=_approver(args))
        _summary(bb, rt)
        return 0 if bb.phase.value == "report" else 1
    if args.cmd == "run":
        rt = _runtime(args)
        bb = rt.run(yaml.safe_load(Path(args.task).read_text()), approve=_approver(args))
        _summary(bb, rt)
        return 0 if bb.phase.value == "report" else 1
    if args.cmd == "resume":
        rt = _runtime(args)
        bb = rt.drive(args.run_id, approve=_approver(args))
        _summary(bb, rt)
        return 0 if bb.phase.value == "report" else 1
    if args.cmd == "serve":
        import uvicorn

        from .api.server import create_app

        rt = _runtime(args)
        print(f"dashboard: http://{args.host}:{args.port}/")
        uvicorn.run(create_app(rt), host=args.host, port=args.port, log_level="warning")
        return 0
    if args.cmd == "tools":
        from .config import load_settings
        from .registry import Registry

        st = load_settings(args.config)
        reg = Registry(st.resolve(st.tools_dir), st.resolve(st.workflows_dir), st.mode)
        print(reg.describe_for_llm(st.compute.use_containers))
        if reg.errors:
            print("\nREGISTRY ERRORS:\n  " + "\n  ".join(reg.errors))
        return 0
    if args.cmd == "lessons":
        from .memory import make_memory

        st = _settings(args)
        mem = make_memory(st.memory, st.resolve(st.memory.path))
        if args.validate:
            print(mem.set_status(args.validate, "validated"))
        elif args.reject:
            print(mem.set_status(args.reject, "rejected"))
        else:
            for les in mem.all():
                print(f"{les.lesson_id} [{les.status}] {les.text}")
        return 0
    if args.cmd in ("verify", "prm-export"):
        st = _settings(args)
        run_dir = st.resolve(st.workspace) / args.run_id
        if args.cmd == "verify":
            from .provenance import Provenance

            ok, msg = Provenance.verify(run_dir)
            print(msg)
            return 0 if ok else 1
        from .observability import export_prm_dataset

        print(export_prm_dataset(run_dir))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

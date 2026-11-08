"""Command-line interface.

  teval run    golden.jsonl [--config config.json] [--report out.md]
  teval record golden.jsonl --baseline baseline.json [--config config.json]
  teval gate   golden.jsonl --baseline baseline.json [--tolerance 0.05]

Exit code is 0 when the run/gate passes, 1 otherwise - wire it into CI.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import gate as gate_mod
from .harness import evaluate
from .report import to_markdown
from .schema import SchemaError, load_jsonl


def _load_config(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="teval", description="Trajectory eval harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "record", "gate"):
        sp = sub.add_parser(name)
        sp.add_argument("golden", help="JSONL of golden trajectories")
        sp.add_argument("--config", default="", help="JSON eval config")
        if name == "run":
            sp.add_argument("--report", default="", help="write markdown report here")
        else:
            sp.add_argument("--baseline", required=True)
        if name == "gate":
            sp.add_argument("--tolerance", type=float, default=0.0)
    args = p.parse_args(argv)

    try:
        trajectories = load_jsonl(args.golden)
        config = _load_config(args.config)
    except (OSError, json.JSONDecodeError, SchemaError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    result = evaluate(trajectories, config)

    if args.cmd == "record":
        gate_mod.save_baseline(result, args.baseline)
        print(f"baseline written to {args.baseline} ({len(result.results)} trajectories)")
        return 0

    if args.cmd == "gate":
        try:
            baseline = gate_mod.load_baseline(args.baseline)
        except (OSError, json.JSONDecodeError) as e:
            print(f"error: cannot read baseline: {e}", file=sys.stderr)
            return 2
        g = gate_mod.check(result, baseline, tolerance=args.tolerance)
        if g.ok:
            print(f"gate PASS: no regressions vs baseline (score {result.score:.3f})")
            return 0
        print(f"gate FAIL: {len(g.regressions)} regressions vs baseline:", file=sys.stderr)
        for r in g.regressions:
            print(f"  - {r}", file=sys.stderr)
        return 1

    # run
    report = to_markdown(result)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"report written to {args.report}")
    else:
        print(report)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

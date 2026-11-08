"""CI gate: compare a run against a baseline and fail on regression.

The baseline is a JSON file mapping trajectory_id -> per-grader scores,
written by `teval record`. `teval gate` exits non-zero when any grader score
drops below its baseline by more than the tolerance, or when a previously
passing trajectory now fails. This is what blocks bad PRs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .harness import RunResult


@dataclass
class GateResult:
    ok: bool
    regressions: list = field(default_factory=list)  # list[str]


def snapshot(result: RunResult) -> dict:
    """Serializable baseline: per-trajectory, per-grader scores."""
    return {
        r.trajectory_id: {g.name: round(g.score, 4) for g in r.grades}
        for r in result.results
    }


def save_baseline(result: RunResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot(result), f, indent=2, sort_keys=True)


def load_baseline(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check(result: RunResult, baseline: dict, tolerance: float = 0.0) -> GateResult:
    regressions = []
    for r in result.results:
        base = baseline.get(r.trajectory_id)
        if base is None:
            continue  # new trajectory: not a regression, just unbaselined
        for g in r.grades:
            prev = base.get(g.name)
            if prev is None:
                continue
            if g.score < prev - tolerance:
                regressions.append(
                    f"{r.trajectory_id}/{g.name}: {prev:.3f} -> {g.score:.3f}"
                )
    return GateResult(ok=not regressions, regressions=regressions)

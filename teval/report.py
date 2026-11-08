"""Markdown report rendering for a RunResult."""

from __future__ import annotations

from .harness import RunResult


def to_markdown(result: RunResult, title: str = "Trajectory Eval Report") -> str:
    lines = [f"# {title}", ""]
    lines.append(f"Overall score: **{result.score:.3f}** "
                 f"({'PASS' if result.passed else 'FAIL'}) "
                 f"across {len(result.results)} trajectories, "
                 f"{result.violation_count} violations.")
    lines.append("")
    lines.append("| Trajectory | Score | Status |")
    lines.append("| --- | --- | --- |")
    for r in result.results:
        lines.append(f"| {r.trajectory_id} | {r.score:.3f} | {'PASS' if r.passed else 'FAIL'} |")
    lines.append("")
    for r in result.results:
        bad = [g for g in r.grades if g.violations and not g.passed]
        if not bad:
            continue
        lines.append(f"## {r.trajectory_id}")
        lines.append("")
        for g in bad:
            lines.append(f"- **{g.name}** ({g.score:.3f})")
            for v in g.violations:
                lines.append(f"  - {v}")
        lines.append("")
    return "\n".join(lines)

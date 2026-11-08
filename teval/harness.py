"""Harness: run all graders over trajectories and aggregate scores."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import graders as G
from .schema import Trajectory


@dataclass
class TrajectoryResult:
    trajectory_id: str
    grades: list = field(default_factory=list)  # list[Grade]

    @property
    def score(self) -> float:
        if not self.grades:
            return 1.0
        return sum(g.score for g in self.grades) / len(self.grades)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.grades)


@dataclass
class RunResult:
    results: list = field(default_factory=list)  # list[TrajectoryResult]

    @property
    def score(self) -> float:
        if not self.results:
            return 1.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def violation_count(self) -> int:
        return sum(len(g.violations) for r in self.results for g in r.grades)


DEFAULT_CONFIG = {
    "allowed_tools": [],
    "max_tokens": 0,
    "max_steps": 0,
    "max_repeat": 2,
    "canary": "",
    "expected_answer": "",
    "answer_regex": "",
}


def evaluate_trajectory(traj: Trajectory, config: dict) -> TrajectoryResult:
    cfg = {**DEFAULT_CONFIG, **(config or {}), **(traj.metadata.get("eval", {}) or {})}
    grades = [
        G.grade_tool_validity(traj, cfg["allowed_tools"]),
        G.grade_loop_detection(traj, cfg["max_repeat"]),
        G.grade_budget(traj, cfg["max_tokens"], cfg["max_steps"]),
        G.grade_injection_canary(traj, cfg["canary"]),
        G.grade_final_answer(traj, cfg["expected_answer"], cfg["answer_regex"]),
        G.grade_answer_grounding(traj),
    ]
    return TrajectoryResult(trajectory_id=traj.id, grades=grades)


def evaluate(trajectories: list, config: dict) -> RunResult:
    return RunResult(results=[evaluate_trajectory(t, config) for t in trajectories])

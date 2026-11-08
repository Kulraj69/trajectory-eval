"""Trajectory schema: typed records for agent runs, loaded from JSON/JSONL.

A Trajectory is one agent run: an ordered list of Steps ("hops") plus an
optional final answer. Graders score every hop, not just the final answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable


class SchemaError(ValueError):
    """Raised when a trajectory record is malformed."""


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class Step:
    """One hop in an agent run.

    kind: "thought" | "tool_call" | "tool_result" | "answer"
    tokens: tokens consumed by this hop (prompt + completion), if metered.
    """

    kind: str
    content: str = ""
    tool_call: ToolCall | None = None
    tokens: int = 0

    @staticmethod
    def from_dict(d: dict) -> "Step":
        kind = d.get("kind")
        if kind not in ("thought", "tool_call", "tool_result", "answer"):
            raise SchemaError(f"step kind must be thought/tool_call/tool_result/answer, got {kind!r}")
        tc = d.get("tool_call")
        tool_call = None
        if tc is not None:
            if not isinstance(tc, dict) or not isinstance(tc.get("name"), str):
                raise SchemaError("tool_call must be an object with a string 'name'")
            tool_call = ToolCall(name=tc["name"], arguments=tc.get("arguments") or {})
        tokens = d.get("tokens", 0)
        if not isinstance(tokens, int) or tokens < 0:
            raise SchemaError("tokens must be a non-negative int")
        return Step(kind=kind, content=d.get("content", "") or "", tool_call=tool_call, tokens=tokens)


@dataclass
class Trajectory:
    id: str
    task: str
    steps: list = field(default_factory=list)  # list[Step]
    final_answer: str = ""
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "Trajectory":
        if not isinstance(d.get("id"), str) or not d["id"]:
            raise SchemaError("trajectory needs a non-empty string 'id'")
        steps = [Step.from_dict(s) for s in d.get("steps", [])]
        return Trajectory(
            id=d["id"],
            task=d.get("task", "") or "",
            steps=steps,
            final_answer=d.get("final_answer", "") or "",
            metadata=d.get("metadata") or {},
        )

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.steps)

    def tool_calls(self) -> Iterable[ToolCall]:
        for s in self.steps:
            if s.kind == "tool_call" and s.tool_call is not None:
                yield s.tool_call


def load_jsonl(path: str) -> list:
    """Load trajectories from a JSONL file (one JSON object per line)."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Trajectory.from_dict(json.loads(line)))
            except (json.JSONDecodeError, SchemaError) as e:
                raise SchemaError(f"{path}:{lineno}: {e}") from e
    return out

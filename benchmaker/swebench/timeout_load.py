# benchmaker/swebench/timeout_load.py
"""Pure helpers for the command-timeout-under-load experiment.

A command of natural (uncontended) duration ``d`` seconds times out under load
factor ``L`` and per-command timeout ``T`` iff ``L * d > T`` -- equivalently iff
``d > tau`` where ``tau = T / L`` is the *effective per-command time budget*.

Strict rule: a task survives only if *every* command finishes within ``tau``
(``max(d) <= tau``); otherwise its first over-budget command times out, the
replayed trajectory diverges, and the task is counted as failed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union


def effective_tau(timeout_s: float, load_factor: float) -> float:
    """tau = T / L: the per-command time budget under load factor L."""
    if load_factor <= 0:
        raise ValueError("load_factor must be > 0")
    return timeout_s / load_factor


def would_time_out(duration_s: float, tau_s: float) -> bool:
    """A command times out iff its natural duration exceeds the budget tau."""
    return duration_s > tau_s


def task_survives(max_duration_s: float, tau_s: float) -> bool:
    """Strict rule: a task survives iff all commands finish within tau."""
    return max_duration_s <= tau_s


@dataclass(frozen=True)
class CommandTiming:
    tool: str
    duration_s: float


def recover_command_timings(log_path: Union[str, Path]) -> list[CommandTiming]:
    """Per-command wall-times from a pi-container agent log (JSONL).

    duration = (toolResult message timestamp) - (issuing assistant timestamp),
    using epoch-ms ``timestamp`` fields on ``message_end`` events. An assistant
    message with no ``toolCall`` is skipped (no command was issued).
    """
    timings: list[CommandTiming] = []
    pending: Optional[tuple[int, str]] = None  # (assistant_ts_ms, tool_name)
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") != "message_end":
                continue
            msg = obj.get("message", {})
            role = msg.get("role")
            ts = msg.get("timestamp")
            if role == "assistant":
                tools = [c.get("name") for c in msg.get("content", [])
                         if c.get("type") == "toolCall"]
                pending = (ts, tools[0]) if (ts and tools) else None
            elif role == "toolResult" and pending is not None:
                a_ts, tool = pending
                if a_ts and ts:
                    timings.append(CommandTiming(tool, (ts - a_ts) / 1000.0))
                pending = None
    return timings


@dataclass(frozen=True)
class CurvePoint:
    tau_s: float
    n_survive: int
    n_solved: int
    accuracy: float
    n_broken: int


def accuracy_curve(tasks: Iterable[tuple[float, float]],
                   taus: Iterable[float]) -> list[CurvePoint]:
    """Strict accuracy(tau) curve.

    ``tasks`` is an iterable of ``(reward, max_duration_s)``. Accuracy is the
    survivor-solved count over the *total* task count, so points are comparable
    across taus. A task survives iff ``max_duration_s <= tau``.
    """
    tasks = list(tasks)
    n = len(tasks)
    out: list[CurvePoint] = []
    for tau in taus:
        survive = [(r, d) for (r, d) in tasks if task_survives(d, tau)]
        n_solved = sum(1 for (r, _d) in survive if r == 1.0)
        acc = n_solved / n if n else 0.0
        out.append(CurvePoint(tau, len(survive), n_solved, acc, n - len(survive)))
    return out

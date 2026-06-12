# tests/test_timeout_load.py
"""Unit tests for the command-timeout-under-load helpers."""
from __future__ import annotations

import glob
import json
import math
import os

import pytest

from benchmaker.swebench.timeout_load import (
    CommandTiming, CurvePoint, accuracy_curve, effective_tau,
    recover_command_timings, task_survives, would_time_out,
)


def test_effective_tau():
    assert effective_tau(600, 1) == 600
    assert effective_tau(600, 20) == 30


def test_effective_tau_rejects_nonpositive_load():
    with pytest.raises(ValueError):
        effective_tau(600, 0)


def test_would_time_out_boundary():
    # strictly greater than the budget times out; equal does not
    assert would_time_out(5.0001, 5.0) is True
    assert would_time_out(5.0, 5.0) is False


def test_task_survives_uses_max_within_budget():
    assert task_survives(5.0, 5.0) is True
    assert task_survives(5.1, 5.0) is False
    assert task_survives(50.0, math.inf) is True


def _log(tmp_path, rows):
    p = tmp_path / "pi-container.log"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_recover_command_timings(tmp_path):
    rows = [
        {"type": "message_start", "message": {"role": "user", "content": []}},
        {"type": "message_end",
         "message": {"role": "assistant", "timestamp": 1000,
                     "content": [{"type": "toolCall", "name": "bash"}]}},
        {"type": "message_end",
         "message": {"role": "toolResult", "timestamp": 3500, "content": []}},
        {"type": "message_end",
         "message": {"role": "assistant", "timestamp": 4000,
                     "content": [{"type": "toolCall", "name": "read"}]}},
        {"type": "message_end",
         "message": {"role": "toolResult", "timestamp": 4010, "content": []}},
    ]
    timings = recover_command_timings(_log(tmp_path, rows))
    assert timings == [CommandTiming("bash", 2.5), CommandTiming("read", 0.01)]


def test_recover_ignores_assistant_without_toolcall(tmp_path):
    rows = [
        {"type": "message_end",
         "message": {"role": "assistant", "timestamp": 1000,
                     "content": [{"type": "text", "text": "thinking"}]}},
        {"type": "message_end",
         "message": {"role": "toolResult", "timestamp": 9000, "content": []}},
    ]
    assert recover_command_timings(_log(tmp_path, rows)) == []


def test_accuracy_curve_strict_and_monotonic():
    # (reward, max_duration_s) per task
    tasks = [(1.0, 1.0), (1.0, 8.0), (0.0, 2.0), (1.0, 50.0)]
    pts = {p.tau_s: p for p in accuracy_curve(tasks, [math.inf, 10, 5, 0.5])}
    assert pts[math.inf].n_solved == 3 and pts[math.inf].n_broken == 0
    assert pts[10].n_solved == 2   # 50s task broken
    assert pts[5].n_solved == 1    # 8s and 50s broken
    assert pts[0.5].n_solved == 0  # all broken
    # accuracy is non-increasing as tau shrinks
    accs = [pts[t].accuracy for t in [math.inf, 10, 5, 0.5]]
    assert accs == sorted(accs, reverse=True)


def test_accuracy_curve_point_fields():
    pts = accuracy_curve([(1.0, 3.0), (0.0, 3.0)], [2.0])
    assert pts == [CurvePoint(tau_s=2.0, n_survive=0, n_solved=0,
                              accuracy=0.0, n_broken=2)]


def test_effective_tau_rejects_negative_timeout():
    with pytest.raises(ValueError):
        effective_tau(-1, 1)


def test_recover_drops_first_when_two_assistants_without_result(tmp_path):
    # Two assistant tool calls with no intervening toolResult: the first is
    # overwritten, so only the second pairs with the result.
    rows = [
        {"type": "message_end",
         "message": {"role": "assistant", "timestamp": 1000,
                     "content": [{"type": "toolCall", "name": "bash"}]}},
        {"type": "message_end",
         "message": {"role": "assistant", "timestamp": 2000,
                     "content": [{"type": "toolCall", "name": "read"}]}},
        {"type": "message_end",
         "message": {"role": "toolResult", "timestamp": 2500, "content": []}},
    ]
    assert recover_command_timings(_log(tmp_path, rows)) == [CommandTiming("read", 0.5)]


C1_DIR = "jobs/replay_2026-06-11__17-21-48_c1_0d73"


@pytest.mark.skipif(not os.path.isdir(C1_DIR), reason="local c1 jobs dir not present")
def test_curve_on_real_c1_is_monotonic_and_baseline_matches():
    tasks = []
    for tdir in glob.glob(os.path.join(C1_DIR, "*")):
        lp = os.path.join(tdir, "agent", "pi-container.log")
        rj = os.path.join(tdir, "result.json")
        if not (os.path.exists(lp) and os.path.exists(rj)):
            continue
        timings = recover_command_timings(lp)
        max_d = max((c.duration_s for c in timings), default=0.0)
        with open(rj) as f:
            reward = (json.load(f).get("verifier_result") or {}).get("rewards", {}).get("reward")
        tasks.append((float(reward or 0.0), max_d))
    pts = accuracy_curve(tasks, [math.inf, 30, 5, 1])
    accs = [p.accuracy for p in pts]
    assert accs == sorted(accs, reverse=True)                       # monotonic
    assert pts[0].n_solved == sum(1 for r, _ in tasks if r == 1.0)  # baseline = recorded solved

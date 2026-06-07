"""Unit tests for benchmaker.swebench.observability (pure helpers + JobObserver).

No network or live harbor: TrialResult / ClusterStatus inputs are small fakes
with the same attribute names the real objects expose.
"""
from __future__ import annotations

import json
import types
from datetime import datetime, timezone

from benchmaker.swebench import observability as obs


def _ti(start_s, end_s):
    base = datetime(2026, 6, 7, 12, 0, 0, tzinfo=timezone.utc)
    from datetime import timedelta
    return types.SimpleNamespace(
        started_at=None if start_s is None else base + timedelta(seconds=start_s),
        finished_at=None if end_s is None else base + timedelta(seconds=end_s),
    )


def _fake_result(trial="t1", task="task1", phases=None, rewards=None, totals=None,
                 metadata=None):
    phases = phases or {}
    vr = None if rewards is None else types.SimpleNamespace(rewards=rewards)
    ar = types.SimpleNamespace(metadata=metadata or {})
    r = types.SimpleNamespace(
        trial_name=trial, task_name=task,
        environment_setup=phases.get("environment_setup"),
        agent_setup=phases.get("agent_setup"),
        agent_execution=phases.get("agent_execution"),
        verifier=phases.get("verifier"),
        verifier_result=vr, agent_result=ar,
        compute_token_cost_totals=lambda: (totals or (None, None, None, None)),
    )
    return r


def test_phase_spans_from_result_basic():
    r = _fake_result(phases={
        "environment_setup": _ti(0, 42),
        "agent_execution": _ti(50, 410),
        "verifier": _ti(410, 455),
        # agent_setup intentionally missing
    })
    spans = obs.phase_spans_from_result(r)
    names = [s["name"] for s in spans]
    assert names == ["environment_setup", "agent_execution", "verifier"]
    env = spans[0]
    assert env["kind"] == "phase" and env["trial"] == "t1" and env["task"] == "task1"
    assert env["duration_s"] == 42.0
    assert env["start"].startswith("2026-06-07T12:00:00")
    # all token/seq fields present and None for phases
    assert env["seq"] is None and env["n_input_tokens"] is None


def test_phase_spans_skips_unfinished():
    r = _fake_result(phases={"agent_setup": _ti(None, None)})
    assert obs.phase_spans_from_result(r) == []

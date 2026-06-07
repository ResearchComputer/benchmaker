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


def test_util_row_from_status():
    node = types.SimpleNamespace(node_id="n1", available=True, running_count=3)
    status = types.SimpleNamespace(
        node_count=2, available_node_count=1, unavailable_node_count=1,
        sandbox_count=8, nodes=(node,),
    )
    wall = datetime(2026, 6, 7, 12, 2, 12, tzinfo=timezone.utc)
    row = obs.util_row_from_status(status, t=132.5, wall=wall)
    assert row["t"] == 132.5
    assert row["wall"].startswith("2026-06-07T12:02:12")
    assert row["node_count"] == 2 and row["available_node_count"] == 1
    assert row["sandbox_count"] == 8
    assert row["nodes"] == [{"id": "n1", "available": True, "running_count": 3}]


def test_util_row_handles_empty_nodes():
    status = types.SimpleNamespace(
        node_count=0, available_node_count=0, unavailable_node_count=0,
        sandbox_count=0, nodes=(),
    )
    row = obs.util_row_from_status(status, t=0.0, wall=datetime.now(timezone.utc))
    assert row["nodes"] == [] and row["sandbox_count"] == 0


def test_summarize_phases_and_agent_and_util():
    spans = [
        obs._span("t1", "task1", "phase", "agent_execution", duration_s=100.0),
        obs._span("t2", "task2", "phase", "agent_execution", duration_s=300.0),
        obs._span("t1", "task1", "llm_call", "llm_call", duration_s=2.0,
                  n_input_tokens=1000, n_output_tokens=200),
        obs._span("t1", "task1", "llm_call", "llm_call", duration_s=4.0,
                  n_input_tokens=1500, n_output_tokens=None),
        obs._span("t1", "task1", "sandbox_exec", "sandbox_exec", duration_s=1.0, rc=0),
    ]
    util_rows = [
        {"sandbox_count": 5, "node_count": 12, "available_node_count": 11},
        {"sandbox_count": 10, "node_count": 12, "available_node_count": 10},
    ]
    s = obs.summarize(spans, util_rows)
    ph = s["phases"]["agent_execution"]
    assert ph["count"] == 2 and ph["mean_s"] == 200.0
    assert abs(ph["p90_s"] - 280.0) < 1e-6     # interp between 100 and 300 at q=.9
    ag = s["agent"]
    assert ag["n_llm_calls"] == 2 and ag["n_exec"] == 1
    assert ag["total_input_tokens"] == 2500 and ag["total_output_tokens"] == 200
    assert ag["llm_mean_s"] == 3.0
    u = s["utilization"]
    assert u["sandbox_peak"] == 10 and u["sandbox_mean"] == 7.5
    assert u["node_count"] == 12 and u["polls"] == 2


def test_summarize_empty_is_safe():
    s = obs.summarize([], [])
    assert s["phases"] == {} and s["agent"]["n_llm_calls"] == 0
    assert s["utilization"]["sandbox_peak"] is None
    # format_summary must not raise on empty input
    assert isinstance(obs.format_summary(s), str)


def test_format_summary_mentions_sections():
    s = obs.summarize(
        [obs._span("t1", "k", "phase", "verifier", duration_s=44.0)],
        [{"sandbox_count": 3, "node_count": 4, "available_node_count": 4}],
    )
    text = obs.format_summary(s)
    assert "verifier" in text and "sandbox" in text.lower()

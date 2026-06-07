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


def test_parse_pi_token_spans_nominal():
    lines = [
        json.dumps({"type": "session_header", "id": "abc"}),       # header, no usage
        json.dumps({"type": "message", "role": "assistant",
                    "timestamp": "2026-06-07T12:00:01+00:00",
                    "usage": {"input": 1200, "output": 150,
                              "cacheRead": 100, "cacheWrite": 0,
                              "cost": {"total": 0.0031}}}),
        "not json at all",                                         # skipped
        json.dumps({"type": "tool_result"}),                       # no usage -> skipped
    ]
    spans = obs.parse_pi_token_spans("\n".join(lines), trial="t9", task="k9")
    assert len(spans) == 1
    sp = spans[0]
    assert sp["kind"] == "llm_call" and sp["trial"] == "t9" and sp["task"] == "k9"
    assert sp["n_input_tokens"] == 1200 and sp["n_output_tokens"] == 150
    assert sp["n_cache_tokens"] == 100 and abs(sp["cost_usd"] - 0.0031) < 1e-9
    assert sp["start"].startswith("2026-06-07T12:00:01")
    assert sp["seq"] == 1


def test_parse_pi_token_spans_missing_and_renamed_fields_degrade():
    lines = [
        json.dumps({"usage": {"output": 7}}),                      # no input, no ts
        json.dumps({"usage": {"foo": 1, "bar": 2}}),               # renamed -> no tokens, skipped
        json.dumps({"usage": {"input": 5}, "timestamp": 1750000000000}),  # epoch millis
    ]
    spans = obs.parse_pi_token_spans("\n".join(lines))
    # first (output only) kept, second skipped (no numeric input/output), third kept
    assert len(spans) == 2
    assert spans[0]["n_output_tokens"] == 7 and spans[0]["start"] is None
    assert spans[1]["n_input_tokens"] == 5 and spans[1]["start"] is not None


def test_merge_span_files_stamps_trial_from_path(tmp_path):
    # <job>/trial-a/agent-logs/timeline-spans.jsonl  -> trial "trial-a"
    d = tmp_path / "trial-a" / "agent-logs"
    d.mkdir(parents=True)
    (d / "timeline-spans.jsonl").write_text(
        json.dumps({"kind": "llm_call", "seq": 1}) + "\n"
        + json.dumps({"kind": "sandbox_exec", "seq": 1, "trial": "explicit"}) + "\n"
    )
    spans = obs.merge_span_files(tmp_path)
    assert len(spans) == 2
    # path-derived trial when absent; preserved when already set
    assert spans[0]["trial"] == "trial-a"
    assert spans[1]["trial"] == "explicit"


def test_trajectory_manifest_rows(tmp_path):
    tdir = tmp_path / "t1"
    (tdir / "logs").mkdir(parents=True)
    (tdir / "logs" / "benchmaker-host.trajectory.json").write_text("{}")
    (tdir / "logs" / "pi-host.log").write_text("{}")
    r = _fake_result(
        trial="t1", task="task1", rewards={"reward": 1.0},
        totals=(2500, 100, 200, 0.01),
        metadata={"exit_status": "submitted", "n_calls": 5, "n_actions": 4},
    )
    rows = obs.trajectory_manifest_rows([r], tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["trial"] == "t1" and row["reward"] == 1.0 and row["passed"] is True
    assert row["exit_status"] == "submitted" and row["n_calls"] == 5
    assert row["n_input_tokens"] == 2500 and row["n_output_tokens"] == 200
    assert row["cost_usd"] == 0.01
    paths = set(row["trajectory_paths"])
    assert "t1/logs/benchmaker-host.trajectory.json" in paths
    assert "t1/logs/pi-host.log" in paths


def test_write_jsonl_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    obs._write_jsonl(p, [{"a": 1}, {"b": 2}])
    lines = p.read_text().splitlines()
    assert [json.loads(x) for x in lines] == [{"a": 1}, {"b": 2}]


import pytest
from benchmaker.swebench.agent import CodingAgent, parse_openai_usage


def test_parse_openai_usage_shapes():
    data = {"usage": {"prompt_tokens": 1000, "completion_tokens": 120,
                      "prompt_tokens_details": {"cached_tokens": 80}}}
    u = parse_openai_usage(data)
    assert u == {"n_input_tokens": 1000, "n_output_tokens": 120, "n_cache_tokens": 80}
    assert parse_openai_usage({}) is None
    assert parse_openai_usage({"usage": {}}) == {
        "n_input_tokens": None, "n_output_tokens": None, "n_cache_tokens": None}


async def test_run_loop_emits_tracer_spans_in_order():
    # canned model: step 1 runs a command, step 2 submits.
    replies = ["```bash\necho hi\n```", "```bash\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n```"]
    state = {"i": 0}

    async def send(messages):
        r = replies[state["i"]]
        state["i"] += 1
        return r

    async def fake_exec(action, timeout_s):
        return 0, "hi\n"

    seen: list[dict] = []
    agent = CodingAgent(send_fn=send, step_limit=5)
    result = await agent.run_loop("do it", fake_exec, tracer=seen.append)

    kinds = [(s["kind"], s["seq"]) for s in seen]
    assert kinds == [("llm_call", 1), ("sandbox_exec", 1), ("llm_call", 2)]
    # send_fn path carries no usage -> token fields present but None
    assert seen[0]["n_input_tokens"] is None
    assert seen[1]["rc"] == 0
    # every span has start/end/duration
    assert all(s["start"] and s["end"] and s["duration_s"] is not None for s in seen)
    # send_fn path -> LoopResult tokens stay None
    assert result.n_input_tokens is None and result.exit_status == "submitted"


async def test_host_agent_writes_spans_and_tokens(tmp_path):
    from benchmaker.swebench.harbor_agent import BenchmakerHostAgent

    # Fake environment: exec echoes rc=0.
    class FakeExecRes:
        def __init__(self):
            self.return_code = 0
            self.stdout = "ok\n"
            self.stderr = ""

    class FakeEnv:
        async def exec(self, command, cwd=None, timeout_sec=None):
            return FakeExecRes()

    class FakeCtx:
        def __init__(self):
            self.metadata = None
            self.n_input_tokens = None
            self.n_output_tokens = None
            self.n_cache_tokens = None

    # Build the agent but swap in a loop whose model returns usage and submits.
    agent = BenchmakerHostAgent(logs_dir=tmp_path, model="m", api_base="http://x",
                                api_key="k")

    replies = ["```bash\nls\n```",
               "```bash\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nok\n```"]
    usages = [{"n_input_tokens": 100, "n_output_tokens": 10, "n_cache_tokens": 0},
              {"n_input_tokens": 120, "n_output_tokens": 5, "n_cache_tokens": 0}]
    state = {"i": 0}

    from benchmaker.swebench.agent import CodingAgent

    class TokenLoop(CodingAgent):
        async def _send_with_usage(self, messages):
            i = state["i"]; state["i"] += 1
            return replies[i], usages[i]

    agent._build_loop_agent = lambda: TokenLoop(url="http://x/chat/completions",
                                                model="m", api_key="k", step_limit=5)

    ctx = FakeCtx()
    await agent.run("do it", FakeEnv(), ctx)

    spans_file = tmp_path / "timeline-spans.jsonl"
    assert spans_file.exists()
    spans = [json.loads(x) for x in spans_file.read_text().splitlines() if x.strip()]
    kinds = [s["kind"] for s in spans]
    assert kinds == ["llm_call", "sandbox_exec", "llm_call"]
    assert spans[0]["n_input_tokens"] == 100
    # token totals populated onto the harbor context
    assert ctx.n_input_tokens == 220 and ctx.n_output_tokens == 15
    assert ctx.metadata["exit_status"] == "submitted"


async def test_pi_exec_bridge_emits_spans(tmp_path):
    import aiohttp
    from benchmaker.swebench.pi_agent import _ExecBridge

    class FakeExecRes:
        return_code = 0
        stdout = "hi\n"
        stderr = ""

    class FakeEnv:
        async def exec(self, command, cwd=None, timeout_sec=None):
            return FakeExecRes()

    spans_path = tmp_path / "timeline-spans.jsonl"
    bridge = _ExecBridge(FakeEnv(), cwd="/testbed", exec_timeout_s=30.0,
                         spans_path=spans_path)
    await bridge.start()
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{bridge.url}/exec",
                              json={"command": "echo hi"}) as resp:
                body = await resp.json()
                assert body["return_code"] == 0
    finally:
        await bridge.stop()

    spans = [json.loads(x) for x in spans_path.read_text().splitlines() if x.strip()]
    assert len(spans) == 1
    assert spans[0]["kind"] == "sandbox_exec" and spans[0]["rc"] == 0
    assert spans[0]["seq"] == 1 and spans[0]["duration_s"] is not None


def test_job_observer_write_produces_all_artifacts(tmp_path):
    # one trial dir with a fine-grained span file and a pi log
    tdir = tmp_path / "t1" / "logs"
    tdir.mkdir(parents=True)
    (tdir / "timeline-spans.jsonl").write_text(
        json.dumps(obs._span("", "", "llm_call", "llm_call", seq=1, duration_s=2.0,
                              n_input_tokens=50, n_output_tokens=5)) + "\n"
    )
    (tdir / "pi-host.log").write_text(
        json.dumps({"usage": {"input": 300, "output": 20}}) + "\n"
    )

    r = _fake_result(trial="t1", task="task1",
                     phases={"agent_execution": _ti(0, 10)},
                     rewards={"reward": 1.0}, totals=(350, 0, 25, 0.02),
                     metadata={"exit_status": "submitted"})

    observer = obs.JobObserver(flash_url=None, interval=5.0)
    observer._results = [r]                       # what the END hook would collect
    observer._util_rows = [{"sandbox_count": 4, "node_count": 6,
                            "available_node_count": 6}]
    summary_text = observer.write(tmp_path)

    timeline = [json.loads(x) for x in
                (tmp_path / "timeline.jsonl").read_text().splitlines() if x.strip()]
    kinds = sorted(s["kind"] for s in timeline)
    assert kinds == ["llm_call", "llm_call", "phase"]   # phase + fine + pi
    util = (tmp_path / "utilization.jsonl").read_text().splitlines()
    assert len(util) == 1
    manifest = [json.loads(x) for x in
                (tmp_path / "trajectories.jsonl").read_text().splitlines() if x.strip()]
    assert manifest[0]["trial"] == "t1" and manifest[0]["passed"] is True
    assert "observability" in summary_text


def test_harbor_eval_parse_args_has_timeline_flags(monkeypatch):
    import sys
    from benchmaker.swebench import harbor_eval as he

    monkeypatch.setattr(sys, "argv", ["prog", "--no-timeline",
                                      "--utilization-interval-sec", "2.5"])
    args = he._parse_args()
    assert args.timeline is False
    assert args.utilization_interval_sec == 2.5

    monkeypatch.setattr(sys, "argv", ["prog"])
    args = he._parse_args()
    assert args.timeline is True                  # default on
    assert args.utilization_interval_sec == 5.0

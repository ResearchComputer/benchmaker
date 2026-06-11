# tests/test_trajectory.py
"""Unit tests for benchmaker.swebench.trajectory (pure converter + key derivation)."""
from __future__ import annotations

import json

from benchmaker.swebench import trajectory as T


def test_recorded_turn_roundtrip():
    turn = T.RecordedTurn(
        index=0, content="hi", reasoning="because",
        tool_calls=[{"id": "call_1", "name": "bash", "arguments": {"command": "ls"}}],
        finish_reason="tool_calls",
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )
    rebuilt = T.RecordedTurn.from_dict(json.loads(json.dumps(turn.to_dict())))
    assert rebuilt == turn


def test_trajectory_roundtrip():
    traj = T.Trajectory(
        key="django__django-1", instance_id="django__django-1", model="m",
        trial="django__django-1__abc",
        turns=[T.RecordedTurn(0, "x", None, [], "stop", {})],
    )
    rebuilt = T.Trajectory.from_dict(json.loads(json.dumps(traj.to_dict())))
    assert rebuilt == traj
    assert traj.to_dict()["n_turns"] == 1


def test_task_key_parses_instance_id_from_prompt():
    messages = [
        {"role": "system", "content": "you are pi"},
        {"role": "user", "content": "Fix the bug.\n\n# Task: astropy__astropy-12907\nRepository: astropy"},
    ]
    assert T.task_key_from_messages(messages) == "astropy__astropy-12907"


def test_task_key_handles_list_content_and_hash_fallback():
    messages = [{"role": "user", "content": [{"type": "text", "text": "no task marker here"}]}]
    key = T.task_key_from_messages(messages)
    assert key.startswith("sha1:") and len(key) > 10


def _pi_log(*events) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def test_parse_pi_conversation_extracts_turns_and_key():
    log = _pi_log(
        {"type": "session", "id": "s1"},
        {"type": "message_start", "message": {"role": "user", "content": [
            {"type": "text", "text": "Fix it.\n# Task: django__django-11095\nRepository: django"}]}},
        "{ this is a corrupt line",  # must be skipped, not raise
        {"type": "turn_end", "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "let me look"},
                {"type": "text", "text": "investigating"},
                {"type": "toolCall", "id": "call_a", "name": "bash",
                 "arguments": {"command": "ls"}},
            ],
            "stopReason": "toolUse", "model": "zai-org/GLM-4.7-Flash",
            "usage": {"input": 1513, "output": 124, "cacheRead": 0, "cacheWrite": 5,
                      "totalTokens": 1637, "cost": {"total": 0.0}}}},
        {"type": "turn_end", "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "stopReason": "endTurn", "model": "zai-org/GLM-4.7-Flash",
            "usage": {"input": 2000, "output": 10, "totalTokens": 2010}}},
    )
    traj = T.parse_pi_conversation(log)
    assert traj.key == "django__django-11095"
    assert traj.instance_id == "django__django-11095"
    assert traj.model == "zai-org/GLM-4.7-Flash"
    assert len(traj.turns) == 2

    t0 = traj.turns[0]
    assert t0.index == 0
    assert t0.content == "investigating"
    assert t0.reasoning == "let me look"
    assert t0.finish_reason == "tool_calls"
    assert t0.tool_calls == [{"id": "call_a", "name": "bash", "arguments": {"command": "ls"}}]
    assert t0.usage == {"prompt_tokens": 1513, "completion_tokens": 124,
                        "total_tokens": 1637, "cache_read": 0, "cache_write": 5, "cost": 0.0}

    t1 = traj.turns[1]
    assert t1.finish_reason == "stop" and t1.content == "done"


def test_parse_pi_conversation_empty_log_is_safe():
    traj = T.parse_pi_conversation("")
    assert traj.turns == [] and traj.key == _key_helper("")


def test_parse_pi_conversation_skips_malformed_turn_end():
    log = _pi_log(
        {"type": "message_start", "message": {"role": "user", "content": "# Task: x__x-1\n"}},
        {"type": "turn_end"},                       # no message key
        {"type": "turn_end", "message": None},      # null message
        {"type": "turn_end", "message": {"role": "assistant",
            "content": [{"type": "text", "text": "ok"}], "stopReason": "endTurn",
            "model": "m", "usage": {"input": 1, "output": 1, "totalTokens": 2}}},
    )
    traj = T.parse_pi_conversation(log)
    assert len(traj.turns) == 1 and traj.turns[0].content == "ok"


def _key_helper(text: str) -> str:
    return T._key_from_text(text)


def _write_job(tmp_path, trial, log_text):
    d = tmp_path / trial / "agent"
    d.mkdir(parents=True)
    (d / "pi-container.log").write_text(log_text)


def test_convert_job_and_load_store(tmp_path):
    log_a = _pi_log(
        {"type": "message_start", "message": {"role": "user",
         "content": "x\n# Task: aa__aa-1\n"}},
        {"type": "turn_end", "message": {"role": "assistant",
         "content": [{"type": "text", "text": "a0"}], "stopReason": "endTurn",
         "model": "m", "usage": {"input": 1, "output": 1, "totalTokens": 2}}},
    )
    log_b = _pi_log(
        {"type": "message_start", "message": {"role": "user",
         "content": "y\n# Task: bb__bb-2\n"}},
        {"type": "turn_end", "message": {"role": "assistant",
         "content": [{"type": "toolCall", "id": "c1", "name": "bash",
                      "arguments": {"command": "ls"}}], "stopReason": "toolUse",
         "model": "m", "usage": {"input": 3, "output": 4, "totalTokens": 7}}},
    )
    _write_job(tmp_path, "aa__aa-1__x", log_a)
    _write_job(tmp_path, "bb__bb-2__y", log_b)

    out = tmp_path / "replay-trajectories.jsonl"
    n = T.convert_job(tmp_path, out)
    assert n == 2
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2

    store = T.load_store(out)
    assert set(store) == {"aa__aa-1", "bb__bb-2"}
    assert store["aa__aa-1"].trial == "aa__aa-1__x"
    assert store["bb__bb-2"].turns[0].tool_calls[0]["name"] == "bash"


def test_load_store_last_wins_on_duplicate_key(tmp_path):
    out = tmp_path / "t.jsonl"
    a = T.Trajectory(key="k", instance_id="k", model="m",
                     turns=[T.RecordedTurn(0, "first", None, [], "stop", {})])
    b = T.Trajectory(key="k", instance_id="k", model="m",
                     turns=[T.RecordedTurn(0, "second", None, [], "stop", {})])
    out.write_text(json.dumps(a.to_dict()) + "\n" + json.dumps(b.to_dict()) + "\n")
    store = T.load_store(out)
    assert store["k"].turns[0].content == "second"


def test_convert_job_empty_dir_writes_no_file(tmp_path):
    out = tmp_path / "out.jsonl"
    n = T.convert_job(tmp_path, out)
    assert n == 0
    assert not out.exists()

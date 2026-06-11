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
            "usage": {"input": 1513, "output": 124, "cacheRead": 0,
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
                        "total_tokens": 1637, "cache_read": 0, "cost": 0.0}

    t1 = traj.turns[1]
    assert t1.finish_reason == "stop" and t1.content == "done"


def test_parse_pi_conversation_empty_log_is_safe():
    traj = T.parse_pi_conversation("")
    assert traj.turns == [] and traj.key == _key_helper("")


def _key_helper(text: str) -> str:
    return T._key_from_text(text)

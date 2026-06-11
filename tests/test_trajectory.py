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

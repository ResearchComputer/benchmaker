# tests/test_replay_server.py
"""Unit + integration tests for the trajectory replay server."""
from __future__ import annotations

import json

from benchmaker.swebench import replay_server as R
from benchmaker.swebench.trajectory import RecordedTurn, Trajectory


def _store():
    traj = Trajectory(
        key="aa__aa-1", instance_id="aa__aa-1", model="m", trial="aa__aa-1__x",
        turns=[
            RecordedTurn(0, "look", "reason", [
                {"id": "call_a", "name": "bash", "arguments": {"command": "ls"}}],
                "tool_calls", {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}),
            RecordedTurn(1, "done", None, [], "stop",
                {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21}),
        ],
    )
    return {traj.key: traj}


def _messages(n_assistant: int):
    msgs = [{"role": "user", "content": "Fix.\n# Task: aa__aa-1\n"}]
    for i in range(n_assistant):
        msgs.append({"role": "assistant", "content": f"a{i}"})
        msgs.append({"role": "tool", "content": "result"})
    return msgs


def test_select_turn_by_assistant_count():
    store = _store()
    turn, key, idx = R.select_turn(store, _messages(0))
    assert key == "aa__aa-1" and idx == 0 and turn.content == "look"
    turn, _, idx = R.select_turn(store, _messages(1))
    assert idx == 1 and turn.content == "done"


def test_select_turn_overflow_and_unknown_key():
    store = _store()
    turn, _, idx = R.select_turn(store, _messages(2))
    assert turn is None and idx == 2
    turn, key, _ = R.select_turn(store, [{"role": "user", "content": "# Task: nope-9\n"}])
    assert turn is None and key == "nope-9"


def test_turn_to_openai_response_shape():
    store = _store()
    turn, _, _ = R.select_turn(store, _messages(0))
    resp = R.turn_to_openai_response(turn, "m", response_id="id-0")
    assert resp["object"] == "chat.completion"
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    msg = choice["message"]
    assert msg["content"] == "look" and msg["reasoning_content"] == "reason"
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_a" and tc["type"] == "function"
    assert tc["function"]["name"] == "bash"
    assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}
    assert resp["usage"]["completion_tokens"] == 2


def test_turn_to_sse_lines_terminate_with_done():
    store = _store()
    turn, _, _ = R.select_turn(store, _messages(1))
    lines = R.turn_to_sse_lines(turn, "m", response_id="id-1")
    assert lines[-1] == "data: [DONE]\n\n"
    first = json.loads(lines[0][len("data: "):])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["content"] == "done"
    final = json.loads(lines[1][len("data: "):])
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["total_tokens"] == 21

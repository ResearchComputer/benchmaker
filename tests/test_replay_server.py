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


import socket

import pytest
from aiohttp import web
import aiohttp


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.asyncio
async def test_app_serves_recorded_turns_streaming_and_not():
    app = R.as_app(_store())
    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    base = f"http://127.0.0.1:{port}/v1/chat/completions"
    try:
        async with aiohttp.ClientSession() as s:
            # turn 0 (non-streaming) -> tool call
            async with s.post(base, json={"model": "m", "stream": False,
                    "messages": _messages(0)}) as resp:
                body = await resp.json()
            assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"

            # turn 1 (streaming) -> final text + [DONE]
            async with s.post(base, json={"model": "m", "stream": True,
                    "messages": _messages(1)}) as resp:
                text = await resp.text()
            assert "data: [DONE]" in text
            assert '"content": "done"' in text

            # overflow -> terminal stop, miss counted
            async with s.post(base, json={"model": "m", "stream": False,
                    "messages": _messages(5)}) as resp:
                body = await resp.json()
            assert body["choices"][0]["finish_reason"] == "stop"
            assert R.get_misses(app) == 1
    finally:
        await runner.cleanup()


def test_empty_content_turn_serializes_content_null():
    turn = RecordedTurn(0, "", None,
                        [{"id": "c1", "name": "bash", "arguments": {}}],
                        "tool_calls", {})
    resp = R.turn_to_openai_response(turn, "m", response_id="x")
    assert resp["choices"][0]["message"]["content"] is None
    assert resp["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"

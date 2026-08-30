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
import httpx2


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
        async with httpx2.AsyncClient() as s:
            # turn 0 (non-streaming) -> tool call
            resp = await s.post(base, json={"model": "m", "stream": False,
                    "messages": _messages(0)})
            body = resp.json()
            assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"

            # turn 1 (streaming) -> final text + [DONE]
            resp = await s.post(base, json={"model": "m", "stream": True,
                    "messages": _messages(1)})
            text = resp.text
            assert "data: [DONE]" in text
            assert '"content": "done"' in text
            # default path (no tokenizer) stays single-chunk: one content delta
            # chunk + one finish chunk, no per-token pacing.
            chunk_count = sum(1 for l in text.splitlines() if l.startswith("data: {"))
            assert chunk_count == 2

            # overflow -> terminal stop, miss counted
            resp = await s.post(base, json={"model": "m", "stream": False,
                    "messages": _messages(5)})
            body = resp.json()
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


class _StubFast:
    """Mimics a HF fast tokenizer's offset-mapping call (no transformers dep)."""
    is_fast = True

    def __init__(self, spans_for):
        self._spans_for = spans_for  # callable: text -> list[(start, end)]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert return_offsets_mapping
        assert not add_special_tokens
        return {"offset_mapping": self._spans_for(text)}


def test_offset_tokenizer_reconstructs_exactly():
    # 2-char tokens
    tok = R._HFOffsetTokenizer(
        _StubFast(lambda t: [(i, min(i + 2, len(t))) for i in range(0, len(t), 2)]))
    s = "Hello, world!\n  indented"
    pieces = tok.tokenize_pieces(s)
    assert "".join(pieces) == s
    assert len(pieces) == (len(s) + 1) // 2
    assert tok.tokenize_pieces("") == []


def test_offset_tokenizer_handles_gaps_and_specials():
    # spans skip some chars (e.g. zero-width/special tokens); reconstruction must
    # still be exact by attaching skipped chars to the next piece.
    tok = R._HFOffsetTokenizer(_StubFast(lambda t: [(0, 1), (0, 0), (2, 3)]))
    pieces = tok.tokenize_pieces("abcd")
    assert "".join(pieces) == "abcd"
    assert pieces == ["a", "bc", "d"]  # gap at index 1 folds into the next piece


def test_offset_tokenizer_absorbs_leading_gap():
    # first span starts at 2: chars 0-1 fold into the first piece
    tok = R._HFOffsetTokenizer(_StubFast(lambda t: [(2, 4), (4, 6)]))
    pieces = tok.tokenize_pieces("abcdef")
    assert "".join(pieces) == "abcdef"
    assert pieces[0] == "abcd"


class _CharTok:
    """One character per token; exact reconstruction, deterministic count."""

    def tokenize_pieces(self, text):
        return list(text)


async def test_stream_turn_order_reconstruction_and_timing(monkeypatch):
    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr(R.asyncio, "sleep", fake_sleep)

    turn = RecordedTurn(
        0, "Hi", "rs",
        [{"id": "c1", "name": "bash", "arguments": {"x": 1}}],
        "tool_calls",
        {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4})

    lines = []
    async for line in R.stream_turn(turn, "m", response_id="id",
                                    tokenizer=_CharTok(), inter_token_ms=50.0):
        lines.append(line)

    assert lines[-1] == "data: [DONE]\n\n"
    payloads = [json.loads(l[len("data: "):]) for l in lines if l.startswith("data: {")]
    deltas = [p["choices"][0]["delta"] for p in payloads]

    # role appears on the first delta only
    assert deltas[0].get("role") == "assistant"
    assert all("role" not in d for d in deltas[1:])

    # each field reconstructs byte-exact
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    args = "".join(tc["function"].get("arguments", "")
                   for d in deltas for tc in d.get("tool_calls", []))
    assert reasoning == "rs"
    assert content == "Hi"
    assert args == json.dumps({"x": 1})

    # ordering: reasoning before content before tool args
    order = []
    for d in deltas:
        if d.get("reasoning_content"):
            order.append("r")
        elif d.get("content"):
            order.append("c")
        elif d.get("tool_calls"):
            order.append("t")
    assert order == sorted(order, key="rct".index)

    # N-1 sleeps total; the first token of the turn is instant (prefill free,
    # TTFT≈0). struct openers add no sleep; the first args token still paces
    # relative to the prior content token.
    n_tokens = len("rs") + len("Hi") + len(json.dumps({"x": 1}))
    assert len(sleeps) == n_tokens - 1
    assert all(d == 0.05 for d in sleeps)

    # final chunk carries recorded finish_reason + usage
    final = payloads[-1]
    assert final["choices"][0]["finish_reason"] == "tool_calls"
    assert final["usage"]["total_tokens"] == 4


async def test_stream_turn_empty_turn_streams_instantly(monkeypatch):
    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr(R.asyncio, "sleep", fake_sleep)

    turn = R._terminal_turn()
    lines = []
    async for line in R.stream_turn(turn, "m", response_id="id",
                                    tokenizer=_CharTok(), inter_token_ms=50.0):
        lines.append(line)
    assert sleeps == []  # 0 tokens -> no pacing
    assert lines[-1] == "data: [DONE]\n\n"
    final = json.loads(lines[-2][len("data: "):])
    assert final["choices"][0]["finish_reason"] == "stop"


async def test_stream_turn_no_pacing_when_delay_zero(monkeypatch):
    # content-only turn with inter_token_ms=0: the delay>0 guard must skip
    # asyncio.sleep entirely even though there are many tokens; output is still
    # byte-exact and properly terminated.
    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr(R.asyncio, "sleep", fake_sleep)

    turn = RecordedTurn(0, "hello world", None, [], "stop",
                        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
    lines = []
    async for line in R.stream_turn(turn, "m", response_id="id",
                                    tokenizer=_CharTok(), inter_token_ms=0.0):
        lines.append(line)

    assert sleeps == []  # delay==0 -> guard skips sleep even with tokens
    payloads = [json.loads(l[len("data: "):]) for l in lines if l.startswith("data: {")]
    deltas = [p["choices"][0]["delta"] for p in payloads]
    assert "".join(d.get("content", "") for d in deltas) == "hello world"
    assert deltas[0].get("role") == "assistant"
    assert lines[-1] == "data: [DONE]\n\n"
    final = payloads[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["total_tokens"] == 3


@pytest.mark.asyncio
async def test_app_timed_streaming_reconstructs_over_http():
    app = R.as_app(_store(), tokenizer=_CharTok(), inter_token_ms=1.0)
    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    base = f"http://127.0.0.1:{port}/v1/chat/completions"
    try:
        async with httpx2.AsyncClient() as s:
            resp = await s.post(base, json={"model": "m", "stream": True,
                    "messages": _messages(0)})
            text = resp.text
        payloads = [json.loads(l[len("data: "):]) for l in text.splitlines()
                    if l.startswith("data: {")]
        deltas = [p["choices"][0]["delta"] for p in payloads]
        # turn 0 in _store(): content "look", reasoning "reason", bash tool call
        assert "".join(d.get("content", "") for d in deltas) == "look"
        assert "".join(d.get("reasoning_content", "") for d in deltas) == "reason"
        assert "".join(tc["function"].get("arguments", "")
                       for d in deltas for tc in d.get("tool_calls", [])
                       ) == json.dumps({"command": "ls"})
        # many deltas (one per char), not a single burst
        assert len(deltas) > 5
        assert "data: [DONE]" in text
    finally:
        await runner.cleanup()


def test_resolve_tokenizer_guard():
    # disabled (itt<=0) -> None without importing transformers; a spec is
    # silently ignored when disabled (not an error).
    assert R._resolve_tokenizer(0.0, None, True) is None
    assert R._resolve_tokenizer(0.0, "ignored", True) is None
    # enabled without a spec -> clear error
    with pytest.raises(ValueError, match="requires --tokenizer"):
        R._resolve_tokenizer(50.0, None, True)


def _drive_replay(store, prompt):
    """Simulate pi's open-loop agent against the replay server: keep calling
    `select_turn` (keyed by assistant-message count), append the served assistant
    message, and continue. pi advances ONLY when the served turn carries tool
    calls to execute; an action-less turn ends the loop regardless of
    `finish_reason` (verified against the real pi binary). Returns the number of
    recorded turns the agent actually consumes."""
    msgs = [{"role": "user", "content": prompt}]
    consumed = 0
    while True:
        turn, _key, _idx = R.select_turn(store, msgs)
        if turn is None:                       # miss / turn overflow -> terminal stop
            break
        consumed += 1
        am = {"role": "assistant", "content": turn.content or None}
        if turn.tool_calls:
            am["tool_calls"] = [{"id": tc["id"], "type": "function",
                                 "function": {"name": tc["name"], "arguments": "{}"}}
                                for tc in turn.tool_calls]
        msgs.append(am)
        if not turn.tool_calls:                # no tool to run -> pi halts (any finish_reason)
            break
        msgs.append({"role": "tool", "content": "result"})  # tool ran, loop for next turn
    return consumed


def _traj_with_actionless_midturn():
    return Trajectory(key="aa__aa-1", instance_id="aa__aa-1", model="m", turns=[
        RecordedTurn(0, "", None, [{"id": "c0", "name": "bash", "arguments": {}}], "tool_calls", {}),
        RecordedTurn(1, "", None, [], "stop", {}),    # action-less mid turn (the bug trigger)
        RecordedTurn(2, "", None, [{"id": "c2", "name": "write", "arguments": {}}], "tool_calls", {}),
        RecordedTurn(3, "done", None, [], "stop", {}),  # genuine final stop
    ])


def test_replay_truncates_at_actionless_midturn_without_fix():
    # Documents the bug: a verbatim action-less mid turn halts the agent early,
    # dropping the later turns (incl. the write at idx 2).
    traj = _traj_with_actionless_midturn()
    store = {traj.key: traj}
    assert _drive_replay(store, "# Task: aa__aa-1\n") == 2  # stops at the idx-1 action-less turn


def test_replay_consumes_all_turns_after_drop():
    from benchmaker.swebench.trajectory import drop_nonfinal_actionless_turns
    traj = _traj_with_actionless_midturn()
    traj.turns = drop_nonfinal_actionless_turns(traj.turns)
    store = {traj.key: traj}
    # consumed == recorded: every remaining turn is consumed; the agent reaches
    # the write and the genuine final stop (3 turns after the no-op is elided).
    assert _drive_replay(store, "# Task: aa__aa-1\n") == len(traj.turns) == 3


def _store_with_results():
    traj = Trajectory(
        key="aa__aa-1", instance_id="aa__aa-1", model="m",
        turns=[
            RecordedTurn(0, "look", None,
                         [{"id": "c0", "name": "bash", "arguments": {}}],
                         "tool_calls", {}, tool_results=[{"name": "bash", "status": 0}]),
            RecordedTurn(1, "done", None, [], "stop", {}),
        ],
    )
    return {traj.key: traj}


def _req(tool_content):
    return {"messages": [
        {"role": "user", "content": "# Task: aa__aa-1\n"},
        {"role": "assistant", "content": "look",
         "tool_calls": [{"id": "c0", "type": "function",
                         "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c0", "content": tool_content},
    ]}


def test_divergence_none_when_status_matches():
    store = _store_with_results()
    prev = store["aa__aa-1"].turns[0]
    msgs = _req("returncode: 0\nok\n")["messages"]
    assert R._divergence(prev, msgs) is None


def test_divergence_detected_when_status_flips():
    store = _store_with_results()
    prev = store["aa__aa-1"].turns[0]
    msgs = _req("returncode: -1\ncommand timed out\n")["messages"]
    d = R._divergence(prev, msgs)
    assert d is not None and d[0] == "bash" and d[1] == 0 and d[2] == -1


def test_divergence_skipped_when_recorded_status_none():
    store = _store_with_results()
    prev = store["aa__aa-1"].turns[0]
    prev.tool_results = [{"name": "bash", "status": None}]
    msgs = _req("returncode: -1\n")["messages"]
    assert R._divergence(prev, msgs) is None


def test_divergence_id_keyed_reordering_ok():
    prev = RecordedTurn(0, "", None,
                        [{"id": "a", "name": "bash", "arguments": {}},
                         {"id": "b", "name": "bash", "arguments": {}}],
                        "tool_calls", {},
                        tool_results=[{"name": "bash", "status": 0},
                                      {"name": "bash", "status": 0}])
    msgs = [
        {"role": "user", "content": "# Task: aa__aa-1\n"},
        {"role": "assistant", "content": "x"},
        {"role": "tool", "tool_call_id": "b", "content": "returncode: 0\n"},
        {"role": "tool", "tool_call_id": "a", "content": "returncode: 0\n"},
    ]
    assert R._divergence(prev, msgs) is None


def test_validate_off_ignores_tool_results():
    store = _store_with_results()
    app = R.as_app(store, model_fallback="m")  # validate defaults False
    assert R.get_divergences(app) == 0

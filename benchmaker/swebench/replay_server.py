# benchmaker/swebench/replay_server.py
"""Stateless OpenAI-compatible replay server for recorded SWE-bench trajectories.

It serves recorded LLM outputs back to an unchanged pi + harbor pipeline so a run
can be re-evaluated deterministically at any concurrency. Because pi sends the
full message history on every call, the response is chosen purely from the
request body — `(task key from the first user message, count of assistant
messages so far)` — so the server holds no mutable per-session state and is
correct under arbitrary concurrency. See
`docs/superpowers/specs/2026-06-11-swebench-trajectory-replay-design.md`.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from aiohttp import web

from benchmaker.swebench.trajectory import (
    RecordedTurn, Trajectory, load_store, task_key_from_messages,
)

log = logging.getLogger("benchmaker.replay_server")

# aiohttp >=3.9 prefers typed AppKey over string keys (avoids NotAppKeyWarning).
# MISSES_KEY holds a one-element list so the miss count is mutated *in place* —
# the handler never reassigns an app key after startup (which aiohttp deprecates).
STORE_KEY: "web.AppKey" = web.AppKey("store", dict)
MISSES_KEY: "web.AppKey" = web.AppKey("misses", list)


def count_assistant_messages(messages: Any) -> int:
    return sum(1 for m in (messages or [])
               if isinstance(m, dict) and m.get("role") == "assistant")


def select_turn(store: dict, messages: Any):
    """Return `(RecordedTurn | None, key, turn_index)` for a request's messages.
    `turn_index` is the count of assistant messages already present."""
    key = task_key_from_messages(messages)
    idx = count_assistant_messages(messages)
    traj = store.get(key)
    if traj is None or idx >= len(traj.turns):
        return None, key, idx
    return traj.turns[idx], key, idx


def _message_dict(turn: RecordedTurn) -> dict:
    # OpenAI returns content=null (not "") on tool-only turns; match that so a
    # client that branches on `content is None` behaves as it did against the
    # live server.
    msg: dict[str, Any] = {"role": "assistant", "content": turn.content or None}
    if turn.reasoning:
        msg["reasoning_content"] = turn.reasoning
    if turn.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.get("id") or f"call_{turn.index}_{i}",
                "type": "function",
                "function": {
                    "name": tc.get("name") or "",
                    "arguments": json.dumps(tc.get("arguments") or {}),
                },
            }
            for i, tc in enumerate(turn.tool_calls)
        ]
    return msg


def _openai_usage(usage: dict) -> dict:
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def turn_to_openai_response(turn: RecordedTurn, model: str, *, response_id: str) -> dict:
    return {
        "id": response_id, "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": _message_dict(turn),
                     "finish_reason": turn.finish_reason}],
        "usage": _openai_usage(turn.usage),
    }


def turn_to_sse_lines(turn: RecordedTurn, model: str, *, response_id: str) -> list[str]:
    """Reconstruct a minimal OpenAI SSE stream for `turn`.

    Deterministic replay does not need token-level streaming, so we emit the
    whole assistant message as a single content+tool_calls delta, then a finish
    chunk carrying `finish_reason` + `usage`, then `[DONE]`. Each tool_call
    carries an `index` (required by OpenAI-compatible streaming parsers).
    """
    created = int(time.time())
    delta = _message_dict(turn)
    for i, tc in enumerate(delta.get("tool_calls", [])):
        tc["index"] = i  # streaming tool_calls require an index
    first = {"id": response_id, "object": "chat.completion.chunk", "created": created,
             "model": model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    final = {"id": response_id, "object": "chat.completion.chunk", "created": created,
             "model": model,
             "choices": [{"index": 0, "delta": {}, "finish_reason": turn.finish_reason}],
             "usage": _openai_usage(turn.usage)}
    return [f"data: {json.dumps(first)}\n\n",
            f"data: {json.dumps(final)}\n\n",
            "data: [DONE]\n\n"]


def _terminal_turn() -> RecordedTurn:
    """A clean stop, served on a replay miss so a divergent run ends not hangs."""
    return RecordedTurn(index=-1, content="", reasoning=None, tool_calls=[],
                        finish_reason="stop", usage={})


def as_app(store: dict, *, model_fallback: str = "") -> web.Application:
    """Build the aiohttp app. Replay misses (unknown key or turn overflow) are
    counted in place; read them with `get_misses(app)`."""
    app = web.Application()
    app[STORE_KEY] = store
    app[MISSES_KEY] = [0]

    async def handle(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        messages = body.get("messages") or []
        model = body.get("model") or model_fallback
        stream = bool(body.get("stream"))
        turn, key, idx = select_turn(store, messages)
        if turn is None:
            app[MISSES_KEY][0] += 1
            log.warning("replay miss key=%s turn_index=%s (serving terminal stop)", key, idx)
            turn = _terminal_turn()
        response_id = f"chatcmpl-replay-{key}-{idx}"
        if stream:
            resp = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for line in turn_to_sse_lines(turn, model, response_id=response_id):
                await resp.write(line.encode())
            await resp.write_eof()
            return resp
        return web.json_response(
            turn_to_openai_response(turn, model, response_id=response_id))

    # Accept both /v1/... and /... so it works whichever base URL the agent uses.
    app.router.add_post("/v1/chat/completions", handle)
    app.router.add_post("/chat/completions", handle)
    return app


def get_misses(app: web.Application) -> int:
    """Number of replay misses served by `app` (read after the run)."""
    return app[MISSES_KEY][0]


async def start_server(store: dict, host: str, port: int, *,
                       model_fallback: str = "") -> web.AppRunner:
    """Start the app on host:port and return its (already set up) runner.
    Caller is responsible for `await runner.cleanup()`."""
    app = as_app(store, model_fallback=model_fallback)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import asyncio
    p = argparse.ArgumentParser(
        description="Serve recorded SWE-bench trajectories as an "
                    "OpenAI-compatible replay endpoint.")
    p.add_argument("trajectories", help="Path to replay-trajectories.jsonl.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9100)
    a = p.parse_args(argv)
    store = load_store(a.trajectories)
    fallback = next((t.model for t in store.values() if t.model), "")
    print(f"loaded {len(store)} trajectories; serving on "
          f"http://{a.host}:{a.port}/v1/chat/completions")

    async def _run() -> None:
        runner = await start_server(store, a.host, a.port, model_fallback=fallback)
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:  # pragma: no cover
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

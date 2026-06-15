# benchmaker/swebench/replay_server.py
"""Stateless OpenAI-compatible replay server for recorded SWE-bench trajectories.

It serves recorded LLM outputs back to an unchanged pi + harbor pipeline so a run
can be re-evaluated deterministically at any concurrency. Because pi sends the
full message history on every call, the response is chosen purely from the
request body — `(task key from the first user message, count of assistant
messages so far)` — so the server holds no mutable per-session state and is
correct under arbitrary concurrency. See
`docs/superpowers/specs/2026-06-11-swebench-trajectory-replay-design.md`.

Streaming is single-chunk by default. Pass a `tokenizer` + `inter_token_ms`
(CLI: --tokenizer / --inter-token-time) to pace streaming one token per SSE
chunk with prefill free (TTFT≈0); output stays byte-exact and the reported
`usage` stays the recorded value.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional, Protocol

from aiohttp import web

from benchmaker.swebench.trajectory import (
    RecordedTurn, Trajectory, derive_tool_status, load_store, task_key_from_messages,
)

log = logging.getLogger("benchmaker.replay_server")

# aiohttp >=3.9 prefers typed AppKey over string keys (avoids NotAppKeyWarning).
# MISSES_KEY holds a one-element list so the miss count is mutated *in place* —
# the handler never reassigns an app key after startup (which aiohttp deprecates).
STORE_KEY: "web.AppKey" = web.AppKey("store", dict)
MISSES_KEY: "web.AppKey" = web.AppKey("misses", list)
DIVERGENCES_KEY: "web.AppKey" = web.AppKey("divergences", list)
TOKENIZER_KEY: "web.AppKey" = web.AppKey("tokenizer", object)
ITT_KEY: "web.AppKey" = web.AppKey("inter_token_ms", float)


class Tokenizer(Protocol):
    """Splits text into ordered substrings whose concatenation equals the input."""

    def tokenize_pieces(self, text: str) -> list[str]: ...


class _HFOffsetTokenizer:
    """Wrap a HF *fast* tokenizer; split text into exact substrings via offset
    mapping. Per-token `decode()` of byte-level BPE ids drops/merges leading
    spaces and would break reconstruction, so we slice the original string by the
    tokenizer's char offsets instead — `"".join(pieces) == text` always."""

    def __init__(self, tok: Any) -> None:
        self._tok = tok

    def tokenize_pieces(self, text: str) -> list[str]:
        if not text:
            return []
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        spans = [(s, e) for (s, e) in enc["offset_mapping"] if e > s]
        if not spans:
            return [text]
        pieces: list[str] = []
        cursor = 0
        for _s, e in spans:
            if e < cursor:
                raise ValueError(
                    "offset_mapping spans are not in ascending order "
                    f"(end={e!r} < cursor={cursor!r}); tokenizer not supported")
            # text[cursor:e] folds any chars the tokenizer skipped (gaps, zero-
            # width specials) into the front of this token's piece.
            pieces.append(text[cursor:e])
            cursor = e
        if cursor < len(text):
            pieces.append(text[cursor:])
        return pieces


def load_tokenizer(spec: str, *, trust_remote_code: bool = True) -> Tokenizer:
    """Load an HF fast tokenizer by id/path for timed streaming. Lazy-imports
    `transformers` so default installs never need it."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "timed streaming needs `transformers`; install with "
            "`pip install 'benchmaker[tokenizer]'`") from e
    tok = AutoTokenizer.from_pretrained(spec, trust_remote_code=trust_remote_code)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError(
            f"tokenizer {spec!r} is not a fast tokenizer; offset mapping "
            "(needed for exact reconstruction) is unavailable")
    return _HFOffsetTokenizer(tok)


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


def _tool_content_text(content: Any) -> str:
    """OpenAI tool messages carry `content` as a str (sometimes a blocks list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _trailing_tool_messages(messages: Any) -> list[dict]:
    """The role:'tool' messages after the last assistant message (this turn's results)."""
    out: list[dict] = []
    for m in reversed(messages or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            break
        if role == "tool":
            out.append(m)
    out.reverse()
    return out


def _divergence(prev_turn: RecordedTurn, messages: Any):
    """Return (tool_name, recorded_status, incoming_status) for the first tool
    result of `prev_turn` whose derived status differs from the recording, else
    None. Matched by tool_call_id (order-independent). None status on either side
    is skipped (never a divergence)."""
    recorded: dict[str, tuple[str, Optional[int]]] = {}
    results = prev_turn.tool_results or []
    for tc, tr in zip(prev_turn.tool_calls, results):
        tcid = tc.get("id")
        if tcid:
            recorded[tcid] = (tr.get("name") or tc.get("name") or "", tr.get("status"))
    for m in _trailing_tool_messages(messages):
        tcid = m.get("tool_call_id")
        if tcid not in recorded:
            continue
        name, rec_status = recorded[tcid]
        inc_status = derive_tool_status(name, _tool_content_text(m.get("content")))
        if rec_status is not None and inc_status is not None and rec_status != inc_status:
            return (name, rec_status, inc_status)
    return None


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


def _chunk_line(delta: dict, *, response_id: str, model: str, created: int,
                finish_reason: Optional[str] = None,
                usage: Optional[dict] = None) -> str:
    obj: dict[str, Any] = {
        "id": response_id, "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        obj["usage"] = usage
    return f"data: {json.dumps(obj)}\n\n"


def _stream_chunks(turn: RecordedTurn, tokenizer: Tokenizer):
    """Yield `(kind, delta)` in stream order. kind is 'token' (paced) or
    'struct' (instant tool-call opener)."""
    if turn.reasoning:
        for piece in tokenizer.tokenize_pieces(turn.reasoning):
            yield "token", {"reasoning_content": piece}
    if turn.content:
        for piece in tokenizer.tokenize_pieces(turn.content):
            yield "token", {"content": piece}
    for i, tc in enumerate(turn.tool_calls):
        yield "struct", {"tool_calls": [{
            "index": i,
            "id": tc.get("id") or f"call_{turn.index}_{i}",
            "type": "function",
            "function": {"name": tc.get("name") or "", "arguments": ""},
        }]}
        args_json = json.dumps(tc.get("arguments") or {})
        for piece in tokenizer.tokenize_pieces(args_json):
            yield "token", {"tool_calls": [
                {"index": i, "function": {"arguments": piece}}]}


async def stream_turn(turn: RecordedTurn, model: str, *, response_id: str,
                      tokenizer: Tokenizer, inter_token_ms: float):
    """Stream `turn` as paced OpenAI SSE chunks: one chunk per token, sleeping
    `inter_token_ms` before each token after the first (prefill free, TTFT≈0).
    Reconstructs byte-exact; final chunk carries recorded finish_reason + usage."""
    created = int(time.time())
    delay = max(0.0, inter_token_ms) / 1000.0
    first = True
    emitted_token = False
    for kind, delta in _stream_chunks(turn, tokenizer):
        if kind == "token" and emitted_token and delay > 0:
            await asyncio.sleep(delay)
        if first:
            delta = {"role": "assistant", **delta}
            first = False
        if kind == "token":
            emitted_token = True
        yield _chunk_line(delta, response_id=response_id, model=model, created=created)
    if first:  # nothing streamed (empty / terminal turn) — still send a role chunk
        yield _chunk_line({"role": "assistant"}, response_id=response_id,
                          model=model, created=created)
    yield _chunk_line({}, response_id=response_id, model=model, created=created,
                      finish_reason=turn.finish_reason, usage=_openai_usage(turn.usage))
    yield "data: [DONE]\n\n"


def _terminal_turn() -> RecordedTurn:
    """A clean stop, served on a replay miss so a divergent run ends not hangs."""
    return RecordedTurn(index=-1, content="", reasoning=None, tool_calls=[],
                        finish_reason="stop", usage={})


def as_app(store: dict, *, model_fallback: str = "",
           tokenizer: Optional[Tokenizer] = None,
           inter_token_ms: float = 0.0,
           validate: bool = False) -> web.Application:
    """Build the aiohttp app. Replay misses (unknown key or turn overflow) are
    counted in place; read them with `get_misses(app)`. When `tokenizer` is set
    and `inter_token_ms > 0`, streaming responses are paced one token per chunk."""
    app = web.Application()
    app[STORE_KEY] = store
    app[MISSES_KEY] = [0]
    app[DIVERGENCES_KEY] = [0]
    app[TOKENIZER_KEY] = tokenizer
    app[ITT_KEY] = float(inter_token_ms)

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
        elif validate and idx >= 1:
            traj = store.get(key)
            prev = traj.turns[idx - 1] if (traj and idx - 1 < len(traj.turns)) else None
            if prev is not None and prev.tool_results:
                d = _divergence(prev, messages)
                if d is not None:
                    app[DIVERGENCES_KEY][0] += 1
                    log.warning("replay divergence key=%s turn_index=%s tool=%s "
                                "recorded_status=%s incoming_status=%s "
                                "(serving terminal stop)", key, idx - 1, d[0], d[1], d[2])
                    turn = _terminal_turn()
        response_id = f"chatcmpl-replay-{key}-{idx}"
        if stream:
            resp = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            tok = app[TOKENIZER_KEY]
            itt = app[ITT_KEY]
            if tok is not None and itt > 0:
                async for line in stream_turn(turn, model, response_id=response_id,
                                              tokenizer=tok, inter_token_ms=itt):
                    await resp.write(line.encode())
            else:
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


def get_divergences(app: web.Application) -> int:
    """Number of content divergences detected by `app` (read after the run)."""
    return app[DIVERGENCES_KEY][0]


async def start_server(store: dict, host: str, port: int, *,
                       model_fallback: str = "",
                       tokenizer: Optional[Tokenizer] = None,
                       inter_token_ms: float = 0.0) -> web.AppRunner:
    """Start the app on host:port and return its (already set up) runner.
    Caller is responsible for `await runner.cleanup()`."""
    app = as_app(store, model_fallback=model_fallback,
                 tokenizer=tokenizer, inter_token_ms=inter_token_ms)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner


def _resolve_tokenizer(inter_token_ms: float, spec: Optional[str],
                       trust_remote_code: bool) -> Optional[Tokenizer]:
    """Map CLI args to a tokenizer (or None). Raises ValueError when timed
    streaming is requested without a tokenizer spec."""
    if inter_token_ms <= 0:
        return None
    if not spec:
        raise ValueError(
            "timed streaming (--inter-token-time > 0) requires --tokenizer")
    return load_tokenizer(spec, trust_remote_code=trust_remote_code)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Serve recorded SWE-bench trajectories as an "
                    "OpenAI-compatible replay endpoint.")
    p.add_argument("trajectories", help="Path to replay-trajectories.jsonl.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9100)
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer id/path; required for timed streaming.")
    p.add_argument("--inter-token-time", type=float, default=0.0,
                   metavar="MS",
                   help="Milliseconds between streamed tokens "
                        "(0 = single-chunk stream, the default). Requires --tokenizer.")
    p.add_argument("--trust-remote-code", dest="trust_remote_code",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Allow tokenizer-repo code (GLM-4.7-Flash needs it). "
                        "Default: on.")
    a = p.parse_args(argv)
    # Resolve the tokenizer first so a bad flag combination fails fast, before
    # the (potentially large) trajectory file is read.
    try:
        tokenizer = _resolve_tokenizer(
            a.inter_token_time, a.tokenizer, a.trust_remote_code)
    except ValueError as e:
        p.error(str(e))
        return 1  # unreachable: p.error() exits; keeps `tokenizer` bound for type-checkers
    if tokenizer is not None:
        print(f"timed streaming: {a.inter_token_time} ms/token "
              f"(tokenizer={a.tokenizer})")
    elif a.tokenizer:
        log.warning("--tokenizer is set but --inter-token-time is 0; "
                    "streaming stays single-chunk")
    store = load_store(a.trajectories)
    fallback = next((t.model for t in store.values() if t.model), "")
    print(f"loaded {len(store)} trajectories; serving on "
          f"http://{a.host}:{a.port}/v1/chat/completions")

    async def _run() -> None:
        runner = await start_server(
            store, a.host, a.port, model_fallback=fallback,
            tokenizer=tokenizer, inter_token_ms=a.inter_token_time)
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

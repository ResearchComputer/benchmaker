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
    msg: dict[str, Any] = {"role": "assistant", "content": turn.content or ""}
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

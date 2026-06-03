"""Canonical SFT-warmup record + source normalizers.

This module defines the *on-disk contract* for the agent-warmup dataset (the
"protocol", in benchmaker's vocabulary) and the adapters that map heterogeneous
upstream trace formats onto it.

One JSONL row == one trajectory:

    {
      "id":      "hermes:glm-5.1:1b510b01-...",   # globally unique, source-prefixed
      "source":  "hermes-agent-reasoning",         # provenance tag
      "messages": [                                 # OpenAI chat + tool extensions
         {"role": "system",    "content": "..."},
         {"role": "user",      "content": "..."},
         {"role": "assistant", "content": "...",    # may be null when only tool_calls
                               "reasoning": "...",   # chain-of-thought, kept separate
                               "tool_calls": [
                                   {"id": "call_0", "type": "function",
                                    "function": {"name": "bash",
                                                 "arguments": "{\"command\": \"ls\"}"}}]},
         {"role": "tool",      "tool_call_id": "call_0", "name": "bash",
                               "content": "<stdout>"},
         {"role": "assistant", "content": "done"}
      ],
      "tools":  [{"type": "function",                # tool schemas available to the
                  "function": {"name": "bash",       # agent (null when unknown)
                               "description": "...",
                               "parameters": {...}}}],
      "verified": false,                             # default; True only when tested
      "verification": null,                          # filled by the verified track
      "meta": {"model": "...", "n_turns": 7, ...}    # free-form provenance
    }

Design choices (locked with the user):
  * Tool calls use the OpenAI structured form (`assistant.tool_calls` +
    `role:"tool"` results). Lossless and re-renderable to any chat template at
    train time.
  * Reasoning / chain-of-thought lives in a per-message `reasoning` field,
    separate from `content`, so a trainer can mask or keep it independently.
  * `verified` defaults to False. Only the SWE-bench track (see `swebench.py`),
    which actually runs the tests, flips it True.

Normalizers are pure functions `(<source row>) -> WarmupRecord | None`. They
return None for rows that can't be salvaged (empty, malformed, no usable turn)
so callers can count skips.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# --------------------------------------------------------------------------- #
# Canonical record
# --------------------------------------------------------------------------- #

# Message roles we emit. Anything else is dropped during normalization.
_ROLES = ("system", "user", "assistant", "tool")


@dataclass
class WarmupRecord:
    """One normalized SFT trajectory. See module docstring for the JSON shape."""

    id: str
    source: str
    messages: list[dict[str, Any]]
    tools: Optional[list[dict[str, Any]]] = None
    verified: bool = False
    verification: Optional[dict[str, Any]] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "messages": self.messages,
            "verified": self.verified,
        }
        # Keep optional fields out of the row when empty so files stay compact.
        if self.tools:
            out["tools"] = self.tools
        if self.verification is not None:
            out["verification"] = self.verification
        if self.meta:
            out["meta"] = self.meta
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def make_tool_call(call_id: str, name: str, arguments: Any) -> dict[str, Any]:
    """Build one OpenAI `tool_calls` entry. `arguments` is JSON-stringified."""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def assistant_msg(
    content: Optional[str],
    *,
    reasoning: Optional[str] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content or None}
    if reasoning:
        msg["reasoning"] = reasoning
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def tool_msg(tool_call_id: str, content: str, name: Optional[str] = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
    if name:
        msg["name"] = name
    return msg


def validate(rec: WarmupRecord) -> Optional[str]:
    """Return an error string if the record is malformed, else None.

    Cheap structural checks only — enough to catch normalizer bugs before a row
    is written. Not a schema validator for downstream training.
    """
    if not rec.id or not isinstance(rec.id, str):
        return "missing/invalid id"
    if not rec.source:
        return "missing source"
    if not rec.messages:
        return "empty messages"
    seen_call_ids: set[str] = set()
    for i, m in enumerate(rec.messages):
        role = m.get("role")
        if role not in _ROLES:
            return f"messages[{i}]: bad role {role!r}"
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                cid = tc.get("id")
                fn = (tc.get("function") or {}).get("name")
                if not cid or not fn:
                    return f"messages[{i}]: tool_call missing id/name"
                if cid in seen_call_ids:
                    return f"messages[{i}]: duplicate tool_call id {cid!r}"
                seen_call_ids.add(cid)
            if m.get("content") is None and not m.get("tool_calls"):
                return f"messages[{i}]: assistant has neither content nor tool_calls"
        elif role == "tool":
            if not m.get("tool_call_id"):
                return f"messages[{i}]: tool result missing tool_call_id"
        else:
            if not isinstance(m.get("content"), str):
                return f"messages[{i}]: {role} content must be a string"
    return None


# --------------------------------------------------------------------------- #
# Shared text helpers
# --------------------------------------------------------------------------- #

# `<think>...</think>` / `<thinking>...</thinking>` reasoning channel.
_THINK_RE = re.compile(r"<(think|thinking)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)


def split_reasoning(text: str) -> tuple[Optional[str], str]:
    """Pull `<think>` blocks out of `text`.

    Returns `(reasoning, content)`. Multiple blocks are joined with blank lines.
    `content` is the text with the blocks removed and surrounding whitespace
    trimmed. Returns `(None, text.strip())` when there's no think block.
    """
    blocks = [m.group(2).strip() for m in _THINK_RE.finditer(text)]
    if not blocks:
        return None, text.strip()
    content = _THINK_RE.sub("", text).strip()
    reasoning = "\n\n".join(b for b in blocks if b) or None
    return reasoning, content


# --------------------------------------------------------------------------- #
# Normalizer 1: OpenAI-style `messages` with inline <think> (Claude reasoning)
# --------------------------------------------------------------------------- #

_OAI_ROLE_MAP = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "tool": "tool",
}


def normalize_oai_messages(
    row: dict[str, Any],
    *,
    source: str,
    id_prefix: str = "",
    row_index: int = 0,
    messages_key: str = "messages",
    meta_keys: Iterable[str] = (),
) -> Optional[WarmupRecord]:
    """Normalize a row whose `messages` is already an OpenAI-ish list.

    Handles the common reasoning-trace shape: plain `{role, content}` turns
    where assistant `content` embeds `<think>...</think>`. Tool calls, if
    present as native `tool_calls` / `role:"tool"`, are carried through.
    """
    raw = row.get(messages_key)
    if not isinstance(raw, list) or not raw:
        return None

    messages: list[dict[str, Any]] = []
    for turn in raw:
        if not isinstance(turn, dict):
            return None
        role = _OAI_ROLE_MAP.get(str(turn.get("role", "")).lower())
        if role is None:
            return None
        content = turn.get("content")
        if role == "assistant":
            tool_calls = _carry_tool_calls(turn.get("tool_calls"))
            reasoning = turn.get("reasoning")
            text = content if isinstance(content, str) else _content_to_text(content)
            if reasoning is None and isinstance(text, str):
                reasoning, text = split_reasoning(text)
            messages.append(assistant_msg(text or None, reasoning=reasoning,
                                           tool_calls=tool_calls))
        elif role == "tool":
            cid = turn.get("tool_call_id") or turn.get("id") or ""
            messages.append(tool_msg(str(cid), _content_to_text(content),
                                     name=turn.get("name")))
        else:
            messages.append({"role": role, "content": _content_to_text(content)})

    if not _has_user_and_assistant(messages):
        return None

    rid = _row_id(id_prefix, row.get("id"), row_index)
    meta = {k: row[k] for k in meta_keys if k in row}
    if "model" in row and "model" not in meta:
        meta["model"] = row["model"]
    return WarmupRecord(id=rid, source=source, messages=messages, meta=meta)


def _carry_tool_calls(raw: Any) -> Optional[list[dict[str, Any]]]:
    """Pass through native OpenAI `tool_calls`, normalizing arguments to str."""
    if not isinstance(raw, list) or not raw:
        return None
    out: list[dict[str, Any]] = []
    for i, tc in enumerate(raw):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name")
        if not name:
            continue
        args = fn.get("arguments", tc.get("arguments", "{}"))
        out.append(make_tool_call(tc.get("id") or f"call_{i}", name, args))
    return out or None


def _content_to_text(content: Any) -> str:
    """Coerce a message `content` (str | list-of-blocks | other) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for blk in content:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                parts.append(blk.get("text") or blk.get("content") or "")
        return "".join(parts)
    return str(content)


# --------------------------------------------------------------------------- #
# Normalizer 2: Hermes / ShareGPT function-calling (`conversations` + `tools`)
# --------------------------------------------------------------------------- #

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)

_HERMES_ROLE_MAP = {"system": "system", "human": "user", "user": "user",
                    "gpt": "assistant", "assistant": "assistant", "tool": "tool"}


def normalize_hermes(
    row: dict[str, Any],
    *,
    source: str,
    id_prefix: str = "",
    row_index: int = 0,
    meta_keys: Iterable[str] = ("category", "subcategory", "task"),
) -> Optional[WarmupRecord]:
    """Normalize a Hermes-format ShareGPT row.

    `conversations` is a list of `{from, value}`; assistant turns embed
    `<think>` plus zero or more `<tool_call>{json}</tool_call>` blocks, and tool
    turns embed `<tool_response>{json}</tool_response>` blocks carrying a
    `tool_call_id`. The XML calls have no id of their own, so we pair each
    assistant call with the next tool response by position and reuse its id.
    """
    conv = row.get("conversations")
    if not isinstance(conv, list) or not conv:
        return None

    tools = _parse_hermes_tools(row.get("tools"))

    messages: list[dict[str, Any]] = []
    pending_calls: list[dict[str, Any]] = []  # assistant tool_calls awaiting ids
    call_seq = 0

    for turn in conv:
        if not isinstance(turn, dict):
            return None
        role = _HERMES_ROLE_MAP.get(str(turn.get("from", "")).lower())
        value = turn.get("value")
        if role is None or not isinstance(value, str):
            return None

        if role == "assistant":
            reasoning, body = split_reasoning(value)
            calls_raw = _TOOL_CALL_RE.findall(value)
            text = _TOOL_CALL_RE.sub("", body).strip()
            tool_calls: list[dict[str, Any]] = []
            for raw in calls_raw:
                parsed = _loads(raw)
                if not isinstance(parsed, dict) or "name" not in parsed:
                    continue
                cid = f"call_{call_seq}"
                call_seq += 1
                tc = make_tool_call(cid, parsed["name"], parsed.get("arguments", {}))
                tool_calls.append(tc)
            pending_calls = tool_calls
            messages.append(assistant_msg(text or None, reasoning=reasoning,
                                           tool_calls=tool_calls or None))
        elif role == "tool":
            responses = _TOOL_RESP_RE.findall(value) or [value]
            for i, raw in enumerate(responses):
                parsed = _loads(raw)
                # Prefer pairing by position with the assistant's pending calls;
                # fall back to the id the response carries, then a synthetic one.
                if i < len(pending_calls):
                    cid = pending_calls[i]["id"]
                elif isinstance(parsed, dict) and parsed.get("tool_call_id"):
                    cid = str(parsed["tool_call_id"])
                else:
                    cid = f"call_{call_seq}"
                    call_seq += 1
                name = parsed.get("name") if isinstance(parsed, dict) else None
                body = parsed.get("content") if isinstance(parsed, dict) else None
                content = body if isinstance(body, str) else json.dumps(
                    body if body is not None else parsed, ensure_ascii=False)
                messages.append(tool_msg(cid, content, name=name))
            pending_calls = []
        else:
            messages.append({"role": role, "content": value})

    if not _has_user_and_assistant(messages):
        return None

    rid = _row_id(id_prefix, row.get("id"), row_index)
    meta = {k: row[k] for k in meta_keys if k in row}
    return WarmupRecord(id=rid, source=source, messages=messages, tools=tools,
                        meta=meta)


def _parse_hermes_tools(raw: Any) -> Optional[list[dict[str, Any]]]:
    """The `tools` column is a JSON string of `{name, description, parameters}`.

    Wrap each into the OpenAI `{"type":"function","function":{...}}` envelope.
    """
    if not raw:
        return None
    data = raw if isinstance(raw, list) else _loads(raw)
    if not isinstance(data, list):
        return None
    out: list[dict[str, Any]] = []
    for t in data:
        if not isinstance(t, dict) or "name" not in t:
            continue
        if t.get("type") == "function" and "function" in t:
            out.append(t)  # already enveloped
        else:
            out.append({"type": "function", "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            }})
    return out or None


# --------------------------------------------------------------------------- #
# Normalizer 3: pi-mono session traces (`traces` list of events)
# --------------------------------------------------------------------------- #


def normalize_pi_traces(
    row: dict[str, Any],
    *,
    source: str,
    id_prefix: str = "",
    row_index: int = 0,
    traces_key: str = "traces",
) -> Optional[WarmupRecord]:
    """Normalize a pi-mono / agents-sdk session.

    `traces` is a list of session events; the conversation lives on events that
    carry a `message`. A message's `content` is a list of typed blocks:
        - {"type": "thinking", "thinking": "..."}      -> reasoning
        - {"type": "text", "text": "..."}              -> content
        - {"type": "toolCall", "id", "name", "arguments"} -> assistant tool_call
    Tool *results* arrive as their own message events carrying `toolCallId` /
    `toolName` (role typically "toolResult"/"tool").
    """
    traces = row.get(traces_key)
    if isinstance(traces, str):
        traces = _loads(traces)
    if not isinstance(traces, list) or not traces:
        return None

    messages: list[dict[str, Any]] = []
    model: Optional[str] = None

    for ev in traces:
        if not isinstance(ev, dict):
            continue
        msg = ev.get("message")
        if not isinstance(msg, dict):
            continue
        model = msg.get("model") or msg.get("responseModel") or model
        role = str(msg.get("role", "")).lower()

        # Tool result message (pi emits these as standalone events).
        if msg.get("toolCallId") or role in ("toolresult", "tool"):
            cid = str(msg.get("toolCallId") or "")
            if not cid:
                continue
            messages.append(tool_msg(cid, _pi_block_text(msg.get("content")),
                                     name=msg.get("toolName")))
            continue

        if role in ("user", "human"):
            text = _pi_block_text(msg.get("content"))
            if text.strip():
                messages.append({"role": "user", "content": text})
            continue

        if role == "system":
            text = _pi_block_text(msg.get("content"))
            if text.strip():
                messages.append({"role": "system", "content": text})
            continue

        if role == "assistant":
            reasoning_parts: list[str] = []
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            content = msg.get("content")
            if not isinstance(content, list):
                content = [content] if content else []
            for blk in content:
                if not isinstance(blk, dict):
                    if isinstance(blk, str):
                        text_parts.append(blk)
                    continue
                btype = blk.get("type")
                if btype == "thinking":
                    if blk.get("thinking"):
                        reasoning_parts.append(blk["thinking"])
                elif btype == "text":
                    if blk.get("text"):
                        text_parts.append(blk["text"])
                elif btype in ("toolCall", "tool_use", "toolUse"):
                    name = blk.get("name")
                    if not name:
                        continue
                    cid = blk.get("id") or f"call_{len(tool_calls)}"
                    args = _pi_clean_args(blk.get("arguments", blk.get("input", {})))
                    tool_calls.append(make_tool_call(str(cid), name, args))
            if reasoning_parts or text_parts or tool_calls:
                messages.append(assistant_msg(
                    "\n".join(text_parts).strip() or None,
                    reasoning="\n\n".join(reasoning_parts).strip() or None,
                    tool_calls=tool_calls or None,
                ))

    if not _has_user_and_assistant(messages):
        return None

    rid = _row_id(id_prefix, row.get("session_id") or row.get("id"), row_index)
    meta: dict[str, Any] = {}
    if model:
        meta["model"] = model
    for k in ("harness", "num_tool_calls", "num_user_messages"):
        if k in row:
            meta[k] = row[k]
    return WarmupRecord(id=rid, source=source, messages=messages, meta=meta)


def _pi_block_text(content: Any) -> str:
    """Extract plain text from a pi message `content` (blocks or string)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    if isinstance(content, list):
        parts: list[str] = []
        for blk in content:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                parts.append(blk.get("text") or blk.get("content") or "")
        return "".join(parts)
    return str(content)


def _pi_clean_args(args: Any) -> dict[str, Any]:
    """Drop null/empty fields from pi's fixed-shape arguments struct."""
    if not isinstance(args, dict):
        return {} if args is None else {"value": args}
    return {k: v for k, v in args.items() if v not in (None, "", [], {})}


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #


def _has_user_and_assistant(messages: list[dict[str, Any]]) -> bool:
    roles = {m["role"] for m in messages}
    return "user" in roles and "assistant" in roles


def _row_id(prefix: str, raw_id: Any, index: int) -> str:
    base = str(raw_id) if raw_id not in (None, "") else f"row{index}"
    return f"{prefix}:{base}" if prefix else base


def _loads(text: Any) -> Any:
    """Lenient JSON parse; returns None on failure."""
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None

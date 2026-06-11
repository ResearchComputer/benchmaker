# benchmaker/swebench/trajectory.py
"""Convert pi (`@earendil-works/pi-coding-agent`) `--mode json` logs into a
replayable trajectory store.

A *trajectory* is one SWE-bench task's ordered list of recorded LLM outputs
(`RecordedTurn`s) plus the key used to recognise the task at replay time. The
converter walks a harbor job dir (`<trial>/agent/pi-container.log`) and writes
one trajectory per line to a single consolidated JSONL. The replay server
(`benchmaker.swebench.replay_server`) loads that file and serves the recorded
outputs back to an unchanged pipeline. See
`docs/superpowers/specs/2026-06-11-swebench-trajectory-replay-design.md`.

Defensive throughout (mirrors `observability.parse_pi_token_spans`): a corrupt
log line is skipped, never raised.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("benchmaker.trajectory")

# pi's SWE-bench prompt embeds `# Task: <instance_id>` (see pi_agent
# SWEBENCH_PROMPT_TEMPLATE); that id is unique per instance and is our key.
_TASK_RE = re.compile(r"# Task:\s*(\S+)")


@dataclass
class RecordedTurn:
    """One recorded LLM output (assistant turn)."""
    index: int
    content: str
    reasoning: Optional[str]
    tool_calls: list[dict]          # [{"id","name","arguments": dict}]
    finish_reason: str              # "tool_calls" | "stop"
    usage: dict                     # OpenAI-named: prompt_tokens/completion_tokens/...

    def to_dict(self) -> dict:
        return {
            "index": self.index, "content": self.content, "reasoning": self.reasoning,
            "tool_calls": self.tool_calls, "finish_reason": self.finish_reason,
            "usage": self.usage,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RecordedTurn":
        return cls(
            index=int(d.get("index", 0)),
            content=d.get("content") or "",
            reasoning=d.get("reasoning"),
            tool_calls=list(d.get("tool_calls") or []),
            finish_reason=d.get("finish_reason") or "stop",
            usage=dict(d.get("usage") or {}),
        )


@dataclass
class Trajectory:
    key: str
    instance_id: Optional[str]
    model: str
    turns: list[RecordedTurn]
    trial: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "trial": self.trial, "instance_id": self.instance_id, "key": self.key,
            "model": self.model, "n_turns": len(self.turns),
            "turns": [t.to_dict() for t in self.turns],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(
            key=d.get("key") or "",
            instance_id=d.get("instance_id"),
            model=d.get("model") or "",
            trial=d.get("trial"),
            turns=[RecordedTurn.from_dict(t) for t in (d.get("turns") or [])],
        )


def _message_text(content: Any) -> str:
    """Flatten an OpenAI/pi message `content` (str or list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
                parts.append(b["text"])
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return ""


def _instance_id_from_text(text: str) -> Optional[str]:
    m = _TASK_RE.search(text or "")
    return m.group(1) if m else None


def _key_from_text(text: str) -> str:
    iid = _instance_id_from_text(text)
    if iid:
        return iid
    return "sha1:" + hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def task_key_from_messages(messages: Any) -> str:
    """Replay key from a request's `messages`: the first user message's task id
    (or a content hash). Shared by converter and server so keys always agree."""
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "user":
            return _key_from_text(_message_text(m.get("content")))
    return _key_from_text("")


def _usage_from_pi(u: Any) -> dict:
    """Map pi's usage block to OpenAI-named token fields."""
    if not isinstance(u, dict):
        return {}
    out: dict[str, Any] = {}
    if isinstance(u.get("input"), (int, float)):
        out["prompt_tokens"] = int(u["input"])
    if isinstance(u.get("output"), (int, float)):
        out["completion_tokens"] = int(u["output"])
    if isinstance(u.get("totalTokens"), (int, float)):
        out["total_tokens"] = int(u["totalTokens"])
    if isinstance(u.get("cacheRead"), (int, float)):
        out["cache_read"] = int(u["cacheRead"])
    cost = u.get("cost")
    cost_total = cost.get("total") if isinstance(cost, dict) else None
    if isinstance(cost_total, (int, float)):
        out["cost"] = float(cost_total)
    return out


def _turn_from_assistant(msg: dict, index: int) -> RecordedTurn:
    content = ""
    reasoning: Optional[str] = None
    tool_calls: list[dict] = []
    for b in msg.get("content") or []:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            content += b.get("text") or ""
        elif bt == "thinking":
            reasoning = (reasoning or "") + (b.get("thinking") or "")
        elif bt == "toolCall":
            tool_calls.append({
                "id": b.get("id") or f"call_{index}",
                "name": b.get("name") or "",
                "arguments": b.get("arguments") if isinstance(b.get("arguments"), dict) else {},
            })
    finish = "tool_calls" if (tool_calls or msg.get("stopReason") == "toolUse") else "stop"
    return RecordedTurn(
        index=index, content=content, reasoning=reasoning, tool_calls=tool_calls,
        finish_reason=finish, usage=_usage_from_pi(msg.get("usage")),
    )


def parse_pi_conversation(log_text: str) -> Trajectory:
    """Parse one pi `--mode json` log into a `Trajectory`.

    Assistant outputs come from `turn_end` events (one per LLM call, carrying
    full `usage`); the task key comes from the first `user` message. Corrupt or
    unexpected lines are skipped.
    """
    first_user_text: Optional[str] = None
    model: Optional[str] = None
    turns: list[RecordedTurn] = []
    for raw in log_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except Exception:
            continue
        if not isinstance(ev, dict):
            continue
        if first_user_text is None:
            msg = ev.get("message")
            if isinstance(msg, dict) and msg.get("role") == "user":
                first_user_text = _message_text(msg.get("content"))
        if ev.get("type") == "turn_end":
            msg = ev.get("message")
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                if model is None and isinstance(msg.get("model"), str):
                    model = msg["model"]
                turns.append(_turn_from_assistant(msg, index=len(turns)))
    text = first_user_text or ""
    return Trajectory(
        key=_key_from_text(text), instance_id=_instance_id_from_text(text),
        model=model or "", turns=turns,
    )

"""Multi-turn trajectory replay workload.

Expands each agent trajectory into one chat request per assistant turn, sending
the sanitized message prefix up to (but excluding) that turn. Built for
prefix-cache / HiCache parity benchmarking against an OpenAI-compatible chat
endpoint — pair with ``OpenAIChatWorkloadType(passthrough_meta=True)``.

Source: a HuggingFace dataset id (needs the ``datasets`` package) or a local
JSONL file. The ``SWE-bench/SWE-smith-trajectories`` ``messages`` column is a
JSON-encoded *string*; ``parse_messages`` handles both string and list.

Recorded per request (into Request.meta -> samples.jsonl): conversation_id,
instance_id, turn_index, prefix_messages, model_label, and -- when a tokenizer
is configured -- expected_prefix_tokens (the theoretical upper bound of cacheable
prefix, i.e. the previous turn's prompt length; compare to the server's
cached_tokens).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Iterator, Optional

from benchmaker.workloads.datasets import Workload


# Message keys accepted by OpenAI-compatible chat endpoints. All dataset-specific
# keys (agent, message_type, ...) are dropped.
_OPENAI_MSG_KEYS = ("role", "content", "name", "tool_calls", "tool_call_id")


def sanitize_message(msg: dict) -> dict:
    """Keep only OpenAI-valid keys; default missing content to ''."""
    out: dict[str, Any] = {}
    for k in _OPENAI_MSG_KEYS:
        if k in msg and msg[k] is not None:
            out[k] = msg[k]
    out.setdefault("role", "user")
    if "content" not in out and "tool_calls" not in out:
        out["content"] = ""
    return out


def parse_messages(raw: Any) -> list[dict]:
    """Parse a `messages` value (JSON string or list) into sanitized dicts."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise TypeError(f"messages must be a list or JSON-list string, got {type(raw).__name__}")
    return [sanitize_message(m) for m in raw if isinstance(m, dict)]


def expand_trajectory(
    messages: list[dict],
    *,
    meta_base: dict,
    max_tokens: int,
    max_turns: Optional[int],
    count_tokens: Optional[Callable[[list[dict]], int]],
) -> list[dict]:
    """One item per assistant turn; prompt = sanitized prefix before that turn.

    `expected_prefix_tokens` (only when `count_tokens` is given) is the token
    count of the *previous emitted turn's* prompt — the nested-prefix upper bound
    of what the server could serve from cache (0 for the first turn).
    """
    items: list[dict] = []
    turn = 0
    prev_tokens = 0
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        prefix = messages[:i]
        if not prefix:
            continue
        meta = dict(meta_base)
        meta["turn_index"] = turn
        meta["prefix_messages"] = len(prefix)
        if count_tokens is not None:
            meta["expected_prefix_tokens"] = prev_tokens
        items.append({"messages": prefix, "max_tokens": max_tokens, "meta": meta})
        if count_tokens is not None:
            prev_tokens = count_tokens(prefix)
        turn += 1
        if max_turns is not None and turn >= max_turns:
            break
    return items


class TrajectoryReplayWorkload(Workload):
    name = "trajectory"

    def __init__(
        self,
        *,
        dataset: Optional[str] = None,
        split: str = "tool",
        path: Optional[str] = None,
        messages_field: str = "messages",
        id_field: str = "instance_id",
        model_field: str = "model",
        max_tokens: int = 1024,
        max_turns_per_trajectory: Optional[int] = None,
        max_trajectories: Optional[int] = None,
        tokenizer: Optional[str] = None,
        name: Optional[str] = None,
        **hf_kwargs: Any,
    ):
        if (dataset is None) == (path is None):
            raise ValueError("provide exactly one of `dataset` or `path`.")
        self._dataset = dataset
        self._split = split
        self._path = path
        self._messages_field = messages_field
        self._id_field = id_field
        self._model_field = model_field
        self._max_tokens = max_tokens
        self._max_turns = max_turns_per_trajectory
        self._max_trajectories = max_trajectories
        self._hf_kwargs = hf_kwargs
        if name:
            self.name = name
        elif dataset:
            self.name = f"trajectory:{dataset}/{split}"
        else:
            self.name = f"trajectory:{path}"

        self._count_tokens = self._make_counter(tokenizer)
        self._gen: Optional[Iterator[dict]] = None
        self._lock = asyncio.Lock()

    def _make_counter(self, tokenizer: Optional[str]):
        if not tokenizer:
            return None
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "trajectory replay with --tokenizer needs `transformers`. "
                "Install with `pip install transformers` or `pip install -e .[hf]`."
            ) from e
        tok = AutoTokenizer.from_pretrained(tokenizer)

        def _count(msgs: list[dict]) -> int:
            try:
                text = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = "\n".join(str(m.get("content") or "") for m in msgs)
            return len(tok(text)["input_ids"])

        return _count

    def _iter_rows(self) -> Iterator[dict]:
        if self._path:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        else:
            try:
                from datasets import load_dataset
            except ImportError as e:
                raise ImportError(
                    "trajectory replay from a HF dataset needs `datasets`. "
                    "Install with `pip install datasets` or `pip install -e .[hf]`."
                ) from e
            ds = load_dataset(self._dataset, split=self._split, streaming=True,
                              **self._hf_kwargs)
            for row in ds:
                yield dict(row)

    def _iter_items(self) -> Iterator[dict]:
        n_traj = 0
        for row in self._iter_rows():
            conv_id = row.get(self._id_field) or row.get("traj_id")
            meta_base = {
                "conversation_id": conv_id,
                "instance_id": row.get(self._id_field),
                "model_label": row.get(self._model_field),
            }
            messages = parse_messages(row[self._messages_field])
            items = expand_trajectory(
                messages, meta_base=meta_base, max_tokens=self._max_tokens,
                max_turns=self._max_turns, count_tokens=self._count_tokens,
            )
            if not items:
                continue  # degenerate trajectory (no assistant turns) — don't count it
            if self._max_trajectories is not None and n_traj >= self._max_trajectories:
                return
            n_traj += 1
            for item in items:
                yield item

    async def next_item(self) -> Any:
        async with self._lock:
            if self._gen is None:
                self._gen = self._iter_items()
            try:
                return next(self._gen)
            except StopIteration:
                raise StopAsyncIteration

    async def aclose(self) -> None:
        if self._gen is not None:
            self._gen.close()
            self._gen = None

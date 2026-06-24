"""Deep-retrieval-augmented generation workload.

``DeepRAGWorkload`` reads prepared JSONL rows with a question, a short answer,
and multiple retrieved passages.  It packs the first ``depth`` passages (or a
prefix bounded by ``context_tokens_target``) into an OpenAI chat request.  The
result is deliberately prefill-heavy while still retaining a reference answer
for correctness evaluation.

Use :mod:`tools.rag.prepare` to create compatible rows from HotpotQA's
multi-passage distractor split.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from benchmaker.workloads.datasets import JsonlWorkload, Workload


DEFAULT_SYSTEM_PROMPT = "Answer the question using only the context below."


class DeepRAGWorkload(Workload):
    """Yield prefill-heavy RAG chat items from prepared JSONL data.

    Expected rows have at least ``question``, ``answer``, and ``passages``.
    Passages can be strings or ``{title, text}`` mappings.  The emitted item is
    compatible with ``OpenAIChatWorkloadType(passthrough_meta=True)`` and has
    this shape::

        {
          "messages": [...], "reference": "...", "rag_depth": 10,
          "prompt_tokens_hint": 1820, "max_tokens": 64,
        }

    ``prompt_tokens_hint`` is a whitespace-token estimate, not a model-specific
    tokenizer count.  When an OpenAI-compatible server returns usage, its
    reported ``prompt_tokens`` remains the authoritative realized measurement.
    """

    name = "deeprag"

    def __init__(
        self,
        path: str,
        *,
        depth: int = 10,
        context_tokens_target: Optional[int] = None,
        max_tokens: Optional[int] = 64,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        question_key: str = "question",
        reference_key: str = "answer",
        passages_key: str = "passages",
        id_key: str = "id",
        loop: bool = True,
        max_items: Optional[int] = None,
        workload_name: Optional[str] = None,
    ):
        if depth <= 0:
            raise ValueError("depth must be > 0")
        if context_tokens_target is not None and context_tokens_target <= 0:
            raise ValueError("context_tokens_target must be > 0 when set")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be > 0 when set")
        if not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")

        self._rows = JsonlWorkload(
            path=path,
            field=None,
            loop=loop,
            max_items=max_items,
        )
        self._depth = depth
        self._context_tokens_target = context_tokens_target
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._question_key = question_key
        self._reference_key = reference_key
        self._passages_key = passages_key
        self._id_key = id_key
        self.name = workload_name or "deeprag"

    async def next_item(self) -> dict[str, Any]:
        row = await self._rows.next_item()
        if not isinstance(row, dict):
            raise TypeError("DeepRAG JSONL rows must be JSON objects")
        try:
            question = row[self._question_key]
            reference = row[self._reference_key]
            raw_passages = row[self._passages_key]
        except KeyError as exc:
            raise KeyError(
                f"DeepRAG row is missing required field {exc.args[0]!r}; "
                f"expected {self._question_key!r}, {self._reference_key!r}, "
                f"and {self._passages_key!r}"
            ) from exc
        if not isinstance(question, str) or not question.strip():
            raise ValueError("DeepRAG question must be a non-empty string")
        if not isinstance(reference, str):
            reference = str(reference)

        passages = _normalise_passages(raw_passages)
        if not passages:
            raise ValueError("DeepRAG row has no usable passages")
        packed = _pack_passages(
            passages,
            depth=self._depth,
            context_tokens_target=self._context_tokens_target,
        )
        if not packed:
            raise ValueError("DeepRAG context target left no usable passage text")

        context = "\n\n".join(
            f"<passage {index}>\n{passage}"
            for index, passage in enumerate(packed, start=1)
        )
        user = f"{context}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]
        item: dict[str, Any] = {
            "messages": messages,
            "reference": reference,
            "rag_depth": len(packed),
            "prompt_tokens_hint": _estimate_prompt_tokens(messages),
        }
        if self._max_tokens is not None:
            item["max_tokens"] = self._max_tokens
        if self._id_key in row:
            item["source_id"] = row[self._id_key]
        return item

    async def aclose(self) -> None:
        await self._rows.aclose()


def _normalise_passages(value: Any) -> list[str]:
    """Convert prepared string/object passages into non-empty display text."""
    if not isinstance(value, list):
        raise TypeError("DeepRAG passages must be a list")
    passages: list[str] = []
    for passage in value:
        text = _passage_text(passage)
        if text:
            passages.append(text)
    return passages


def _passage_text(passage: Any) -> str:
    if isinstance(passage, str):
        return passage.strip()
    if not isinstance(passage, dict):
        raise TypeError("each DeepRAG passage must be a string or object")
    title = passage.get("title")
    text = passage.get("text", passage.get("content", ""))
    if isinstance(text, list):
        text = " ".join(str(part) for part in text)
    if not isinstance(text, str):
        raise TypeError("DeepRAG passage text must be a string or list of strings")
    parts = [str(part).strip() for part in (title, text) if part is not None]
    return "\n".join(part for part in parts if part)


def _pack_passages(passages: Iterable[str], *, depth: int,
                   context_tokens_target: Optional[int]) -> list[str]:
    packed: list[str] = []
    used = 0
    for passage in list(passages)[:depth]:
        words = passage.split()
        if context_tokens_target is not None:
            remaining = context_tokens_target - used
            if remaining <= 0:
                break
            words = words[:remaining]
        text = " ".join(words).strip()
        if text:
            packed.append(text)
            used += len(words)
    return packed


def _estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
    """Stable dependency-free prompt-size estimate for pre-run validation."""
    text = " ".join(message["content"] for message in messages)
    return max(1, len(re.findall(r"\S+", text)))

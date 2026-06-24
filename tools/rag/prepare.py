"""Prepare HotpotQA-style multi-passage QA data for ``DeepRAGWorkload``.

The default source is HotpotQA's distractor configuration, which already
provides ten context paragraphs per question.  The output is compact JSONL:

    {"id": "...", "question": "...", "answer": "...",
     "passages": [{"title": "...", "text": "..."}, ...]}

The benchmark chooses the retrieval depth later, so one prepared corpus can
serve shallow and deep experiments without downloading/reprocessing data.

Usage:
    python tools/rag/prepare.py --out .local/hotpotqa_distractor.jsonl
    python tools/rag/prepare.py --max-items 1000 --split validation
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Iterable, Mapping


DEFAULT_DATASET = "hotpot_qa"
DEFAULT_CONFIG = "distractor"
DEFAULT_SPLIT = "validation"
DEFAULT_OUT = ".local/hotpotqa_distractor.jsonl"


def normalise_hotpot_row(row: Mapping[str, Any], *, max_passage_chars: int | None
                         ) -> dict[str, Any] | None:
    """Normalize a HotpotQA row while retaining every retrieved passage."""
    question = row.get("question")
    answer = row.get("answer")
    context = row.get("context")
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(answer, str):
        return None
    if isinstance(context, Mapping):
        titles = context.get("title", context.get("titles"))
        sentence_groups = context.get("sentences", context.get("paragraphs"))
    elif isinstance(context, (list, tuple)) and len(context) == 2:
        titles, sentence_groups = context
    else:
        return None
    if not isinstance(titles, (list, tuple)) or not isinstance(sentence_groups, (list, tuple)):
        return None
    passages: list[dict[str, str]] = []
    for title, sentences in zip(titles, sentence_groups):
        if isinstance(sentences, str):
            text = sentences.strip()
        elif isinstance(sentences, (list, tuple)):
            text = " ".join(str(sentence).strip() for sentence in sentences).strip()
        else:
            continue
        if max_passage_chars is not None:
            text = text[:max_passage_chars].rstrip()
        if text:
            passages.append({"title": str(title).strip(), "text": text})
    if not passages:
        return None
    return {
        "id": row.get("id"),
        "question": question,
        "answer": answer,
        "passages": passages,
    }


def convert(rows: Iterable[Mapping[str, Any]], out_path: str, *,
            max_items: int | None, max_passage_chars: int | None) -> tuple[int, int]:
    """Write normalized rows and return ``(written, skipped)``."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in rows:
            if max_items is not None and written >= max_items:
                break
            prepared = normalise_hotpot_row(row, max_passage_chars=max_passage_chars)
            if prepared is None:
                skipped += 1
                continue
            out.write(json.dumps(prepared, ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def load_rows(dataset: str, config: str | None, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets is required; install benchmaker's hf extra or `datasets`."
        ) from exc
    if config:
        return load_dataset(dataset, config, split=split)
    return load_dataset(dataset, split=split)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Hugging Face dataset configuration (default: distractor)")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-passage-chars", type=int, default=None,
                        help="truncate each passage before writing (default: preserve all)")
    args = parser.parse_args()
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be > 0")
    if args.max_passage_chars is not None and args.max_passage_chars <= 0:
        parser.error("--max-passage-chars must be > 0")

    rows = load_rows(args.dataset, args.config or None, args.split)
    written, skipped = convert(
        rows,
        args.out,
        max_items=args.max_items,
        max_passage_chars=args.max_passage_chars,
    )
    print(f"[done] wrote {written:,} rows to {args.out} (skipped {skipped:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

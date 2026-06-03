"""Download the ShareGPT V3 dump and convert it to a generic JSONL workload.

Source:
    https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered

The upstream file is a single big JSON array. Each element looks like:

    {
        "id": "abc123",
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt",   "value": "Hi there!"},
            ...
        ]
    }

This script normalises each conversation into a row shaped for bench-maker's
OpenAI chat workload-type:

    {"id": "...", "messages": [{"role": "user", "content": "..."},
                               {"role": "assistant", "content": "..."}, ...]}

`messages` is the only content field — everything a benchmark needs is in it.
It is truncated to end on a *user* turn, so each row is a valid generation
request: the server completes the final assistant reply given the prior
history. Short source conversations collapse to a single user turn (a
plain single-turn prompt); longer ones carry multi-turn context.

Feed it with `field="messages"` so the workload yields the list directly and
the request body stays clean (the `id` is kept for provenance, not sent):

    JsonlWorkload(path=".local/sharegpt_v3.jsonl", field="messages")

Usage:
    python tools/sharegpt/prepare.py                       # default paths
    python tools/sharegpt/prepare.py --max-items 5000      # subset for fast runs
    python tools/sharegpt/prepare.py \\
        --raw   .local/sharegpt_v3_raw.json \\
        --out   .local/sharegpt_v3.jsonl \\
        --min-chars 8 --max-chars 16000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

DEFAULT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered"
    "/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
DEFAULT_RAW = ".local/sharegpt_v3_raw.json"
DEFAULT_OUT = ".local/sharegpt_v3.jsonl"

_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "assistant": "assistant",
    "bing": "assistant",
    "system": "system",
}


def download(url: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"[skip download] {dst} already exists ({os.path.getsize(dst):,} B)")
        return
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    print(f"[download] {url}\n        -> {dst}")
    tmp = dst + ".part"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if total:
                pct = 100 * read / total
                print(f"\r  {read/1e6:7.1f} / {total/1e6:.1f} MB ({pct:5.1f}%)",
                      end="", flush=True)
        print()
    os.replace(tmp, dst)


def normalise_messages(conv: list[dict]) -> list[dict] | None:
    """Map ShareGPT `conversations` -> OpenAI-style `messages`, truncated to end
    on a user turn. Returns None if the conversation is unusable (empty, no
    leading user turn, unmappable role, or non-string content)."""
    out: list[dict] = []
    for turn in conv:
        role = _ROLE_MAP.get((turn.get("from") or "").lower())
        if role is None:
            return None
        text = turn.get("value")
        if not isinstance(text, str):
            return None
        out.append({"role": role, "content": text})
    # Drop trailing non-user turns so the request asks for an assistant reply.
    while out and out[-1]["role"] != "user":
        out.pop()
    if not out or out[0]["role"] != "user":
        return None
    return out


def convert(raw_path: str, out_path: str, *, max_items: int | None,
            min_chars: int, max_chars: int) -> tuple[int, int]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"[load] {raw_path} (this is a big JSON file; takes a few seconds)")
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"unexpected top-level type: {type(data).__name__}")

    written = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in data:
            if max_items is not None and written >= max_items:
                break
            messages = normalise_messages(row.get("conversations") or [])
            if not messages:
                skipped += 1
                continue
            chars = sum(len(m["content"]) for m in messages)
            if not (min_chars <= chars <= max_chars):
                skipped += 1
                continue
            f.write(json.dumps({"id": row.get("id"), "messages": messages}) + "\n")
            written += 1
    return written, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--raw", default=DEFAULT_RAW,
                    help=f"where to cache the upstream JSON (default: {DEFAULT_RAW})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output JSONL path (default: {DEFAULT_OUT})")
    ap.add_argument("--max-items", type=int, default=None,
                    help="stop after this many conversations (default: all)")
    ap.add_argument("--min-chars", type=int, default=4,
                    help="skip rows whose total message content is shorter than this")
    ap.add_argument("--max-chars", type=int, default=16000,
                    help="skip rows whose total message content is longer than this")
    ap.add_argument("--skip-download", action="store_true",
                    help="assume --raw already exists")
    args = ap.parse_args()

    if not args.skip_download:
        download(args.url, args.raw)
    elif not os.path.exists(args.raw):
        raise SystemExit(f"--skip-download set but {args.raw} is missing")

    written, skipped = convert(
        args.raw, args.out,
        max_items=args.max_items,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    print(f"[done] wrote {written:,} rows to {args.out} (skipped {skipped:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

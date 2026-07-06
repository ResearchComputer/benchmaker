"""Download the TraceLab coding-agent trace and optionally subset it.

Source:
    https://github.com/uw-syfi/TraceLab  (release tag ``v0.0.1``)

The published dataset is a single sanitized JSONL — one row per LLM round from
real Claude Code / Codex sessions (357K rows, 43 pseudonymous developers). Each
row carries token accounting (``input_tokens_total = prefix_tokens +
newly_append_tokens``, ``output_tokens``), session ids, model, and tool
metadata — but **no prompt text** (sanitized for privacy).

`benchmaker`'s :class:`~benchmaker.workloads.tracelab.TraceLabWorkload`
synthesizes token-faithful prompts from that accounting. Run this script once
to fetch the asset, then point the workload (or the ``benchmaker tracelab``
recipe) at the local file:

    python tools/tracelab/prepare.py
    benchmaker tracelab --trace .local/syfi_coding_trace.jsonl \\
        --url http://localhost:8000/v1/chat/completions --model ... \\
        --prefix-cache --rate poisson:8 --duration 60s

This script mirrors the official download/verify recipe and adds optional
subsetting (provider/model/token-range filters + a row cap) so a small,
focused slice can be carved out for fast iteration:

    python tools/tracelab/prepare.py --provider claude --max-items 5000

Usage:
    python tools/tracelab/prepare.py
    python tools/tracelab/prepare.py --tag v0.0.1 --out .local/syfi_coding_trace.jsonl
    python tools/tracelab/prepare.py --skip-verify --provider codex \\
        --min-input-tokens 4096 --max-items 10000
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import urllib.request
from typing import Any, Iterable, Optional

DEFAULT_TAG = "v0.0.1"
DEFAULT_BASE = (
    "https://github.com/uw-syfi/TraceLab/releases/download/{tag}/"
    "syfi_coding_trace.jsonl.gz"
)
DEFAULT_GZ = ".local/syfi_coding_trace.jsonl.gz"
DEFAULT_OUT = ".local/syfi_coding_trace.jsonl"

# Pinned SHA256 of the v0.0.1 JSONL.gz asset (from the TraceLab README).
PINNED_SHA256 = "9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b"


def download(url: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"[skip download] {dst} already exists "
              f"({os.path.getsize(dst):,} B)")
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


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decompress(gz_path: str, out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with gzip.open(gz_path, "rt", encoding="utf-8") as src, \
            open(out_path, "w", encoding="utf-8") as out:
        for line in src:
            out.write(line)
            n += 1
    return n


def subset(in_path: str, out_path: str, *, provider: Optional[str],
           model_filter: Optional[str], min_input: Optional[int],
           max_input: Optional[int], min_output: Optional[int],
           max_output: Optional[int], max_items: Optional[int]) -> tuple[int, int]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    skipped = 0
    with open(in_path, "r", encoding="utf-8") as src, \
            open(out_path, "w", encoding="utf-8") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not _keep(row, provider, model_filter, min_input, max_input,
                         min_output, max_output):
                skipped += 1
                continue
            if max_items is not None and written >= max_items:
                # Keep scanning only to count skips for the report.
                skipped += 1
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def _keep(row: dict[str, Any], provider: Optional[str], model_filter: Optional[str],
          min_input: Optional[int], max_input: Optional[int], min_output: Optional[int],
          max_output: Optional[int]) -> bool:
    if provider is not None and row.get("provider") != provider:
        return False
    if model_filter is not None and row.get("model") != model_filter:
        return False
    inp = _to_int(row.get("input_tokens_total"))
    if inp is None:
        inp = (_to_int(row.get("prefix_tokens")) or 0) + \
              (_to_int(row.get("newly_append_tokens")) or 0)
    if min_input is not None and (inp is None or inp < min_input):
        return False
    if max_input is not None and (inp is None or inp > max_input):
        return False
    out = _to_int(row.get("output_tokens"))
    if min_output is not None and (out is None or out < min_output):
        return False
    if max_output is not None and (out is None or out > max_output):
        return False
    return True


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help=f"release tag to download (default: {DEFAULT_TAG})")
    ap.add_argument("--url", default=None, help="override the download URL")
    ap.add_argument("--gz", default=DEFAULT_GZ,
                    help=f"where to cache the downloaded .jsonl.gz "
                         f"(default: {DEFAULT_GZ})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output JSONL path (default: {DEFAULT_OUT})")
    ap.add_argument("--no-decompress", action="store_true",
                    help="leave the .jsonl.gz as-is; the workload reads .gz "
                         "directly, so this only matters if you want a plain "
                         ".jsonl beside it")
    ap.add_argument("--skip-download", action="store_true",
                    help="assume --gz already exists")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the SHA256 check against the pinned checksum")
    # Subsetting (optional). When any are set, the output is rewritten with
    # only the matching rows; otherwise the decompressed file is left whole.
    ap.add_argument("--provider", default=None, choices=["claude", "codex"])
    ap.add_argument("--model", dest="model_filter", default=None,
                    help="keep only rows whose model matches exactly")
    ap.add_argument("--min-input-tokens", type=int, default=None)
    ap.add_argument("--max-input-tokens", type=int, default=None)
    ap.add_argument("--min-output-tokens", type=int, default=None)
    ap.add_argument("--max-output-tokens", type=int, default=None)
    ap.add_argument("--max-items", type=int, default=None,
                    help="cap the number of rows written")
    args = ap.parse_args()

    if not args.skip_download:
        url = args.url or DEFAULT_BASE.format(tag=args.tag)
        download(url, args.gz)
    elif not os.path.exists(args.gz):
        raise SystemExit(f"--skip-download set but {args.gz} is missing")

    if not args.skip_verify:
        digest = sha256(args.gz)
        if digest != PINNED_SHA256:
            if args.tag == DEFAULT_TAG:
                raise SystemExit(
                    f"SHA256 mismatch for {args.gz}!\n"
                    f"  expected {PINNED_SHA256}\n  got      {digest}\n"
                    f"Re-download, or pass --skip-verify if you know better."
                )
            print(f"[warn] tag {args.tag!r} is not pinned; skipping SHA check "
                  f"(got {digest})")
        else:
            print(f"[ok] SHA256 matches pinned v0.0.1 checksum")

    if args.no_decompress and not _has_subset_filters(args):
        print(f"[done] kept {args.gz} compressed")
        return 0

    # Decompress to a temp plain JSONL, then optionally subset onto --out.
    raw_jsonl = args.gz[:-3] if args.gz.lower().endswith(".gz") else args.gz + ".jsonl"
    n = decompress(args.gz, raw_jsonl)
    print(f"[decompressed] {n:,} rows -> {raw_jsonl}")

    if _has_subset_filters(args):
        written, skipped = subset(
            raw_jsonl, args.out,
            provider=args.provider, model_filter=args.model_filter,
            min_input=args.min_input_tokens, max_input=args.max_input_tokens,
            min_output=args.min_output_tokens, max_output=args.max_output_tokens,
            max_items=args.max_items,
        )
        # The intermediate decompressed file is now redundant; remove it.
        if os.path.abspath(raw_jsonl) != os.path.abspath(args.out):
            os.remove(raw_jsonl)
        print(f"[done] wrote {written:,} matching rows to {args.out} "
              f"(skipped {skipped:,})")
    else:
        if os.path.abspath(raw_jsonl) != os.path.abspath(args.out):
            os.replace(raw_jsonl, args.out)
        print(f"[done] wrote {n:,} rows to {args.out}")
    return 0


def _has_subset_filters(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, attr) is not None
        for attr in ("provider", "model_filter", "min_input_tokens",
                     "max_input_tokens", "min_output_tokens",
                     "max_output_tokens", "max_items")
    )


if __name__ == "__main__":
    sys.exit(main())

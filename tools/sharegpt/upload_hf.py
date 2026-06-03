"""Upload the converted ShareGPT JSONL to the Hugging Face Hub.

Prereq: run `tools/sharegpt/prepare.py` first to produce the JSONL, and have a
**write** token (the env `HF_TOKEN` defaults are often read-only). Provide one
via any of:

    huggingface-cli login           # interactive, cached for future runs
    export HF_TOKEN=hf_xxx           # write token in the environment
    python tools/sharegpt/upload_hf.py --token hf_xxx

The dataset lands as `split=sharegpt` under the default config, so consumers do:

    from datasets import load_dataset
    ds = load_dataset("researchcomputer/llmsys-bench", split="sharegpt")

Usage:
    python tools/sharegpt/upload_hf.py
    python tools/sharegpt/upload_hf.py \\
        --jsonl .local/sharegpt_v3.jsonl \\
        --repo  researchcomputer/llmsys-bench \\
        --split sharegpt
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_JSONL = ".local/sharegpt_v3.jsonl"
DEFAULT_REPO = "researchcomputer/llmsys-bench"
DEFAULT_SPLIT = "sharegpt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jsonl", default=DEFAULT_JSONL,
                    help=f"path to the converted JSONL (default: {DEFAULT_JSONL})")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"target dataset repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--split", default=DEFAULT_SPLIT,
                    help=f"split name to upload as (default: {DEFAULT_SPLIT})")
    ap.add_argument("--config", default="default",
                    help="dataset config name (default: 'default')")
    ap.add_argument("--token", default=None,
                    help="HF write token; falls back to HF_TOKEN / cached login")
    ap.add_argument("--private", action="store_true",
                    help="create the repo as private if it does not exist yet")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("json", data_files=args.jsonl, split="train")
    print(f"loaded {ds.num_rows:,} rows from {args.jsonl}: {ds}")

    url = ds.push_to_hub(
        args.repo,
        config_name=args.config,
        split=args.split,
        private=args.private,
        token=args.token,
        commit_message=(
            f"Add ShareGPT V3 (cleaned, truncated to last user turn) "
            f"as split={args.split}"
        ),
    )
    print(f"\nuploaded -> {url}")
    print(f"load with: load_dataset({args.repo!r}, split={args.split!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

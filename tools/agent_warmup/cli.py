"""Build the agent-warmup SFT dataset.

Run as a module from the repo root: ``python -m tools.agent_warmup.cli <cmd>``.

Two tracks, one canonical JSONL schema (see `protocol.py`):

  import-hf   Track A — normalize published trace datasets into warmup rows.
              `verified=false` (no tests were run on these).

  generate    Track B — run *our* coding agent (model from `.env`) on SWE-bench
              tasks inside a Flash Sandbox, then run the gold tests. Rows are
              `verified=true` iff the agent's patch resolves the task.
              (Implemented in `runner.py`; needs the sandbox orchestrator.)

  merge       Concatenate + de-duplicate per-track JSONL files into one dataset.
  stats       Summarize a JSONL dataset (rows, verified split, sources, tools).

Examples
--------
    # Track A: pull all three published sources (subset for a smoke test)
    python -m tools.agent_warmup.cli import-hf claude-reasoning --out .local/claude.jsonl --max-items 200
    python -m tools.agent_warmup.cli import-hf hermes           --out .local/hermes.jsonl
    python -m tools.agent_warmup.cli import-hf pi-traces        --out .local/pi.jsonl

    # Track B: generate verified SWE-bench traces (needs flash-sandbox running)
    python -m tools.agent_warmup.cli generate --dataset princeton-nlp/SWE-bench_Verified \\
        --split test --num-tasks 50 --out .local/swebench.jsonl

    # Combine everything
    python -m tools.agent_warmup.cli merge .local/*.jsonl --out .local/agent_warmup.jsonl
    python -m tools.agent_warmup.cli stats .local/agent_warmup.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Callable, Iterator, Optional

from . import protocol as P


# --------------------------------------------------------------------------- #
# HuggingFace raw-file readers
#
# We read the repos' underlying files directly (JSONL / parquet) instead of
# going through `datasets.load_dataset`. The pi-traces set has heterogeneous
# nested schemas across files that make Arrow casting blow up; reading raw JSON
# sidesteps that and keeps all three sources on one code path.
# --------------------------------------------------------------------------- #


def _read_jsonl_rows(repo: str, files: list[str]) -> Iterator[tuple[dict, str]]:
    """Yield `(row, tag)` for line-delimited JSON files (one row per line)."""
    from huggingface_hub import hf_hub_download

    for fname in files:
        path = hf_hub_download(repo, fname, repo_type="dataset")
        tag = os.path.splitext(os.path.basename(fname))[0]
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line), tag
                except json.JSONDecodeError:
                    continue


def _read_pi_sessions(repo: str, files: Optional[list[str]]) -> Iterator[tuple[dict, str]]:
    """Yield one `(session_row, tag)` per pi `.jsonl` file.

    Each file is a raw pi-mono session log (one event per line). We assemble the
    events into a `traces` list — the shape `normalize_pi_traces` expects.
    """
    from huggingface_hub import HfApi, hf_hub_download

    if files is None:
        files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                 if f.endswith(".jsonl")]
    for fname in files:
        path = hf_hub_download(repo, fname, repo_type="dataset")
        events: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not events:
            continue
        session_id = next((e.get("id") for e in events if e.get("type") == "session"), None)
        tag = os.path.splitext(os.path.basename(fname))[0]
        yield {"traces": events, "session_id": session_id, "file_path": fname}, tag


def _read_parquet_rows(repo: str, files: list[str],
                       batch_size: int = 256) -> Iterator[tuple[dict, str]]:
    """Yield `(row, tag)` from parquet files, batched to bound memory."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for fname in files:
        path = hf_hub_download(repo, fname, repo_type="dataset")
        # Tag with the config dir (e.g. "glm-5.1") so ids stay unique per config.
        tag = os.path.basename(os.path.dirname(fname)) or os.path.splitext(
            os.path.basename(fname))[0]
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row, tag


# --------------------------------------------------------------------------- #
# Source presets: how to read + normalize each published dataset
# --------------------------------------------------------------------------- #

Normalizer = Callable[..., Optional[P.WarmupRecord]]


class Preset:
    def __init__(self, repo: str, reader: str, normalizer: Normalizer,
                 source: str, files: Optional[list[str]] = None,
                 norm_kwargs: Optional[dict[str, Any]] = None):
        self.repo = repo
        self.reader = reader
        self.normalizer = normalizer
        self.source = source
        self.files = files
        self.norm_kwargs = norm_kwargs or {}

    def iter_rows(self) -> Iterator[tuple[dict, str]]:
        if self.reader == "jsonl":
            assert self.files is not None
            return _read_jsonl_rows(self.repo, self.files)
        if self.reader == "pi-events":
            return _read_pi_sessions(self.repo, self.files)
        if self.reader == "parquet":
            assert self.files is not None
            return _read_parquet_rows(self.repo, self.files)
        raise ValueError(f"unknown reader {self.reader!r}")


PRESETS: dict[str, Preset] = {
    "claude-reasoning": Preset(
        repo="angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k",
        reader="jsonl",
        files=["full_train.jsonl"],
        normalizer=P.normalize_oai_messages,
        source="claude-reasoning",
        norm_kwargs={"meta_keys": ("category", "model")},
    ),
    "hermes": Preset(
        repo="lambda/hermes-agent-reasoning-traces",
        reader="parquet",
        files=["data/glm-5.1/train.parquet", "data/kimi/train.parquet"],
        normalizer=P.normalize_hermes,
        source="hermes-agent-reasoning",
    ),
    "pi-traces": Preset(
        repo="armand0e/qwen3.7-max-pi-traces",
        reader="pi-events",
        files=None,  # all .jsonl files in the repo
        normalizer=P.normalize_pi_traces,
        source="pi-traces",
    ),
}


# --------------------------------------------------------------------------- #
# Output helper
# --------------------------------------------------------------------------- #


class JsonlWriter:
    """Write validated WarmupRecords, deduping ids, counting skips."""

    def __init__(self, out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self._fh = open(out_path, "w", encoding="utf-8")
        self.written = 0
        self.invalid = 0
        self.dupes = 0
        self._seen_ids: set[str] = set()

    def add(self, rec: Optional[P.WarmupRecord]) -> bool:
        if rec is None:
            return False
        err = P.validate(rec)
        if err:
            self.invalid += 1
            return False
        if rec.id in self._seen_ids:
            self.dupes += 1
            return False
        self._seen_ids.add(rec.id)
        self._fh.write(rec.to_json() + "\n")
        self.written += 1
        return True

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------- #
# Subcommand: import-hf
# --------------------------------------------------------------------------- #


def cmd_import_hf(args: argparse.Namespace) -> int:
    preset = PRESETS.get(args.preset)
    if preset is None:
        raise SystemExit(f"unknown preset {args.preset!r}; known: {sorted(PRESETS)}")
    if args.files:
        preset.files = args.files

    writer = JsonlWriter(args.out)
    skipped = 0
    seen = 0
    print(f"[import-hf] {args.preset} <- {preset.repo} (reader={preset.reader})")
    for row, tag in preset.iter_rows():
        if args.max_items is not None and writer.written >= args.max_items:
            break
        rec = preset.normalizer(
            row,
            source=preset.source,
            id_prefix=f"{args.preset}:{tag}",
            row_index=seen,
            **preset.norm_kwargs,
        )
        seen += 1
        if not writer.add(rec):
            skipped += 1
        if seen % 500 == 0:
            print(f"\r  read {seen}  written {writer.written}  skipped {skipped}",
                  end="", flush=True)
    writer.close()
    print(f"\n[done] wrote {writer.written:,} rows to {args.out} "
          f"(skipped {skipped:,}: invalid={writer.invalid} dupes={writer.dupes})")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: generate (Track B — verified SWE-bench traces)
# --------------------------------------------------------------------------- #


def cmd_generate(args: argparse.Namespace) -> int:
    from . import runner
    return runner.run_generate(args)


# --------------------------------------------------------------------------- #
# Subcommand: merge
# --------------------------------------------------------------------------- #


def cmd_merge(args: argparse.Namespace) -> int:
    paths: list[str] = []
    for pat in args.inputs:
        paths.extend(sorted(glob.glob(pat)))
    paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(args.out)]
    if not paths:
        raise SystemExit("no input files matched")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    seen: set[str] = set()
    written = dupes = bad = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for path in paths:
            n = 0
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        bad += 1
                        continue
                    rid = obj.get("id")
                    if not rid:
                        bad += 1
                        continue
                    if rid in seen:
                        dupes += 1
                        continue
                    seen.add(rid)
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    written += 1
                    n += 1
            print(f"  + {path}: {n:,} rows")
    print(f"[done] merged {written:,} rows to {args.out} "
          f"(dupes {dupes:,}, bad {bad:,})")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: stats
# --------------------------------------------------------------------------- #


def cmd_stats(args: argparse.Namespace) -> int:
    from collections import Counter

    total = verified = with_tools = with_toolcalls = with_reasoning = 0
    sources: Counter = Counter()
    turn_counts: list[int] = []
    approx_chars = 0

    for path in args.inputs:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                total += 1
                sources[obj.get("source", "?")] += 1
                if obj.get("verified"):
                    verified += 1
                if obj.get("tools"):
                    with_tools += 1
                msgs = obj.get("messages") or []
                turn_counts.append(len(msgs))
                has_tc = has_re = False
                for m in msgs:
                    if m.get("tool_calls"):
                        has_tc = True
                    if m.get("reasoning"):
                        has_re = True
                    c = m.get("content")
                    if isinstance(c, str):
                        approx_chars += len(c)
                with_toolcalls += int(has_tc)
                with_reasoning += int(has_re)

    if total == 0:
        print("(no rows)")
        return 0
    avg_turns = sum(turn_counts) / len(turn_counts)
    print(f"rows:            {total:,}")
    print(f"  verified:      {verified:,} ({100*verified/total:.1f}%)")
    print(f"  with tools:    {with_tools:,}")
    print(f"  w/ tool_calls: {with_toolcalls:,}")
    print(f"  w/ reasoning:  {with_reasoning:,}")
    print(f"avg messages/row: {avg_turns:.1f}  (max {max(turn_counts)})")
    print(f"approx content tokens: ~{approx_chars // 4:,} (chars/4)")
    print("by source:")
    for src, n in sources.most_common():
        print(f"  {src:32s} {n:,}")
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m tools.agent_warmup.cli",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_imp = sub.add_parser("import-hf", help="normalize a published trace dataset")
    p_imp.add_argument("preset", choices=sorted(PRESETS))
    p_imp.add_argument("--out", required=True, help="output JSONL path")
    p_imp.add_argument("--max-items", type=int, default=None)
    p_imp.add_argument("--files", nargs="+", default=None,
                       help="override the preset's source files")
    p_imp.set_defaults(func=cmd_import_hf)

    p_gen = sub.add_parser("generate", help="run our agent on SWE-bench (verified)")
    p_gen.add_argument("--out", required=True)
    p_gen.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    p_gen.add_argument("--split", default="test")
    p_gen.add_argument("--num-tasks", type=int, default=None)
    p_gen.add_argument("--instance-ids", nargs="+", default=None)
    p_gen.add_argument("--concurrency", type=int, default=4)
    p_gen.add_argument("--max-turns", type=int, default=50)
    p_gen.add_argument("--max-tokens", type=int, default=8192,
                       help="max_tokens per completion. Too small truncates a turn "
                            "mid-output (dropped trajectory); too large is clamped "
                            "automatically to fit the model's context window, so "
                            "there's no point setting it near the context size "
                            "(8192 is plenty for these agent turns).")
    p_gen.add_argument("--sandbox-type", default="docker",
                       help="Flash Sandbox backend (CSCS service is 'kubernetes')")
    p_gen.add_argument("--skip-verification", action="store_true",
                       help="don't run tests; emit unverified trajectories "
                            "(verified=false). Also allows repos the swebench "
                            "harness can't verify, via --fallback-image.")
    p_gen.add_argument("--fallback-image", default="python:3.12",
                       help="image for repos without a swebench eval image; the "
                            "repo is cloned at base_commit (used with "
                            "--skip-verification)")
    p_gen.add_argument("--model", default=None, help="override OPENAI_COMPATIBLE_MODEL")
    p_gen.add_argument("--thinking", choices=["auto", "on", "off"], default="auto",
                       help="send chat_template_kwargs thinking flag. 'off' disables "
                            "reasoning (recommended for Kimi-K2.5 + tools: thinking "
                            "on makes the SGLang template emit empty turns after a "
                            "tool result). 'auto' = off for kimi* models, else "
                            "unset. 'on' leaves it unset (relies on the nudge).")
    p_gen.add_argument("--keep-unverified", action="store_true",
                       help="also write rows whose patch failed the tests")
    p_gen.add_argument("--resume", action="store_true",
                       help="preserve an existing --out file: skip instances "
                            "already present in it and append new rows (instead "
                            "of overwriting). Resumes an interrupted run.")
    p_gen.add_argument("--env", default=None, help="path to .env (default: search up)")
    p_gen.set_defaults(func=cmd_generate)

    p_mrg = sub.add_parser("merge", help="concat + dedup JSONL datasets")
    p_mrg.add_argument("inputs", nargs="+", help="input JSONL paths / globs")
    p_mrg.add_argument("--out", required=True)
    p_mrg.set_defaults(func=cmd_merge)

    p_st = sub.add_parser("stats", help="summarize a JSONL dataset")
    p_st.add_argument("inputs", nargs="+", help="input JSONL paths")
    p_st.set_defaults(func=cmd_stats)

    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

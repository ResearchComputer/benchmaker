#!/usr/bin/env python3
"""Filter a graded-trajectory JSONL down to the successful tasks.

``collect_trajectories.py`` writes one record per SWE-bench trial, each tagged
with its grade (``passed``, ``resolved``, ``reward``). This script keeps only
the ones you want — by default the *passed* tasks — so you can carve a clean
success-only slice out of a mixed file such as ``.local/pi-host-traj-v1-500.jsonl``.

The filter is keyed on a single field (``--field``, default ``passed``): a record
is kept when the field is truthy. ``--field reward`` is special-cased to a numeric
threshold (``--threshold``, default ``>= 1.0``) so it works for graded reward,
and ``--negate`` flips the test to keep the *unsuccessful* tasks instead (useful
for building a failure-analysis set).

By default the result is written next to the input (``<stem>.success.jsonl``);
``--out`` picks an explicit path and ``--in-place`` rewrites the input file
(after an atomic temp-file write + rename so a crash can't truncate it).

Examples::

    # keep only passed tasks -> .local/pi-host-traj-v1-500.success.jsonl
    python scripts/filter_trajectories.py .local/pi-host-traj-v1-500.jsonl

    # rewrite the input in place
    python scripts/filter_trajectories.py .local/pi-host-traj-v1-500.jsonl --in-place

    # keep unresolved trials (failure set) somewhere explicit
    python scripts/filter_trajectories.py IN.jsonl --field resolved --negate \
        --out failures.jsonl

    # keep any task scoring at least 0.5 reward
    python scripts/filter_trajectories.py IN.jsonl --field reward --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

# Which fields are "did it succeed" booleans vs. numeric reward.
_BOOL_FIELDS = ("passed", "resolved")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path,
                   help="Input graded-trajectory JSONL.")
    p.add_argument("--field", default="passed", choices=_BOOL_FIELDS + ("reward",),
                   help="Grade field to filter on (default: passed).")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="For --field reward: keep records with reward >= this "
                        "(default: 1.0).")
    p.add_argument("--negate", action="store_true",
                   help="Keep the *failing* records instead (field falsy / below "
                        "threshold).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSONL (default: <input>.success.jsonl).")
    p.add_argument("--in-place", action="store_true",
                   help="Rewrite the input file (atomic temp + rename).")
    args = p.parse_args(argv)
    if args.in_place and args.out is not None:
        p.error("--in-place and --out are mutually exclusive.")
    return args


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed records, skipping blank/unparseable lines defensively."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: skipping unparseable line in {path}",
                      file=sys.stderr)
                continue
            if isinstance(rec, dict):
                yield rec


def _keep(rec: dict[str, Any], field: str, threshold: float, negate: bool) -> bool:
    """The filter test for one record."""
    value = rec.get(field)
    if field == "reward":
        try:
            ok = value is not None and float(value) >= threshold
        except (TypeError, ValueError):
            ok = False
    else:
        ok = bool(value)
    return (not ok) if negate else ok


def _default_out(path: Path, negate: bool) -> Path:
    suffix = "failure" if negate else "success"
    return path.with_suffix(f".{suffix}{path.suffix}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    out_path = args.input if args.in_place else (args.out or _default_out(args.input,
                                                                          args.negate))

    # Write to a temp file in the same dir, then rename — atomic, so a crash or
    # SIGKILL can never leave a half-written trajectory store.
    tmp_fd, tmp_name = _temp_in_dir(out_path.parent)
    total = kept = 0
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            for rec in _iter_records(args.input):
                total += 1
                if _keep(rec, args.field, args.threshold, args.negate):
                    fh.write(json.dumps(rec, default=str) + "\n")
                    kept += 1
        os.replace(tmp_name, out_path)
    except BaseException:
        # Clean up the temp file on any failure so we never leave litter.
        try:
            os.close(tmp_fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

    dropped = total - kept
    pct = (kept / total * 100.0) if total else 0.0
    label = "failing" if args.negate else "successful"
    print(f"read {total}; kept {kept} {label} ({pct:.1f}%), dropped {dropped} "
          f"-> {out_path}")
    return 0


def _temp_in_dir(directory: Path) -> tuple[int, str]:
    """``mkstemp`` anchored in ``directory`` so ``os.replace`` stays on one fs."""
    directory.mkdir(parents=True, exist_ok=True)
    return _mkstemp(directory)


def _mkstemp(directory: Path) -> tuple[int, str]:
    # Thin wrapper so the atomic-write logic above is easy to follow.
    import tempfile
    return tempfile.mkstemp(prefix=".filter-", suffix=".tmp", dir=str(directory))


if __name__ == "__main__":
    raise SystemExit(main())

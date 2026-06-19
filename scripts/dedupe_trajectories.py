#!/usr/bin/env python3
"""Inspect and deduplicate a graded-trajectory JSONL.

``collect_trajectories.py`` keys each record by its SWE-bench instance id and is
careful never to write the same instance twice. But files get concatenated,
merged across runs, or hand-edited, and then the same instance can appear more
than once — which silently double-counts a task in any pass-rate or replay use.
This tool finds those duplicates and, on request, removes them.

**Inspect (default, read-only).** Prints every duplicated key, its copies, and a
one-line summary of each (trial, reward/passed, termination), marking which copy
would be KEPT vs DROPPED. Writes nothing. Exits ``0`` if the file is already
unique, ``1`` if any duplicate exists (so it doubles as a CI guard).

**Write (``--out`` / ``--in-place``).** Emits the deduplicated store — one record
per key — via an atomic temp-file write + rename, so a crash can never truncate
it. Kept records preserve first-appearance order.

**Keep policy (best-graded).** Within a duplicate group the most informative copy
wins, ranked by: graded over ungraded (``reward`` is not null), then passed over
failed, then a real termination (``completed``/``timeout``) over a cut-off
(``incomplete``/unknown). True ties keep the last seen, matching
``trajectory.load_store``'s last-wins semantics.

The dedup key is the instance id (``--key``), falling back to the trial name when
the instance id is absent — the same key ``collect_trajectories`` dedups on.

Examples::

    # inspect (read-only); exit 1 if duplicates exist
    python scripts/dedupe_trajectories.py .local/pi-host-traj-v1-500.jsonl

    # rewrite the file in place, keeping the best-graded copy of each instance
    python scripts/dedupe_trajectories.py .local/pi-host-traj-v1-500.jsonl --in-place

    # write the deduped store somewhere explicit
    python scripts/dedupe_trajectories.py IN.jsonl --out deduped.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

# Terminations that mean the run actually ended (vs was cut off / unknown).
_REAL_TERMINATIONS = ("completed", "timeout")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="Input graded-trajectory JSONL.")
    p.add_argument("--key", default="instance_id",
                   choices=["instance_id", "key", "trial"],
                   help="Field to dedup on (default: instance_id; falls back to "
                        "the trial name when absent).")
    p.add_argument("--out", type=Path, default=None,
                   help="Write the deduped store here (default: inspect only).")
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


def record_key(rec: dict[str, Any], field: str) -> Optional[str]:
    """The dedup key for a record: its ``field`` value, the trial name as backup."""
    return rec.get(field) or rec.get("trial")


def _rank(rec: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Best-graded ordering key: higher wins within a duplicate group."""
    return (
        rec.get("reward") is not None,                       # graded > ungraded
        bool(rec.get("passed")),                             # pass   > fail
        rec.get("termination") in _REAL_TERMINATIONS,        # real end > cut-off
    )


def find_duplicates(records: list[dict[str, Any]],
                    field: str) -> dict[str, list[dict[str, Any]]]:
    """Map each key that appears more than once to its records, in input order."""
    by_key: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        key = record_key(rec, field)
        if key is None:
            continue
        by_key.setdefault(key, []).append(rec)
    return {k: v for k, v in by_key.items() if len(v) > 1}


def dedupe(records: list[dict[str, Any]],
           field: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (kept, dropped). One record per key — the best-graded copy, ties to
    last seen — with kept records in first-appearance order of their key."""
    order: list[str] = []
    best: dict[str, dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    for rec in records:
        key = record_key(rec, field)
        if key is None:
            # No usable key — can't dedup it; keep it verbatim in order.
            order.append(id(rec))            # unique sentinel; never collides
            best[id(rec)] = rec
            continue
        if key not in best:
            order.append(key)
            best[key] = rec
            continue
        # Ties (>=) keep the later record, matching load_store's last-wins.
        if _rank(rec) >= _rank(best[key]):
            dropped.append(best[key])
            best[key] = rec
        else:
            dropped.append(rec)
    return [best[k] for k in order], dropped


def _summary(rec: dict[str, Any]) -> str:
    return (f"trial={rec.get('trial')} reward={rec.get('reward')} "
            f"passed={rec.get('passed')} termination={rec.get('termination')}")


def format_report(groups: dict[str, list[dict[str, Any]]],
                  field: str) -> str:
    """Human-readable inspect report: each duplicated key with KEEP/DROP marks."""
    lines: list[str] = []
    n_dropped = 0
    for key in sorted(groups):
        recs = groups[key]
        # The kept copy is the best-graded one; ties to the last seen.
        keep = max(recs, key=lambda r: (_rank(r), recs.index(r)))
        lines.append(f"{field}={key}  ({len(recs)} copies)")
        for rec in recs:
            mark = "KEEP" if rec is keep else "DROP"
            if rec is not keep:
                n_dropped += 1
            lines.append(f"    {mark}  {_summary(rec)}")
    header = (f"{len(groups)} duplicated {field}(s); "
              f"{n_dropped} record(s) would be dropped")
    return "\n".join([header, *lines]) if groups else "no duplicates"


def _atomic_write(records: list[dict[str, Any]], out_path: Path) -> None:
    """Write one record per line to ``out_path`` via temp-file + rename."""
    import tempfile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".dedupe-", suffix=".tmp",
                               dir=str(out_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, default=str) + "\n")
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    records = list(_iter_records(args.input))
    groups = find_duplicates(records, args.key)

    # Inspect mode (no write target): report and signal via exit code.
    if not args.in_place and args.out is None:
        print(format_report(groups, args.key))
        return 1 if groups else 0

    kept, dropped = dedupe(records, args.key)
    out_path = args.input if args.in_place else args.out
    _atomic_write(kept, out_path)
    print(f"read {len(records)}; kept {len(kept)}, dropped {len(dropped)} "
          f"duplicate(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Backfill the completion-status block onto an already-collected trajectory JSONL.

``collect_trajectories.py`` now tags every fresh record with its termination
status (``exit_status``/``termination``/``completed``/``status_source``). This
one-off applies the same labels to a file collected *before* that change — e.g.
``.local/pi-host-traj-v1-500.jsonl`` — so old and new records carry the same
fields.

For each record the status is **authoritative** when the trial's harbor
``result.json`` still exists under ``--jobs-root`` (``status_source`` =
``result_json``): ``exit_status`` is read straight from it and mapped to
``completed`` / ``timeout`` / ``error``. When the job dir has been cleaned up
the status is **inferred** from the record's own stored last turn
(``status_source`` = ``inferred``): a trailing ``stop`` turn means the agent
finished on its own (``completed``); a trailing tool-call turn means the run was
cut off — a timeout *or* an error, which the trajectory alone cannot tell apart
(``incomplete``).

The rewrite is in place via an atomic temp-file + rename, so a crash can never
truncate the input. It is idempotent: re-running only refreshes the fields.

Examples::

    # label the v1 set, reading authoritative status from ./jobs where available
    python scripts/backfill_trajectory_status.py .local/pi-host-traj-v1-500.jsonl

    # write a labelled copy instead of rewriting in place
    python scripts/backfill_trajectory_status.py IN.jsonl --out OUT.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_collector():
    """Import collect_trajectories.py for its shared status helpers."""
    spec = _ilu.spec_from_file_location(
        "collect_trajectories", str(REPO_ROOT / "scripts" / "collect_trajectories.py"))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_C = _load_collector()


def trial_result_dir(trial: str, jobs_root: Path) -> Optional[Path]:
    """The ``jobs_root/*/<trial>`` dir that still has a ``result.json``, or None."""
    for rj in sorted(jobs_root.glob(f"*/{trial}/result.json")):
        return rj.parent
    return None


def status_for_record(rec: dict[str, Any], jobs_root: Path) -> dict[str, Any]:
    """Completion fields for one already-collected record.

    Authoritative from the trial's ``result.json`` when its job dir survives,
    else inferred from the record's stored last turn.
    """
    trial = rec.get("trial")
    tdir = trial_result_dir(trial, jobs_root) if trial else None
    if tdir is not None:
        exit_status = _C._read_exit_status(tdir)
        if exit_status is not None:
            termination = _C._TERMINATION_BY_EXIT.get(exit_status, "error")
            return {"exit_status": exit_status, "termination": termination,
                    "completed": termination == "completed",
                    "status_source": "result_json"}
    turns = rec.get("turns") or []
    last = turns[-1] if turns and isinstance(turns[-1], dict) else None
    finish = last.get("finish_reason") if last else None
    has_tool_calls = bool(last.get("tool_calls")) if last else False
    termination = _C._termination_from_finish(finish, has_tool_calls)
    return {"exit_status": "unknown", "termination": termination,
            "completed": termination == "completed", "status_source": "inferred"}


def backfill_file(path: Path, *, jobs_root: Path, out: Optional[Path] = None) -> int:
    """Add/refresh the status block on every record; return how many were written.

    Atomic: writes a temp file in the destination dir and renames over it.
    """
    out_path = out or path
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: skipping unparseable line in {path}", file=sys.stderr)
                continue
            if not isinstance(rec, dict):
                continue
            rec.update(status_for_record(rec, jobs_root))
            records.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".backfill-", suffix=".tmp", dir=str(out_path.parent))
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
    return len(records)


def _summarise(records_path: Path) -> None:
    from collections import Counter
    term = Counter()
    src = Counter()
    for line in records_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        term[rec.get("termination")] += 1
        src[rec.get("status_source")] += 1
    print(f"  termination: {dict(term)}")
    print(f"  source:      {dict(src)}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="Collected trajectory JSONL to label.")
    p.add_argument("--jobs-root", type=Path, default=REPO_ROOT / "jobs",
                   help="Where to look for surviving <job>/<trial>/result.json "
                        "(default: ./jobs).")
    p.add_argument("--out", type=Path, default=None,
                   help="Write here instead of rewriting the input in place.")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    out_path = args.out or args.input
    n = backfill_file(args.input, jobs_root=args.jobs_root, out=out_path)
    print(f"labelled {n} record(s) -> {out_path}")
    _summarise(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

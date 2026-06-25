#!/usr/bin/env python3
"""Thin CLI over benchmaker.swebench.foldlogs — losslessly fold pi-host.log files
by removing REDUNDANT records (message_update / turn_end / agent_end), NOT by
compression. See benchmaker/swebench/foldlogs.py for the full rationale and the
per-file safety guarantees.

Usage:
  python3 fold_pihost_log.py FILE [FILE ...]        # fold in place (atomic)
  python3 fold_pihost_log.py --dry-run FILE         # report bytes, change nothing
  find ... -name pi-host.log | python3 fold_pihost_log.py --stdin

The swebench-replay recipe also calls foldlogs.fold_tree() automatically after each
run (disable with --no-fold-logs or BENCH_FOLD_LOGS=0); this CLI is for folding
existing corpora or re-folding by hand.
"""
import argparse
import os
import sys

try:
    from benchmaker.swebench.foldlogs import fold_file
except ImportError:
    # allow running the script standalone from a source checkout
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from benchmaker.swebench.foldlogs import fold_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--stdin", action="store_true", help="read file paths from stdin")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = list(a.files)
    if a.stdin:
        files += [l.strip() for l in sys.stdin if l.strip()]
    tot_o = tot_n = 0
    nf = ns = ne = nn = 0
    for p in files:
        try:
            status, ob, nb, miss = fold_file(p, dry_run=a.dry_run)
        except Exception as e:
            print(f"ERROR {p}: {e}", file=sys.stderr)
            ne += 1
            continue
        tot_o += ob
        tot_n += nb
        if status == "SKIPPED_UNSAFE":
            ns += 1
            print(f"SKIP (would drop {miss} unique msgs): {p}", file=sys.stderr)
        elif status == "NOOP":
            nn += 1
        else:
            nf += 1
    saved = tot_o - tot_n
    print(f"files folded={nf} noop={nn} skipped={ns} errors={ne} | "
          f"{tot_o/1e9:.2f}GB -> {tot_n/1e9:.2f}GB  saved {saved/1e9:.2f}GB "
          f"({100*saved/tot_o:.1f}%)" if tot_o else "no files")


if __name__ == "__main__":
    main()

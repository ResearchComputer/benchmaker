#!/usr/bin/env python3
"""Collapse SWE-bench task dirs under a root into single <trial>.jsonl files.

Thin CLI over benchmaker.swebench.cleanjobs.clean_tree. DESTRUCTIVE without
--dry-run: each cleaned task dir is deleted after its replacement is written and
verified."""
import argparse
import os
import sys

try:
    from benchmaker.swebench.cleanjobs import clean_tree
except ImportError:
    # allow running the script standalone from a source checkout
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from benchmaker.swebench.cleanjobs import clean_tree


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default="jobs",
                    help="root to clean (default: jobs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report sizes/counts; change nothing on disk")
    args = ap.parse_args(argv)
    s = clean_tree(args.root, dry_run=args.dry_run)
    print(f"cleanjobs: {s['orig_bytes']/1e9:.2f}GB -> {s['new_bytes']/1e9:.2f}GB "
          f"(saved {s['saved_bytes']/1e9:.2f}GB; cleaned {s['cleaned']}, "
          f"skipped {s['skipped']}, errors {s['errors']})"
          + ("  [dry-run]" if args.dry_run else ""))
    for p in s["skipped_paths"]:
        print(f"  SKIPPED_UNSAFE: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

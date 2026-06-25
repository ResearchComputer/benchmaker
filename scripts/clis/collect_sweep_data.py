#!/usr/bin/env python3
"""Collect per-trial data from one or more timeout x concurrency sweep roots
into a single tidy CSV for later analysis.

Each row is one trial with its cell params (T, c), reward, and per-phase wall
durations (environment_setup / agent_setup / agent_execution / verifier / total)
pulled from the rich per-trial result.json. Verifier duration is the phase the
per-container CPU cap actually affects, so keep it separate from agent time.

Usage:
  python3 collect_sweep_data.py --root LABEL=jobs/sweep_... [--root LABEL2=...] \
      --out analysis/sweep_data.csv
"""
import argparse
import csv
import os
import re
import statistics as st
from datetime import datetime

CELL_RE = re.compile(r"timeout_T([0-9.eE+-]+)_c(\d+)")


def _secs(section):
    """finished_at - started_at in seconds for a {started_at, finished_at} dict."""
    if not isinstance(section, dict):
        return None
    s, f = section.get("started_at"), section.get("finished_at")
    if not s or not f:
        return None
    try:
        ds = datetime.fromisoformat(s.replace("Z", "+00:00"))
        df = datetime.fromisoformat(f.replace("Z", "+00:00"))
        return (df - ds).total_seconds()
    except ValueError:
        return None


def _verifier_timed_out(d):
    """1 if the trial failed with VerifierTimeoutError, else 0.

    exception_info is an ExceptionInfo dict with keys
    exception_type / exception_message / exception_traceback. We match on the
    type field only (substring, to tolerate fully-qualified names); the
    free-form message is not consulted to avoid false positives.
    """
    ei = d.get("exception_info") or {}
    return 1 if "VerifierTimeoutError" in ei.get("exception_type", "") else 0


def _trial_rows(label, root):
    from benchmaker.swebench import trial_io
    rows = []
    for trial in trial_io.iter_trials(root):
        m = CELL_RE.search(trial.path)
        if not m:
            continue
        T, C = float(m.group(1)), int(m.group(2))
        d = trial.result
        reward = trial.reward
        rows.append({
            "run": label,
            "T": f"{T:g}",
            "c": C,
            "task": trial.task_name,
            "reward": reward,
            "solved": 1 if reward == 1.0 else 0,
            "graded": 0 if reward is None else 1,
            "verifier_timeout": _verifier_timed_out(d),
            "env_setup_s": _secs(d.get("environment_setup")),
            "agent_setup_s": _secs(d.get("agent_setup")),
            "agent_exec_s": _secs(d.get("agent_execution")),
            "verifier_s": _secs(d.get("verifier")),
            "total_s": _secs(d),
        })
    return rows


def _fmt(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", action="append", required=True,
                    help="LABEL=path/to/sweep_root (repeatable)")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    all_rows = []
    for spec in args.root:
        label, _, root = spec.partition("=")
        if not root:
            label, root = os.path.basename(spec.rstrip("/")), spec
        n = len(all_rows)
        all_rows.extend(_trial_rows(label, root))
        print(f"{label}: {len(all_rows) - n} trials from {root}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cols = ["run", "T", "c", "task", "reward", "solved", "graded",
            "verifier_timeout",
            "env_setup_s", "agent_setup_s", "agent_exec_s", "verifier_s", "total_s"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows -> {args.out}")

    # quick per-(run,T,c) readout so the data is glanceable now
    print(f"\n{'run':<10}{'T':>5}{'c':>5}{'graded':>7}{'solved':>7}"
          f"{'acc%':>6}{'vto':>5}{'verif_med':>10}{'verif_p90':>10}{'verif_max':>10}")
    cells = {}
    for r in all_rows:
        cells.setdefault((r["run"], float(r["T"]), r["c"]), []).append(r)
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2])):
        rs = cells[key]
        g = sum(x["graded"] for x in rs)
        s = sum(x["solved"] for x in rs)
        vto = sum(x["verifier_timeout"] for x in rs)
        v = [x["verifier_s"] for x in rs]
        print(f"{key[0]:<10}{key[1]:>5g}{key[2]:>5}{g:>7}{s:>7}"
              f"{(100.0 * s / g if g else 0):>6.1f}{vto:>5}"
              f"{_fmt(v, .5):>10.1f}{_fmt(v, .9):>10.1f}{_fmt(v, 1.0):>10.1f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize a real timeout x concurrency replay sweep.

Reads the per-cell jobs directories produced by sweep_swebench_replay_timeout.sh
and prints solved counts plus exec-timeout / duration-tail stats per (T, c) cell.

Cell directories are named timeout_T{T:g}_c{C} under --out-root, matching the
normalization the shell loop applies when it creates them.
"""
import argparse
import os


def _solved(jobs_dir):
    from benchmaker.swebench import trial_io
    solved = total = 0
    for trial in trial_io.iter_trials(jobs_dir):
        r = trial.reward
        if r is None:
            continue
        total += 1
        solved += 1 if r == 1.0 else 0
    return solved, total


def _spans(jobs_dir, T):
    # A real exec timeout shows up as a bridge span with rc<0 whose duration
    # ran up to the wall; rc<0 with ~0 duration is an instant exec error, not a
    # timeout, so gate on duration to avoid conflating the two.
    from benchmaker.swebench import trial_io
    durs, n_exec, n_timeout = [], 0, 0
    for trial in trial_io.iter_trials(jobs_dir):
        for obj in trial.timeline_spans:
            if obj.get("name") != "sandbox_exec":
                continue
            d = float(obj.get("duration_s") or 0.0)
            n_exec += 1
            durs.append(d)
            if int(obj.get("rc", 0)) < 0 and d >= 0.9 * T:
                n_timeout += 1
    durs.sort()
    mean = sum(durs) / len(durs) if durs else 0.0
    p95 = durs[int(0.95 * (len(durs) - 1))] if durs else 0.0
    mx = durs[-1] if durs else 0.0
    return n_exec, n_timeout, mean, p95, mx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=os.environ.get("OUT_ROOT", "jobs"),
                    help="root holding the per-cell jobs dirs (default: $OUT_ROOT or 'jobs')")
    ap.add_argument("--timeouts", nargs="+", type=float, required=True,
                    help="exec-timeout grid (seconds), space-separated")
    ap.add_argument("--concurrencies", nargs="+", type=int, required=True,
                    help="concurrency grid, space-separated")
    args = ap.parse_args()

    hdr = f"{'T(s)':>6} {'c':>5} {'solved':>9} {'timeouts':>9} {'execs':>7} {'mean(s)':>8} {'p95(s)':>8} {'max(s)':>8}"
    print(hdr)
    print("-" * len(hdr))
    for T in args.timeouts:
        for C in args.concurrencies:
            jobs_dir = os.path.join(args.out_root, f"timeout_T{T:g}_c{C}")
            if not os.path.isdir(jobs_dir):
                print(f"{T:>6g} {C:>5} {'(not run)':>9}")
                continue
            s, n = _solved(jobs_dir)
            n_exec, n_to, mean, p95, mx = _spans(jobs_dir, T)
            solved = f"{s}/{n}" if n else "(empty)"
            print(f"{T:>6g} {C:>5} {solved:>9} {n_to:>9} {n_exec:>7} {mean:>8.2f} {p95:>8.1f} {mx:>8.1f}")


if __name__ == "__main__":
    main()

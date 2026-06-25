#!/usr/bin/env python3
"""Predicted (strict Tier-1) vs actual (Tier-2) solved for the load-factor sweep.

Reads the uncontended c1 baseline run to build the offline Tier-1 accuracy curve
and compares its predicted solved count at each effective per-command budget
tau = T / L against the actual solved count from each loadfactor_L{L:g} jobs dir
produced by sweep_swebench_loadfactor.sh.
"""
import argparse
import math
import os

from benchmaker.swebench.timeout_load import accuracy_curve


def _c1_tasks(c1_dir):
    # Tier-1 baseline tasks (reward, max_command_duration) from the uncontended c1 run.
    from benchmaker.swebench import trial_io
    tasks = []
    for trial in trial_io.iter_trials(c1_dir):
        r = trial.reward
        if r is None:
            continue
        max_d = max((c.duration_s for c in
                     trial_io.recover_command_timings_from_trial(trial)), default=0.0)
        tasks.append((float(r or 0.0), max_d))
    return tasks


def _predicted_solved(c1_tasks, T, L):
    tau = math.inf if L <= 1 else T / L
    return accuracy_curve(c1_tasks, [tau])[0].n_solved if c1_tasks else None


def _actual_solved(jobs_dir):
    from benchmaker.swebench import trial_io
    solved = total = 0
    for trial in trial_io.iter_trials(jobs_dir):
        r = trial.reward
        if r is None:
            continue
        total += 1
        solved += 1 if r == 1.0 else 0
    return solved, total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=os.environ.get("OUT_ROOT", "jobs"),
                    help="root holding the per-L jobs dirs (default: $OUT_ROOT or 'jobs')")
    ap.add_argument("--inject-timeout", type=float, required=True,
                    help="per-command timeout T (seconds)")
    ap.add_argument("--c1-dir", required=True,
                    help="uncontended c1 baseline jobs dir used for Tier-1 prediction")
    ap.add_argument("--load-factors", nargs="+", type=float, required=True,
                    help="synthetic load-factor grid L, space-separated")
    args = ap.parse_args()

    T = args.inject_timeout
    c1_tasks = _c1_tasks(args.c1_dir)

    print(f"{'L':>6} {'tau(s)':>7} {'predicted':>10} {'actual':>10}")
    for L in args.load_factors:
        tau = "inf" if L <= 1 else f"{T / L:g}"
        pred = _predicted_solved(c1_tasks, T, L)
        jobs_dir = os.path.join(args.out_root, f"loadfactor_L{L:g}")
        if os.path.isdir(jobs_dir):
            s, n = _actual_solved(jobs_dir)
            actual = f"{s}/{n}" if n else "(empty)"
        else:
            actual = "(not run)"
        print(f"{L:>6g} {tau:>7} {str(pred):>10} {actual:>10}")


if __name__ == "__main__":
    main()

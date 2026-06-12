#!/usr/bin/env python3
"""Offline strict accuracy(tau) curve for command-timeouts-under-load.

Reads an uncontended (e.g. c1) jobs directory, recovers per-command durations
from each task's agent log, and computes accuracy(tau) where a task survives iff
all its commands finish within the effective budget tau = T / L.

Usage:
    python scripts/analyze_timeout_curve.py jobs/replay_..._c1_xxxx \
        --timeout 600 --out results/timeout_curve --plot
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os

from benchmaker.swebench.timeout_load import accuracy_curve, recover_command_timings

DEFAULT_TAUS = [math.inf, 300, 120, 60, 30, 20, 10, 5, 3, 2, 1, 0.5]


def collect_tasks(jobs_dir: str) -> list[tuple[float, float, str]]:
    """Return (reward, max_duration_s, task_name) for each task in jobs_dir."""
    tasks: list[tuple[float, float, str]] = []
    for tdir in sorted(glob.glob(os.path.join(jobs_dir, "*"))):
        if not os.path.isdir(tdir):
            continue
        lp = os.path.join(tdir, "agent", "pi-container.log")
        rj = os.path.join(tdir, "result.json")
        if not (os.path.exists(lp) and os.path.exists(rj)):
            continue
        name = os.path.basename(tdir).rsplit("__", 1)[0]
        timings = recover_command_timings(lp)
        max_d = max((c.duration_s for c in timings), default=0.0)
        with open(rj) as f:
            reward = (json.load(f).get("verifier_result") or {}).get("rewards", {}).get("reward")
        tasks.append((float(reward or 0.0), max_d, name))
    return tasks


def _load_factor(timeout: float, tau: float) -> float:
    if math.isinf(tau):
        return 1.0
    return timeout / tau if tau else math.inf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs_dir", help="uncontended (c1) jobs directory")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="per-command timeout T in seconds (default 600)")
    ap.add_argument("--taus", type=float, nargs="*", default=None,
                    help="explicit tau grid (seconds); default is a log-ish grid")
    ap.add_argument("--out", default=None, help="output prefix (writes .csv and .json)")
    ap.add_argument("--plot", action="store_true", help="also write <out>.pdf (needs matplotlib)")
    a = ap.parse_args()

    tasks = collect_tasks(a.jobs_dir)
    if not tasks:
        ap.error(f"no tasks with logs found under {a.jobs_dir}")
    taus = a.taus if a.taus else DEFAULT_TAUS
    curve = accuracy_curve([(r, d) for r, d, _ in tasks], taus)
    n = len(tasks)
    baseline = sum(1 for r, _d, _ in tasks if r == 1.0)

    print(f"tasks={n}  T={a.timeout:g}s  baseline_solved={baseline}")
    print(f"{'tau(s)':>8} {'L=T/tau':>8} {'survive':>8} {'solved':>7} {'acc%':>6} {'broken':>7}")
    for p in curve:
        tau_s = "inf" if math.isinf(p.tau_s) else f"{p.tau_s:g}"
        L = _load_factor(a.timeout, p.tau_s)
        Ls = "inf" if math.isinf(L) else f"{L:.1f}"
        print(f"{tau_s:>8} {Ls:>8} {p.n_survive:>8} {p.n_solved:>7} "
              f"{p.accuracy * 100:>5.0f}% {p.n_broken:>7}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tau_s", "load_factor", "n_survive", "n_solved", "accuracy", "n_broken"])
            for p in curve:
                w.writerow([p.tau_s, _load_factor(a.timeout, p.tau_s),
                            p.n_survive, p.n_solved, p.accuracy, p.n_broken])
        with open(a.out + ".json", "w") as f:
            json.dump([p.__dict__ for p in curve], f, indent=2, default=str)
        print(f"wrote {a.out}.csv and {a.out}.json")

    if a.plot:
        _plot(curve, a.timeout, a.out or "timeout_curve")
    return 0


def _plot(curve, timeout: float, out_prefix: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [p for p in curve if not math.isinf(p.tau_s)]
    xs = [p.tau_s for p in pts]
    ys = [p.accuracy * 100 for p in pts]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(xs, ys, marker="o")
    ax.set_xscale("log")
    ax.invert_xaxis()  # more load (smaller tau) toward the right
    ax.set_xlabel(r"effective per-command budget $\tau = T/L$ (s)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Strict accuracy vs load (T={timeout:g}s)")
    fig.tight_layout()
    fig.savefig(out_prefix + ".pdf")
    print(f"wrote {out_prefix}.pdf")


if __name__ == "__main__":
    raise SystemExit(main())

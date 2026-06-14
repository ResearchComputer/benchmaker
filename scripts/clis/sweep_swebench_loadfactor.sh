#!/usr/bin/env bash
# Tier-2 validation for the command-timeout-under-load experiment.
#
# Replays the SAME 100 trajectories at a FIXED low real concurrency (so command
# durations stay uncontended) while sweeping a synthetic load factor L via
# BENCH_LOAD_FACTOR. A command of real duration d is reported as a timeout when
# L*d > T (T = BENCH_INJECT_TIMEOUT_S). Each L maps to an effective per-command
# budget tau = T / L:
#
#     L:   1     2     5    10    20    30    60   120   200   300   600
#   tau: inf   300   120    60    30    20    10     5     3     2     1   (T=600)
#
# At L=1 injection is a strict no-op, so the run must reproduce the c1 baseline
# (~49 solved) — a plumbing sanity check. As L grows, injected timeouts diverge
# the replay (replay miss -> terminal stop), and solved count should fall toward
# the Tier-1 prediction. NOTE: the offline Tier-1 curve is a STRICT lower bound
# (it assumes any timeout = fail); the actual run may solve MORE when the fix
# already landed before the timeout, so expect actual >= predicted.
#
# Requires the live replay endpoint. BENCH_LOAD_FACTOR is read by the in-harness
# _ExecBridge, so exporting it before the benchmaker call is sufficient.
set -uo pipefail

# --- config (override via env) ---------------------------------------------
: "${FLASH_SANDBOX_URL:=http://100.71.204.79:8080}"
: "${REACHABLE_HOST:=100.101.144.78}"
: "${TRAJECTORIES:=.local/replay-trajectories.jsonl}"
: "${N_TASKS:=100}"
: "${CONCURRENCY:=1}"                 # keep low so durations are uncontended
: "${BENCH_INJECT_TIMEOUT_S:=600}"    # per-command timeout T (seconds)
: "${C1_DIR:=jobs/replay_2026-06-11__17-21-48_c1_0d73}"  # baseline for prediction
: "${LOAD_FACTORS:=1 2 5 10 20 30 60}"  # space-separated L grid
OUT_ROOT="${OUT_ROOT:-jobs}"

export FLASH_SANDBOX_URL BENCH_INJECT_TIMEOUT_S

echo "Tier-2 load-factor sweep: T=${BENCH_INJECT_TIMEOUT_S}s  concurrency=${CONCURRENCY}  n_tasks=${N_TASKS}"
echo "L grid: ${LOAD_FACTORS}"
echo

# --- run one replay per load factor ----------------------------------------
for L in ${LOAD_FACTORS}; do
    jobs_dir="${OUT_ROOT}/loadfactor_L${L}"
    echo "=== L=${L}  (tau=$(python3 -c "print('inf' if ${L}<=1 else ${BENCH_INJECT_TIMEOUT_S}/${L})")s)  -> ${jobs_dir} ==="
    BENCH_LOAD_FACTOR="${L}" benchmaker swebench-replay \
        --trajectories "${TRAJECTORIES}" \
        --mode pi-container \
        --concurrency "${CONCURRENCY}" \
        --reachable-host "${REACHABLE_HOST}" \
        --host 0.0.0.0 \
        --n-tasks "${N_TASKS}" \
        --jobs-dir "${jobs_dir}" \
        || echo "WARNING: L=${L} run exited non-zero; continuing"
    echo
done

# --- predicted (Tier-1) vs actual (Tier-2) summary -------------------------
echo "=== predicted (strict Tier-1) vs actual (Tier-2) solved ==="
OUT_ROOT="${OUT_ROOT}" python3 - "${BENCH_INJECT_TIMEOUT_S}" "${C1_DIR}" ${LOAD_FACTORS} <<'PY'
import glob
import json
import math
import os
import sys

from benchmaker.swebench.timeout_load import accuracy_curve, recover_command_timings

T = float(sys.argv[1])
c1_dir = sys.argv[2]
load_factors = [float(x) for x in sys.argv[3:]]
out_root = os.environ.get("OUT_ROOT", "jobs")


def _reward(result_json):
    try:
        with open(result_json) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return (data.get("verifier_result") or {}).get("rewards", {}).get("reward")


# Tier-1 baseline tasks (reward, max_command_duration) from the uncontended c1 run.
c1_tasks = []
for tdir in glob.glob(os.path.join(c1_dir, "*")):
    lp = os.path.join(tdir, "agent", "pi-container.log")
    rj = os.path.join(tdir, "result.json")
    if not (os.path.exists(lp) and os.path.exists(rj)):
        continue
    max_d = max((c.duration_s for c in recover_command_timings(lp)), default=0.0)
    r = _reward(rj)
    if r is None:
        continue
    c1_tasks.append((float(r or 0.0), max_d))


def predicted_solved(L):
    tau = math.inf if L <= 1 else T / L
    return accuracy_curve(c1_tasks, [tau])[0].n_solved if c1_tasks else None


def actual_solved(jobs_dir):
    # Recurse: --jobs-dir output layout may nest task dirs one level down.
    solved = total = 0
    for rj in glob.glob(os.path.join(jobs_dir, "**", "result.json"), recursive=True):
        r = _reward(rj)
        if r is None:
            continue
        total += 1
        solved += 1 if r == 1.0 else 0
    return solved, total


print(f"{'L':>6} {'tau(s)':>7} {'predicted':>10} {'actual':>10}")
for L in load_factors:
    tau = "inf" if L <= 1 else f"{T / L:g}"
    pred = predicted_solved(L)
    jobs_dir = os.path.join(out_root, f"loadfactor_L{L:g}")
    if os.path.isdir(jobs_dir):
        s, n = actual_solved(jobs_dir)
        actual = f"{s}/{n}" if n else "(empty)"
    else:
        actual = "(not run)"
    print(f"{L:>6g} {tau:>7} {str(pred):>10} {actual:>10}")
PY

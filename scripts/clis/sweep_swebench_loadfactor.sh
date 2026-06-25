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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
OUT_ROOT="${OUT_ROOT}" python3 "${SCRIPT_DIR}/summarize_loadfactor_sweep.py" \
    --out-root "${OUT_ROOT}" \
    --inject-timeout "${BENCH_INJECT_TIMEOUT_S}" \
    --c1-dir "${C1_DIR}" \
    --load-factors ${LOAD_FACTORS}

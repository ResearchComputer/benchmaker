set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- config (override via env) ---------------------------------------------
: "${FLASH_SANDBOX_URL:=http://100.71.204.79:8080}"
: "${REACHABLE_HOST:=100.71.204.79}"
: "${TRAJECTORIES:=.local/pi-host-traj-v1-500.jsonl}"
: "${N_TASKS:=500}"
: "${EXCLUDE_TASKS:=sphinx-doc__sphinx-7590 scikit-learn__scikit-learn-14710}"
: "${TIMEOUTS:=10}"
: "${CONCURRENCIES:=16}"
# Flash Sandbox backend the replay requests (forwarded as
# environment.kwargs.backend_type). "docker" preserves prior behavior; set to
# "firecracker" to run the sweep on microVMs (requires an FC-capable node).
: "${BACKEND_TYPE:=docker}"
# When 1, replay fails fast on environment divergence (a live tool-result status
# that differs from the recording), so a command that times out under a tight
# --exec-timeout-sec counts as a task failure. This is what makes the pass rate
# actually sensitive to the timeout; default off preserves prior behavior.
: "${VALIDATE_OBSERVATIONS:=0}"
# When 1, enable sandbox QoS for the replay (verifier gets a relaxed exec
# timeout via the multiplier). Mirrors VALIDATE_OBSERVATIONS: off by default so
# the standard sweep invocation is unchanged.
: "${QOS_ENABLED:=0}"
# QOS_VERIFIER_TIMEOUT_MULTIPLIER: when QOS_ENABLED=1, the verifier's timeout
# budget is scaled by this factor (it runs at best_effort cpu.weight, so it
# needs more wall-clock). 2.0 gives the verifier twice the base budget.
: "${QOS_VERIFIER_TIMEOUT_MULTIPLIER:=2.0}"
# cpu.weight tiers (defaults match harbor's FlashSandboxEnvironment defaults).
# Exposed so the B4 ship-gate loop can raise best_effort and re-run without
# editing this script. Only forwarded when QOS_ENABLED=1.
: "${QOS_ON_DEMAND_CPU_WEIGHT:=10000}"
: "${QOS_BEST_EFFORT_CPU_WEIGHT:=10}"
OUT_ROOT="${OUT_ROOT:-jobs}"

# Each sweep run gets its own timestamped root so repeated runs don't pile up
# into the same cell dirs (which made the summary count N runs x 497 tasks).
# Override SWEEP_ROOT to point the run + summary at an existing sweep instead.
SWEEP_TS="$(date +%Y-%m-%d__%H-%M-%S)"
SWEEP_ROOT="${SWEEP_ROOT:-${OUT_ROOT}/sweep_${SWEEP_TS}}"

export FLASH_SANDBOX_URL

exclude_args=()
for t in ${EXCLUDE_TASKS}; do exclude_args+=(--exclude-task "${t}"); done

validate_args=()
if [ "${VALIDATE_OBSERVATIONS}" = "1" ]; then
    validate_args+=(--validate-observations)
fi

qos_args=()
if [ "${QOS_ENABLED}" = "1" ]; then
    qos_args+=(--qos-enabled
               --qos-verifier-timeout-multiplier "${QOS_VERIFIER_TIMEOUT_MULTIPLIER}"
               --on-demand-cpu-weight "${QOS_ON_DEMAND_CPU_WEIGHT}"
               --best-effort-cpu-weight "${QOS_BEST_EFFORT_CPU_WEIGHT}")
fi

echo "Real timeout x concurrency sweep (pi-host)  n_tasks=${N_TASKS}"
echo "  T grid:           ${TIMEOUTS}"
echo "  concurrency grid: ${CONCURRENCIES}"
echo "  backend:          ${BACKEND_TYPE}"
echo "  excluding:        ${EXCLUDE_TASKS:-<none>}"
echo "  validate-observations: ${VALIDATE_OBSERVATIONS}"
if [ "${QOS_ENABLED}" = "1" ]; then
    echo "  qos-enabled:      ${QOS_ENABLED} (verifier timeout x${QOS_VERIFIER_TIMEOUT_MULTIPLIER}, on_demand=${QOS_ON_DEMAND_CPU_WEIGHT} best_effort=${QOS_BEST_EFFORT_CPU_WEIGHT})"
else
    echo "  qos-enabled:      ${QOS_ENABLED}"
fi
echo "  sweep root:       ${SWEEP_ROOT}"
echo

# --- run one replay per (T, C) cell ----------------------------------------
for T in ${TIMEOUTS}; do
    # Normalize T the same way the summary does (Python f"{T:g}"), so the
    # directory created here matches the one the summary stats. Without this,
    # e.g. T=0.00001 creates timeout_T0.00001_c100 but the summary looks up
    # timeout_T1e-05_c100 -> "(not run)".
    Tg=$(python3 -c "print(f'{float(\"${T}\"):g}')")
    for C in ${CONCURRENCIES}; do
        jobs_dir="${SWEEP_ROOT}/timeout_T${Tg}_c${C}"
        echo "=== T=${T}s  c=${C}  -> ${jobs_dir} ==="
        benchmaker swebench-replay \
            --trajectories "${TRAJECTORIES}" \
            --mode pi-host \
            --route-tools all \
            --exec-timeout-sec "${T}" \
            --backend-type "${BACKEND_TYPE}" \
            --concurrency "${C}" \
            --reachable-host "${REACHABLE_HOST}" \
            --host 0.0.0.0 \
            --n-tasks "${N_TASKS}" \
            "${exclude_args[@]}" \
            "${validate_args[@]}" \
            "${qos_args[@]}" \
            --jobs-dir "${jobs_dir}" \
            || echo "WARNING: T=${T} c=${C} run exited non-zero; continuing"
        echo
    done
done

# --- summary: solved + timeouts + duration tail per cell -------------------
echo "=== solved / timeouts / duration tail per (T, c) cell ==="
echo "  sweep root: ${SWEEP_ROOT}"
OUT_ROOT="${SWEEP_ROOT}" python3 "${SCRIPT_DIR}/summarize_replay_timeout_sweep.py" \
    --out-root "${SWEEP_ROOT}" \
    --timeouts ${TIMEOUTS} \
    --concurrencies ${CONCURRENCIES}

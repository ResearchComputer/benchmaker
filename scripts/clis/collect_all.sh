#!/usr/bin/env bash
set -euo pipefail

export FLASH_SANDBOX_URL="${FLASH_SANDBOX_URL:-http://100.71.204.79:8080}"

# --- tunables -------------------------------------------------------------- #
N_TASKS="${N_TASKS:-100}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_TURNS="${MAX_TURNS:-150}"
MEM_MAX="${MEM_MAX:-24G}"
SWAP_MAX="${SWAP_MAX:-8G}"
OUT="${OUT:-.local/pi-host-trajectories-${N_TASKS}.jsonl}"
RESUME="${RESUME:-1}"

# --- run inside a memory-capped scope -------------------------------------- #
# Re-exec ourselves once under systemd-run so the cap covers the whole process
# tree. Guarded by BENCHMAKER_SCOPED to avoid an infinite re-exec loop. Falls
# through (uncapped) if systemd-run is unavailable.
if [[ -z "${BENCHMAKER_SCOPED:-}" ]] && command -v systemd-run >/dev/null 2>&1; then
  export BENCHMAKER_SCOPED=1
  exec systemd-run --user --scope --quiet \
    -p MemoryMax="$MEM_MAX" -p MemorySwapMax="$SWAP_MAX" \
    -- "$0" "$@"
fi

# --- the actual job -------------------------------------------------------- #
mkdir -p .local
LOG=".local/collect-$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ -n "${BENCHMAKER_SCOPED:-}" ]]; then
  echo "running under systemd scope (MemoryMax=$MEM_MAX, MemorySwapMax=$SWAP_MAX)"
else
  echo "WARNING: systemd-run unavailable — running WITHOUT a memory cap"
fi
echo "concurrency=$CONCURRENCY, n-tasks=$N_TASKS, max-turns=$MAX_TURNS, resume=$RESUME, out=$OUT, logging to $LOG"

RESUME_FLAG=()
[[ "$RESUME" == "0" ]] && RESUME_FLAG=(--no-resume)

python scripts/collect_trajectories.py \
  --mode pi-host \
  --n-tasks "$N_TASKS" \
  --concurrency "$CONCURRENCY" \
  "${RESUME_FLAG[@]}" \
  --out "$OUT" \
  -- --agent-kwarg "pi_max_turns=$MAX_TURNS" 2>&1 | tee "$LOG"

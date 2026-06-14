set -euo pipefail
export FLASH_SANDBOX_URL=http://100.71.204.79:8080

python scripts/collect_trajectories.py \
  --mode pi-host \
  --n-tasks 500 \
  --concurrency 32 \
  --out pi-host-trajectories-500.jsonl

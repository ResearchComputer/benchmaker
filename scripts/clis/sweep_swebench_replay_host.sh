set -euo pipefail

export FLASH_SANDBOX_URL=http://100.71.204.79:8080

benchmaker swebench-replay \
  --trajectories .local/pi-host-trajectories-100.jsonl \
  --mode pi-host \
  --concurrency-sweep 64,100 \
  --reachable-host 100.71.204.79 \
  --host 0.0.0.0 \
  --n-tasks 100

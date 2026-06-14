set -euo pipefail

export FLASH_SANDBOX_URL=http://100.71.204.79:8080

benchmaker swebench-replay \
  --trajectories .local/replay-trajectories.jsonl \
  --mode pi-container \
  --concurrency-sweep 1,4,8,16,32,64,100 \
  --reachable-host 100.101.144.78 \
  --host 0.0.0.0 \
  --n-tasks 100

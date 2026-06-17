set -euo pipefail

export FLASH_SANDBOX_URL=http://100.71.204.79:8080

benchmaker swebench-replay \
  --trajectories .local/pi-host-trajectories-100.jsonl \
  --mode pi-host \
  --route-tools all \
  --exclude-task psf__requests-2317 \
  --concurrency-sweep 100 \
  --reachable-host 100.71.204.79 \
  --host 0.0.0.0 \
  --n-tasks 100

benchmaker swebench-replay \
  --trajectories .local/pi-host-trajectories-100.jsonl \
  --mode pi-host \
  --route-tools all \
  --exclude-task psf__requests-2317 \
  --concurrency-sweep 64 \
  --reachable-host 100.71.204.79 \
  --host 0.0.0.0 \
  --n-tasks 100

benchmaker swebench-replay \
  --trajectories .local/pi-host-trajectories-100.jsonl \
  --mode pi-host \
  --route-tools all \
  --exclude-task psf__requests-2317 \
  --concurrency-sweep 32 \
  --reachable-host 100.71.204.79 \
  --host 0.0.0.0 \
  --n-tasks 100

benchmaker swebench-replay \
  --trajectories .local/pi-host-trajectories-100.jsonl \
  --mode pi-host \
  --route-tools all \
  --exclude-task psf__requests-2317 \
  --concurrency-sweep 16 \
  --reachable-host 100.71.204.79 \
  --host 0.0.0.0 \
  --n-tasks 100

# benchmaker swebench-replay \
#   --trajectories .local/pi-container-trajectories-100.jsonl \
#   --mode pi-container \
#   --exclude-task psf__requests-2317 \
#   --concurrency-sweep 100 \
#   --reachable-host 100.71.204.79 \
#   --host 0.0.0.0 \
#   --n-tasks 100

# benchmaker swebench-replay \
#   --trajectories .local/pi-container-trajectories-100.jsonl \
#   --mode pi-container \
#   --exclude-task psf__requests-2317 \
#   --concurrency-sweep 64 \
#   --reachable-host 100.71.204.79 \
#   --host 0.0.0.0 \
#   --n-tasks 100

# benchmaker swebench-replay \
#   --trajectories .local/pi-container-trajectories-100.jsonl \
#   --mode pi-container \
#   --exclude-task psf__requests-2317 \
#   --concurrency-sweep 32 \
#   --reachable-host 100.71.204.79 \
#   --host 0.0.0.0 \
#   --n-tasks 100

# benchmaker swebench-replay \
#   --trajectories .local/pi-container-trajectories-100.jsonl \
#   --mode pi-container \
#   --exclude-task psf__requests-2317 \
#   --concurrency-sweep 16 \
#   --reachable-host 100.71.204.79 \
#   --host 0.0.0.0 \
#   --n-tasks 100

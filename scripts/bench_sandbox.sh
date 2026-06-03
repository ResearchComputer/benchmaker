#!/usr/bin/env bash
# Benchmark the SwissAI/CSCS Flash Sandbox service (https://sandbox.swissai.cscs.ch).
#
# Lazily creates one docker sandbox, fires `exec` calls at it for the run, and
# deletes it on exit. Reports per-request latency / throughput.
#
# Usage:
#   scripts/bench_sandbox.sh                     # defaults: 20 rps for 10s
#   FLASH=https://sandbox.swissai.cscs.ch RPS=50 DURATION=20 scripts/bench_sandbox.sh
set -euo pipefail
cd "$(dirname "$0")/.."

FLASH="${FLASH:-${FLASH_SANDBOX_URL:-https://sandbox.swissai.cscs.ch}}" RPS="${RPS:-20}" DURATION="${DURATION:-10}" \
python -c ''

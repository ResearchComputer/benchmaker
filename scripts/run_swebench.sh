#!/usr/bin/env bash
# Start a SWE-bench experiment via the `benchmaker swebench` recipe (harbor).
#
# harbor owns the per-instance Flash Sandbox environment, the agent run, and the
# verifier. The model endpoint (URL / model / key) is read from `.env`
# (OPENAI_API_BASE_URL, OPENAI_COMPATIBLE_MODEL, OPENAI_API_KEY) and the sandbox
# from FLASH_SANDBOX_URL.
#
# Agent:        --agent pi (default) | pi-host | coding-agent | mini-swe-agent
#               | claude-code | <module:Class>     (run `--list-agents` to see all)
# Concurrency:  --concurrency N      N trials in flight at once.
# Dataset:      --dataset swebench-verified        (harbor resolves the images)
#
# The defaults below are a 5-task smoke run; every flag passes straight through,
# so override or add anything on the command line:
#
#   scripts/run_swebench.sh                                  # 5 tasks, agent=pi
#   scripts/run_swebench.sh --n-tasks 50 --concurrency 16
#   scripts/run_swebench.sh --agent coding-agent
#   scripts/run_swebench.sh --list-agents
benchmaker swebench \
    --agent pi \
    --n-tasks 5 \
    --concurrency 4 \
    --dataset swebench-verified \
    "$@"

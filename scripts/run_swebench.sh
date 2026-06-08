benchmaker swebench \
    --agent pi \
    --agent-kwarg pi_max_turns=150 \
    --n-tasks 100 \
    --concurrency 100 \
    --dataset swebench-verified \
    --jobs-dir "$JOBS_DIR" \
    "$@"

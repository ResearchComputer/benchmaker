# Trajectory replay (prefix-cache parity)

`benchmaker trajectory-replay` expands a multi-turn agent-trajectory dataset into
one chat request per assistant turn — each request sends the growing message
prefix up to that turn, which exercises the server's prefix / radix cache.

```bash
benchmaker trajectory-replay --preset swe-smith \
  --url http://host:8000/v1/chat/completions --model $MODEL \
  --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct \
  --max-trajectories 50 --rate closed:4 --out-dir runs/
```

Source: `--preset swe-smith` (the
[`SWE-bench/SWE-smith-trajectories`](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)
hub dataset, split `tool`; needs `pip install -e .[hf]`) or a local
`--prompts-jsonl` whose rows carry a `messages` field (a list, or the JSON-encoded
string the dataset ships). Messages are sanitized to OpenAI-valid keys
(`role`/`content`/`name`/`tool_calls`/`tool_call_id`); dataset-specific keys (`agent`,
`message_type`) are dropped.

**The parity pair**, per request:
- `meta.expected_prefix_tokens` — theoretical upper bound of cacheable prefix
  (the previous turn's prompt length; needs `--tokenizer`).
- `extra.cached_tokens` — tokens the server actually served from cache.

`cached_tokens / expected_prefix_tokens` is the prefix-cache hit efficiency.

**Use `--rate closed:N`.** Items are emitted in conversation order; closed-loop
keeps same-conversation turns adjacent so the cache warms naturally. The run ends
when the dataset is exhausted.

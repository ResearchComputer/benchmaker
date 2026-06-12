# Trajectory replay (prefix-cache parity)

`benchmaker trajectory-replay` expands a multi-turn agent-trajectory dataset
into one chat request per assistant turn — each request sends the growing
message prefix up to that turn, which exercises the server's prefix / radix
cache.

---

## Quick start

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
string the dataset ships).

## CLI reference (`benchmaker trajectory-replay`)

| Flag                                | Meaning                                                          |
| ----------------------------------- | --------------------------------------------------------------- |
| `--url`                             | Endpoint URL. Falls back to `$OPENAI_API_BASE_URL`/`$OPENAI_BASE_URL`. |
| `--model`                           | Target model. Falls back to `$OPENAI_COMPATIBLE_MODEL`/`$OPENAI_MODEL`. |
| `--api-key`                         | API key. Falls back to `$OPENAI_API_KEY`.                        |
| `-H, --header`                      | Extra header `'Name: value'` — repeatable                       |
| `--dataset`                         | HuggingFace dataset id (needs `datasets`). Mutually exclusive with `--prompts-jsonl`. |
| `--prompts-jsonl`                   | Local JSONL of trajectory rows.                                 |
| `--split`                           | Dataset split (default `tool`).                                 |
| `--preset`                          | Dataset preset: `swe-smith`. Fills in dataset + split defaults.  |
| `--tokenizer`                       | HF tokenizer id; enables `expected_prefix_tokens`.              |
| `--messages-field`                  | Row field carrying the messages list (default `messages`).      |
| `--id-field`                        | Row field for the instance/task id (default `instance_id`).     |
| `--model-field`                     | Row field for the model name label (default `model`).           |
| `--max-tokens`                      | Per-request generation cap (default `1024`).                    |
| `--max-turns-per-trajectory`       | Cap assistant turns replayed per trajectory.                    |
| `--max-trajectories`                | Cap number of trajectories replayed.                            |

Defaults to `--rate closed:8 --duration 24h` (exhaustion ends the run; the
clock is just a ceiling).

## YAML config

```yaml
workload_type:
  type: openai
  url: ${OPENAI_API_BASE_URL}/chat/completions
  model: ${OPENAI_MODEL}
  passthrough_meta: true

workload:
  type: trajectory
  dataset: SWE-bench/SWE-smith-trajectories
  split: tool
  tokenizer: Qwen/Qwen2.5-Coder-7B-Instruct
  max_trajectories: 50
  max_tokens: 1024

load: closed:4
duration: 24h
timeout_s: 600
```

## How it works

### Data flow

```text
Source (HF dataset or local JSONL)
  │  each row has a `messages` field
  ▼
parse_messages()
  │  JSON-string → list, sanitize each msg to OpenAI-valid keys
  ▼
expand_trajectory()
  │  one item per assistant turn:
  │    - prefix = messages[:assistant_turn]
  │    - meta.conversation_id, turn_index, prefix_messages
  │    - meta.expected_prefix_tokens (if tokenizer set)
  ▼
OpenAIChatWorkloadType (passthrough_meta=True)
  │  builds chat-completion request with messages=prefix
  ▼
Server response
  │  extra.cached_tokens = how many tokens the server served from cache
  ▼
Parity pair: meta.expected_prefix_tokens vs extra.cached_tokens
```

### Message sanitization

Messages are sanitized to OpenAI-valid keys (`role`, `content`, `name`,
`tool_calls`, `tool_call_id`). Dataset-specific keys (`agent`,
`message_type`, etc.) are dropped. Missing `content` defaults to `""`.

### The parity pair

Each request records two values:

- `meta.expected_prefix_tokens` — theoretical upper bound of cacheable
  prefix (the previous turn's prompt length; needs `--tokenizer`). This is
  the token count of the *previous* emitted turn's prompt — the nested-prefix
  upper bound of what the server could serve from cache (0 for the first turn).
- `extra.cached_tokens` — tokens the server actually served from cache.

`cached_tokens / expected_prefix_tokens` is the prefix-cache hit efficiency.

### Why closed-loop?

Items are emitted in conversation order; closed-loop (`--rate closed:N`)
keeps same-conversation turns adjacent in the request stream so the cache
warms naturally. Open-loop would interleave conversations and defeat the
prefix cache.

## Presets

| Preset       | Dataset                                 | Split  |
| ------------ | --------------------------------------- | ------ |
| `swe-smith`  | `SWE-bench/SWE-smith-trajectories`     | `tool` |

## Library usage

```python
from benchmaker import (
    BenchConfig, BenchRunner, OpenAIChatWorkloadType, TrajectoryReplayWorkload,
    parse_rate_spec,
)

wt = OpenAIChatWorkloadType(
    url="http://localhost:8000/v1/chat/completions",
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    passthrough_meta=True,
    max_tokens=1024,
)
workload = TrajectoryReplayWorkload(
    dataset="SWE-bench/SWE-smith-trajectories",
    split="tool",
    tokenizer="Qwen/Qwen2.5-Coder-7B-Instruct",
    max_trajectories=50,
    max_tokens=1024,
)
cfg = BenchConfig(
    workload_type=wt,
    workload=workload,
    load=parse_rate_spec("closed:4", duration_s=86400),
    timeout_s=600,
)
result = await BenchRunner(cfg).run()
```

## Custom trajectory sources

Use `--prompts-jsonl` (CLI) or `path=` (library/YAML) to load from a local
JSONL file. Each row must have a `messages` field — either a list of message
dicts or a JSON-encoded string of such a list.

```jsonl
{"instance_id": "task-1", "model": "gpt-4o", "messages": "[{\"role\":\"user\",\"content\":\"...\"},{\"role\":\"assistant\",\"content\":\"...\"}]"}
{"instance_id": "task-2", "model": "gpt-4o", "messages": [{"role":"user","content":"..."}]}
```

Both forms are accepted — `parse_messages` handles both JSON strings and
already-parsed lists.

# SGLang `/generate` benchmark

`benchmaker sglang` drives an SGLang **native** `/generate` endpoint (not the
OpenAI-compatible `/v1/chat/completions` path). Prefer the OpenAI path
(`benchmaker llm`) where possible; use this for raw-text parity checks against
deployments that only expose `/generate`.

---

## Quick start

```bash
benchmaker sglang \
  --url http://host:30000/generate \
  --prompts-jsonl prompts.jsonl --prompt-field text \
  --max-tokens 256 --rate poisson:8 --duration 60s --out-dir runs/
```

URL falls back to `.env` (`SGLANG_API_BASE_URL`, `SGLANG_BASE_URL`).

## CLI reference (`benchmaker sglang`)

| Flag                         | Meaning                                                                |
| ---------------------------- | ---------------------------------------------------------------------- |
| `--url`                      | Endpoint URL. Falls back to `$SGLANG_API_BASE_URL`/`$SGLANG_BASE_URL`. |
| `-H, --header`               | Extra header `'Name: value'` — repeatable                              |
| `--prompt`                   | Prompt text — repeatable (mutually exclusive with `--prompts-jsonl`)   |
| `--prompts-jsonl`            | Path to JSONL of prompts                                               |
| `--prompt-field`             | Field name in JSONL rows (default `text`)                              |
| `--full-jsonl-row / --no-full-jsonl-row` | Yield full JSONL row into samples (default off)          |
| `--shuffle / --no-shuffle`   | Shuffle the static prompt list (default on)                            |
| `--seed`                     | RNG seed for shuffle                                                   |
| `--max-tokens`               | Max completion tokens (default 128)                                    |
| `--temperature`              | Sampling temperature (default 0.0)                                     |
| `--top-p`                    | Nucleus sampling threshold                                             |
| `--top-k`                    | Top-k sampling                                                         |
| `--extra`                    | Pass-through `'key=value'` (value JSON-decoded if possible) — repeatable |

Plus the shared load/output flags (`--rate`, `--duration`, `--out-dir`, etc.).

## YAML config

```yaml
workload_type:
  type: sglang-generate
  url: http://localhost:30000/generate
  max_tokens: 256
  temperature: 0.0
  top_p: 0.9

workload:
  type: jsonl
  path: examples/data/prompts.jsonl
  field: text

load: poisson:8
duration: 60s
timeout_s: 600
```

## Library usage

```python
from benchmaker import (
    BenchConfig, BenchRunner, SGLangGenerateWorkloadType,
    JsonlWorkload, parse_rate_spec,
)

wt = SGLangGenerateWorkloadType.from_env(
    url=None, dotenv_path=".env",
    max_tokens=256, temperature=0.0, top_p=0.9,
)
cfg = BenchConfig(
    workload_type=wt,
    workload=JsonlWorkload("prompts.jsonl", field="text"),
    load=parse_rate_spec("poisson:8", duration_s=60),
    timeout_s=600,
)
result = await BenchRunner(cfg).run()
```

### `from_env()` classmethod

Reads `SGLANG_API_BASE_URL` or `SGLANG_BASE_URL` from the environment
(loaded from `.env`), appending `/generate`:

```python
SGLangGenerateWorkloadType.from_env(url=None, dotenv_path=".env")
```

Explicit `url=` overrides the env var lookup.

## Constructor parameters

```python
SGLangGenerateWorkloadType(
    url="http://host:30000/generate",
    max_tokens=128,
    temperature=0.0,
    extra_body=None,            # dict merged into sampling_params
    headers=None,
    timeout_s=600.0,
    passthrough_meta=False,     # record non-sampling keys into Request.meta
    **sampling,                 # top_p, top_k, min_p, stop, ...
)
```

Any extra keyword argument (`top_p`, `top_k`, `min_p`, `stop`, etc.) is
forwarded into `sampling_params`. `extra_body` is still accepted as an
explicit dict; `**sampling` overrides it on key conflict.

## Item interpretation

| Item type | Behavior                                                       |
| --------- | -------------------------------------------------------------- |
| `None`    | `{"text": ""}`                                                 |
| `str`     | `{"text": item}`                                               |
| `dict`    | `text`/`input_ids` pulled out; remaining sampling keys merged into `sampling_params`. With `passthrough_meta`, non-sampling keys are recorded into `Request.meta` instead of sent. `max_tokens` in a dict item is aliased to `max_new_tokens`. |

### Sampling keys

Recognized in dict items and forwarded into `sampling_params`:

`temperature`, `max_new_tokens`, `top_p`, `top_k`, `min_p`, `stop`,
`stop_token_ids`, `frequency_penalty`, `presence_penalty`,
`repetition_penalty`, `ignore_eos`, `skip_special_tokens`, `n`, `seed`,
`min_new_tokens`, `regex`, `json_schema`, `ebnf`.

### Full-row mode (`--full-jsonl-row`)

When enabled, each full JSONL object is yielded to the workload type with
`passthrough_meta=True`. Sampling keys flow into `sampling_params`; everything
else is recorded into `Request.meta` (and ends up in `samples.jsonl`) instead
of being sent to the server.

## Captured metrics

Per-request metrics (in `Sample.extra`, then aggregated as `workload_metrics`):

- `ttft_s`                  — time to first token
- `itl_ms_mean / itl_ms_p50 / itl_ms_p99` — inter-token latency
- `tokens_out`              — completion tokens (from `meta_info`, else counted)
- `prompt_tokens`           — from `meta_info`
- `cached_tokens`           — from `meta_info`
- `tokens_per_s`            — `tokens_out / (latency - ttft)`

Per-request metadata (in `Sample.meta`):

- `finish_reason`           — from `meta_info`

A 200-status response with zero tokens is marked `ok=False`
(error: `"no tokens received"`).

## SGLang vs OpenAI path

| Feature | `benchmaker llm` (OpenAI) | `benchmaker sglang` (native) |
| ------- | ------------------------- | ----------------------------- |
| Endpoint | `/v1/chat/completions` | `/generate` |
| Body format | `{"messages": [...], "stream": true}` | `{"text": ..., "sampling_params": {...}, "stream": true}` |
| Streaming | SSE `data: {choices: [...]}` | SSE `data: {text, meta_info}` |
| Token counting | `usage` block | `meta_info` |
| Multi-turn | Native (message list) | Manual (concat text) |
| Prefix cache | `cached_tokens` in `usage` | `cached_tokens` in `meta_info` |

Use the OpenAI path when possible — it handles multi-turn conversations
naturally and is compatible with any OpenAI-compatible server. Use the native
SGLang path only for raw-text parity checks or when `/generate` is the only
available endpoint.

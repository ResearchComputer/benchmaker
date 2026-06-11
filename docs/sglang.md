# SGLang `/generate` benchmark

`benchmaker sglang` drives an SGLang **native** `/generate` endpoint (not the
OpenAI-compatible `/v1/chat/completions` path). Prefer the OpenAI path
(`benchmaker llm`) where possible; use this for raw-text parity checks against
deployments that only expose `/generate`.

```bash
benchmaker sglang \
  --url http://host:30000/generate \
  --prompts-jsonl prompts.jsonl --prompt-field text \
  --max-tokens 256 --rate poisson:8 --duration 60s --out-dir runs/
```

Captured per request (`samples.jsonl` -> `extra`): `ttft_s`, `itl_ms_mean/p50/p99`,
`tokens_out`, `prompt_tokens`, **`cached_tokens`**, `tokens_per_s`; `finish_reason`
in `meta`. These come from SGLang's streamed `meta_info`.

Full-row mode (`--full-jsonl-row`) records every other row field into
`samples.jsonl` `meta` instead of sending it. YAML: `workload_type: {type:
sglang-generate, url: ...}`.

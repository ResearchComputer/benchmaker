# Changelog

## 0.1.4 — 2026-06-24

- Fixed silent loss of streamed `usage` (real `prompt_tokens`, `cached_tokens`)
  and truncated eval text when an SSE event was split across HTTP chunk
  boundaries — common on the `mix` path under high concurrency with large
  prompts, which zeroed prefix-cache hit rates. SSE parsing now reassembles
  lines across chunks (shared by the `openai-chat`, `sglang`, and eval paths),
  and the `openai-chat` workload warns once when `include_usage` was requested
  but no usage block was parsed (#13).
- Added `DeepRAGWorkload` and a HotpotQA distractor JSONL preparation tool for
  prefill-heavy retrieval benchmarks.
- Added concurrent named workload lanes with independent load models, durable
  `meta.lane` tags, and per-lane metric summaries.
- Added a phase-swinging DeepRAG + ShareGPT configuration example.

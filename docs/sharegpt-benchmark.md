# End-to-end: benchmarking an LLM endpoint with ShareGPT

A self-contained walkthrough: take a real multi-turn prompt distribution
(ShareGPT), point it at an OpenAI-compatible chat endpoint, run a load test, and
read the latency / throughput / token metrics. No prior familiarity with the
other docs is assumed.

If you just want the concepts behind the moving parts, see
[Concepts](concepts.md); for everything this page glosses over, see
[Workloads](workloads.md), [Load models](load-models.md), and
[Metrics & output](metrics.md).

---

## 1. Install

```bash
pip install -e .          # library + `benchmaker` CLI
pip install -e .[hf]      # adds `datasets`, needed to pull from the Hub
```

## 2. Get the dataset

A cleaned, benchmark-ready ShareGPT V3 copy is published on the Hugging Face Hub
at [`researchcomputer/llmsys-bench`](https://huggingface.co/datasets/researchcomputer/llmsys-bench)
under `split="sharegpt"`. Each row is one conversation:

```json
{"id": "...", "messages": [{"role": "user", "content": "..."},
                           {"role": "assistant", "content": "..."},
                           {"role": "user", "content": "..."}]}
```

`messages` is the only content field. Every row is truncated to end on a
**user** turn, so it is a valid generation request: the server completes the
final assistant reply given the prior history. Short source conversations
collapse to a single user turn (a plain single-turn prompt); longer ones carry
multi-turn context. The `id` is provenance only — it is not sent to the server.

```python
from datasets import load_dataset
ds = load_dataset("researchcomputer/llmsys-bench", split="sharegpt")
# Dataset({features: ['id', 'messages'], num_rows: 58485})
```

### Rebuilding it yourself (optional)

The split is produced by `tools/sharegpt/prepare.py`, which downloads the
upstream JSON (~700 MB) into `.local/` (gitignored) and converts it. Run it for
a subset, different filtering, or a refresh:

```bash
python tools/sharegpt/prepare.py                 # full -> .local/sharegpt_v3.jsonl
python tools/sharegpt/prepare.py --max-items 2000   # quick subset
python tools/sharegpt/prepare.py --min-chars 8 --max-chars 8000
```

`--min-chars` / `--max-chars` filter on total message content per row. To
re-publish after regenerating, `tools/sharegpt/upload_hf.py` pushes the JSONL
back to the Hub (needs a write token).

## 3. Configure the endpoint

`OpenAIChatWorkloadType` talks to any OpenAI-compatible
`/v1/chat/completions` server (vLLM, SGLang, TGI, OpenAI, hosted gateways, …).
Put the connection details in a `.env` file next to where you run from:

```ini
OPENAI_API_BASE_URL=https://your-endpoint/v1/
OPENAI_COMPATIBLE_MODEL=your-org/your-model
OPENAI_API_KEY=sk-...
```

`from_env(...)` reads these (the base URL also accepts `OPENAI_BASE_URL` /
`OPENAI_API_BASE`; the model also accepts `OPENAI_MODEL`). Explicit kwargs
override the env. Never commit `.env` — it is gitignored.

## 4. Run it (Python)

This loads a random 200-row sample, drives the endpoint closed-loop with 8
concurrent in-flight requests, stops after 100 requests, and prints a report.

```python
import asyncio, sys
from datasets import load_dataset
from benchmaker import (
    BenchConfig, BenchRunner, OpenAIChatWorkloadType,
    IterableWorkload, parse_rate_spec,
)

async def main():
    ds = load_dataset("researchcomputer/llmsys-bench", split="sharegpt")
    sample = ds.shuffle(seed=0).select(range(200))

    wt = OpenAIChatWorkloadType.from_env(
        dotenv_path=".env", max_tokens=256, temperature=0.0,
    )
    cfg = BenchConfig(
        workload_type=wt,
        workload=IterableWorkload(row["messages"] for row in sample),
        load=parse_rate_spec("closed:8", duration_s=180, max_requests=100),
        timeout_s=120,
    )
    runner = BenchRunner(cfg)
    await runner.run()
    runner.metrics.render(sys.stdout)
    print("bundle:", runner.write_bundle("./runs"))

asyncio.run(main())
```

Why `IterableWorkload(row["messages"] for row in sample)`: it yields each
conversation's bare message **list**, which `OpenAIChatWorkloadType` uses as
`messages` directly. (Passing the whole row dict would leak `id` into the
request body.) `HFDatasetWorkload` is for `{prompt, reference}`-shaped datasets
and is not the right fit for a messages-list dataset.

### Choosing the load

`parse_rate_spec` accepts:

| Spec               | Meaning                                            |
| ------------------ | -------------------------------------------------- |
| `"50"`             | constant 50 req/s (open loop)                      |
| `"poisson:50"`     | Poisson arrivals, mean 50 req/s (open loop)        |
| `"closed:8"`       | 8 concurrent workers, next request on completion   |
| `"ramp:10..500:30s"` | ramp the rate over 30s                           |
| `"sweep:10,50,100@20s"` | staged rates, 20s each                        |

For a **shared / rate-limited** endpoint, prefer **closed-loop** (`closed:N`):
it adapts to the server's speed and never queues more than `N` requests, so you
won't trip rate limits. Open-loop Poisson is the standard choice when you
control the server and want to characterize behavior under a fixed arrival rate.
Always cap the run with `max_requests=` and/or a `duration_s`.

## 5. Run it (CLI)

The same thing without writing a script, using a local JSONL (the CLI reads a
file, not the Hub — rebuild with `tools/sharegpt/prepare.py` first):

```bash
benchmaker llm \
    --prompts-jsonl .local/sharegpt_v3.jsonl \
    --prompt-field  messages \
    --max-tokens 256 \
    --rate closed:8 --duration 180s --max-requests 100 \
    --out-dir ./runs --label dataset=sharegpt
```

URL / model / API key fall back to `.env`, so they don't need to be passed.
`--prompt-field messages` selects the list per row, exactly like the Python
`IterableWorkload` above.

## 6. Reading the results

A run prints a report and (with `write_bundle` / `--out-dir`) writes a bundle to
`runs/<run-id>/`:

```
runs/20260528T081053Z/
  meta.json       # run id, host, versions, labels, timestamps
  summary.json    # the aggregated metrics below
  samples.jsonl   # one row per request (latency, status, extra metrics)
```

Example report from a real run against a hosted GLM-4.7-Flash endpoint:

```
[benchmaker] results  (100 requests, 51.37s wall)
  throughput     :       1.95 req/s
  success        : 100
  failed         : 0  (0.00%)
  latency (s)
    mean  : 3.9753   p50 : 3.9841   p90 : 4.1303   p99 : 4.1388   max : 4.1437
  status codes
    200  : 100
  workload metrics
    tokens_out     mean 256.0   (capped at --max-tokens)
    prompt_tokens  mean 917.8   p50 815.5   p90 1879.7   max 2752.0
    tokens_per_s   mean 64.53   p50 64.26   p99 77.63
```

What to look at:

- **throughput / goodput** — completed (and successful) requests per second.
- **latency** — wall time per request. With `closed:8` and a fixed
  `max_tokens`, latency is dominated by decoding that many tokens, so the spread
  is tight regardless of prompt length (above: ~3.97s ≈ 256 tokens / 64.5 tok/s).
- **prompt_tokens** — the real ShareGPT length distribution (here ranging from
  10 to 2752 tokens), reported by the server's `usage` block.
- **tokens_per_s** — per-request generation throughput.

For an OpenAI-streaming endpoint that streams `content` token-by-token, the
report also includes **`ttft_s`** (time to first token) and **`itl_ms`**
(inter-token latency p50/p99). See below for when those go missing.

## 7. Reasoning models and TTFT / ITL

`OpenAIChatWorkloadType` measures TTFT and inter-token latency from the
streamed `choices[].delta` field. **Reasoning models** (GLM-4.x, DeepSeek-R1,
Qwen3-thinking, gpt-5 reasoning, …) stream their chain-of-thought under a
separate `delta.reasoning_content` field and leave `delta.content` `null` until
the final answer. Those reasoning tokens are real engine output, so the
workload-type counts them the same as content tokens (#14):

- `ttft_s` — by default (`--ttft-token any`, the default) measured to the
  *first token of any kind* (reasoning or content), the engine-cost signal a
  serving benchmark wants. Pass `--ttft-token content` to measure time to the
  *first visible* token instead (the latency a user perceives).
- `content_ttft_s` — the first-visible-token time, surfaced separately whenever
  it differs from `ttft_s` (i.e. reasoning preceded content), so both signals
  are available in one run regardless of the knob.
- `itl_ms_*` — inter-token latency across the *whole* generation (reasoning
  and content decoded at the same per-token cost), reflecting true decode
  cadence instead of conflating the reasoning phase.
- `tokens_out` — from the server `usage` block when present; otherwise falls
  back to the count of streamed tokens, which now includes reasoning tokens.
- `reasoning_tokens` / `content_tokens` — surfaced from
  `usage.completion_tokens_details` when the server reports the breakdown.

If `max_tokens` is small, a reasoning model may spend the entire budget
*thinking* and emit no answer at all (its `usage` shows
`reasoning_tokens == completion_tokens`); such a sample still carries
latency/throughput metrics but is worth flagging separately when interpreting
results.

## 8. Tips

- **Be a good citizen** on shared endpoints: closed-loop, a `max_requests` cap,
  and a sane `timeout_s`. Smoke-test with one request before a volume run.
- **Vary sampling** by forwarding extra params: any unknown kwarg to
  `OpenAIChatWorkloadType` (or `--extra key=value` on the CLI) is passed straight
  into the request body (`top_p`, `stop`, `min_tokens`, `ignore_eos`, …).
- **Compare runs** with `--label k=v` and `benchmaker collect ./runs --label k`
  to pivot many bundles into a table.
- **Reproducibility**: `ds.shuffle(seed=...)` fixes the sample;
  `temperature=0.0` makes generations deterministic where the server supports it.

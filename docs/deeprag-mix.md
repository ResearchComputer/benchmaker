# DeepRAG and mixed workload lanes

This release adds the two workload features needed to benchmark a
phase-changing prefill/decode fleet: a prefill-heavy retrieval workload and
independently scheduled, tagged dataset lanes.

## Prepare a multi-passage QA corpus

HotpotQA's `distractor` split contains ten supplied context paragraphs per
question. Prepare it once; the result is JSONL with `question`, `answer`, and
`passages`, and can be reused at any retrieval depth.

```bash
python tools/rag/prepare.py --out .local/hotpotqa_distractor.jsonl
```

`DeepRAGWorkload` packs up to `depth` passages into the prompt. Set
`context_tokens_target` to bound the packed passage text with a stable
whitespace-token estimate. Every request records `rag_depth` and
`prompt_tokens_hint`; servers that return OpenAI usage additionally produce
the authoritative `prompt_tokens` metric.

Use `passthrough_meta: true` for this workload. It keeps the reference,
source ID, and prompt-size metadata out of the provider request while retaining
them in `samples.jsonl` and per-lane summaries.

## Run anti-correlated lanes

All lanes share one `workload_type`/endpoint but have independent workloads and
load models. The runner schedules their ticket streams concurrently and stamps
every request and sample with `meta.lane`. `summary.json` contains a top-level
`lanes` map with complete per-lane latency and workload-metric summaries.

### End-to-end quickstart

Set the OpenAI-compatible endpoint and model, then create local copies of the
two source datasets:

```bash
export OPENAI_API_BASE_URL=http://router:8000/v1
export OPENAI_MODEL=Qwen/Qwen3-14B

python tools/rag/prepare.py --out .local/hotpotqa_distractor.jsonl
python tools/sharegpt/prepare.py --out .local/sharegpt_v3.jsonl
```

Run the provided phase-swing configuration and save the result bundle:

```bash
benchmaker run examples/configs/config_deeprag_mix.yaml \
  --out-dir runs --run-id pd-phase-swing
```

Inspect each lane independently from the bundle's `summary.json`:

```python
from benchmaker import read_bundle

summary = read_bundle("runs/pd-phase-swing")["summary"]
for lane, metrics in summary["lanes"].items():
    workload = metrics.get("workload_metrics", {})
    print(
        lane,
        "requests=", metrics["total_requests"],
        "ttft_p99_s=", workload.get("ttft_s", {}).get("p99"),
        "tpot_proxy_ms=", workload.get("itl_ms_mean", {}).get("p50"),
    )
```

`samples.jsonl` preserves the same lane name under `meta.lane`, which is useful
for custom plots or SLO calculations. Compare static and dynamic P/D policies
using the same prepared files, lane rates, depth, and `max_tokens` settings.

```yaml
workload_type:
  type: openai
  url: ${OPENAI_API_BASE_URL}/chat/completions
  model: ${OPENAI_MODEL}
  timeout_s: 600
  passthrough_meta: true

mix:
  lanes:
    - name: deeprag
      workload:
        type: deeprag
        path: .local/hotpotqa_distractor.jsonl
        depth: 10
        context_tokens_target: 12000
        max_tokens: 64
      rate: sweep:2,8,2,8@60s

    - name: sharegpt
      workload:
        type: jsonl
        path: .local/sharegpt_v3.jsonl
        loop: true
      rate: sweep:8,2,8,2@60s

# Optional: evaluates DeepRAG rows. Rows without a reference (for example,
# ShareGPT) remain ungraded rather than being sent a reference-bearing payload.
correctness:
  scorer: {type: contains}
```

The included [example configuration](../examples/configs/config_deeprag_mix.yaml)
uses the same shape. For static baseline comparisons, use the same two datasets
but substitute constant lane rates. Keep the lanes and `max_tokens` settings
identical so the only experimental variable is P/D allocation.

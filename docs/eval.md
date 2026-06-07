# Correctness / accuracy evaluation

benchmaker can grade each response against a reference and surface accuracy
alongside latency in the same run. This stays out of the core: it's a
**workload-type wrapper** + a **post-response hook** + a **scorer function**,
all swappable.

```text
    Workload yields {prompt, reference, ...}
                       │
                       ▼
          EvalWorkloadType (wrapper)
        ┌─────────────────────────────┐
        │  strips reference from item │
        │  delegates to base WT       │
        │  stashes reference on .meta │
        └─────────────────────────────┘
                       │
                       ▼
                Base WorkloadType
              (HTTP, OpenAIChat, ...)
                       │
                       ▼
                  HTTP request
                       │
                       ▼
                    Response
                       │
                       ▼
        correctness_hook (post-response hook)
        ┌─────────────────────────────┐
        │ extract prediction          │
        │ scorer(reference, pred)     │
        │ merge scores → sample.extra │
        │ gate sample.ok on `correct` │
        └─────────────────────────────┘
                       │
                       ▼
          MetricsAggregator summarises
          workload_metrics.correct.*  (mean / p50 / ...)
```

All scores land in `Sample.extra`, which `MetricsAggregator` already
auto-summarises generically (see [Metrics & output](metrics.md)) — no core
changes needed to plot or compare accuracy across runs.

## Quickstart (library)

```python
import asyncio
from benchmaker import (
    BenchConfig, BenchRunner, ConstantRPS,
    EvalWorkloadType, OpenAIChatWorkloadType, JsonlWorkload,
    correctness_hook, exact_match,
)

async def main():
    base = OpenAIChatWorkloadType(
        url="http://localhost:8000/v1/chat/completions",
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=64, temperature=0.0,
    )
    cfg = BenchConfig(
        workload_type=EvalWorkloadType(base),                 # carries `reference`
        workload=JsonlWorkload("eval.jsonl"),                 # {prompt, reference}
        load=ConstantRPS(rps=2, duration_s=120),
        post_hooks=[correctness_hook(exact_match(case_insensitive=True))],
    )
    result = await BenchRunner(cfg).run()
    print("accuracy:", result.summary["workload_metrics"]["correct"]["mean"])

asyncio.run(main())
```

Dataset shape — one record per line:

```json
{"prompt": "What is 2+2?", "reference": "4"}
{"prompt": "Capital of France?", "reference": "Paris"}
```

The `reference` key is stripped from the body sent to the model and copied to
`Request.meta`, where the post-hook reads it. To use a different key (e.g.
`expected`), pass `reference_key="expected"` to both the wrapper and the hook.

## Pieces

### `EvalWorkloadType`

Wraps any base `WorkloadType` to plumb eval-only fields through to post-hooks
without leaking them to the service.

```python
EvalWorkloadType(
    base,                              # WorkloadType to delegate to
    reference_key="reference",         # item key holding the gold answer
    extra_meta_keys=("qid", "split"),  # other item keys to lift onto meta
)
```

It overrides only `make_request` (strips + lifts) and `make_sample`
(delegates), so the base's metrics — TTFT, ITL, tokens/s, etc. — still get
attached to every sample.

**Caveat**: workload-types with a custom `run_ticket` (e.g. sandbox lifecycle)
won't compose cleanly. Use a bespoke post-hook instead, reading the reference
directly from `req.meta` you set in your own pre-hook.

### `correctness_hook`

```python
correctness_hook(
    scorer,                              # (reference, prediction) -> dict[str, float]
    *,
    reference_key="reference",           # meta key (match the wrapper!)
    extractor=extract_text,              # Response -> prediction string
    gate_key="correct",                  # extra key whose <=0 value flips sample.ok=False
    prefix="",                           # prepended to every extra key
    require_reference=True,              # skip grading + flag if reference missing
)
```

Behaviour per request:

1. If the response failed (`ok=False`), the hook does nothing — keep the
   failure visible.
2. If `reference` is missing and `require_reference=True`, records
   `<prefix>missing_reference=1` and stops.
3. Calls `extractor(response)` to get the prediction string. The default
   `extract_text` tries OpenAI chat-completion shape (streaming or not) and
   falls back to UTF-8-decoded raw body.
4. Calls `scorer(reference, prediction)` (sync or async). Each numeric value
   in the returned dict is merged into `Sample.extra` as `<prefix><key>`.
5. If `gate_key` is set and present in the score dict, the sample is marked
   `ok=False` (with `error="failed-<key>"`) when that score is `<= 0`. This
   makes accuracy participate in `goodput_rps`.
6. Stores a truncated copy of the prediction (≤2 KB) on `Sample.meta` for
   offline inspection.

Set `gate_key=None` to record accuracy without affecting success counts.

### Extractors

- `extract_openai_text(response)` — assistant content from
  `/v1/chat/completions` (streaming SSE or single JSON).
- `extract_raw_text(response)` — UTF-8-decode `response.body`.
- `extract_text(response)` — try OpenAI first, fall back to raw. Default.

Custom: any `Response -> str`. Pass via `extractor=...`.

## Stock scorers

A scorer is a callable `(reference, prediction) -> dict[str, float]` (sync or
async). The stock scorers are factory functions that return one.

| Scorer                               | Returns                                                  |
| ------------------------------------ | -------------------------------------------------------- |
| `exact_match(strip, case_insensitive)` | `{correct: 1\|0}` — full-string equality                |
| `contains(strip, case_insensitive)`  | `{correct: 1\|0}` — reference is a substring of prediction |
| `regex_match(pattern, group, case_insensitive)` | `{correct, matched}` — capture group equals reference; if reference is None, matching at all counts |
| `json_valid(required_keys)`          | `{valid_json, correct}` — parses as JSON (and contains required top-level keys) |
| `multiple_choice(choices, case_insensitive)` | `{correct, answered}` — first choice letter found in prediction equals reference |
| `judge_llm(send, template, parse, max_concurrency)` | Async — asks an LLM judge; default parses 0..10 integer, `correct=1` when ≥7 |

Each scorer is independent of which workload-type produced the response.

### LLM as judge

`judge_llm` takes a user-supplied async `send(prompt) -> str` so it doesn't
hold opinions about the HTTP client.

```python
from benchmaker import openai_chat_judge, judge_llm, correctness_hook

send, judge_aclose = openai_chat_judge(
    url="http://judge:8000/v1/chat/completions",
    model="judge-7b",
    api_key="...",
)
hook = correctness_hook(judge_llm(send, max_concurrency=4))

try:
    result = await BenchRunner(cfg_with_hook).run()
finally:
    await judge_aclose()           # close the judge's session
```

Override `template` (format string with `{reference}` / `{prediction}`, or a
callable) and `parse` (judge text → score dict) for non-default rubrics.

## Writing a custom scorer

Anything that returns a dict of numeric scores will do:

```python
import sacrebleu

def bleu_scorer(reference, prediction):
    bleu = sacrebleu.sentence_bleu(prediction, [reference]).score
    return {
        "bleu": bleu,
        "correct": 1.0 if bleu >= 30 else 0.0,
    }

hook = correctness_hook(bleu_scorer, gate_key=None)   # accuracy without gating
```

Async scorers are awaited transparently.

## YAML

Add a top-level `correctness:` block. The loader wraps `workload_type` in
`EvalWorkloadType` and appends the post-hook automatically.

```yaml
workload_type:
  type: openai
  url: http://localhost:8000/v1/chat/completions
  model: meta-llama/Llama-3.1-8B-Instruct
  max_tokens: 64

workload:
  type: jsonl
  path: data/eval.jsonl                  # rows: {prompt, reference, ...}

load: poisson:4
duration: 5m

correctness:
  reference_key: reference               # default; change to "expected" etc.
  extra_meta_keys: [qid, split]          # optional — also lifted to req.meta
  gate_key: correct                      # default; set "null" to disable gating
  prefix: ""                             # default
  require_reference: true                # default

  scorer:
    type: exact_match                    # exact_match | contains | regex |
                                         # json_valid | multiple_choice | judge_llm
    case_insensitive: true               # kwargs forwarded to the scorer
```

Bare-string shorthand when no scorer kwargs are needed:

```yaml
correctness:
  scorer: exact_match
```

Custom scorer via Python factory:

```yaml
correctness:
  scorer:
    factory: my_pkg.scorers:make_bleu_scorer
    threshold: 30                        # kwargs forwarded to the factory
```

The factory must return either a scorer callable or a `(scorer, aclose)` tuple.
The optional `aclose` is awaited when the run finishes.

### LLM-as-judge in YAML

`openai_chat:` shortcut uses the built-in `openai_chat_judge` helper:

```yaml
correctness:
  scorer:
    type: judge_llm
    openai_chat:
      url: http://judge:8000/v1/chat/completions
      model: judge-7b
      api_key: ${JUDGE_API_KEY}
      temperature: 0.0
      max_tokens: 8
    template: |
      Grade the following on a 0..10 scale.
      Reference: {reference}
      Answer:    {prediction}
      Reply with only the integer.
    max_concurrency: 4
    parse_factory: my_pkg.judges:parse_strict   # optional override
```

Or build the send callable yourself:

```yaml
correctness:
  scorer:
    type: judge_llm
    send_factory: my_pkg.judges:make_send       # returns send OR (send, aclose)
    send_kwargs:
      url: http://judge:8000/v1/chat/completions
```

## Inspecting results

After the run, accuracy is in the standard summary:

```python
result.summary["workload_metrics"]["correct"]["mean"]   # accuracy in [0, 1]
result.summary["workload_metrics"]["judge_score"]["p50"]
```

When `gate_key` is enabled, accuracy also drives `goodput_rps` — successes per
second include only requests that both completed AND scored correct.

For per-request inspection, every sample carries:

- `sample.extra["correct"]` (and any other scorer outputs)
- `sample.meta["prediction"]` — model output (truncated to 2048 chars by default)
- `sample.meta["reference"]` — the gold answer (added by the wrapper)
- All the base workload-type metrics (TTFT, ITL, tokens_out, ...)

### Persisting raw outputs to disk

These per-request records (including `meta.prediction` and `meta.reference`)
land in `samples.jsonl` inside the run bundle — **but only when you pass
`--out-dir`**. Without it, nothing hits disk; the summary just renders to
stdout.

```bash
benchmaker run my_eval.yaml --out-dir ./runs --label model=my-model
# →  ./runs/<run-id>/summary.json    (aggregate metrics)
#    ./runs/<run-id>/samples.jsonl   (one JSON object per request)
#    ./runs/<run-id>/meta.json       (run identifiers + resolved config)
```

A line from `samples.jsonl` looks like:

```json
{"start_ts": 12345.6, "latency_s": 1.23, "status": 200, "ok": true,
 "workload": "openai-chat",
 "meta": {"reference": "42", "prediction": "Let me think... #### 42",
          "finish_reason": "stop", "prompt_messages": [...]},
 "extra": {"ttft_s": 0.21, "tokens_out": 84, "correct": 1.0}}
```

The default 2048-char prediction cap keeps bundles small. Override per run:

```yaml
correctness:
  max_prediction_chars: full   # or "none" / "null" — store the full output
  # or an integer to truncate further, e.g. 512
  scorer:
    type: exact_match
```

From the library, pass `max_prediction_chars=None` (or any int) to
`correctness_hook`.

For ad-hoc dumps from a library script, call
`runner.metrics.dump_samples_jsonl(path)` directly after `runner.run()`.

## HuggingFace datasets

[`HFDatasetWorkload`](workloads.md#hfdatasetworkload) loads any
`datasets`-compatible dataset and reshapes rows into `{prompt, reference, ...}`
out of the box. The combo is the shortest path from "I have an eval set" to
"I have a graded run":

```python
from benchmaker import (
    BenchConfig, BenchRunner, EvalWorkloadType, HFDatasetWorkload,
    OpenAIChatWorkloadType, correctness_hook, regex_match, parse_rate_spec,
)

workload = HFDatasetWorkload(preset="gsm8k", split="test", max_items=200)
base = OpenAIChatWorkloadType(url=..., model=..., max_tokens=512)
wt = EvalWorkloadType(base)
hook = correctness_hook(regex_match(r"-?\d+", group=0))   # pull the integer

cfg = BenchConfig(
    workload_type=wt, workload=workload,
    load=parse_rate_spec("4", duration_s=600),
    post_hooks=[hook],
)
result = await BenchRunner(cfg).run()
```

Equivalent YAML:

```yaml
workload_type:
  type: openai
  url: http://localhost:8000/v1/chat/completions
  model: meta-llama/Llama-3.1-8B-Instruct
  max_tokens: 512

workload:
  type: hf
  preset: gsm8k
  split: test
  max_items: 200

load: 4
duration: 10m

correctness:
  scorer:
    type: regex
    pattern: '-?\d+'
    group: 0
```

The `gsm8k` preset already does `reference_transform="gsm8k_answer"` so
references are bare integers — perfect for the regex scorer. For a more
robust grader (last-integer-in-output, whitespace-tolerant), see
[`examples/gsm8k_eval.py`](../examples/gsm8k_eval.py).

## Example

Runnable end-to-end scripts:

- [`examples/llm_eval.py`](../examples/llm_eval.py) — `--scorer exact|contains|regex|judge`
  over any JSONL dataset, with optional run bundle output.
- [`examples/gsm8k_eval.py`](../examples/gsm8k_eval.py) — GSM8K from
  HuggingFace + integer-match scorer.

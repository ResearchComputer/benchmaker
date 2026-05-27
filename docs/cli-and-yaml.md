# CLI & YAML reference

The `bench-maker` CLI is installed by `pip install -e .`. It has four
subcommands: `quick` (one-liner HTTP), `llm` (one-liner OpenAI-compatible
chat), `run` (config file), and `collect` (pivot many run-dirs into a table).

All three benchmark subcommands accept the same set of output flags:

| Flag             | Meaning                                                              |
| ---------------- | -------------------------------------------------------------------- |
| `--out-dir DIR`  | Parent dir for the run bundle. The bundle is written to `DIR/<run-id>/`. |
| `--run-id ID`    | Override the default UTC-timestamp run id.                           |
| `--label k=v`    | Free-form tag stored in `meta.json` (repeatable; surfaceable in `collect`). |
| `--notes TEXT`   | Free-form note stored in `meta.json`.                                |

See [Metrics & output](metrics.md) for the on-disk format.

## `bench-maker quick`

Run a single endpoint without writing a config file.

```bash
bench-maker quick \
    --url https://httpbin.org/post \
    --method POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${TOKEN}" \
    --json-body '{"q": "hello"}' \
    --rate poisson:200 \
    --duration 30s \
    --out-dir ./runs --run-id baseline
```

Options:

| Flag                  | Meaning                                                |
| --------------------- | ------------------------------------------------------ |
| `--url`               | Target URL (required)                                  |
| `--method`            | HTTP method (default `GET`)                            |
| `-H, --header`        | `'Name: value'` — repeatable                           |
| `--json-body`         | JSON body string (mutually exclusive with `--data`)    |
| `--data`              | Raw string body                                        |
| `--rate`              | Rate spec (see [Load models](load-models.md))          |
| `--duration`          | `30s`, `2m`, `1h`, or a bare number (seconds)          |
| `--max-requests`      | Stop after N requests                                  |
| `--timeout`           | Per-request timeout (seconds)                          |
| `--connection-limit`  | Connector cap (default 1000)                           |
| `--out-dir`, `--run-id`, `--label`, `--notes` | Run-bundle output (see top of this page) |
| `--quiet`             | Suppress progress output                               |

`quick` doesn't expose hooks or monitors — use `run` with a config file for that.

## `bench-maker llm`

Drive an OpenAI-compatible chat-completions endpoint (vLLM, SGLang, TGI,
OpenAI itself, ...) without a config file. URL / model / API key fall back to
`.env` (`OPENAI_API_BASE_URL`, `OPENAI_COMPATIBLE_MODEL`, `OPENAI_API_KEY`).

```bash
bench-maker llm \
    --url http://localhost:8000/v1/chat/completions \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt "Explain RDMA in one paragraph." \
    --prompt "Name three trade-offs of paged attention." \
    --max-tokens 256 --min-tokens 64 --ignore-eos \
    --temperature 0.0 --top-p 0.9 \
    --stop "\n\n" \
    --extra repetition_penalty=1.1 \
    --rate poisson:8 --duration 60s \
    --out-dir ./runs --label model=llama-3.1-8b
```

Options:

| Flag                         | Meaning                                                                |
| ---------------------------- | ---------------------------------------------------------------------- |
| `--url`                      | Endpoint URL. Falls back to `$OPENAI_API_BASE_URL`/`$OPENAI_BASE_URL`. |
| `--model`                    | Model name. Falls back to `$OPENAI_COMPATIBLE_MODEL`/`$OPENAI_MODEL`.   |
| `--api-key`                  | API key. Falls back to `$OPENAI_API_KEY`.                              |
| `-H, --header`               | Extra header `'Name: value'` — repeatable                              |
| `--prompt`                   | Prompt text — repeatable (mutually exclusive with `--prompts-jsonl`)   |
| `--prompts-jsonl`            | Path to JSONL of prompts                                               |
| `--prompt-field`             | Field name in JSONL rows (default `prompt`)                            |
| `--shuffle / --no-shuffle`   | Shuffle the static prompt list (default on)                            |
| `--seed`                     | RNG seed for shuffle                                                   |
| `--max-tokens`               | Max completion tokens (default 128)                                    |
| `--min-tokens`               | vLLM/SGLang extension: min tokens before EOS is honored                |
| `--ignore-eos`               | vLLM/SGLang extension: keep generating until `max_tokens`              |
| `--temperature`              | Sampling temperature (default 0.0)                                     |
| `--top-p`                    | Nucleus sampling threshold                                             |
| `--top-k`                    | Top-k sampling (vLLM/SGLang)                                           |
| `--stop`                     | Stop string — repeatable                                               |
| `--extra`                    | Pass-through `'key=value'` (value JSON-decoded if possible) — repeatable |
| `--rate`                     | Rate spec (see [Load models](load-models.md))                          |
| `--duration`                 | `30s`, `2m`, `1h`, or a bare number                                    |
| `--max-requests`             | Stop after N requests                                                  |
| `--timeout`                  | Per-request timeout (default 600s)                                     |
| `--connection-limit`         | Connector cap (default 1000)                                           |
| `--dotenv`                   | Path to `.env` (default `.env`; use `--dotenv ''` to disable)          |
| `--out-dir`, `--run-id`, `--label`, `--notes` | Run-bundle output (see top of this page)              |
| `--quiet`                    | Suppress progress output                                               |

`--extra` is the escape hatch for any sampling param the dedicated flags don't
cover (e.g. `--extra guided_json='{"type":"object"}'`, `--extra frequency_penalty=0.5`).
It maps 1:1 onto the request body.

## `bench-maker run`

```bash
bench-maker run config.yaml \
    --out-dir ./runs --run-id baseline --label variant=v0
```

Options:

| Flag                                          | Meaning                                          |
| --------------------------------------------- | ------------------------------------------------ |
| `--out-dir`, `--run-id`, `--label`, `--notes` | Run-bundle output (see top of this page)         |
| `--dotenv`                                    | Path to `.env` (default `.env`)                  |
| `--quiet`                                     | Suppress progress output                         |

## `bench-maker collect`

Pivot one or more run bundles into a comparison table on stdout:

```bash
bench-maker collect ./runs                              # markdown
bench-maker collect ./runs --format csv > results.csv
bench-maker collect ./runs --label variant \
    --metric workload_metrics.ttft_s.p50 \
    --metric workload_metrics.tokens_per_s.mean \
    --sort-by p99_s
```

Options:

| Flag                          | Meaning                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| `--format md\|csv\|json`      | Output format (default `md`)                                 |
| `--metric DOTTED.PATH`        | Add a dotted-path metric from `summary.json` as a column. Repeatable. |
| `--columns C1,C2,...`         | Restrict to these column names (after `--metric` is applied) |
| `--label KEY`                 | Promote `meta.labels[KEY]` into a column. Repeatable.        |
| `--sort-by COLUMN`            | Sort rows by a column (ascending)                            |
| `--recursive / --no-recursive`| Whether to descend one level into non-bundle dirs (default on) |

See [Metrics & output](metrics.md) for the on-disk bundle layout.

## YAML schema

```yaml
# ---------- workload-type (required) ----------
workload_type:
  type: http | openai | factory      # see below
  # type-specific kwargs:
  url: ...
  method: GET
  headers: {...}
  timeout_s: 60

# ---------- workload (optional; defaults to single None item) ----------
workload:
  type: static | jsonl | callable | factory
  # type-specific kwargs

# ---------- load (required) ----------
load: <rate spec>                    # string or dict; see below
duration: 30s                        # any string accepted by parse_duration
max_requests: 100000                 # optional

# ---------- hooks (optional) ----------
pre_hooks:
  - my_pkg.hooks:sign_request
post_hooks:
  - my_pkg.hooks:parse_response

# ---------- monitors (optional) ----------
monitors:
  - type: prometheus | function | factory
    # type-specific kwargs

# ---------- correctness / accuracy eval (optional) ----------
# Wraps workload_type in EvalWorkloadType and adds a post-hook.
# See docs/eval.md for the full schema.
correctness:
  reference_key: reference
  scorer:
    type: exact_match | contains | regex | json_valid | multiple_choice | judge_llm
    # ...scorer-specific kwargs

# ---------- runner knobs (all optional) ----------
connection_limit: 1000               # aiohttp TCPConnector limit
timeout_s: 60                        # global default per-request timeout
max_in_flight: 10000                 # safety cap on concurrent tasks
progress_every_s: 1.0                # 0 disables progress output
```

### `workload_type` types

- `http` — kwargs map to `HttpWorkloadType(...)`.
- `openai` (or `llm`, `llm-chat`, `openai-chat`) — kwargs map to
  `OpenAIChatWorkloadType(...)`.
- `factory: 'module:fn'` — call `fn(**kwargs)`; must return a `WorkloadType`.

### `workload` types

- `static` — `StaticWorkload(items=[...], shuffle=..., seed=..., max_items=...)`
- `jsonl` — `JsonlWorkload(path=..., field=..., loop=..., max_items=...)`
- `callable` — `CallableWorkload(fn='module:fn')`
- `hf` (or `huggingface`) — `HFDatasetWorkload(...)`. Use `preset:` for common
  eval sets (`gsm8k`, `mmlu`, `humaneval`) or pass `path/name/split/prompt_field/
  reference_field/...` explicitly. Requires `pip install -e .[hf]`. See
  [Workloads & workload-types](workloads.md#hfdatasetworkload).
- `factory: 'module:fn'` — call `fn(**kwargs)`; must return a `Workload`.

You can also write a workload as bare YAML for two shortcuts:

```yaml
workload: data/prompts.jsonl         # string ending in .jsonl -> JsonlWorkload
workload:                            # bare list -> StaticWorkload(items=[...])
  - "first prompt"
  - "second prompt"
```

### `load` spec

Most expressive as a string (see [Load models](load-models.md) for syntax):

```yaml
load: 100rps
load: poisson:200
load: closed:32
load: ramp:10..500:30s
load: ramp-poisson:10..500:30s
load: sweep:10,50,100,500@30s
```

Equivalent dict form, when you want more knobs:

```yaml
load:
  type: poisson
  rps: 200
  seed: 42
```

### `monitors` items

```yaml
monitors:
  - type: prometheus
    name: vllm
    url: http://localhost:8000/metrics
    interval_s: 1.0
    metric_names: [vllm:num_requests_running, vllm:gpu_cache_usage_perc]
    labelled_keys: true
    headers: {Authorization: "Bearer ..."}
    tick_at_start: true
  - type: function
    name: gpu
    fn: my_pkg.monitors:gpu_tick
    interval_s: 0.5
  - factory: my_pkg.monitors:make_my_monitor
    interval_s: 1.0
```

## Hook / function references

Anywhere YAML accepts `module:function`, the resolver does:

1. Import `module` (must be on `sys.path`).
2. Walk attribute access on the result for everything after `:`.
3. Verify the result is callable.

Both `"my_pkg.subpkg.module:fn"` and `"my_pkg.subpkg.module.fn"` work; the
`:` form is recommended for unambiguity.

## Example: full LLM benchmark with monitor

```yaml
workload_type:
  type: openai
  url: http://localhost:8000/v1/chat/completions
  model: meta-llama/Llama-3.1-8B-Instruct
  max_tokens: 256
  temperature: 0.0
  # Any extra kwarg is forwarded to the request body — useful for
  # vLLM/SGLang extensions:
  min_tokens: 64
  ignore_eos: true
  top_p: 0.9
  stop: ["\n\n"]

workload:
  type: jsonl
  path: data/sharegpt_subset.jsonl
  field: prompt

load: poisson:8
duration: 5m
timeout_s: 600

monitors:
  - type: prometheus
    name: vllm
    url: http://localhost:8000/metrics
    interval_s: 1.0
    metric_names:
      - vllm:num_requests_running
      - vllm:num_requests_waiting
      - vllm:gpu_cache_usage_perc
```

Then:

```bash
bench-maker run llm_bench.yaml \
    --out-dir ./runs --run-id llama-baseline --label model=llama-3.1-8b
```

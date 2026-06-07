# CLI & YAML reference

The `benchmaker` CLI is installed by `pip install -e .`. Its subcommands fall
into two groups:

- **Recipes** — named, self-contained benchmark scenarios you run as
  `benchmaker <recipe> --args`: `http` (one-off HTTP), `llm` (OpenAI-compatible
  chat), `sandbox` (Flash Sandbox), and `swebench` (SWE-bench Verified eval).
- **Infra commands** — `run` (drive any benchmark from a YAML config file) and
  `collect` (pivot many run-dirs into a table).

> `quick` is a deprecated alias for `http`; it still works but prints a warning.

Every recipe shares the same **load**, **timeout**, and **output** flags (added
by the CLI on top of each recipe's own options):

| Flag                  | Meaning                                                       |
| --------------------- | ------------------------------------------------------------ |
| `--rate`              | Load spec (see [Load models](load-models.md)). Default `10`. |
| `--duration`         | `30s`, `2m`, `1h`, or a bare number (seconds). Default `10s`. |
| `--max-requests`      | Stop after N requests                                        |
| `--timeout`           | Per-request timeout in seconds (default `600`)              |
| `--connection-limit`  | Connector cap (default `1000`)                              |
| `--dotenv`            | Path to `.env` (default `.env`; use `--dotenv ''` to disable) |
| `--out-dir DIR`       | Parent dir for the run bundle (`DIR/<run-id>/`)             |
| `--run-id ID`         | Override the default UTC-timestamp run id                   |
| `--label k=v`         | Free-form tag stored in `meta.json` (repeatable)           |
| `--notes TEXT`        | Free-form note stored in `meta.json`                       |
| `--quiet`             | Suppress progress output                                    |

Some recipes override the load/timeout *defaults* (e.g. `swebench` defaults to
`--rate closed:16 --timeout 7200`); you can still pass the flags to override.
Recipes don't expose hooks or monitors — use `run` with a config file for that.
See [Metrics & output](metrics.md) for the on-disk bundle format.

## `benchmaker http`

Benchmark a single HTTP endpoint without writing a config file.

```bash
benchmaker http \
    --url https://httpbin.org/post \
    --method POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${TOKEN}" \
    --json-body '{"q": "hello"}' \
    --rate poisson:200 \
    --duration 30s \
    --out-dir ./runs --run-id baseline
```

Recipe options (plus the shared flags above):

| Flag                  | Meaning                                                |
| --------------------- | ------------------------------------------------------ |
| `--url`               | Target URL (required)                                  |
| `--method`            | HTTP method (default `GET`)                            |
| `-H, --header`        | `'Name: value'` — repeatable                           |
| `--json-body`         | JSON body string (mutually exclusive with `--data`)    |
| `--data`              | Raw string body                                        |

## `benchmaker llm`

Drive an OpenAI-compatible chat-completions endpoint (vLLM, SGLang, TGI,
OpenAI itself, ...) without a config file. URL / model / API key fall back to
`.env` (`OPENAI_API_BASE_URL`, `OPENAI_COMPATIBLE_MODEL`, `OPENAI_API_KEY`).

```bash
benchmaker llm \
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

Recipe options (plus the shared flags above):

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

`--extra` is the escape hatch for any sampling param the dedicated flags don't
cover (e.g. `--extra guided_json='{"type":"object"}'`, `--extra frequency_penalty=0.5`).
It maps 1:1 onto the request body.

## `benchmaker sandbox`

Benchmark a [Flash Sandbox](workloads.md) endpoint. `--operation exec` (default)
reuses one sandbox per run; `create` makes a fresh pod per request; `lifecycle`
times the full create → exec → delete sequence per request.

```bash
benchmaker sandbox \
    --base-url http://localhost:8080 \
    --operation lifecycle \
    --image alpine:3.20 \
    --command "echo hello" \
    --command "uname -a" \
    --ttl-seconds 600 \
    --rate closed:8 --duration 30s
```

Recipe options (plus the shared flags above):

| Flag                          | Meaning                                                          |
| ----------------------------- | --------------------------------------------------------------- |
| `--base-url`                  | Flash Sandbox base URL (required)                               |
| `--operation`                 | `exec` (default), `create`, or `lifecycle`                      |
| `-c, --command`               | Command to run — repeatable (one workload item each)            |
| `--image`                     | Image for the create spec (e.g. `alpine:3.20`)                  |
| `--spec-json`                 | JSON object merged into the create spec (overrides `--image`)   |
| `--endpoint-prefix`           | `/sandboxes` (cluster, default) or `/native/sandboxes` (node)   |
| `--ttl-seconds`               | Server-side reap TTL for created sandboxes                      |
| `--persistent / --no-persistent` | Use `/pshell` so `cd`/`export` persist across exec calls     |
| `--sandbox-id`                | Target an existing sandbox (exec only)                          |
| `-H, --header`                | Extra header `'Name: value'` — repeatable                       |

## `benchmaker swebench`

Evaluate a coding agent on SWE-bench through **harbor**. Unlike the other
recipes, this one is *self-driving*: harbor owns the per-instance Flash Sandbox
environment, the agent run, and the verifier, so it does **not** flow through
`BenchRunner` and produces **no run-bundle** — it prints harbor's accuracy
summary + job dir instead (so the shared load/timeout/`--out-dir`/`collect`
flags don't apply; only `--dotenv` is shared). Requires the `harbor` package.

Model URL/model/key fall back to `.env` (`OPENAI_API_BASE_URL`,
`OPENAI_COMPATIBLE_MODEL`, `OPENAI_API_KEY`); the sandbox comes from
`$FLASH_SANDBOX_URL`. Harbor resolves the per-instance images from its
registered dataset (`--dataset`), so there are no image-registry flags here.

```bash
# A small slice with the default pi agent:
benchmaker swebench --n-tasks 5 --concurrency 4

# Our own CodingAgent loop, evaluated by harbor:
benchmaker swebench --agent coding-agent --n-tasks 5

# List the agent registry and exit:
benchmaker swebench --list-agents
```

Recipe options (plus the shared `--dotenv`):

| Flag                          | Meaning                                                          |
| ----------------------------- | --------------------------------------------------------------- |
| `--agent`                     | Registry key (`pi` default, `pi-host`, `coding-agent`, `mini-swe-agent`, `claude-code`), a bare harbor built-in, or `module:Class` |
| `--dataset`                   | Harbor dataset slug (default `swebench-verified`)              |
| `--model` / `--api-base` / `--api-key` | Model endpoint (fall back to the `OPENAI_*` env vars)  |
| `--n-tasks`                   | Cap the number of dataset tasks                                |
| `--task`                      | Restrict to specific task name(s)/glob(s) — repeatable        |
| `--concurrency`               | Concurrent trials, harbor `n_concurrent_trials` (default `4`)  |
| `--n-attempts`                | Attempts per task (default `1`)                                |
| `--timeout-multiplier`        | Multiplier on harbor timeouts; cold-start needs 4–6× (default `4.0`) |
| `--force-build`               | Force-rebuild the environment image                            |
| `--backend-type`              | Flash Sandbox backend (`docker` default, `kubernetes`)         |
| `--request-timeout-sec`       | Per-request timeout (default `120`)                            |
| `--agent-ready-timeout-sec`   | Wait for the in-sandbox agent to come up (default `600`)       |
| `--agent-kwarg`               | Extra agent kwarg `key=value` — repeatable                     |
| `--agent-config-file`         | YAML forwarded to the agent's `config_file` kwarg             |
| `--job-name`                  | Harbor job name (defaults to a timestamp)                      |
| `--list-agents`               | List the registry agent keys and exit                         |

## `benchmaker run`

```bash
benchmaker run config.yaml \
    --out-dir ./runs --run-id baseline --label variant=v0
```

Options:

| Flag                                          | Meaning                                          |
| --------------------------------------------- | ------------------------------------------------ |
| `--out-dir`, `--run-id`, `--label`, `--notes` | Run-bundle output (see top of this page)         |
| `--dotenv`                                    | Path to `.env` (default `.env`)                  |
| `--quiet`                                     | Suppress progress output                         |

## `benchmaker collect`

Pivot one or more run bundles into a comparison table on stdout:

```bash
benchmaker collect ./runs                              # markdown
benchmaker collect ./runs --format csv > results.csv
benchmaker collect ./runs --label variant \
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
benchmaker run llm_bench.yaml \
    --out-dir ./runs --run-id llama-baseline --label model=llama-3.1-8b
```

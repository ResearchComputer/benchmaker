# Metrics & output

The runner produces a `BenchResult` with two members:

- `result.samples` — every per-request `Sample` (in order of completion)
- `result.summary` — aggregated dict (described below)

`runner.metrics` is the live `MetricsAggregator`; it can be inspected directly
or rendered to a terminal with `runner.metrics.render(sys.stdout)`.

To persist a run to disk in a format that downstream scripts can read back,
use the **run bundle** layout:

```python
runner.write_bundle("./runs", run_id="baseline", labels={"variant": "v0"})
```

or from the CLI:

```bash
bench-maker run cfg.yaml --out-dir ./runs --run-id baseline --label variant=v0
```

## Run bundle layout

Each run produces one directory:

```
runs/<run_id>/
    meta.json       # run identifiers, timestamps, resolved source config
    summary.json    # aggregated metrics (the dict described below)
    samples.jsonl   # one JSON record per request
    monitors.jsonl  # one JSON record per monitor tick (only if monitors ran)
```

`run_id` defaults to a UTC timestamp like `20260526T142233Z`. Pass `--run-id`
to set it explicitly. The directory is the unit of result interchange — copy
it, tar it, share it.

### `meta.json`

```json
{
  "run_id": "baseline",
  "bundle_version": 1,
  "bench_maker_version": "0.1.0",
  "started_at": "2026-05-26T14:22:33.012345+00:00",
  "ended_at":   "2026-05-26T14:22:43.512345+00:00",
  "wall_time_s": 10.5,
  "hostname": "...",
  "python_version": "3.12.2",
  "workload_type": "http",
  "workload": "static",
  "labels": {"variant": "v0"},
  "notes": "",
  "source_config": { ... raw config (YAML dict / CLI args) ... }
}
```

`labels` is a free-form dict for whatever you want to group/filter on
downstream (model name, hardware tier, commit SHA…). Pass `--label k=v`
repeatedly on the CLI; pass `labels={...}` from Python.

### `summary.json`

```python
{
    "wall_time_s": 30.0,
    "total_requests": 6000,
    "success": 5994,
    "failed": 6,
    "error_rate": 0.001,
    "throughput_rps": 200.0,    # all responses / wall
    "goodput_rps": 199.8,       # successes only / wall
    "bytes_sent": 1234567,
    "bytes_recv": 9876543,
    "status_codes": {"200": 5994, "429": 4, "500": 2},
    "errors": {"timeout": 4, "ClientConnectorError: ...": 2},

    # On any successful request:
    "latency_s": {
        "mean": 0.045, "min": 0.012, "max": 1.230,
        "p50": 0.038, "p90": 0.080, "p95": 0.110, "p99": 0.250, "p999": 0.900
    },

    # Per-workload-type extras (when present) — e.g. for OpenAIChatWorkloadType:
    "workload_metrics": {
        "ttft_s":       {"mean": 0.18, "p50": 0.16, "p90": 0.32, "p99": 0.55, "min": 0.12, "max": 0.71},
        "itl_ms_mean":  { ... },
        "tokens_out":   { ... },
        "tokens_per_s": { ... }
    },

    # Per-monitor time-series summaries (when configured):
    "monitors": {
        "vllm": {
            "tick_count": 30,
            "metrics": {
                "vllm:num_requests_running": {
                    "mean": 3.4, "min": 0, "max": 7,
                    "p50": 3, "p90": 6, "p99": 7,
                    "first": 0, "last": 4
                }
            }
        }
    }
}
```

### `samples.jsonl`

One JSON object per line:

```json
{
  "start_ts": 12345.6,
  "latency_s": 0.043,
  "status": 200,
  "ok": true,
  "bytes_sent": 256,
  "bytes_recv": 1024,
  "error": null,
  "workload": "openai-chat",
  "meta": {"prompt_messages": [...], "max_tokens": 128, "finish_reason": "stop"},
  "extra": {"ttft_s": 0.17, "itl_ms_mean": 25.3, "tokens_out": 64.0, "tokens_per_s": 32.1}
}
```

### `monitors.jsonl`

One JSON object per tick:

```json
{"monitor": "vllm", "elapsed_s": 1.02, "values": {"vllm:num_requests_running": 3.0}}
{"monitor": "vllm", "elapsed_s": 2.04, "values": {"vllm:num_requests_running": 4.0}}
```

`elapsed_s` shares its time origin with `Sample.start_ts`, so the two files
can be joined on the time axis.

## What counts as "success"

A `Sample.ok` is `True` when:

- the HTTP transport succeeded (no timeout / connection error), **AND**
- the response status is `2xx` or `3xx`, **AND**
- the workload-type did not flag it as failed in `make_sample`

For `OpenAIChatWorkloadType`, a 200-status response with zero output tokens is
demoted to `ok=False` (error: `"no tokens received"`), so a server that
returns empty completions is correctly recorded as failing the workload even
though the HTTP layer succeeded.

A post-hook can also flip `sample.ok` to `False` after the fact.

## Collecting many runs into a table

```bash
bench-maker collect ./runs                     # markdown table to stdout
bench-maker collect ./runs --format csv > results.csv
bench-maker collect ./runs/a ./runs/b ./runs/c # explicit run-dirs also work
```

Default columns: `run_id, workload_type, workload, wall_s, total, ok, fail,
err_rate, rps, good_rps, p50_s, p90_s, p99_s, max_s`.

Add columns:

```bash
# Promote a label to a column.
bench-maker collect ./runs --label variant

# Add a workload-specific or monitor metric (dotted path inside summary.json).
bench-maker collect ./runs \
    --metric workload_metrics.ttft_s.p50 \
    --metric workload_metrics.tokens_per_s.mean \
    --metric monitors.vllm.metrics.vllm:num_requests_running.mean

# Restrict the column set (after extras are added) and sort.
bench-maker collect ./runs \
    --label variant \
    --columns run_id,label.variant,rps,p99_s \
    --sort-by rps
```

For richer analysis (per-stage breakdowns under `Sweep`, alignment across
multiple processes, plotting), iterate the JSONL files directly:

```python
from benchmaker import read_bundle, iter_jsonl

bundle = read_bundle("./runs/baseline")
for row in iter_jsonl(bundle["samples_path"]):
    ...
```

## Rendering to a terminal

```python
runner.metrics.render(sys.stdout)
```

Plain-text output, no dependencies. Sections appear in this order:
throughput → latency → status codes → errors → workload metrics → monitors.

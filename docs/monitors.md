# Monitors

`Monitor`s run *alongside* the benchmark, sampling something external at a
fixed interval. They are entirely optional. Each tick returns a flat
`{metric_name: float}` dict that the runner records as a time-series; the
aggregator summarizes each metric (mean / min / max / p50 / p90 / p99 / first /
last) in the final report.

Use cases:

- scrape vLLM / SGLang / TGI `/metrics` (Prometheus text format)
- read GPU utilization, memory, or power via NVML
- track Slurm or Kubernetes queue depth
- pull custom telemetry from your service

## Lifecycle

For each `Monitor` in `BenchConfig.monitors`:

1. `await monitor.setup()` — once, before the first tick
2. one immediate `tick()` if `tick_at_start=True` (default)
3. `tick()` every `interval_s` seconds
4. one final `tick()` when the benchmark ends
5. `await monitor.aclose()` — once, for cleanup

Each tick failure is logged to stderr but never crashes the benchmark.

## Built-in: `PrometheusMonitor`

Scrape any Prometheus-format `/metrics` endpoint.

```python
from benchmaker import PrometheusMonitor

PrometheusMonitor(
    url="http://localhost:8000/metrics",
    interval_s=1.0,
    metric_names={                   # optional filter; None = keep everything
        "vllm:num_requests_running",
        "vllm:gpu_cache_usage_perc",
        "vllm:time_to_first_token_seconds_sum",
    },
    labelled_keys=True,              # default: preserve labels in metric names
    headers={"Authorization": "Bearer ..."},  # optional
    name="vllm",
)
```

- `metric_names` is matched against the *bare* metric name (before `{labels}`).
- With `labelled_keys=True` (default), the series `vllm:gpu_util{gpu="0"} 0.8`
  is recorded under the key `vllm:gpu_util{gpu="0"}`. Different label sets
  produce different keys.
- With `labelled_keys=False`, label info is dropped and series with identical
  bare names are summed — handy when you only care about totals across shards.

Per-tick HTTP failures (timeout, 5xx, etc.) are silently skipped — that tick
just records no sample.

## Built-in: `FunctionMonitor`

Wrap any sync or async callable returning a `dict[str, float]` (or `None`).

```python
from benchmaker import FunctionMonitor

def gpu_tick():
    import pynvml
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    u = pynvml.nvmlDeviceGetUtilizationRates(h)
    m = pynvml.nvmlDeviceGetMemoryInfo(h)
    return {"gpu0_util": float(u.gpu), "gpu0_mem_gb": m.used / 1e9}

mon = FunctionMonitor(fn=gpu_tick, interval_s=0.5, name="nvml")
```

The callable is called with no arguments. If it needs persistent state
(NVML handles, an HTTP session, …), use a closure or a custom subclass.

## Custom Monitor subclass

When you need `setup` / `aclose`:

```python
from benchmaker import Monitor

class NvmlMonitor(Monitor):
    name = "nvml"
    interval_s = 0.5

    async def setup(self):
        import pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(0)

    async def tick(self):
        import pynvml
        u = pynvml.nvmlDeviceGetUtilizationRates(self._h)
        return {"gpu_util": float(u.gpu)}

    async def aclose(self):
        import pynvml
        pynvml.nvmlShutdown()
```

## Wiring

```python
BenchConfig(
    workload_type=..., workload=..., load=...,
    monitors=[vllm_mon, gpu_mon],
)
```

YAML form:

```yaml
monitors:
  - type: prometheus
    name: vllm
    url: http://localhost:8000/metrics
    interval_s: 1.0
    metric_names:
      - vllm:num_requests_running
      - vllm:gpu_cache_usage_perc
  - type: function
    name: nvml
    fn: my_pkg.monitors:gpu_tick
    interval_s: 0.5
```

For arbitrary subclasses, use the `factory` form:

```yaml
monitors:
  - factory: my_pkg.monitors:make_nvml_monitor
    interval_s: 0.5
```

## Output

`result.summary["monitors"][<name>]` contains:

```python
{
    "tick_count": 60,
    "metrics": {
        "vllm:num_requests_running": {
            "mean": 3.4, "min": 0, "max": 7,
            "p50": 3, "p90": 6, "p99": 7,
            "first": 0, "last": 4,
        },
        # ...
    },
}
```

Per-tick raw time-series are written automatically as part of the run bundle
(`monitors.jsonl`) whenever monitors are configured — see
[Metrics & output](metrics.md):

```python
runner.write_bundle("./runs", run_id="baseline")
# ./runs/baseline/monitors.jsonl
# {"monitor": "vllm", "elapsed_s": 1.02, "values": {"vllm:num_requests_running": 3, ...}}
```

This format is convenient for plotting alongside per-request samples: align
on `elapsed_s` against the JSONL of request samples to overlay
client-perceived latency with server-side queue depth.

## A complete vLLM example

See [`examples/vllm_with_monitor.py`](../examples/vllm_with_monitor.py) for a
runnable script that:

1. Drives a vLLM server with a small prompt set
2. Concurrently scrapes its `/metrics` endpoint
3. Reports client-side TTFT/ITL/tokens-per-sec *and* server-side queue depth
   and KV-cache utilization in one summary

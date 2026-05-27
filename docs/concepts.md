# Concepts

A benchmark is the composition of four pieces. Each is independently
swappable.

## 1. WorkloadType — the protocol

A `WorkloadType` knows how to talk to a *kind* of service. It owns:

- `make_request(item) -> Request` — build the outgoing request from a dataset item
- `streaming: bool`               — whether the runner should read the response chunk-by-chunk
- `make_sample(item, req, resp, start_ts) -> Sample` — turn the response into a measurable

Built-in workload-types:

- `HttpWorkloadType`         — generic HTTP
- `OpenAIChatWorkloadType`   — OpenAI-compatible `/v1/chat/completions` with SSE streaming and TTFT/ITL/tokens-per-sec capture

Write your own by subclassing `WorkloadType`. See [Workloads & workload-types](workloads.md#custom-workload-type).

## 2. Workload — the dataset

A `Workload` is a source of per-request items. Each item is opaque to the
runner — only the `WorkloadType` knows how to interpret it.

Built-in workloads:

- `StaticWorkload(items, shuffle, seed, max_items)` — cycle a fixed list
- `JsonlWorkload(path, field, loop, max_items)` — stream from a JSONL file (with optional field projection)
- `CallableWorkload(fn)` — sync/async callable that returns the next item
- `IterableWorkload(iterable, loop)` — wrap any (async) iterable

When omitted, the runner uses an empty `StaticWorkload` (one `None` item,
cycled forever) — fine for fixed-request benchmarks.

## 3. LoadModel — the schedule

Decides *when* the next request should fire. Two families:

- **Open-loop**: arrivals are scheduled on wall-clock time, regardless of
  in-flight count. Reveals saturation and queueing behavior. Use
  `ConstantRPS`, `PoissonRPS`, `Ramp`, or `Sweep` (when composing constant
  stages).
- **Closed-loop**: N concurrent workers, each firing the next request as soon
  as the previous completes. Measures latency at a fixed concurrency. Use
  `ClosedLoop(concurrency=N)`.

User-friendly string syntax for configs:

| Spec                          | Meaning                                   |
| ----------------------------- | ----------------------------------------- |
| `100` / `100rps`              | constant 100 rps (open-loop)              |
| `poisson:100`                 | Poisson arrivals, mean 100 rps            |
| `closed:32` / `concurrency:32`| closed-loop, 32 concurrent workers        |
| `ramp:10..500:30s`            | linear ramp 10 → 500 rps over 30s         |
| `ramp-poisson:10..500:30s`    | Poisson arrivals at the ramped rate       |
| `sweep:10,50,100,500@20s`     | sweep through stages, 20s each            |

See [Load models](load-models.md) for when to pick which.

## 4. Monitor — periodic side-channel sampling

Optional. A `Monitor` runs alongside the benchmark, sampling something
external every `interval_s` (vLLM `/metrics`, GPU utilization, queue depth,
…). Each tick is recorded as a time-series and summarized in the final
report.

Built-in: `FunctionMonitor` (wrap any callable), `PrometheusMonitor` (scrape
a Prom endpoint). See [Monitors](monitors.md).

## Hooks — request and response interceptors

Pre/post-processing callables let you mutate requests (sign, add auth) or
extract metrics from responses (parse JSON, count tokens). They run on every
request. See [Hooks](hooks.md).

## Putting it together

```python
from benchmaker import (
    BenchConfig, BenchRunner,
    OpenAIChatWorkloadType, JsonlWorkload,
    PoissonRPS, PrometheusMonitor,
)

cfg = BenchConfig(
    workload_type=OpenAIChatWorkloadType(url="...", model="..."),
    workload=JsonlWorkload(path="prompts.jsonl", field="prompt"),
    load=PoissonRPS(rps=8, duration_s=300),
    monitors=[PrometheusMonitor(url="http://localhost:8000/metrics",
                                interval_s=1.0)],
)
result = await BenchRunner(cfg).run()
```

Result has both client-side request metrics and the monitor time-series.

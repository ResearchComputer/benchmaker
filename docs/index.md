# benchmaker docs

Async HTTP benchmarking with pluggable workload-types, datasets, load models,
hooks, and optional periodic monitors.

A benchmark in benchmaker is the composition of four things:

```text
+--------+   item   +---------------+   request   +-----------+   +---------+
|workload|--------->| workload-type |------------>| pre-hooks |-->| aiohttp |
|(dataset|          | (protocol)    |             +-----------+   +---------+
| / log) |          | make_request  |                                 |
+--------+          | make_sample   |              +------------+     v
   ^                +---------------+              | post-hooks |<----+
   |                                               +------------+
   +-- load model decides WHEN to fire ----+              v
                                           |        +----------+
              monitors run alongside ------+------->| metrics  |
              (Prometheus, NVML, ...)               | aggregator|
                                                    +----------+
```

| Concept       | What it is                                       | Examples                                          |
| ------------- | ------------------------------------------------ | ------------------------------------------------- |
| WorkloadType  | The *protocol* — how to talk to the service       | `HttpWorkloadType`, `OpenAIChatWorkloadType`      |
| Workload      | The *dataset* — what to send                      | `StaticWorkload`, `JsonlWorkload`, `CallableWorkload` |
| LoadModel     | When to fire requests                             | `ConstantRPS`, `PoissonRPS`, `ClosedLoop`, `Ramp`, `Sweep` |
| Monitor       | Optional periodic side-channel sampler            | `PrometheusMonitor`, `FunctionMonitor`            |

## Reading order

1. [Quickstart](quickstart.md) — install, first benchmark, first CLI run.
2. [Concepts](concepts.md) — the four-piece mental model in detail.
3. [Load models](load-models.md) — rate-spec syntax, open vs closed loop.
4. [Workloads & workload-types](workloads.md) — built-ins, custom subclasses,
   item interpretation rules.
5. [Hooks](hooks.md) — pre/post-processing for custom auth, parsing, metrics.
6. [Monitors](monitors.md) — scraping vLLM `/metrics`, GPU telemetry,
   custom side-channels.
7. [Metrics & output](metrics.md) — what gets emitted; JSON/JSONL dumps.
8. [Correctness / accuracy eval](eval.md) — grade responses against references
   alongside latency: `EvalWorkloadType`, `correctness_hook`, stock scorers,
   LLM-as-judge.
9. [CLI & YAML reference](cli-and-yaml.md) — `bench-maker quick`,
   `bench-maker run`, full YAML schema.

## Where things live

```
benchmaker/        # library code
entrypoints/       # CLI (bench-maker)
examples/          # runnable examples
tests/             # pytest smoke tests
docs/              # this directory
```

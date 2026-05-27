# bench-maker

Async HTTP benchmarking with pluggable workload-types (protocols), workloads
(datasets), load models, hooks, and optional periodic monitors.

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

## Install

```bash
pip install -e .
pip install -e .[dev]   # for tests
```

This installs the `benchmaker` Python package and the `bench-maker` CLI.

## 30-second tour

```python
import asyncio
from benchmaker import BenchConfig, BenchRunner, ConstantRPS, HttpWorkloadType

async def main():
    cfg = BenchConfig(
        workload_type=HttpWorkloadType(url="https://httpbin.org/get"),
        load=ConstantRPS(rps=50, duration_s=10),
    )
    result = await BenchRunner(cfg).run()
    print(result.summary)

asyncio.run(main())
```

Or via the CLI:

```bash
bench-maker quick --url https://httpbin.org/get --rate poisson:50 --duration 10s
```

## Documentation

Full docs live in [`docs/`](docs/):

- [Quickstart](docs/quickstart.md)
- [Concepts](docs/concepts.md) — WorkloadType, Workload, LoadModel, Monitor
- [Load models](docs/load-models.md) — rate-spec syntax, open vs closed loop
- [Workloads & workload-types](docs/workloads.md) — built-ins and custom subclasses
- [Hooks](docs/hooks.md) — pre/post request processing
- [Monitors](docs/monitors.md) — vLLM `/metrics`, GPU telemetry, custom samplers
- [Metrics & output](docs/metrics.md) — summary structure, JSONL dumps
- [Correctness / accuracy eval](docs/eval.md) — grade responses against references
- [CLI & YAML reference](docs/cli-and-yaml.md)

## Examples

Under [`examples/`](examples/):

- `simple_get.py`         — minimal library usage
- `custom_hooks.py`       — request signing + response parsing
- `llm_chat.py`           — OpenAI-compatible LLM endpoint with streaming
- `vllm_with_monitor.py`  — LLM benchmark with concurrent vLLM `/metrics` scrape
- `sandbox_exec.py`       — Flash Sandbox `/exec` latency benchmark
- `sandbox_lifecycle.py`  — full create → exec → delete cold-start benchmark
- `llm_eval.py`           — LLM benchmark + accuracy grading (exact/regex/judge)
- `gsm8k_eval.py`         — GSM8K from HuggingFace + integer-match scorer
- `config.yaml`           — generic HTTP YAML config
- `config_llm.yaml`       — LLM YAML config with a Prometheus monitor

## Project layout

```
benchmaker/          # library code
entrypoints/         # CLI (bench-maker)
examples/            # runnable examples
tests/               # pytest smoke tests
docs/                # reference docs
```

## Run the tests

```bash
pytest -q
```

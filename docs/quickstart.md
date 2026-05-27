# Quickstart

## Install

```bash
pip install -e .
pip install -e .[dev]   # for tests
```

This installs the `benchmaker` Python package and the `bench-maker` CLI.

## Smallest possible benchmark

```python
import asyncio
from benchmaker import BenchConfig, BenchRunner, ConstantRPS, HttpWorkloadType

async def main():
    runner = BenchRunner(BenchConfig(
        workload_type=HttpWorkloadType(url="https://httpbin.org/get"),
        load=ConstantRPS(rps=50, duration_s=10),
    ))
    result = await runner.run()
    print(result.summary)

asyncio.run(main())
```

The `workload` (dataset) is optional — when omitted, a default
`StaticWorkload` is used that emits a single `None` item forever, which means
"fire the base request unchanged."

## With a dataset of inputs

```python
from benchmaker import (
    BenchConfig, BenchRunner, ConstantRPS,
    HttpWorkloadType, StaticWorkload,
)

cfg = BenchConfig(
    workload_type=HttpWorkloadType(url="https://api.example.com/v1/predict",
                                   method="POST"),
    workload=StaticWorkload(items=[{"q": q} for q in ["hi", "bye", "ciao"]]),
    load=ConstantRPS(rps=200, duration_s=30),
)
```

Each item from the workload is interpreted by the workload-type. For
`HttpWorkloadType`, a plain dict becomes the JSON body. See
[Workloads & workload-types](workloads.md) for the full item-interpretation
rules.

## LLM streaming benchmark

```python
from benchmaker import (
    BenchConfig, BenchRunner, PoissonRPS,
    OpenAIChatWorkloadType, JsonlWorkload,
)

cfg = BenchConfig(
    workload_type=OpenAIChatWorkloadType(
        url="http://localhost:8000/v1/chat/completions",
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=128,
    ),
    workload=JsonlWorkload(path="data/sharegpt.jsonl", field="prompt"),
    load=PoissonRPS(rps=8, duration_s=300),
    timeout_s=600,
)
```

TTFT, inter-token latency, and tokens/sec are captured automatically. See
[Workloads & workload-types](workloads.md#openaichatworkloadtype).

## CLI quick-start

One-liner:

```bash
bench-maker quick \
    --url https://httpbin.org/get \
    --rate ramp:10..200:10s \
    --duration 10s
```

Config-driven:

```bash
bench-maker run examples/config_llm.yaml \
    --out-dir ./runs --run-id baseline --label variant=v0

# pivot many runs into a table
bench-maker collect ./runs --label variant
```

See [CLI & YAML reference](cli-and-yaml.md) for the full surface.

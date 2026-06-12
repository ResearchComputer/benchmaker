# Quickstart

## Install

```bash
pip install -e .
pip install -e .[dev]   # for tests
```

This installs the `benchmaker` Python package and the `benchmaker` CLI.

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

## User-defined agent benchmark

benchmaker supports pluggable Python agents — multi-step pipelines that own
their own HTTP clients (model API, tools, sandbox, …) instead of going through
a single request/response cycle:

```python
import asyncio
from benchmaker import Agent, AgentContext, AgentResult, AgentWorkloadType
from benchmaker import BenchConfig, BenchRunner, StaticWorkload, ClosedLoop

class EchoAgent(Agent):
    async def run(self, ctx: AgentContext) -> AgentResult:
        task = ctx.item["prompt"]
        # ... your multi-step pipeline here ...
        return AgentResult(output=f"echo: {task}", ok=True,
                           metrics={"steps": 1.0})

cfg = BenchConfig(
    workload_type=AgentWorkloadType(EchoAgent(), reference_key="reference"),
    workload=StaticWorkload(items=[
        {"prompt": "hello", "reference": "echo: hello"},
        {"prompt": "world", "reference": "echo: world"},
    ]),
    load=ClosedLoop(concurrency=4, duration_s=30),
)
result = await BenchRunner(cfg).run()
print(result.summary)
```

The agent is instantiated once and reused across tickets. `AgentWorkloadType`
handles reference extraction and correctness grading via the standard
`correctness_hook`. See [Workloads & workload-types](workloads.md#agentworkloadtype).

## CLI quick-start

One-liner (a *recipe* — `benchmaker <recipe> --args`; recipes: `http`, `llm`,
`sandbox`, `sglang`, `swebench`, `trajectory-replay`, `swebench-replay`):

```bash
benchmaker http \
    --url https://httpbin.org/get \
    --rate ramp:10..200:10s \
    --duration 10s
```

Config-driven:

```bash
benchmaker run examples/config_llm.yaml \
    --out-dir ./runs --run-id baseline --label variant=v0

# pivot many runs into a table
benchmaker collect ./runs --label variant
```

```bash
# SGLang native /generate:
benchmaker sglang \
    --url http://localhost:30000/generate \
    --prompts-jsonl data.jsonl --prompt-field text \
    --rate poisson:8 --duration 60s

# Trajectory replay (prefix-cache parity):
benchmaker trajectory-replay --preset swe-smith \
    --url http://host:8000/v1/chat/completions --model $MODEL \
    --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct
```

See [CLI & YAML reference](cli-and-yaml.md) for the full surface.

# benchmaker

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

This installs the `benchmaker` Python package and the `benchmaker` CLI.

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

Or via the CLI. Workload-specific benchmarks are exposed as **recipes** —
`benchmaker <recipe> --args` (`http`, `llm`, `sandbox`, `swebench`):

```bash
benchmaker http --url https://httpbin.org/get --rate poisson:50 --duration 10s
```

## Walkthrough: benchmarking an LLM endpoint with ShareGPT

A realistic LLM benchmark needs a real prompt distribution.
[ShareGPT V3](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
is a common choice — multi-turn human/assistant conversations scraped from real
ChatGPT users. A cleaned, benchmark-ready copy is published at
[`researchcomputer/llmsys-bench`](https://huggingface.co/datasets/researchcomputer/llmsys-bench)
(`split="sharegpt"`), with one row per conversation:

```json
{"id": "...", "messages": [{"role": "user", "content": "..."},
                           {"role": "assistant", "content": "..."},
                           {"role": "user", "content": "..."}]}
```

`messages` is the only content field — it's everything a chat benchmark needs.
Each row is truncated to end on a **user** turn, so it's a valid generation
request: the server completes the final assistant reply given the prior
history. Short source conversations collapse to a single user turn (a plain
single-turn prompt); longer ones carry multi-turn context.

### Load it directly from the Hub

Pull the published split and feed each row's `messages` list straight into the
chat workload-type (`pip install -e .[hf]`):

```python
import asyncio
from datasets import load_dataset
from benchmaker import (
    BenchConfig, BenchRunner, OpenAIChatWorkloadType,
    IterableWorkload, parse_rate_spec,
)

async def main():
    ds = load_dataset("researchcomputer/llmsys-bench", split="sharegpt")
    cfg = BenchConfig(
        workload_type=OpenAIChatWorkloadType(
            url="http://localhost:8000/v1/chat/completions",
            model="meta-llama/Llama-3.1-8B-Instruct",
            max_tokens=256,
        ),
        workload=IterableWorkload(row["messages"] for row in ds),
        load=parse_rate_spec("poisson:8", duration_s=60),
        timeout_s=600,
    )
    result = await BenchRunner(cfg).run()
    print(result.summary)

asyncio.run(main())
```

`OpenAIChatWorkloadType` receives the message list as-is, so single-turn rows
send one user message and multi-turn rows replay the full history before the
server generates the final assistant turn. TTFT, inter-token latency, and
tokens/sec are captured the same way in both cases. URL / model / API key can
also come from `.env` via `OpenAIChatWorkloadType.from_env(...)`.

### Rebuild or customize it yourself

The published split is produced by `tools/sharegpt/prepare.py`, which downloads
the upstream JSON once into `.local/` (gitignored) and converts it to the JSONL
shape above. Run it when you want a subset, different filtering, or a refresh:

```bash
# Defaults: .local/sharegpt_v3_raw.json  ->  .local/sharegpt_v3.jsonl
python tools/sharegpt/prepare.py

# A quick subset for smoke tests:
python tools/sharegpt/prepare.py --max-items 2000
```

The raw download is ~700 MB. Use `--min-chars` / `--max-chars` to drop empty or
pathologically long conversations (measured over total message content per
row). Point any workload at the local file with `JsonlWorkload(path=...,
field="messages")`, or on the CLI:

```bash
benchmaker llm \
    --url   http://localhost:8000/v1/chat/completions \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prompts-jsonl .local/sharegpt_v3.jsonl \
    --prompt-field  messages \
    --max-tokens 256 \
    --rate poisson:8 --duration 60s \
    --out-dir ./runs --label dataset=sharegpt
```

To re-publish after regenerating, `tools/sharegpt/upload_hf.py` pushes the
JSONL back to the Hub (needs a write token).

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
- [ShareGPT benchmark](docs/sharegpt-benchmark.md) — self-contained end-to-end walkthrough

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

Helper tooling under [`tools/`](tools/), grouped by purpose:

- `sharegpt/`     — `prepare.py` (fetch ShareGPT V3 → JSONL) + `upload_hf.py`
  (push to the HF Hub with a write token)
- `swe_images/`   — mirror SWE-bench/R2E-Gym container images to ghcr
  (`publish.py`) and list the published refs (`pull.py`)
- `agent_warmup/` — build the agent-warmup SFT dataset
  (`python -m tools.agent_warmup.cli`)
- `start_local_llm.sh` — example local SGLang launch command

## Project layout

```
benchmaker/          # library code
  __init__.py        #   public API (re-exports); cli.py — the `benchmaker` CLI
  config.py  env.py  #   YAML config loading + .env interpolation
  core/              #   engine: types, load models, runner, metrics, monitors, trace
  io/                #   run output: per-run bundle + cross-run collection
  workloads/         #   workload-types (http, llm, sandbox, agent, hf, eval)
  recipes/           #   CLI recipes (http, llm, sandbox, swebench) + registry
  swebench/          #   SWE-bench coding agent + grading + harbor adapters
examples/            # runnable examples (incl. swebench/ coding-agent config)
tools/               # out-of-tree tooling: sharegpt/, swe_images/, agent_warmup/
tests/               # pytest smoke tests
docs/                # reference docs
```

## Run the tests

```bash
pytest -q
```

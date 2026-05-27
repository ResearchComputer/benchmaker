"""Benchmark an OpenAI-compatible /v1/chat/completions endpoint with streaming.

Captures TTFT, ITL, tokens/sec on top of base latency metrics.

Usage:
    python -m examples.llm_chat \\
        --url http://localhost:8000/v1/chat/completions \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --rps 4 --duration 30
"""

import argparse
import asyncio
import sys

from benchmaker import (
    BenchConfig,
    BenchRunner,
    OpenAIChatWorkloadType,
    StaticWorkload,
    parse_rate_spec,
)
from benchmaker.load import parse_duration


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--rps", default="4")
    ap.add_argument("--duration", default="30s")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt", default="Tell me a one-sentence fun fact.")
    args = ap.parse_args()

    wt = OpenAIChatWorkloadType(
        url=args.url,
        model=args.model,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
    )
    workload = StaticWorkload(items=[args.prompt])
    load = parse_rate_spec(args.rps, duration_s=parse_duration(args.duration))
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload, load=load, timeout_s=600.0,
    ))
    await runner.run()
    runner.metrics.render(sys.stdout)


if __name__ == "__main__":
    asyncio.run(main())

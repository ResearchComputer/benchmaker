"""LLM benchmark driven by environment variables (.env).

Reads OPENAI_API_BASE_URL, OPENAI_COMPATIBLE_MODEL, OPENAI_API_KEY from .env
(or the shell), then runs a Poisson-arrival benchmark.

Usage:
    python examples/llm_from_env.py
    python examples/llm_from_env.py --rps 4 --duration 30s
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
from benchmaker.env import load_dotenv
from benchmaker.core.load import parse_duration


PROMPTS = [
    "Write a one-sentence fun fact about distributed systems.",
    "Explain RDMA in one paragraph.",
    "What is speculative decoding?",
    "Name three trade-offs of paged attention.",
]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", default="2",
                    help="rate spec: '2', 'poisson:4', 'closed:8', etc.")
    ap.add_argument("--duration", default="20s")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--dotenv", default=".env",
                    help="path to .env (default: .env in cwd)")
    args = ap.parse_args()

    # Load .env into os.environ if present (no-op if missing).
    load_dotenv(args.dotenv)

    wt = OpenAIChatWorkloadType.from_env(
        max_tokens=args.max_tokens,
        # dotenv already loaded above:
        dotenv_path=None,
    )
    workload = StaticWorkload(items=PROMPTS, shuffle=True, seed=0)
    load = parse_rate_spec(args.rps, duration_s=parse_duration(args.duration))

    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload, load=load, timeout_s=600.0,
    ))
    await runner.run()
    runner.metrics.render(sys.stdout)


if __name__ == "__main__":
    asyncio.run(main())

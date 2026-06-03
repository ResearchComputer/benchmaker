"""Benchmark an LLM endpoint AND grade correctness in the same run.

Composition:
  base WorkloadType (OpenAIChatWorkloadType)
    └─ wrapped in EvalWorkloadType    -- carries `reference` through to meta
       + correctness_hook(scorer)      -- post-response scorer
       + workload yielding {prompt, reference} pairs

Per-request and aggregate accuracy show up in the standard summary under
`workload_metrics.correct.*` alongside latency / TTFT / tokens/sec.

Usage (exact-match):
    python -m examples.llm_eval \\
        --url http://localhost:8000/v1/chat/completions \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --dataset prompts.jsonl --scorer exact

Each JSONL line should look like: {"prompt": "...", "reference": "..."}

Usage (LLM-as-judge):
    python -m examples.llm_eval \\
        --url http://localhost:8000/v1/chat/completions \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --dataset prompts.jsonl --scorer judge \\
        --judge-url http://localhost:8001/v1/chat/completions \\
        --judge-model judge-7b
"""

import argparse
import asyncio
import sys

from benchmaker import (
    BenchConfig,
    BenchRunner,
    EvalWorkloadType,
    JsonlWorkload,
    OpenAIChatWorkloadType,
    StaticWorkload,
    contains,
    correctness_hook,
    exact_match,
    judge_llm,
    openai_chat_judge,
    parse_rate_spec,
    regex_match,
)
from benchmaker.core.load import parse_duration


def _build_scorer(args):
    """Returns (scorer, optional aclose-callable)."""
    if args.scorer == "exact":
        return exact_match(strip=True, case_insensitive=args.case_insensitive), None
    if args.scorer == "contains":
        return contains(case_insensitive=args.case_insensitive), None
    if args.scorer == "regex":
        if not args.regex:
            raise SystemExit("--regex is required when --scorer=regex")
        return regex_match(args.regex, group=args.regex_group,
                           case_insensitive=args.case_insensitive), None
    if args.scorer == "judge":
        if not args.judge_url or not args.judge_model:
            raise SystemExit("--judge-url and --judge-model are required for --scorer=judge")
        send, aclose = openai_chat_judge(
            url=args.judge_url, model=args.judge_model, api_key=args.judge_api_key,
        )
        return judge_llm(send, max_concurrency=args.judge_concurrency), aclose
    raise SystemExit(f"unknown scorer {args.scorer!r}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--dataset", required=True,
                    help="JSONL with {prompt, reference} per line")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--rps", default="2")
    ap.add_argument("--duration", default="60s")
    ap.add_argument("--max-tokens", type=int, default=256)

    ap.add_argument("--scorer", choices=["exact", "contains", "regex", "judge"],
                    default="exact")
    ap.add_argument("--case-insensitive", action="store_true")
    ap.add_argument("--regex", default=None,
                    help="Pattern for --scorer=regex (use a capture group)")
    ap.add_argument("--regex-group", type=int, default=1)
    ap.add_argument("--reference-key", default="reference")

    # Judge-only flags
    ap.add_argument("--judge-url", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-api-key", default=None)
    ap.add_argument("--judge-concurrency", type=int, default=4)

    ap.add_argument("--bundle", default=None,
                    help="Optional output bundle directory")
    args = ap.parse_args()

    base = OpenAIChatWorkloadType(
        url=args.url, model=args.model, api_key=args.api_key,
        max_tokens=args.max_tokens,
    )
    wt = EvalWorkloadType(base, reference_key=args.reference_key)
    workload = JsonlWorkload(path=args.dataset, loop=False, max_items=args.max_items)

    scorer, judge_aclose = _build_scorer(args)
    hook = correctness_hook(scorer, reference_key=args.reference_key)

    load = parse_rate_spec(args.rps, duration_s=parse_duration(args.duration))
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload, load=load,
        post_hooks=[hook], timeout_s=600.0,
    ))
    try:
        await runner.run()
    finally:
        if judge_aclose is not None:
            await judge_aclose()

    runner.metrics.render(sys.stdout)
    if args.bundle:
        path = runner.write_bundle(args.bundle)
        print(f"\nwrote bundle: {path}")


if __name__ == "__main__":
    asyncio.run(main())

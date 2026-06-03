"""Benchmark a vLLM server and concurrently scrape its /metrics endpoint.

Records vLLM-internal metrics (queue depth, KV-cache usage, throughput, etc.)
as a time-series alongside the client-side latency/throughput metrics.

Usage:
    python examples/vllm_with_monitor.py \\
        --url http://localhost:8000/v1/chat/completions \\
        --metrics-url http://localhost:8000/metrics \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --rps 8 --duration 60s
"""

import argparse
import asyncio
import sys

from benchmaker import (
    BenchConfig,
    BenchRunner,
    OpenAIChatWorkloadType,
    PrometheusMonitor,
    StaticWorkload,
    parse_rate_spec,
)
from benchmaker.core.load import parse_duration


# vLLM-specific Prometheus metric names worth tracking.
# (See: https://docs.vllm.ai/en/latest/serving/metrics.html)
VLLM_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_per_output_token_seconds_sum",
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="OpenAI-compatible chat URL")
    ap.add_argument("--metrics-url", required=True, help="Prometheus /metrics URL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--rps", default="8")
    ap.add_argument("--duration", default="60s")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--monitor-interval", type=float, default=1.0)
    ap.add_argument("--out-dir", default=None,
                    help="Parent directory for the run bundle. The bundle is written to "
                         "<out-dir>/<run-id>/ (meta.json, summary.json, samples.jsonl, monitors.jsonl).")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    wt = OpenAIChatWorkloadType(
        url=args.url, model=args.model,
        max_tokens=args.max_tokens, api_key=args.api_key,
    )
    workload = StaticWorkload(items=[
        "Tell me a one-sentence fun fact about distributed systems.",
        "Explain RDMA in one paragraph.",
        "What is speculative decoding?",
    ])
    monitor = PrometheusMonitor(
        url=args.metrics_url,
        metric_names=VLLM_METRICS,
        interval_s=args.monitor_interval,
        name="vllm",
    )
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=parse_rate_spec(args.rps, duration_s=parse_duration(args.duration)),
        monitors=[monitor],
        timeout_s=600.0,
    ))
    await runner.run()
    runner.metrics.render(sys.stdout)

    if args.out_dir:
        path = runner.write_bundle(args.out_dir, run_id=args.run_id)
        print(f"wrote run bundle -> {path}")


if __name__ == "__main__":
    asyncio.run(main())

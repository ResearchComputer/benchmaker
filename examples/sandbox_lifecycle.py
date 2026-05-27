"""Benchmark Flash Sandbox full create→exec→delete lifecycle.

Each ticket allocates a fresh sandbox, runs a command, and deletes the sandbox.
Per-step durations land in `Sample.extra` so you can disentangle startup,
execution, and teardown:

    create_s, exec_s, delete_s, lifecycle_s

Useful for measuring cold-start cost and end-to-end "throwaway sandbox" latency.

Usage:
    FLASH=http://localhost:8080 python examples/sandbox_lifecycle.py
"""

import asyncio
import os
import sys

from benchmaker import (
    BenchConfig,
    BenchRunner,
    ConstantRPS,
    SandboxWorkloadType,
    StaticWorkload,
)


async def main():
    base_url = os.environ.get("FLASH", "http://localhost:8080")

    workload_type = SandboxWorkloadType(
        base_url=base_url,
        operation="lifecycle",
        spec={
            "type": "kubernetes",
            "image": "alpine:3.20",
            "command": ["sh", "-c", "sleep 60"],
            "memory_mb": 256,
            "cpu_cores": 0.5,
        },
        ttl_seconds=120,  # safety net in case delete is missed
    )

    workload = StaticWorkload(items=[
        "echo hello",
        "uname -a",
    ])

    runner = BenchRunner(BenchConfig(
        workload_type=workload_type,
        workload=workload,
        load=ConstantRPS(rps=2, duration_s=10),
    ))
    await runner.run()
    runner.metrics.render(sys.stdout)


if __name__ == "__main__":
    asyncio.run(main())

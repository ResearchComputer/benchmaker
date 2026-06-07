"""Benchmark Flash Sandbox /exec latency.

Lazily creates one sandbox on first call, runs many `exec` calls against it,
and deletes the sandbox when the run finishes.

Usage:
    FLASH=http://localhost:8080 python examples/sandbox_exec.py
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
        spec={
            "type": "docker",
            "image": "alpine:3.20",
            "command": ["sh", "-c", "sleep 3600"],
            "memory_mb": 256,
            "cpu_cores": 0.5,
        },
        ttl_seconds=600,  # safety net in case cleanup is skipped
    )

    workload = StaticWorkload(items=[
        "echo hello",
        "uname -a",
        "ls /etc",
        ["sh", "-c", "for i in 1 2 3; do echo $i; done"],
    ])

    runner = BenchRunner(BenchConfig(
        workload_type=workload_type,
        workload=workload,
        load=ConstantRPS(rps=20, duration_s=10),
    ))
    await runner.run()
    runner.metrics.render(sys.stdout)


if __name__ == "__main__":
    asyncio.run(main())

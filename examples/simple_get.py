"""Minimal example: 100 RPS constant load for 10 seconds against a URL."""

import asyncio
import sys

from benchmaker import (
    BenchConfig,
    BenchRunner,
    ConstantRPS,
    HttpWorkloadType,
)


async def main():
    workload_type = HttpWorkloadType(url="https://httpbin.org/get", method="GET")
    load = ConstantRPS(rps=100, duration_s=10)
    # Workload is omitted -> defaults to a single None item (fixed request).
    runner = BenchRunner(BenchConfig(workload_type=workload_type, load=load))
    await runner.run()
    runner.metrics.render(sys.stdout)


if __name__ == "__main__":
    asyncio.run(main())

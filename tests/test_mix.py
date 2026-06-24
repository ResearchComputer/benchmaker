"""Concurrent mixed-lane scheduling and reporting."""

import pytest

from benchmaker import (
    BenchConfig,
    BenchLane,
    BenchRunner,
    ClosedLoop,
    ConstantRPS,
    HttpWorkloadType,
    StaticWorkload,
)


@pytest.mark.asyncio
async def test_mixed_lanes_tag_every_sample_and_report_independently(stub_server: str):
    runner = BenchRunner(BenchConfig(
        workload_type=HttpWorkloadType(url=f"{stub_server}/hello"),
        lanes=[
            BenchLane(
                "prefill",
                StaticWorkload(items=[None]),
                ConstantRPS(rps=1_000, max_requests=2),
            ),
            BenchLane(
                "decode",
                StaticWorkload(items=[None]),
                ConstantRPS(rps=1_000, max_requests=3),
            ),
        ],
        progress_every_s=0,
    ))

    result = await runner.run()

    assert result.summary["total_requests"] == 5
    assert {sample.meta["lane"] for sample in result.samples} == {"prefill", "decode"}
    assert result.summary["lanes"]["prefill"]["total_requests"] == 2
    assert result.summary["lanes"]["decode"]["total_requests"] == 3


@pytest.mark.asyncio
async def test_mixed_closed_loop_completion_returns_to_the_owning_lane(stub_server: str):
    """A completion in one closed loop must not unblock a different lane."""
    runner = BenchRunner(BenchConfig(
        workload_type=HttpWorkloadType(url=f"{stub_server}/slow"),
        lanes=[
            BenchLane("left", StaticWorkload(), ClosedLoop(1, max_requests=2)),
            BenchLane("right", StaticWorkload(), ClosedLoop(1, max_requests=2)),
        ],
        progress_every_s=0,
    ))

    result = await runner.run()

    assert result.summary["total_requests"] == 4
    assert result.summary["lanes"]["left"]["total_requests"] == 2
    assert result.summary["lanes"]["right"]["total_requests"] == 2


def test_yaml_mix_builds_independent_workload_and_rate_pairs(stub_server: str):
    from benchmaker.config import build_config

    cfg = build_config({
        "workload_type": {"type": "http", "url": f"{stub_server}/hello"},
        "mix": {
            "lanes": [
                {
                    "name": "prefill",
                    "workload": {"type": "static", "items": [None]},
                    "rate": {"type": "constant", "rps": 5, "max_requests": 2},
                },
                {
                    "name": "decode",
                    "workload": {"type": "static", "items": [None]},
                    "rate": "10",
                    "max_requests": 3,
                },
            ],
        },
    }, dotenv_path=None, interpolate_env=False)

    assert cfg.load is None
    assert [lane.name for lane in cfg.lanes] == ["prefill", "decode"]
    assert [lane.load.max_requests for lane in cfg.lanes] == [2, 3]

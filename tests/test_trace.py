"""Record/replay round-trip tests."""

import json
import os
import tempfile

import pytest

from benchmaker import BenchConfig, BenchRunner, ConstantRPS, HttpWorkloadType
from benchmaker.trace import (
    ReplayWorkloadType,
    TracePacedLoad,
    TraceRecorder,
    TraceWorkload,
    load_trace,
    request_to_row,
    row_to_request,
)
from benchmaker.types import Request


def test_row_roundtrip_json_body():
    req = Request(
        method="POST", url="https://example/v1",
        headers={"Authorization": "Bearer x"},
        json={"prompt": "hi", "n": 3},
        meta={"reference": "42"},
    )
    row = request_to_row(1.5, req)
    assert row["t_rel"] == 1.5
    assert row["json"]["prompt"] == "hi"
    rebuilt = row_to_request(row)
    assert rebuilt.method == "POST"
    assert rebuilt.json == {"prompt": "hi", "n": 3}
    assert rebuilt.meta["reference"] == "42"


def test_row_roundtrip_binary_body():
    payload = b"\x00\x01\x02notjson"
    req = Request(method="POST", url="https://example/v1", body=payload)
    row = request_to_row(0.0, req)
    assert "body_b64" in row and "json" not in row
    rebuilt = row_to_request(row)
    assert rebuilt.body == payload


@pytest.mark.asyncio
async def test_record_and_replay_reproduces_requests(stub_server: str):
    """Record one run against the stub server, then replay from the trace and
    verify the same requests get fired in the same order."""
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = os.path.join(tmp, "trace.jsonl")

        # ---- 1) Record ----
        wt = HttpWorkloadType(url=f"{stub_server}/hello")
        rec_cfg = BenchConfig(
            workload_type=wt,
            load=ConstantRPS(rps=30, duration_s=0.5),
            recorder=TraceRecorder(trace_path),
            progress_every_s=0,
        )
        rec_runner = BenchRunner(rec_cfg)
        rec_result = await rec_runner.run()
        assert rec_result.summary["success"] == rec_result.summary["total_requests"]

        rows = load_trace(trace_path)
        assert len(rows) == rec_result.summary["total_requests"]
        # Times are monotonic non-negative and reasonable.
        ts = [r["t_rel"] for r in rows]
        assert ts == sorted(ts)
        assert ts[0] >= 0.0
        assert ts[-1] <= 0.6
        # URL captured verbatim.
        assert all(r["url"] == f"{stub_server}/hello" for r in rows)

        # ---- 2) Replay ----
        replay_cfg = BenchConfig(
            workload_type=ReplayWorkloadType(),
            workload=TraceWorkload(rows),
            load=TracePacedLoad(rows, speed=10.0),  # speed up so tests stay quick
            progress_every_s=0,
        )
        replay_runner = BenchRunner(replay_cfg)
        replay_result = await replay_runner.run()
        # Same number of requests, all hit the stub.
        assert replay_result.summary["total_requests"] == len(rows)
        assert replay_result.summary["success"] == len(rows)


@pytest.mark.asyncio
async def test_replay_preserves_request_meta(stub_server: str):
    """A reference attached during recording survives the round trip so an
    eval post-hook can grade replayed samples."""
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = os.path.join(tmp, "trace.jsonl")

        # Recorder pre-hook stuffs a reference into req.meta so it gets written
        # to the trace alongside the request.
        def stamp_meta(req: Request) -> Request:
            req.meta["reference"] = "gold-" + req.url.rsplit("/", 1)[-1]
            return req

        wt = HttpWorkloadType(url=f"{stub_server}/hello")
        rec_cfg = BenchConfig(
            workload_type=wt,
            load=ConstantRPS(rps=20, duration_s=0.3),
            pre_hooks=[stamp_meta],
            recorder=TraceRecorder(trace_path),
            progress_every_s=0,
        )
        await BenchRunner(rec_cfg).run()
        rows = load_trace(trace_path)
        assert rows, "expected at least one recorded row"
        assert all(r["meta"].get("reference") == "gold-hello" for r in rows)

        # Replay: capture meta off the request inside a post-hook.
        seen_refs: list[str] = []

        async def check_ref(req, resp, sample):
            seen_refs.append(req.meta.get("reference"))
            return sample

        replay_cfg = BenchConfig(
            workload_type=ReplayWorkloadType(),
            workload=TraceWorkload(rows),
            load=TracePacedLoad(rows, speed=20.0),
            post_hooks=[check_ref],
            progress_every_s=0,
        )
        await BenchRunner(replay_cfg).run()
        assert seen_refs and all(r == "gold-hello" for r in seen_refs)


@pytest.mark.asyncio
async def test_config_replay_block_overrides_load_and_workload(stub_server: str, tmp_path):
    """A `replay:` YAML block builds the workload-type/workload/load triple
    without needing the original config."""
    from benchmaker.config import build_config

    trace_path = tmp_path / "trace.jsonl"
    # Hand-write a 2-row trace pointing at the stub.
    with open(trace_path, "w") as f:
        for t in (0.0, 0.05):
            f.write(json.dumps({
                "t_rel": t,
                "method": "GET",
                "url": f"{stub_server}/hello",
                "headers": {},
                "params": {},
                "timeout_s": None,
                "meta": {},
            }) + "\n")

    cfg = build_config({
        "replay": {"path": str(trace_path), "speed": 10.0},
    }, dotenv_path=None, interpolate_env=False)

    runner = BenchRunner(cfg)
    result = await runner.run()
    assert result.summary["total_requests"] == 2
    assert result.summary["success"] == 2

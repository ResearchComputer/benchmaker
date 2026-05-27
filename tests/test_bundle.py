"""Bundle (per-run directory) write + collect."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from benchmaker import (
    BenchConfig,
    BenchRunner,
    ConstantRPS,
    HttpWorkloadType,
    is_bundle_dir,
    read_bundle,
)
from benchmaker.bundle import (
    META_FILENAME,
    SUMMARY_FILENAME,
    SAMPLES_FILENAME,
    write_bundle,
)
from benchmaker.collect import collect_table, find_bundles, format_table


async def _quick_run(stub_server: str) -> BenchRunner:
    wt = HttpWorkloadType(url=f"{stub_server}/hello")
    runner = BenchRunner(BenchConfig(
        workload_type=wt, load=ConstantRPS(rps=20, duration_s=0.3),
        progress_every_s=0,
    ))
    await runner.run()
    return runner


@pytest.mark.asyncio
async def test_write_bundle_creates_all_files(stub_server: str, tmp_path):
    runner = await _quick_run(stub_server)
    out_dir = tmp_path / "runs"
    path = runner.write_bundle(
        str(out_dir),
        run_id="run-a",
        source_config={"workload_type": {"type": "http", "url": "x"}, "load": "20"},
        labels={"variant": "baseline"},
        notes="smoke",
    )
    assert path == str((out_dir / "run-a").resolve())
    assert is_bundle_dir(path)
    for name in (META_FILENAME, SUMMARY_FILENAME, SAMPLES_FILENAME):
        assert (out_dir / "run-a" / name).is_file()

    meta = json.loads((out_dir / "run-a" / META_FILENAME).read_text())
    assert meta["run_id"] == "run-a"
    assert meta["workload_type"] == "http"
    assert meta["workload"] == "static"
    assert meta["labels"] == {"variant": "baseline"}
    assert meta["notes"] == "smoke"
    assert meta["bundle_version"] == 1
    assert meta["started_at"].endswith("+00:00")
    assert meta["ended_at"].endswith("+00:00")
    assert meta["wall_time_s"] > 0

    summary = json.loads((out_dir / "run-a" / SUMMARY_FILENAME).read_text())
    assert summary["total_requests"] >= 1

    # samples.jsonl: one decodable record per line.
    with open(out_dir / "run-a" / SAMPLES_FILENAME) as f:
        lines = [ln for ln in f if ln.strip()]
    assert len(lines) == summary["total_requests"]
    for line in lines:
        rec = json.loads(line)
        assert {"start_ts", "latency_s", "status", "ok"} <= rec.keys()


@pytest.mark.asyncio
async def test_read_bundle_roundtrip(stub_server: str, tmp_path):
    runner = await _quick_run(stub_server)
    path = runner.write_bundle(str(tmp_path), run_id="rb")

    bundle = read_bundle(path)
    assert bundle["meta"]["run_id"] == "rb"
    assert "throughput_rps" in bundle["summary"]
    assert bundle["samples_path"].endswith("samples.jsonl")
    # No monitors configured -> no monitors.jsonl.
    assert bundle["monitors_path"] is None


@pytest.mark.asyncio
async def test_default_run_id_when_omitted(stub_server: str, tmp_path):
    runner = await _quick_run(stub_server)
    path = runner.write_bundle(str(tmp_path))
    # Some timestamp-shaped subdir was created.
    children = [p for p in os.listdir(tmp_path) if (tmp_path / p).is_dir()]
    assert len(children) == 1
    assert path.endswith(children[0])


@pytest.mark.asyncio
async def test_collect_table_picks_up_multiple_runs(stub_server: str, tmp_path):
    r1 = await _quick_run(stub_server)
    r2 = await _quick_run(stub_server)
    r1.write_bundle(str(tmp_path), run_id="a", labels={"variant": "x"})
    r2.write_bundle(str(tmp_path), run_id="b", labels={"variant": "y"})

    bundles = find_bundles(str(tmp_path))
    assert sorted(os.path.basename(p) for p in bundles) == ["a", "b"]

    rows, columns = collect_table(bundles, label_keys=["variant"])
    assert {r["run_id"] for r in rows} == {"a", "b"}
    assert {r["label.variant"] for r in rows} == {"x", "y"}
    assert "rps" in columns
    assert "label.variant" in columns

    md = format_table(rows, columns, "md")
    assert "| run_id" in md
    assert "label.variant" in md

    csv_out = format_table(rows, columns, "csv")
    assert csv_out.splitlines()[0].startswith("run_id,workload_type")
    assert any("a," in line for line in csv_out.splitlines()[1:])


@pytest.mark.asyncio
async def test_collect_extra_dotted_metric(stub_server: str, tmp_path):
    r = await _quick_run(stub_server)
    r.write_bundle(str(tmp_path), run_id="m")
    rows, columns = collect_table(
        find_bundles(str(tmp_path)),
        extra_metrics=["latency_s.p50", "summary.throughput_rps"],
    )
    assert "latency_s.p50" in columns
    assert "summary.throughput_rps" in columns
    assert rows[0]["latency_s.p50"] is not None
    assert rows[0]["summary.throughput_rps"] == rows[0]["rps"]


@pytest.mark.asyncio
async def test_cli_run_writes_bundle(stub_server: str, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"workload_type:\n"
        f"  type: http\n"
        f"  url: {stub_server}/hello\n"
        f"load: '20'\n"
        f"duration: 0.3s\n"
        f"progress_every_s: 0\n"
    )

    # Use the installed entrypoint via `python -m`.
    out = subprocess.run(
        [
            sys.executable, "-m", "entrypoints.cli", "run",
            str(cfg_path),
            "--out-dir", str(tmp_path / "runs"),
            "--run-id", "cli-a",
            "--label", "variant=cli",
            "--quiet",
        ],
        capture_output=True, text=True, check=True,
    )
    assert (tmp_path / "runs" / "cli-a" / "meta.json").is_file()

    # Now run `collect` and check it parses.
    collected = subprocess.run(
        [
            sys.executable, "-m", "entrypoints.cli", "collect",
            str(tmp_path / "runs"),
            "--format", "csv",
            "--label", "variant",
        ],
        capture_output=True, text=True, check=True,
    )
    csv_text = collected.stdout.strip()
    assert csv_text.splitlines()[0].startswith("run_id,")
    assert "label.variant" in csv_text.splitlines()[0]
    assert "cli-a" in csv_text
    assert "cli" in csv_text

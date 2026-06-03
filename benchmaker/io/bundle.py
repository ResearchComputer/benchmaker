"""Per-run output bundle.

Each benchmark run writes a directory:

    <out_dir>/<run_id>/
        meta.json       # run identifiers, timestamps, resolved config
        summary.json    # aggregated metrics
        samples.jsonl   # one record per request
        monitors.jsonl  # one record per monitor tick

The bundle is the unit of result interchange. `bench-maker collect` walks
one or more such directories and pivots the summaries into a table.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from benchmaker.metrics import MetricsAggregator, _safe_meta


BUNDLE_VERSION = 1
META_FILENAME = "meta.json"
SUMMARY_FILENAME = "summary.json"
SAMPLES_FILENAME = "samples.jsonl"
MONITORS_FILENAME = "monitors.jsonl"


def default_run_id(prefix: Optional[str] = None) -> str:
    """Filesystem-safe UTC timestamp, e.g. `20260526T142233Z` or `prefix-20260526T142233Z`."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}" if prefix else ts


@dataclass
class RunMeta:
    """Identifiers and provenance written next to the results."""

    run_id: str
    started_at: str                            # ISO-8601 UTC
    ended_at: str                              # ISO-8601 UTC
    wall_time_s: float
    workload_type: str                         # e.g. "http", "openai-chat"
    workload: str                              # e.g. "static", "jsonl"
    hostname: str = field(default_factory=socket.gethostname)
    python_version: str = field(default_factory=platform.python_version)
    bench_maker_version: str = ""              # filled in by write_bundle
    bundle_version: int = BUNDLE_VERSION
    source_config: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "bundle_version": self.bundle_version,
            "bench_maker_version": self.bench_maker_version,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "wall_time_s": self.wall_time_s,
            "hostname": self.hostname,
            "python_version": self.python_version,
            "workload_type": self.workload_type,
            "workload": self.workload,
            "labels": self.labels,
            "notes": self.notes,
            "source_config": _safe_meta(self.source_config),
        }


def write_bundle(
    out_dir: str,
    metrics: MetricsAggregator,
    *,
    workload_type_name: str,
    workload_name: str,
    run_id: Optional[str] = None,
    source_config: Optional[dict] = None,
    labels: Optional[dict] = None,
    notes: str = "",
    started_wall: Optional[float] = None,
    ended_wall: Optional[float] = None,
) -> str:
    """Write the bundle for a finished run. Returns the absolute run directory path.

    `out_dir` is treated as the parent: the actual run lives in `<out_dir>/<run_id>/`.
    If `out_dir` itself already looks like a run dir (already has `<run_id>` baked in),
    pass `run_id=""` to skip the extra nesting.
    """
    from benchmaker import __version__ as bm_version

    if metrics.end_time is None:
        metrics.finalize()

    run_id = run_id if run_id is not None else default_run_id()
    target = os.path.join(out_dir, run_id) if run_id else out_dir
    os.makedirs(target, exist_ok=True)

    duration = metrics.end_time - metrics.start_time if metrics.end_time else 0.0
    end_wall = ended_wall if ended_wall is not None else (metrics.end_wall or time.time())
    start_wall = started_wall if started_wall is not None else (metrics.start_wall or (end_wall - duration))

    meta = RunMeta(
        run_id=run_id or os.path.basename(os.path.abspath(target)),
        started_at=_iso(start_wall),
        ended_at=_iso(end_wall),
        wall_time_s=duration,
        workload_type=workload_type_name,
        workload=workload_name,
        bench_maker_version=bm_version,
        source_config=source_config or {},
        labels=labels or {},
        notes=notes,
    )

    with open(os.path.join(target, META_FILENAME), "w") as f:
        json.dump(meta.to_dict(), f, indent=2, default=_json_default)

    with open(os.path.join(target, SUMMARY_FILENAME), "w") as f:
        json.dump(metrics.summary(), f, indent=2, default=_json_default)

    metrics.dump_samples_jsonl(os.path.join(target, SAMPLES_FILENAME))
    if metrics.monitor_samples:
        metrics.dump_monitor_jsonl(os.path.join(target, MONITORS_FILENAME))

    return os.path.abspath(target)


def read_bundle(run_dir: str) -> dict:
    """Read a run directory and return `{meta, summary, samples_path, monitors_path}`.

    Samples and monitor ticks are returned as file paths rather than loaded, so
    a `collect` over many runs stays cheap. Use `iter_jsonl` to stream rows.
    """
    meta_path = os.path.join(run_dir, META_FILENAME)
    summary_path = os.path.join(run_dir, SUMMARY_FILENAME)
    if not (os.path.isfile(meta_path) and os.path.isfile(summary_path)):
        raise FileNotFoundError(
            f"{run_dir!r} is not a run bundle (missing meta.json or summary.json)"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)
    samples = os.path.join(run_dir, SAMPLES_FILENAME)
    monitors = os.path.join(run_dir, MONITORS_FILENAME)
    return {
        "meta": meta,
        "summary": summary,
        "samples_path": samples if os.path.isfile(samples) else None,
        "monitors_path": monitors if os.path.isfile(monitors) else None,
        "dir": os.path.abspath(run_dir),
    }


def is_bundle_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, META_FILENAME))
        and os.path.isfile(os.path.join(path, SUMMARY_FILENAME))
    )


def iter_jsonl(path: str):
    """Yield decoded JSON objects from a JSONL file. Skips blank lines."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iso(wall_seconds: float) -> str:
    return datetime.fromtimestamp(wall_seconds, tz=timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    return repr(obj)

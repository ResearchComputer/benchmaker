"""Aggregation + reporting of Samples."""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, TextIO

from benchmaker.types import Sample


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


@dataclass
class MetricsAggregator:
    """Accumulates samples and produces a summary."""

    samples: list[Sample] = field(default_factory=list)
    start_time: float = field(default_factory=time.monotonic)
    end_time: Optional[float] = None
    # Wall-clock (time.time) markers, only used when writing the run bundle.
    start_wall: float = field(default_factory=time.time)
    end_wall: Optional[float] = None
    # name -> list of (elapsed_s, {metric_name: value})
    monitor_samples: dict[str, list[tuple[float, dict[str, float]]]] = field(default_factory=dict)

    def add(self, sample: Sample) -> None:
        self.samples.append(sample)

    def monitor_buffer(self, name: str) -> list[tuple[float, dict[str, float]]]:
        if name not in self.monitor_samples:
            self.monitor_samples[name] = []
        return self.monitor_samples[name]

    def finalize(self) -> None:
        self.end_time = time.monotonic()
        self.end_wall = time.time()

    def summary(self) -> dict:
        end = self.end_time or time.monotonic()
        wall_s = max(end - self.start_time, 1e-9)
        ok = [s for s in self.samples if s.ok]
        fail = [s for s in self.samples if not s.ok]
        # Split fail into transport failures vs. delivered-but-graded-wrong.
        wrong = [s for s in fail if s.request_ok]
        request_failed = [s for s in fail if not s.request_ok]
        latencies = [s.latency_s for s in ok]

        status_counts = Counter(s.status for s in self.samples)
        error_counts = Counter(s.error for s in fail if s.error)

        out: dict = {
            "wall_time_s": wall_s,
            "total_requests": len(self.samples),
            "success": len(ok),
            "failed": len(fail),
            "request_failed": len(request_failed),
            "wrong_output": len(wrong),
            "error_rate": (len(fail) / len(self.samples)) if self.samples else 0.0,
            "request_failure_rate": (
                (len(request_failed) / len(self.samples)) if self.samples else 0.0
            ),
            "throughput_rps": len(self.samples) / wall_s,
            "goodput_rps": len(ok) / wall_s,
            "bytes_sent": sum(s.bytes_sent for s in self.samples),
            "bytes_recv": sum(s.bytes_recv for s in self.samples),
            "status_codes": dict(status_counts),
            "errors": dict(error_counts),
        }
        if latencies:
            out["latency_s"] = {
                "mean": statistics.mean(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "p50": _pct(latencies, 50),
                "p90": _pct(latencies, 90),
                "p95": _pct(latencies, 95),
                "p99": _pct(latencies, 99),
                "p999": _pct(latencies, 99.9),
            }

        # Aggregate workload-specific `extra` metrics generically: mean + percentiles.
        extras: dict[str, list[float]] = defaultdict(list)
        for s in ok:
            for k, v in s.extra.items():
                if isinstance(v, (int, float)):
                    extras[k].append(float(v))
        if extras:
            ext_summary = {}
            for k, vals in extras.items():
                ext_summary[k] = {
                    "mean": statistics.mean(vals),
                    "p50": _pct(vals, 50),
                    "p90": _pct(vals, 90),
                    "p99": _pct(vals, 99),
                    "min": min(vals),
                    "max": max(vals),
                }
            out["workload_metrics"] = ext_summary

        # Monitor time-series: summarize each metric per monitor.
        if self.monitor_samples:
            monitors_summary: dict[str, dict] = {}
            for mon_name, ticks in self.monitor_samples.items():
                if not ticks:
                    continue
                by_metric: dict[str, list[float]] = defaultdict(list)
                for _t, values in ticks:
                    for k, v in values.items():
                        if isinstance(v, (int, float)):
                            by_metric[k].append(float(v))
                per_metric = {}
                for k, vals in by_metric.items():
                    per_metric[k] = {
                        "mean": statistics.mean(vals),
                        "min": min(vals),
                        "max": max(vals),
                        "p50": _pct(vals, 50),
                        "p90": _pct(vals, 90),
                        "p99": _pct(vals, 99),
                        "first": vals[0],
                        "last": vals[-1],
                    }
                monitors_summary[mon_name] = {
                    "tick_count": len(ticks),
                    "metrics": per_metric,
                }
            if monitors_summary:
                out["monitors"] = monitors_summary
        return out

    def render(self, out: TextIO) -> None:
        s = self.summary()
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"[benchmaker] results  ({s['total_requests']} requests, "
                     f"{s['wall_time_s']:.2f}s wall)")
        lines.append("=" * 60)
        lines.append(f"  throughput     : {s['throughput_rps']:>10.2f} req/s")
        lines.append(f"  goodput        : {s['goodput_rps']:>10.2f} req/s")
        lines.append(f"  success        : {s['success']}")
        lines.append(f"  failed         : {s['failed']}  ({s['error_rate']*100:.2f}%)")
        lines.append(
            f"    request failed : {s['request_failed']}  "
            f"({s['request_failure_rate']*100:.2f}%)"
        )
        lines.append(f"    wrong output   : {s['wrong_output']}")
        if s.get("latency_s"):
            l = s["latency_s"]
            lines.append("")
            lines.append("  latency (s)")
            for k in ("mean", "p50", "p90", "p95", "p99", "p999", "max"):
                lines.append(f"    {k:<6}: {l[k]:.4f}")
        if s["status_codes"]:
            lines.append("")
            lines.append("  status codes")
            for code, n in sorted(s["status_codes"].items()):
                lines.append(f"    {code:<4} : {n}")
        if s["errors"]:
            lines.append("")
            lines.append("  errors")
            for err, n in sorted(s["errors"].items(), key=lambda kv: -kv[1])[:10]:
                lines.append(f"    {n:<4} x {err}")
        if s.get("workload_metrics"):
            lines.append("")
            lines.append("  workload metrics")
            for k, v in s["workload_metrics"].items():
                lines.append(f"    {k}")
                for kk in ("mean", "p50", "p90", "p99", "max"):
                    lines.append(f"      {kk:<6}: {v[kk]:.4f}")
        if s.get("monitors"):
            for mon_name, mon in s["monitors"].items():
                lines.append("")
                lines.append(f"  monitor: {mon_name}  ({mon['tick_count']} ticks)")
                for k, v in mon["metrics"].items():
                    lines.append(f"    {k}")
                    for kk in ("mean", "min", "max", "p50", "p99", "last"):
                        lines.append(f"      {kk:<6}: {v[kk]:.4f}")
        lines.append("=" * 60)
        out.write("\n".join(lines) + "\n")
        out.flush()

    def dump_samples_jsonl(self, path: str) -> None:
        """Write per-request records for offline analysis."""
        with open(path, "w") as f:
            for s in self.samples:
                f.write(json.dumps({
                    "start_ts": s.start_ts,
                    "latency_s": s.latency_s,
                    "status": s.status,
                    "ok": s.ok,
                    "request_ok": s.request_ok,
                    "bytes_sent": s.bytes_sent,
                    "bytes_recv": s.bytes_recv,
                    "error": s.error,
                    "workload": s.workload,
                    "meta": _safe_meta(s.meta),
                    "extra": s.extra,
                }) + "\n")

    def dump_monitor_jsonl(self, path: str) -> None:
        """Write monitor time-series ticks as JSONL for plotting/analysis."""
        with open(path, "w") as f:
            for mon_name, ticks in self.monitor_samples.items():
                for t, values in ticks:
                    f.write(json.dumps({
                        "monitor": mon_name,
                        "elapsed_s": t,
                        "values": values,
                    }) + "\n")


def _safe_meta(meta: dict) -> dict:
    out = {}
    for k, v in meta.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = repr(v)
    return out

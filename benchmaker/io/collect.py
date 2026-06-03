"""Collect run bundles into a comparison table.

`collect_table(bundle_dirs, ...)` reads every bundle's `meta.json` +
`summary.json` and produces a list of row dicts plus a column ordering.
`format_table(rows, columns, fmt)` renders them as Markdown, CSV, or JSON.

The default columns capture the core throughput/latency/error metrics. Users
can ask for additional dotted-path metrics via `extra_metrics=` (e.g.
`workload_metrics.ttft_s.p50`).
"""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Any, Iterable, Optional

from benchmaker.io.bundle import is_bundle_dir, read_bundle


# Columns surfaced by default. (Order matters.)
DEFAULT_COLUMNS: list[tuple[str, str]] = [
    ("run_id", "meta.run_id"),
    ("workload_type", "meta.workload_type"),
    ("workload", "meta.workload"),
    ("wall_s", "meta.wall_time_s"),
    ("total", "summary.total_requests"),
    ("ok", "summary.success"),
    ("fail", "summary.failed"),
    ("err_rate", "summary.error_rate"),
    ("rps", "summary.throughput_rps"),
    ("good_rps", "summary.goodput_rps"),
    ("p50_s", "summary.latency_s.p50"),
    ("p90_s", "summary.latency_s.p90"),
    ("p99_s", "summary.latency_s.p99"),
    ("max_s", "summary.latency_s.max"),
]


def find_bundles(path: str, *, recursive: bool = True) -> list[str]:
    """Return run-bundle directories under `path`.

    If `path` is itself a bundle, return `[path]`. Otherwise list its immediate
    subdirectories and keep the ones that look like bundles. With
    `recursive=False`, only the top-level is checked.
    """
    if is_bundle_dir(path):
        return [path]
    if not recursive or not os.path.isdir(path):
        return []
    out: list[str] = []
    for name in sorted(os.listdir(path)):
        child = os.path.join(path, name)
        if is_bundle_dir(child):
            out.append(child)
    return out


def _dotted_get(d: dict, path: str, default: Any = None) -> Any:
    """`a.b.c` lookup; segments may be dict keys (including ones with dots, tried first)."""
    if path in d:
        return d[path]
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def collect_table(
    bundle_dirs: Iterable[str],
    *,
    extra_metrics: Optional[list[str]] = None,
    label_keys: Optional[list[str]] = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read each bundle and return (rows, column_names).

    Each row is keyed by friendly column names (e.g. `rps`, `p50_s`). Extra
    metrics keep their full dotted path as the column name, so a caller can
    re-pick columns via the `columns` arg.
    """
    extra_metrics = extra_metrics or []
    label_keys = label_keys or []

    columns: list[str] = [c for c, _ in DEFAULT_COLUMNS]
    for lk in label_keys:
        columns.append(f"label.{lk}")
    columns.extend(extra_metrics)

    rows: list[dict[str, Any]] = []
    for d in bundle_dirs:
        bundle = read_bundle(d)
        meta = bundle["meta"]
        summary = bundle["summary"]
        ctx = {"meta": meta, "summary": summary}

        row: dict[str, Any] = {}
        for col, path in DEFAULT_COLUMNS:
            row[col] = _dotted_get(ctx, path)
        for lk in label_keys:
            row[f"label.{lk}"] = (meta.get("labels") or {}).get(lk)
        for m in extra_metrics:
            # Allow the user to write `summary.foo` or just `foo` (assumes summary.*).
            if m.startswith("meta.") or m.startswith("summary."):
                row[m] = _dotted_get(ctx, m)
            else:
                row[m] = _dotted_get(summary, m)
        rows.append(row)

    return rows, columns


# ----------------------------------------------------------------- formatters


def format_table(rows: list[dict[str, Any]], columns: list[str], fmt: str) -> str:
    fmt = fmt.lower()
    if fmt == "md":
        return _format_md(rows, columns)
    if fmt == "csv":
        return _format_csv(rows, columns)
    if fmt == "json":
        return json.dumps(
            [{c: r.get(c) for c in columns} for r in rows],
            indent=2,
            default=str,
        )
    raise ValueError(f"Unknown format {fmt!r}")


def _format_csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(columns)
    for r in rows:
        w.writerow([_csv_cell(r.get(c)) for c in columns])
    return buf.getvalue().rstrip("\n")


def _csv_cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def _format_md(rows: list[dict[str, Any]], columns: list[str]) -> str:
    rendered: list[list[str]] = [[_md_cell(r.get(c)) for c in columns] for r in rows]
    widths = [len(c) for c in columns]
    for r in rendered:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))) + " |"

    header = line(columns)
    sep = "|" + "|".join("-" * (widths[i] + 2) for i in range(len(columns))) + "|"
    body = [line(r) for r in rendered]
    return "\n".join([header, sep, *body])


def _md_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        # Compact, but keep small-magnitude numbers readable.
        if abs(v) >= 1000:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.3f}"
        return f"{v:.4g}"
    return str(v)

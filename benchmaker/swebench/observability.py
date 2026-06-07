"""Timeline + machine-utilization observability for the SWE-bench harbor run.

Pure helpers (span/util/summary shaping) sit at the top; the I/O-bearing
``JobObserver`` and ``run_job_with_observability`` orchestrate harbor hooks, the
``/status`` poller, and artifact writing. Every orchestration path is
best-effort: it may degrade (skip a poll, drop spans, write a partial file) but
must never raise into a harbor trial. See
``docs/superpowers/specs/2026-06-07-swebench-observability-design.md``.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("benchmaker.observability")

# Harbor's four trial phases, in execution order.
PHASE_FIELDS = ("environment_setup", "agent_setup", "agent_execution", "verifier")


def _iso(dt: Any) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _dur(start: Any, end: Any) -> Optional[float]:
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def _span(trial: str, task: str, kind: str, name: str, **over: Any) -> dict:
    """A timeline span with every field present (None when N/A) for a stable schema."""
    base = {
        "trial": trial, "task": task, "kind": kind, "name": name,
        "start": None, "end": None, "duration_s": None, "seq": None, "rc": None,
        "n_input_tokens": None, "n_output_tokens": None, "n_cache_tokens": None,
        "cost_usd": None, "extra": {},
    }
    base.update(over)
    return base


def phase_spans_from_result(result: Any) -> list[dict]:
    """Phase-level spans for one trial from a harbor ``TrialResult``.

    Reads the four ``TimingInfo`` phases; skips any that are missing or have
    neither timestamp. ``trial``/``task`` come from the result.
    """
    trial = getattr(result, "trial_name", "") or ""
    task = getattr(result, "task_name", "") or ""
    spans: list[dict] = []
    for name in PHASE_FIELDS:
        ti = getattr(result, name, None)
        if ti is None:
            continue
        start = getattr(ti, "started_at", None)
        end = getattr(ti, "finished_at", None)
        if start is None and end is None:
            continue
        spans.append(_span(
            trial, task, "phase", name,
            start=_iso(start), end=_iso(end), duration_s=_dur(start, end),
        ))
    return spans


def util_row_from_status(status: Any, t: float, wall: datetime) -> dict:
    """One ``utilization.jsonl`` row from a flash-sandbox ``ClusterStatus``.

    ``t`` is seconds since the poller started; ``wall`` is the UTC timestamp.
    """
    nodes = []
    for n in getattr(status, "nodes", ()) or ():
        nodes.append({
            "id": getattr(n, "node_id", "") or "",
            "available": bool(getattr(n, "available", False)),
            "running_count": int(getattr(n, "running_count", 0) or 0),
        })
    return {
        "t": round(float(t), 3),
        "wall": wall.isoformat(),
        "node_count": int(getattr(status, "node_count", 0) or 0),
        "available_node_count": int(getattr(status, "available_node_count", 0) or 0),
        "unavailable_node_count": int(getattr(status, "unavailable_node_count", 0) or 0),
        "sandbox_count": int(getattr(status, "sandbox_count", 0) or 0),
        "nodes": nodes,
    }

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


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (pos - lo)) + s[hi] * (pos - lo)


def _mean(values: list[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def summarize(spans: list[dict], util_rows: list[dict]) -> dict:
    """Aggregate spans + utilization rows into a printable summary dict."""
    by_phase: dict[str, list[float]] = {}
    llm_durs: list[float] = []
    exec_durs: list[float] = []
    in_tok = out_tok = 0
    n_llm = n_exec = 0
    for sp in spans:
        kind = sp.get("kind")
        d = sp.get("duration_s")
        if kind == "phase" and d is not None:
            by_phase.setdefault(sp.get("name", "?"), []).append(d)
        elif kind == "llm_call":
            n_llm += 1
            if d is not None:
                llm_durs.append(d)
            in_tok += int(sp.get("n_input_tokens") or 0)
            out_tok += int(sp.get("n_output_tokens") or 0)
        elif kind == "sandbox_exec":
            n_exec += 1
            if d is not None:
                exec_durs.append(d)

    phases = {
        name: {"count": len(ds), "mean_s": _mean(ds), "p90_s": _percentile(ds, 0.9)}
        for name, ds in by_phase.items()
    }
    sandbox_counts = [int(r.get("sandbox_count", 0) or 0) for r in util_rows]
    node_counts = [int(r.get("node_count", 0) or 0) for r in util_rows]
    avail = [float(r.get("available_node_count", 0) or 0) for r in util_rows]
    return {
        "phases": phases,
        "agent": {
            "n_llm_calls": n_llm,
            "llm_mean_s": _mean(llm_durs),
            "n_exec": n_exec,
            "exec_mean_s": _mean(exec_durs),
            "total_input_tokens": in_tok,
            "total_output_tokens": out_tok,
        },
        "utilization": {
            "polls": len(util_rows),
            "sandbox_peak": max(sandbox_counts) if sandbox_counts else None,
            "sandbox_mean": _mean([float(c) for c in sandbox_counts]),
            "node_count": max(node_counts) if node_counts else None,
            "available_mean": _mean(avail),
        },
    }


def _fmt(x: Optional[float], suffix: str = "") -> str:
    return "—" if x is None else f"{x:.1f}{suffix}"


def format_summary(summary: dict) -> str:
    """Render the summary dict as a compact text block for end-of-run printing."""
    lines = ["", "── observability ──", "PHASE                 COUNT   MEAN     P90"]
    for name, p in summary.get("phases", {}).items():
        lines.append(
            f"{name:<20}  {p['count']:>5}  {_fmt(p['mean_s'], 's'):>7}  "
            f"{_fmt(p['p90_s'], 's'):>7}"
        )
    ag = summary.get("agent", {})
    if ag.get("n_llm_calls") or ag.get("n_exec"):
        lines.append(
            f"agent: {ag['n_llm_calls']} llm calls (mean {_fmt(ag['llm_mean_s'], 's')}), "
            f"{ag['n_exec']} execs (mean {_fmt(ag['exec_mean_s'], 's')}); "
            f"tokens in/out {ag['total_input_tokens']}/{ag['total_output_tokens']}"
        )
    u = summary.get("utilization", {})
    if u.get("polls"):
        lines.append(
            f"sandboxes: peak {u['sandbox_peak']}, mean "
            f"{_fmt(u['sandbox_mean'])} / {u['node_count']} nodes "
            f"(mean avail {_fmt(u['available_mean'])}); {u['polls']} polls"
        )
    return "\n".join(lines)

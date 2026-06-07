"""Timeline + machine-utilization observability for the SWE-bench harbor run.

Pure helpers (span/util/summary shaping) sit at the top; the I/O-bearing
``JobObserver`` and ``run_job_with_observability`` orchestrate harbor hooks, the
``/status`` poller, and artifact writing. Every orchestration path is
best-effort: it may degrade (skip a poll, drop spans, write a partial file) but
must never raise into a harbor trial. See
``docs/superpowers/specs/2026-06-07-swebench-observability-design.md``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
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


def _nested(d: dict, *path: str) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _ts_to_iso(ts: Any) -> Optional[str]:
    if isinstance(ts, str):
        return ts
    if isinstance(ts, (int, float)):
        # Heuristic: values past ~year 5138 in seconds are really epoch millis.
        secs = ts / 1000.0 if ts > 1e11 else float(ts)
        try:
            return datetime.fromtimestamp(secs, tz=timezone.utc).isoformat()
        except (ValueError, OSError, OverflowError):
            return None
    return None


def parse_pi_token_spans(text: str, *, trial: str = "", task: str = "") -> list[dict]:
    """Per-request token ``llm_call`` spans from a pi ``--mode json`` JSONL log.

    Defensive: each line is one JSON event; any object whose ``usage`` carries a
    numeric ``input`` or ``output`` becomes one span. pi's usage is shaped
    ``{input, output, cacheRead, cacheWrite, cost:{total}}`` with a sibling
    ``timestamp``. Unknown/renamed fields degrade to ``None`` and never raise.
    """
    spans: list[dict] = []
    seq = 0
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        usage = obj.get("usage") if isinstance(obj, dict) else None
        if not isinstance(usage, dict):
            continue
        i, o = usage.get("input"), usage.get("output")
        if not isinstance(i, (int, float)) and not isinstance(o, (int, float)):
            continue
        cache = 0
        for k in ("cacheRead", "cacheWrite"):
            v = usage.get(k)
            if isinstance(v, (int, float)):
                cache += int(v)
        cost = _nested(usage, "cost", "total")
        ts_iso = _ts_to_iso(obj.get("timestamp"))
        seq += 1
        spans.append(_span(
            trial, task, "llm_call", "pi_llm_call",
            start=ts_iso, end=ts_iso, seq=seq,
            n_input_tokens=int(i) if isinstance(i, (int, float)) else None,
            n_output_tokens=int(o) if isinstance(o, (int, float)) else None,
            n_cache_tokens=cache or None,
            cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
        ))
    return spans


def _trial_from_path(path: Path, job_dir: Path) -> str:
    try:
        return path.relative_to(job_dir).parts[0]
    except Exception:
        return ""


def merge_span_files(job_dir: Any) -> list[dict]:
    """Recursively read every ``timeline-spans.jsonl`` under ``job_dir``.

    Agents write spans under their ``logs_dir`` (nested in the trial dir); the
    first path component under ``job_dir`` is the trial name, applied only when a
    span does not already carry one.
    """
    job_dir = Path(job_dir)
    spans: list[dict] = []
    for f in sorted(job_dir.rglob("timeline-spans.jsonl")):
        trial = _trial_from_path(f, job_dir)
        try:
            text = f.read_text()
        except Exception:
            continue
        for raw in text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                sp = json.loads(raw)
            except Exception:
                continue
            if not sp.get("trial"):
                sp["trial"] = trial
            spans.append(sp)
    return spans


def _reward_of(result: Any) -> Optional[float]:
    vr = getattr(result, "verifier_result", None)
    rewards = getattr(vr, "rewards", None) if vr is not None else None
    if not rewards:
        return None
    for key in ("reward", "default"):
        if key in rewards:
            return float(rewards[key])
    return float(next(iter(rewards.values())))


_TRAJECTORY_GLOBS = ("*.trajectory.json", "pi-*.log",
                     "**/*.trajectory.json", "**/pi-*.log")


def trajectory_manifest_rows(results: list, job_dir: Any) -> list[dict]:
    """One ``trajectories.jsonl`` row per trial: reward, tokens, and the relative
    paths of that trial's trajectory artifacts (indexed, not copied)."""
    job_dir = Path(job_dir)
    rows: list[dict] = []
    for r in results:
        reward = _reward_of(r)
        try:
            n_in, n_cache, n_out, cost = r.compute_token_cost_totals()
        except Exception:
            n_in = n_cache = n_out = cost = None
        meta = {}
        ar = getattr(r, "agent_result", None)
        if ar is not None and getattr(ar, "metadata", None):
            meta = ar.metadata
        trial = getattr(r, "trial_name", "") or ""
        paths: set[str] = set()
        tdir = job_dir / trial
        if tdir.exists():
            for pat in _TRAJECTORY_GLOBS:
                for p in tdir.glob(pat):
                    try:
                        paths.add(str(p.relative_to(job_dir)))
                    except Exception:
                        paths.add(str(p))
        rows.append({
            "trial": trial,
            "task": getattr(r, "task_name", "") or "",
            "reward": reward,
            "passed": reward is not None and reward >= 1.0,
            "exit_status": meta.get("exit_status"),
            "n_calls": meta.get("n_calls"),
            "n_actions": meta.get("n_actions"),
            "n_input_tokens": n_in,
            "n_output_tokens": n_out,
            "n_cache_tokens": n_cache,
            "cost_usd": cost,
            "trajectory_paths": sorted(paths),
        })
    return rows


def _write_jsonl(path: Any, rows: list[dict]) -> None:
    """Write rows as JSONL; best-effort (log + return on failure)."""
    try:
        with Path(path).open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
    except Exception as e:  # pragma: no cover - fs failure
        log.warning("failed to write %s: %s", path, e)


class JobObserver:
    """Collects harbor trial-hook events + ``/status`` utilization for one job,
    then writes ``timeline.jsonl`` / ``utilization.jsonl`` / ``trajectories.jsonl``
    and returns a printable summary. Every method is best-effort."""

    def __init__(self, flash_url: Optional[str], interval: float = 5.0):
        self._flash_url = flash_url
        self._interval = float(interval)
        self._results: list = []          # harbor TrialResult objects (END hook)
        self._events: list[dict] = []     # lightweight progress log
        self._util_rows: list[dict] = []
        self._client: Any = None
        self._poll_task: Any = None
        self._t0: float = 0.0

    # ---- harbor hooks ------------------------------------------------- #

    def attach(self, job: Any) -> "JobObserver":
        async def on_event(ev: Any) -> None:
            try:
                self._events.append({
                    "event": getattr(ev.event, "value", str(ev.event)),
                    "trial": ev.trial_id, "task": ev.task_name,
                    "timestamp": ev.timestamp.isoformat(),
                })
            except Exception:
                pass

        async def on_end(ev: Any) -> None:
            try:
                await on_event(ev)
                if getattr(ev, "result", None) is not None:
                    self._results.append(ev.result)
            except Exception:
                pass

        try:
            job.on_trial_started(on_event)
            job.on_environment_started(on_event)
            job.on_agent_started(on_event)
            job.on_verification_started(on_event)
            job.on_trial_cancelled(on_event)
            job.on_trial_ended(on_end)
        except Exception as e:  # pragma: no cover
            log.warning("failed to attach trial hooks: %s", e)
        return self

    # ---- utilization poller ------------------------------------------ #

    @asynccontextmanager
    async def utilization(self):
        await self._start_poller()
        try:
            yield
        finally:
            await self._stop_poller()

    async def _start_poller(self) -> None:
        if not self._flash_url:
            log.warning("no FLASH_SANDBOX_URL; utilization polling disabled")
            return
        try:
            from flash_sandbox import AsyncHTTPClient
        except Exception:
            log.warning("flash-sandbox not importable; utilization polling disabled")
            return
        self._client = AsyncHTTPClient(address=self._flash_url, timeout=30.0)
        self._t0 = time.monotonic()
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        fails = 0
        warned = False
        while True:
            try:
                status = await self._client.cluster_status()
                self._util_rows.append(util_row_from_status(
                    status, time.monotonic() - self._t0, datetime.now(timezone.utc)))
                fails = 0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                fails += 1
                if fails >= 3 and not warned:
                    log.warning("utilization /status polling failing: %s", e)
                    warned = True
                else:
                    log.debug("utilization poll failed: %s", e)
            await asyncio.sleep(self._interval)

    async def _stop_poller(self) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._poll_task = None
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None

    # ---- artifacts ---------------------------------------------------- #

    def write(self, job_dir: Any) -> str:
        """Write the three artifacts and return the printable summary text."""
        job_dir = Path(job_dir)
        spans: list[dict] = []
        for r in self._results:
            try:
                spans.extend(phase_spans_from_result(r))
            except Exception:
                pass
        spans.extend(merge_span_files(job_dir))
        for f in sorted(job_dir.rglob("pi-*.log")):
            trial = _trial_from_path(f, job_dir)
            try:
                spans.extend(parse_pi_token_spans(f.read_text(), trial=trial))
            except Exception:
                pass

        _write_jsonl(job_dir / "timeline.jsonl", spans)
        _write_jsonl(job_dir / "utilization.jsonl", self._util_rows)
        try:
            manifest = trajectory_manifest_rows(self._results, job_dir)
        except Exception:
            manifest = []
        _write_jsonl(job_dir / "trajectories.jsonl", manifest)

        return format_summary(summarize(spans, self._util_rows))


@asynccontextmanager
async def _nullcontext():
    yield


async def run_job_with_observability(
    job_config: Any, *, flash_url: Optional[str], util_interval: float = 5.0,
    enabled: bool = True,
) -> tuple:
    """Create + run a harbor Job, optionally wrapped in observability.

    Returns ``(job, job_result, summary_text)``; ``summary_text`` is None when
    disabled. The caller still prints harbor's own accuracy table.
    """
    from harbor.job import Job

    job = await Job.create(job_config)
    observer = JobObserver(flash_url, util_interval) if enabled else None
    if observer is not None:
        observer.attach(job)

    cm = observer.utilization() if observer is not None else _nullcontext()
    async with cm:
        job_result = await job.run()

    summary_text = None
    if observer is not None:
        try:
            summary_text = observer.write(job.job_dir)
        except Exception as e:
            log.warning("observability write failed: %s", e)
    return job, job_result, summary_text

"""Record and replay of request traces for deterministic reproduction.

A *trace* is a JSONL file where each line captures one fired request:

    {"t_rel": 0.123, "method": "POST", "url": "...", "headers": {...},
     "params": {...}, "json": {...}, "timeout_s": null, "meta": {...}}

`t_rel` is the time (seconds) between bench start and the moment the request
went out — captured *after* pre-hooks, *before* transport. Binary bodies are
stored under `body_b64` (base64). The recorder is a runner-level feature; the
replay side composes through the standard `workload_type + workload + load`
triple:

  * `ReplayWorkloadType` — `make_request` rebuilds a `Request` from a row.
  * `TraceWorkload`      — yields rows in order.
  * `TracePacedLoad`     — emits one ticket at each row's `t_rel`
    (optionally sped up / slowed down by `speed`).

`build_config` wires all three from a single `replay:` block, so reproducing a
run is just `replay: {path: ..., speed: 1.0}` plus the same `correctness:` (if
grading is desired).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Any, Optional

from benchmaker.load import LoadModel
from benchmaker.types import Request
from benchmaker.workloads.base import WorkloadType
from benchmaker.workloads.datasets import Workload


# --------------------------------------------------------------------------- #
# Wire format
# --------------------------------------------------------------------------- #


def _safe_meta(meta: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in meta.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = repr(v)
    return out


def request_to_row(t_rel: float, req: Request) -> dict:
    """Encode a `Request` as a JSON-serializable trace row."""
    row: dict[str, Any] = {
        "t_rel": float(t_rel),
        "method": req.method,
        "url": req.url,
        "headers": dict(req.headers or {}),
        "params": dict(req.params or {}),
        "timeout_s": req.timeout_s,
        "meta": _safe_meta(req.meta or {}),
    }
    if req.json is not None:
        row["json"] = req.json
    elif req.body is not None:
        row["body_b64"] = base64.b64encode(req.body).decode("ascii")
    return row


def row_to_request(row: dict) -> Request:
    """Rebuild a `Request` from a trace row."""
    body: Optional[bytes] = None
    if "body_b64" in row:
        body = base64.b64decode(row["body_b64"])
    return Request(
        method=row.get("method", "GET"),
        url=row.get("url", ""),
        headers=dict(row.get("headers") or {}),
        params=dict(row.get("params") or {}),
        body=body,
        json=row.get("json"),
        timeout_s=row.get("timeout_s"),
        meta=dict(row.get("meta") or {}),
    )


def load_trace(path: str) -> list[dict]:
    """Load a trace JSONL into a list of rows, sorted by `t_rel`."""
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: float(r.get("t_rel", 0.0)))
    return rows


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #


class TraceRecorder:
    """Append `(t_rel, request)` rows to a JSONL file as the bench runs.

    The runner calls `open()` once at the start, `record(req, fire_mono)` per
    fired request, and `close()` once at the end. `t_rel` is computed as
    `fire_mono - start_mono`; `start_mono` is captured by `open()` (so it
    matches whatever clock the runner's progress logger uses).
    """

    def __init__(self, path: str):
        self._path = path
        self._f = None
        self._lock = asyncio.Lock()
        self._start_mono: Optional[float] = None

    def open(self, start_mono: Optional[float] = None) -> None:
        if self._f is not None:
            return
        d = os.path.dirname(os.path.abspath(self._path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._f = open(self._path, "w")
        self._start_mono = start_mono if start_mono is not None else time.monotonic()

    async def record(self, req: Request, fire_mono: float) -> None:
        if self._f is None:
            self.open()
        assert self._start_mono is not None
        row = request_to_row(fire_mono - self._start_mono, req)
        line = json.dumps(row, default=str)
        async with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None


# --------------------------------------------------------------------------- #
# Replay: workload + workload-type + load model
# --------------------------------------------------------------------------- #


class TraceWorkload(Workload):
    """Yield trace rows (dicts) in order. Halts when the trace is exhausted."""

    def __init__(self, trace: list[dict], name: str = "trace"):
        self.name = name
        self._trace = list(trace)
        self._i = 0

    async def next_item(self) -> Any:
        if self._i >= len(self._trace):
            raise StopAsyncIteration
        row = self._trace[self._i]
        self._i += 1
        return row


class ReplayWorkloadType(WorkloadType):
    """Rebuild a `Request` directly from a trace row.

    `streaming` is set per-instance — match the workload-type used during
    recording so the runner reads chunks the same way (otherwise stream-only
    metrics like TTFT won't be reproduced).
    """

    # Recorded Requests already carry their `reference` (and any other meta the
    # original workload put there) under `meta`. So apply_correctness should
    # install only the post-hook — no EvalWorkloadType item-splitting wrapper.
    handles_reference = True

    def __init__(self, streaming: bool = False, name: str = "replay"):
        self.name = name
        self.streaming = streaming

    async def make_request(self, item: Any) -> Request:
        if not isinstance(item, dict):
            raise TypeError(
                f"ReplayWorkloadType expects trace rows (dict), got {type(item).__name__}"
            )
        return row_to_request(item)


class TracePacedLoad(LoadModel):
    """Open-loop load model that fires one ticket per row at the recorded time.

    `speed` rescales the trace: `2.0` halves all inter-arrival gaps; `0.5`
    doubles them. Use `1.0` for an as-recorded replay.
    """

    def __init__(self, trace: list[dict], speed: float = 1.0):
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self._times = [float(r.get("t_rel", 0.0)) for r in trace]
        self._times.sort()
        self._speed = float(speed)

    async def tickets(self):
        start = time.monotonic()
        for t_rel in self._times:
            target = start + t_rel / self._speed
            now = time.monotonic()
            if target > now:
                await asyncio.sleep(target - now)
            yield None

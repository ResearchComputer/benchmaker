"""Load models.

A LoadModel yields admission "tickets" — each ticket means "fire one request
now". Open-loop models yield based on a target arrival schedule; closed-loop
models yield based on completions (the runner returns a ticket to the model
when a request finishes).
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Union


class LoadModel(ABC):
    """Drives the timing of request admissions."""

    @abstractmethod
    async def tickets(self) -> AsyncIterator[None]:
        """Yield None each time a request should be admitted."""
        ...

    def on_complete(self) -> None:
        """Called by the runner when a request completes. Closed-loop uses this."""
        pass

    @property
    def is_open_loop(self) -> bool:
        return True


# -------- Open-loop --------

class ConstantRPS(LoadModel):
    """Fire requests at a constant target rate (rps), regardless of in-flight count."""

    def __init__(self, rps: float, duration_s: Optional[float] = None,
                 max_requests: Optional[int] = None):
        if rps <= 0:
            raise ValueError("rps must be > 0")
        self.rps = rps
        self.duration_s = duration_s
        self.max_requests = max_requests

    async def tickets(self):
        interval = 1.0 / self.rps
        start = time.monotonic()
        next_fire = start
        n = 0
        while True:
            now = time.monotonic()
            if self.duration_s is not None and (now - start) >= self.duration_s:
                return
            if self.max_requests is not None and n >= self.max_requests:
                return
            sleep_for = next_fire - now
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            yield None
            n += 1
            next_fire += interval


class PoissonRPS(LoadModel):
    """Open-loop Poisson arrivals with mean rate `rps`."""

    def __init__(self, rps: float, duration_s: Optional[float] = None,
                 max_requests: Optional[int] = None, seed: Optional[int] = None):
        if rps <= 0:
            raise ValueError("rps must be > 0")
        self.rps = rps
        self.duration_s = duration_s
        self.max_requests = max_requests
        self._rng = random.Random(seed)

    async def tickets(self):
        start = time.monotonic()
        n = 0
        while True:
            now = time.monotonic()
            if self.duration_s is not None and (now - start) >= self.duration_s:
                return
            if self.max_requests is not None and n >= self.max_requests:
                return
            # Exponential inter-arrival time with mean 1/rps.
            gap = self._rng.expovariate(self.rps)
            await asyncio.sleep(gap)
            yield None
            n += 1


# -------- Closed-loop --------

class ClosedLoop(LoadModel):
    """N concurrent workers; each fires the next request as soon as the previous
    completes. Total in-flight is bounded by `concurrency`.

    This is implemented as: yield up to `concurrency` tickets initially, then
    one more for every completion the runner reports.
    """

    def __init__(self, concurrency: int, duration_s: Optional[float] = None,
                 max_requests: Optional[int] = None):
        if concurrency <= 0:
            raise ValueError("concurrency must be > 0")
        self.concurrency = concurrency
        self.duration_s = duration_s
        self.max_requests = max_requests
        self._sem = asyncio.Semaphore(concurrency)
        self._completions: Optional[asyncio.Queue] = None

    @property
    def is_open_loop(self) -> bool:
        return False

    def on_complete(self) -> None:
        if self._completions is not None:
            self._completions.put_nowait(None)

    async def tickets(self):
        self._completions = asyncio.Queue()
        # Seed: fire N concurrent immediately.
        for _ in range(self.concurrency):
            self._completions.put_nowait(None)

        start = time.monotonic()
        n = 0
        while True:
            now = time.monotonic()
            if self.duration_s is not None and (now - start) >= self.duration_s:
                return
            if self.max_requests is not None and n >= self.max_requests:
                return
            await self._completions.get()
            yield None
            n += 1


# -------- Composite: Sweep / Ramp --------

@dataclass
class _Stage:
    model: LoadModel
    label: str


class Sweep(LoadModel):
    """Run a sequence of sub-load-models in order (each for its own duration).

    Useful for: sweep across RPS values to find saturation, e.g.
        Sweep([ConstantRPS(10, 30), ConstantRPS(50, 30), ConstantRPS(100, 30)])
    """

    def __init__(self, stages: list[LoadModel], labels: Optional[list[str]] = None):
        if not stages:
            raise ValueError("Sweep requires at least one stage")
        self.stages = stages
        self.labels = labels or [f"stage_{i}" for i in range(len(stages))]
        self.current_label: Optional[str] = None

    @property
    def is_open_loop(self) -> bool:
        # Mixed sweeps are treated as open-loop for runner-level scheduling.
        return all(s.is_open_loop for s in self.stages)

    def on_complete(self) -> None:
        for s in self.stages:
            s.on_complete()

    async def tickets(self):
        for label, stage in zip(self.labels, self.stages):
            self.current_label = label
            async for t in stage.tickets():
                yield t
        self.current_label = None


class Ramp(LoadModel):
    """Linearly ramp RPS from `start_rps` to `end_rps` over `duration_s`."""

    def __init__(self, start_rps: float, end_rps: float, duration_s: float,
                 poisson: bool = False, seed: Optional[int] = None):
        if start_rps <= 0 or end_rps <= 0 or duration_s <= 0:
            raise ValueError("start_rps, end_rps, duration_s must all be > 0")
        self.start_rps = start_rps
        self.end_rps = end_rps
        self.duration_s = duration_s
        self.poisson = poisson
        self._rng = random.Random(seed)

    def _rps_at(self, t: float) -> float:
        frac = min(max(t / self.duration_s, 0.0), 1.0)
        return self.start_rps + (self.end_rps - self.start_rps) * frac

    async def tickets(self):
        start = time.monotonic()
        next_fire = start
        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= self.duration_s:
                return
            rps = self._rps_at(elapsed)
            if self.poisson:
                gap = self._rng.expovariate(rps)
                await asyncio.sleep(gap)
                yield None
            else:
                sleep_for = next_fire - now
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                yield None
                next_fire += 1.0 / rps


# -------- User-friendly spec parser --------

_DURATION_RE = re.compile(r"^([0-9]*\.?[0-9]+)(ms|s|m|h)?$")


def parse_duration(s: Union[str, int, float]) -> float:
    """Parse '30s', '500ms', '2m', '1h', or a bare number (seconds)."""
    if isinstance(s, (int, float)):
        return float(s)
    m = _DURATION_RE.match(s.strip())
    if not m:
        raise ValueError(f"Cannot parse duration {s!r}")
    val = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    return val * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]


def parse_rate_spec(
    spec: Union[str, int, float, dict],
    duration_s: Optional[float] = None,
    max_requests: Optional[int] = None,
) -> LoadModel:
    """Friendly load-model factory.

    Accepted forms:
        100                              -> ConstantRPS(100)
        "100"                            -> ConstantRPS(100)
        "100rps"                         -> ConstantRPS(100)
        "poisson:100"                    -> PoissonRPS(100)
        "closed:32"  or "concurrency:32" -> ClosedLoop(32)
        "ramp:10..500:30s"               -> Ramp(10, 500, 30)
        "ramp-poisson:10..500:30s"       -> Ramp(..., poisson=True)
        "sweep:10,50,100,500@30s"        -> Sweep of ConstantRPS, each 30s
        {"type": "constant", "rps": 100, "duration": "60s"}   (dict form)
    """
    if isinstance(spec, dict):
        return _parse_rate_dict(spec)

    if isinstance(spec, (int, float)):
        return ConstantRPS(float(spec), duration_s=duration_s, max_requests=max_requests)

    s = spec.strip().lower()

    if s.startswith("poisson:"):
        rps = float(s.split(":", 1)[1])
        return PoissonRPS(rps, duration_s=duration_s, max_requests=max_requests)

    if s.startswith("closed:") or s.startswith("concurrency:"):
        n = int(s.split(":", 1)[1])
        return ClosedLoop(n, duration_s=duration_s, max_requests=max_requests)

    if s.startswith("ramp-poisson:") or s.startswith("ramp:"):
        poisson = s.startswith("ramp-poisson:")
        rest = s.split(":", 1)[1]
        # rest like "10..500:30s"
        rng, dur = rest.rsplit(":", 1)
        a, b = rng.split("..")
        return Ramp(float(a), float(b), parse_duration(dur), poisson=poisson)

    if s.startswith("sweep:"):
        rest = s.split(":", 1)[1]
        # e.g. "10,50,100,500@30s"
        if "@" in rest:
            vals, dur = rest.split("@", 1)
            per_stage = parse_duration(dur)
        else:
            vals = rest
            if duration_s is None:
                raise ValueError("sweep: needs '@duration' or a top-level duration")
            per_stage = duration_s / len(vals.split(","))
        rates = [float(v) for v in vals.split(",")]
        stages = [ConstantRPS(r, duration_s=per_stage) for r in rates]
        labels = [f"{r:g}rps" for r in rates]
        return Sweep(stages, labels)

    # Plain number with optional 'rps' suffix.
    if s.endswith("rps"):
        s = s[:-3]
    return ConstantRPS(float(s), duration_s=duration_s, max_requests=max_requests)


def _parse_rate_dict(d: dict) -> LoadModel:
    t = d.get("type", "constant").lower()
    duration = d.get("duration")
    if duration is not None and isinstance(duration, str):
        duration = parse_duration(duration)
    max_requests = d.get("max_requests")

    if t == "constant":
        return ConstantRPS(float(d["rps"]), duration_s=duration, max_requests=max_requests)
    if t == "poisson":
        return PoissonRPS(float(d["rps"]), duration_s=duration, max_requests=max_requests,
                          seed=d.get("seed"))
    if t in ("closed", "closed-loop", "concurrency"):
        return ClosedLoop(int(d["concurrency"]), duration_s=duration, max_requests=max_requests)
    if t == "ramp":
        return Ramp(float(d["start_rps"]), float(d["end_rps"]),
                    parse_duration(d.get("duration", duration)),
                    poisson=d.get("poisson", False), seed=d.get("seed"))
    if t == "sweep":
        stages_spec = d["stages"]
        stages = [parse_rate_spec(s) for s in stages_spec]
        labels = [s.get("label") if isinstance(s, dict) else None for s in stages_spec]
        labels = [lab or f"stage_{i}" for i, lab in enumerate(labels)]
        return Sweep(stages, labels)
    raise ValueError(f"Unknown load model type: {t}")

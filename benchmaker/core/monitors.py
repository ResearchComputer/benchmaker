"""Optional periodic monitors.

A `Monitor` runs alongside the benchmark and samples something external every
`interval_s`. Each tick returns a flat `dict[str, float]` of values; the runner
records them as a time-series and the aggregator summarizes them in the final
report.

Typical use cases:
    * scrape vLLM / SGLang `/metrics` (Prometheus) for queue depth, KV-cache
      utilization, throughput, etc.
    * sample GPU utilization (`pynvml`)
    * pull a Slurm/k8s queue depth
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional, Union

import httpx2


class Monitor(ABC):
    """Periodic side-channel sampler.

    Subclasses implement `tick()` which is called every `interval_s` seconds.
    Return a flat `{metric_name: float}` dict (or `None` to skip this tick).
    """

    name: str = "monitor"
    interval_s: float = 1.0
    tick_at_start: bool = True   # whether to fire one immediate tick at t=0

    async def setup(self) -> None:
        """Called once before the first tick. Use for opening sessions, etc."""

    @abstractmethod
    async def tick(self) -> Optional[dict[str, float]]:
        """Return one observation. Called every `interval_s` seconds."""

    async def aclose(self) -> None:
        """Called once after the last tick. Use for cleanup."""


class FunctionMonitor(Monitor):
    """Wrap a sync or async callable that returns a metrics dict.

    The callable receives no arguments. If it raises, the exception is logged
    once (to stderr) but does not kill the benchmark.
    """

    def __init__(
        self,
        fn: Callable[[], Union[Optional[dict[str, float]], Awaitable[Optional[dict[str, float]]]]],
        name: str = "fn",
        interval_s: float = 1.0,
        tick_at_start: bool = True,
    ):
        self._fn = fn
        self.name = name
        self.interval_s = interval_s
        self.tick_at_start = tick_at_start

    async def tick(self) -> Optional[dict[str, float]]:
        result = self._fn()
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[assignment]
        return result  # type: ignore[return-value]


class PrometheusMonitor(Monitor):
    """Scrape a Prometheus `/metrics` endpoint each tick.

    Args:
        url: full URL to the metrics endpoint.
        metric_names: optional set of metric names (without labels) to keep.
            If `None`, all metrics are recorded.
        labelled_keys: if True (default), series with labels are stored as
            `name{label="value"}`. If False, label info is dropped and series
            with identical names are summed.
        headers: HTTP headers (e.g. Authorization).
        interval_s: scrape interval.
    """

    def __init__(
        self,
        url: str,
        metric_names: Optional[set[str]] = None,
        labelled_keys: bool = True,
        headers: Optional[dict[str, str]] = None,
        interval_s: float = 1.0,
        name: str = "prometheus",
        tick_at_start: bool = True,
        timeout_s: float = 5.0,
    ):
        self._url = url
        self._names = set(metric_names) if metric_names else None
        self._labelled = labelled_keys
        self._headers = headers or {}
        self.interval_s = interval_s
        self.name = name
        self.tick_at_start = tick_at_start
        self._timeout = httpx2.Timeout(timeout_s)
        self._session: Optional[httpx2.AsyncClient] = None

    async def setup(self) -> None:
        self._session = httpx2.AsyncClient(timeout=self._timeout)

    async def tick(self) -> Optional[dict[str, float]]:
        assert self._session is not None
        try:
            r = await self._session.get(self._url, headers=self._headers)
            if r.status_code >= 400:
                return None
            text = r.text
        except (httpx2.HTTPError, asyncio.TimeoutError):
            return None
        return parse_prometheus(text, names=self._names, labelled_keys=self._labelled)

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.aclose()
            self._session = None


def parse_prometheus(
    text: str,
    names: Optional[set[str]] = None,
    labelled_keys: bool = True,
) -> dict[str, float]:
    """Minimal Prometheus text-format parser.

    Skips `# HELP` / `# TYPE` lines, comments, and malformed lines. Handles
    name + optional `{labels}` + value (timestamp ignored).
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Split `name[{labels}]` from value(+timestamp). Labels can contain
        # spaces inside quoted values, but the value field always follows the
        # closing brace (or the name itself) with whitespace. Use the position
        # of the last `}` if present, else the first space.
        if "{" in line:
            close = line.find("}")
            if close == -1:
                continue
            name_part = line[: close + 1]
            rest = line[close + 1:].strip()
        else:
            sp = line.find(" ")
            if sp == -1:
                continue
            name_part = line[:sp]
            rest = line[sp + 1:].strip()

        if not rest:
            continue
        value_token = rest.split()[0]
        try:
            value = float(value_token)
        except ValueError:
            continue

        bare = name_part.split("{", 1)[0]
        if names is not None and bare not in names:
            continue

        key = name_part if labelled_keys else bare
        if key in out and not labelled_keys:
            out[key] += value
        else:
            out[key] = value
    return out


async def run_monitor_loop(
    monitor: Monitor,
    samples: list[tuple[float, dict[str, float]]],
    start_mono: float,
    stop_event: asyncio.Event,
) -> None:
    """Drive a single monitor until `stop_event` is set.

    Records `(elapsed_s, values)` tuples into `samples`. Any tick exception is
    swallowed (logged to stderr) so monitor failure doesn't kill the bench.
    """
    import sys

    try:
        await monitor.setup()
    except Exception as e:
        sys.stderr.write(f"[monitor:{monitor.name}] setup failed: {e}\n")
        return

    try:
        if monitor.tick_at_start:
            await _safe_tick(monitor, samples, start_mono)

        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=monitor.interval_s)
                # If we get here, stop was signalled — do one final tick and exit.
                await _safe_tick(monitor, samples, start_mono)
                break
            except asyncio.TimeoutError:
                await _safe_tick(monitor, samples, start_mono)
    finally:
        try:
            await monitor.aclose()
        except Exception as e:
            sys.stderr.write(f"[monitor:{monitor.name}] aclose failed: {e}\n")


async def _safe_tick(monitor: Monitor, samples: list, start_mono: float) -> None:
    import sys
    try:
        values = await monitor.tick()
    except Exception as e:
        sys.stderr.write(f"[monitor:{monitor.name}] tick error: {e}\n")
        return
    if not values:
        return
    samples.append((time.monotonic() - start_mono, dict(values)))

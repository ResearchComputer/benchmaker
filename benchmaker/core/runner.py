"""BenchRunner: ties scheduler -> workload (dataset) -> workload-type ->
httpx2 async client -> metrics."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx2

logger = logging.getLogger(__name__)

from benchmaker.core.load import LoadModel
from benchmaker.core.metrics import MetricsAggregator
from benchmaker.core.types import (
    PostResponseHook,
    PreRequestHook,
    Request,
    Response,
    Sample,
    TicketContext,
    maybe_await,
)
from benchmaker.core.monitors import Monitor, run_monitor_loop
from benchmaker.core.trace import TraceRecorder
from benchmaker.workloads.base import WorkloadType
from benchmaker.workloads.datasets import Workload, StaticWorkload


@dataclass
class BenchLane:
    """One independently scheduled input lane in a mixed benchmark.

    All lanes share the enclosing :class:`BenchConfig`'s workload type and
    endpoint, but each owns its own data source and load model.  This keeps
    OpenAI/HTTP protocol configuration centralized while preserving independent
    arrival processes for phase-swing experiments.
    """

    name: str
    workload: Workload
    load: LoadModel

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("lane name must be a non-empty string")


@dataclass
class BenchConfig:
    workload_type: WorkloadType                # how to talk to the service
    load: Optional[LoadModel] = None           # when to fire (single workload)
    workload: Workload = field(default_factory=StaticWorkload)  # what to send
    lanes: list[BenchLane] = field(default_factory=list)
    pre_hooks: list[PreRequestHook] = field(default_factory=list)
    post_hooks: list[PostResponseHook] = field(default_factory=list)
    monitors: list[Monitor] = field(default_factory=list)  # optional periodic samplers
    # Optional trace recorder. When set, each fired request is appended to a
    # JSONL file (with relative timestamp) so a later run can replay the bench
    # deterministically via `benchmaker.core.trace.TracePacedLoad` + `ReplayWorkloadType`.
    recorder: Optional[TraceRecorder] = None
    connection_limit: int = 1000
    timeout_s: float = 60.0
    max_in_flight: int = 10000
    progress_every_s: float = 1.0
    stop_on_exhausted: bool = True

    def __post_init__(self) -> None:
        if self.lanes:
            if self.load is not None:
                raise ValueError("configure either load/workload or lanes, not both")
            names = [lane.name for lane in self.lanes]
            if len(names) != len(set(names)):
                raise ValueError("mixed benchmark lane names must be unique")
        elif self.load is None:
            raise ValueError("BenchConfig requires a load model or at least one lane")


@dataclass
class BenchResult:
    samples: list[Sample]
    summary: dict


class BenchRunner:
    def __init__(self, config: BenchConfig):
        self.cfg = config
        self.metrics = MetricsAggregator()

    async def run(self) -> BenchResult:
        # httpx2 has no per-connector force_close / ttl_dns_cache knobs; pooling
        # is governed by Limits. force_close=False (keepalive on) maps to
        # allowing up to `connection_limit` keep-alive connections, matching the
        # previous aiohttp pool behaviour. httpx2 resolves DNS through the
        # system resolver per connection (no in-process TTL cache).
        limits = httpx2.Limits(
            max_connections=self.cfg.connection_limit,
            max_keepalive_connections=self.cfg.connection_limit,
        )
        timeout = httpx2.Timeout(self.cfg.timeout_s)
        if self.cfg.recorder is not None:
            self.cfg.recorder.open(start_mono=self.metrics.start_time)
        async with httpx2.AsyncClient(limits=limits, timeout=timeout) as session:
            try:
                await self._drive(session)
            finally:
                await self._aclose_workloads()
                await self.cfg.workload_type.aclose()
                if self.cfg.recorder is not None:
                    self.cfg.recorder.close()
        self.metrics.finalize()
        return BenchResult(samples=self.metrics.samples, summary=self.metrics.summary())

    async def _drive(self, session: httpx2.AsyncClient) -> None:
        sem = asyncio.Semaphore(self.cfg.max_in_flight)
        tasks: set[asyncio.Task] = set()
        progress_task = asyncio.create_task(self._progress_loop())

        # Spawn monitor loops.
        monitor_stop = asyncio.Event()
        monitor_tasks: list[asyncio.Task] = []
        bench_start = self.metrics.start_time
        for mon in self.cfg.monitors:
            buf = self.metrics.monitor_buffer(mon.name)
            monitor_tasks.append(asyncio.create_task(
                run_monitor_loop(mon, buf, bench_start, monitor_stop)
            ))

        try:
            if self.cfg.lanes:
                await self._drive_lanes(session, sem, tasks)
            else:
                await self._drive_single(session, sem, tasks)
        finally:
            progress_task.cancel()
            try:
                await progress_task
            except (asyncio.CancelledError, Exception):
                pass

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Signal monitors to do one last tick and exit, then wait for them.
            monitor_stop.set()
            if monitor_tasks:
                await asyncio.gather(*monitor_tasks, return_exceptions=True)

    async def _drive_single(self, session: httpx2.AsyncClient,
                            sem: asyncio.Semaphore,
                            tasks: set[asyncio.Task]) -> None:
        assert self.cfg.load is not None
        async for _ in self.cfg.load.tickets():
            try:
                item = await self.cfg.workload.next_item()
            except StopAsyncIteration:
                if self.cfg.stop_on_exhausted:
                    break
                continue

            await sem.acquire()
            task = asyncio.create_task(
                self._fire(session, item, sem, self.cfg.load)
            )
            tasks.add(task)
            task.add_done_callback(tasks.discard)

    async def _drive_lanes(self, session: httpx2.AsyncClient,
                           sem: asyncio.Semaphore,
                           tasks: set[asyncio.Task]) -> None:
        """Run each lane's admission iterator concurrently.

        A finite workload ends only its own lane.  Other lanes keep producing
        tickets, which is required when one dataset is short or intentionally
        bursty and another drives a long complementary phase.
        """

        async def produce(lane: BenchLane) -> None:
            async for _ in lane.load.tickets():
                try:
                    item = await lane.workload.next_item()
                except StopAsyncIteration:
                    if self.cfg.stop_on_exhausted:
                        break
                    continue

                await sem.acquire()
                task = asyncio.create_task(
                    self._fire(session, item, sem, lane.load, lane.name)
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)

        producers = [asyncio.create_task(produce(lane)) for lane in self.cfg.lanes]
        try:
            await asyncio.gather(*producers)
        finally:
            for producer in producers:
                if not producer.done():
                    producer.cancel()
            await asyncio.gather(*producers, return_exceptions=True)

    async def _fire(self, session: httpx2.AsyncClient, item: Any,
                    sem: asyncio.Semaphore, load: LoadModel,
                    lane_name: Optional[str] = None) -> None:
        start_mono = time.monotonic()
        try:
            async def fire(req: Request) -> Response:
                if lane_name is not None:
                    # The config-defined lane is authoritative: callers may
                    # still attach arbitrary metadata, but cannot accidentally
                    # collapse a mixed run into an incorrect lane.
                    req.meta["lane"] = lane_name
                for hook in self.cfg.pre_hooks:
                    req = await maybe_await(hook(req))
                fire_start = time.monotonic()
                if self.cfg.recorder is not None:
                    await self.cfg.recorder.record(req, fire_start)
                return await self._execute(session, req, fire_start)

            ctx = TicketContext(
                item=item,
                start_mono=start_mono,
                fire=fire,
                pre_hooks=tuple(self.cfg.pre_hooks),
                post_hooks=tuple(self.cfg.post_hooks),
                workload_name=self.cfg.workload_type.name,
            )
            sample = await self.cfg.workload_type.run_ticket(ctx)
            if lane_name is not None:
                sample.meta["lane"] = lane_name
            self.metrics.add(sample)
        except Exception as e:
            self.metrics.add(_failure_sample(
                f"{type(e).__name__}: {e}",
                self.cfg.workload_type.name,
                lane_name,
            ))
        finally:
            sem.release()
            load.on_complete()

    async def _aclose_workloads(self) -> None:
        workloads = (
            [lane.workload for lane in self.cfg.lanes]
            if self.cfg.lanes
            else [self.cfg.workload]
        )
        closed: set[int] = set()
        for workload in workloads:
            if id(workload) not in closed:
                closed.add(id(workload))
                await workload.aclose()

    async def _execute(self, session: httpx2.AsyncClient, req: Request,
                       start_mono: float) -> Response:
        kwargs: dict = {"headers": req.headers, "params": req.params}
        if req.json is not None:
            kwargs["json"] = req.json
        elif req.body is not None:
            kwargs["content"] = req.body
        if req.timeout_s is not None:
            kwargs["timeout"] = httpx2.Timeout(req.timeout_s)

        try:
            if self.cfg.workload_type.streaming:
                # Stream so each network read surfaces as a separate chunk,
                # preserving per-chunk timing (TTFT/ITL). httpx2's aiter_raw()
                # yields bytes at arbitrary boundaries exactly like aiohttp's
                # content.iter_any(), so the SSE reassembler in workloads/_sse.py
                # still rejoins events split across chunks (#13).
                chunks: list[bytes] = []
                chunk_times: list[float] = []
                async with session.stream(req.method, req.url, **kwargs) as resp:
                    async for chunk in resp.aiter_raw():
                        chunks.append(chunk)
                        chunk_times.append(time.monotonic() - start_mono)
                    body = b"".join(chunks)
                    elapsed = time.monotonic() - start_mono
                    return Response(
                        status=resp.status_code,
                        headers=dict(resp.headers),
                        body=body,
                        elapsed_s=elapsed,
                        ok=200 <= resp.status_code < 400,
                        stream_chunks=chunks,
                        stream_chunk_times=chunk_times,
                    )
            else:
                resp = await session.request(req.method, req.url, **kwargs)
                body = resp.content
                elapsed = time.monotonic() - start_mono
                return Response(
                    status=resp.status_code,
                    headers=dict(resp.headers),
                    body=body,
                    elapsed_s=elapsed,
                    ok=200 <= resp.status_code < 400,
                )
        except asyncio.TimeoutError:
            return Response(
                status=0, headers={}, body=b"",
                elapsed_s=time.monotonic() - start_mono,
                ok=False, error="timeout",
            )
        except httpx2.TimeoutException:
            return Response(
                status=0, headers={}, body=b"",
                elapsed_s=time.monotonic() - start_mono,
                ok=False, error="timeout",
            )
        except httpx2.HTTPError as e:
            return Response(
                status=0, headers={}, body=b"",
                elapsed_s=time.monotonic() - start_mono,
                ok=False, error=f"{type(e).__name__}: {e}",
            )

    async def _progress_loop(self) -> None:
        if self.cfg.progress_every_s <= 0:
            return
        last_n = 0
        last_t = time.monotonic()
        seen_errors: set[str] = set()
        try:
            while True:
                await asyncio.sleep(self.cfg.progress_every_s)
                now = time.monotonic()
                n = len(self.metrics.samples)
                dn = n - last_n
                dt = now - last_t
                inst = dn / dt if dt > 0 else 0.0
                window = self.metrics.samples[last_n:]
                ok = sum(1 for s in window if s.ok)
                # Wrong: request was delivered (HTTP success) but a post-hook /
                # workload graded the output as a failure (e.g. eval gate).
                wrong = sum(1 for s in window if not s.ok and s.request_ok)
                fail = dn - ok - wrong
                logger.info(
                    "+%5d req  (%7.1f rps, %d ok, %d wrong, %d fail) | total=%d",
                    dn, inst, ok, wrong, fail, n,
                )
                # Surface the first occurrence of each distinct error string —
                # one short line per kind, so failed runs are diagnosable
                # without grepping samples.jsonl.
                for s in window:
                    if s.error and s.error not in seen_errors:
                        seen_errors.add(s.error)
                        msg = s.error if len(s.error) <= 200 else s.error[:200] + "..."
                        bucket = "fail" if not s.request_ok else "wrong"
                        logger.warning("  first %s: %s", bucket, msg)
                last_n = n
                last_t = now
        except asyncio.CancelledError:
            return

    def write_bundle(self, out_dir: str, **kwargs) -> str:
        """Write a per-run directory bundle. See `benchmaker.io.bundle.write_bundle`."""
        from benchmaker.io.bundle import write_bundle
        return write_bundle(
            out_dir,
            self.metrics,
            workload_type_name=self.cfg.workload_type.name,
            workload_name=(
                "mix:" + ",".join(lane.name for lane in self.cfg.lanes)
                if self.cfg.lanes else self.cfg.workload.name
            ),
            **kwargs,
        )


def _failure_sample(error: str, workload: str,
                    lane_name: Optional[str] = None) -> Sample:
    return Sample(
        start_ts=time.monotonic(),
        latency_s=0.0,
        status=0,
        ok=False,
        request_ok=False,
        error=error,
        workload=workload,
        meta={"lane": lane_name} if lane_name is not None else {},
    )


async def run_bench(config: BenchConfig) -> BenchResult:
    return await BenchRunner(config).run()

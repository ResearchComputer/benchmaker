"""Workload-type protocol.

A `WorkloadType` knows how to talk to a specific *kind* of HTTP service
(generic HTTP, OpenAI chat completions, …). It does NOT own the inputs —
inputs come from a `Workload` (see `benchmaker.workloads.datasets`), which
yields opaque per-request items.

Pairing:
    bench = WorkloadType("how to talk") + Workload("what to send") + LoadModel("when")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from benchmaker.types import Request, Response, Sample, TicketContext, maybe_await


class WorkloadType(ABC):
    """A request protocol/shape.

    Subclasses turn a workload item (any object yielded by a `Workload`) into a
    `Request`, and the resulting `Response` into a `Sample`. Workload-specific
    metrics (TTFT, tokens/sec, …) are attached via `Sample.extra`.
    """

    name: str = "workload-type"
    streaming: bool = False  # if True, runner reads the response chunk-by-chunk

    @abstractmethod
    async def make_request(self, item: Any) -> Request:
        """Build a `Request` from a workload item.

        `item` may be `None` if the workload yields placeholders (e.g. a fixed-
        request benchmark with no per-request data).
        """

    async def make_sample(self, item: Any, request: Request, response: Response,
                          start_ts: float) -> Sample:
        """Build a Sample. Override to attach workload-type-specific metrics."""
        return Sample(
            start_ts=start_ts,
            latency_s=response.elapsed_s,
            status=response.status,
            ok=response.ok,
            request_ok=response.ok,
            bytes_recv=len(response.body) if response.body else 0,
            bytes_sent=_estimate_request_size(request),
            error=response.error,
            workload=self.name,
            meta=dict(request.meta),
        )

    async def run_ticket(self, ctx: TicketContext) -> Sample:
        """Default ticket flow: one make_request → one fire → one make_sample.

        Override to implement multi-step protocols (e.g. create → exec → delete).
        Use `ctx.fire(req)` to issue requests; it applies pre-hooks and uses
        the shared aiohttp session.
        """
        req = await self.make_request(ctx.item)
        resp = await ctx.fire(req)
        sample = await self.make_sample(ctx.item, req, resp, ctx.start_mono)
        for hook in ctx.post_hooks:
            sample = await maybe_await(hook(req, resp, sample))
        return sample

    async def aclose(self) -> None:
        return None


def _estimate_request_size(req: Request) -> int:
    if req.body is not None:
        return len(req.body)
    if req.json is not None:
        import json as _json
        return len(_json.dumps(req.json).encode("utf-8"))
    return 0

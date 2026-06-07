"""User-defined Agent workload-type.

Drives an arbitrary Python class — typically a multi-turn agent that makes its
own HTTP calls (to a model API, tools, a sandbox, …) — and grades the final
output via the standard `correctness_hook`.

Compared to the LLM workload-types:

  * No request shape is dictated. The agent owns its own client(s).
  * One "request" = one full agent run, which may be many internal calls.
  * Per-trajectory metrics (steps, tool calls, tokens, retries, …) ride on
    `Sample.extra` via `AgentResult.metrics`.

The expected item shape mirrors the LLM workloads: a dict with at least a
``prompt`` field (passed to the agent as the task), an optional ``reference``
(consumed by `correctness_hook`), and any extra columns the dataset carries.
Items are passed through to the agent verbatim — the workload-type only peels
``reference`` (and any ``extra_meta_keys``) into ``Sample.meta`` so the grader
can find them.

Example (subclassing):

    from benchmaker import Agent, AgentContext, AgentResult

    class MyAgent(Agent):
        def __init__(self, model: str):
            self.model = model

        async def run(self, ctx: AgentContext) -> AgentResult:
            task = ctx.item["prompt"]
            output, steps = await my_pipeline(task, model=self.model)
            return AgentResult(
                output=output,
                ok=True,
                metrics={"steps": float(steps)},
            )

YAML:

    workload_type:
      type: agent
      agent: 'mypkg.myagent:MyAgent'
      agent_kwargs:
        model: 'gpt-4o-mini'
"""

from __future__ import annotations

import inspect
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from benchmaker.core.types import (
    FireRequest,
    Request,
    Response,
    Sample,
    TicketContext,
    maybe_await,
)
from benchmaker.workloads.base import WorkloadType


@dataclass
class AgentContext:
    """Per-task state handed to ``Agent.run``.

    ``fire`` is the runner's request-firing callable. When non-None, agents
    can route their own HTTP calls through benchmaker's session + hook
    pipeline instead of using a separate client — handy if you want the
    runner's pre/post hooks (auth headers, request tracing) on every internal
    call. Most agents will simply ignore it.
    """

    item: Any
    workload_name: str = "agent"
    fire: Optional[FireRequest] = None
    start_mono: float = 0.0


@dataclass
class AgentResult:
    """What ``Agent.run`` returns for one task.

    - ``output`` is the text the grader sees (``extract_text`` returns it as-is).
    - ``ok`` is whether the agent considers this trajectory a success
      (typically: did it submit?). May be flipped to False by a correctness
      post-hook when the submission is wrong.
    - ``request_ok`` is whether the agent ran to completion without an
      *infrastructure* failure (model endpoint reachable, no Python exceptions,
      …). Default True. When False the sample is bucketed as "fail" by the
      progress logger and summary; when True but ``ok`` ends up False it's
      bucketed as "wrong" — i.e. "agent delivered an answer, but it isn't
      the right one." Crashed agents are auto-set to ``request_ok=False`` by
      the workload-type (exceptions caught in ``run_ticket``).
    - ``metrics`` lands in ``Sample.extra`` (numeric values only).
    - ``meta`` lands in ``Sample.meta`` (arbitrary JSON-serializable values).
    """

    output: str = ""
    ok: bool = True
    error: Optional[str] = None
    request_ok: bool = True
    metrics: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    bytes_sent: int = 0
    bytes_recv: int = 0


class Agent(ABC):
    """Base class for user-provided agents.

    Subclasses can take whatever ``__init__`` kwargs they need; the
    workload-type instantiates the class once and reuses the same instance
    across tickets, so any expensive setup (loading tools, opening sessions)
    happens once. Override ``aclose`` to release resources.
    """

    @abstractmethod
    async def run(self, ctx: AgentContext) -> AgentResult:
        """Handle one task; return its final result."""

    async def aclose(self) -> None:
        return None


class CallableAgent(Agent):
    """Adapter wrapping a sync/async ``(AgentContext) -> AgentResult|dict|str``."""

    def __init__(self, fn: Callable[[AgentContext], Any], name: str = "callable"):
        self._fn = fn
        self._name = name

    async def run(self, ctx: AgentContext) -> AgentResult:
        result = self._fn(ctx)
        result = await maybe_await(result)
        if isinstance(result, AgentResult):
            return result
        if isinstance(result, dict):
            return AgentResult(**result)
        if isinstance(result, str):
            return AgentResult(output=result)
        raise TypeError(
            f"CallableAgent fn must return AgentResult|dict|str, "
            f"got {type(result).__name__}"
        )


class AgentWorkloadType(WorkloadType):
    """Run a user-provided ``Agent`` per ticket.

    Args:
        agent: an ``Agent`` instance, an ``Agent`` subclass (instantiated with
            ``agent_kwargs``), or a plain callable wrapped in ``CallableAgent``.
        agent_kwargs: forwarded to the class constructor when ``agent`` is a class.
        reference_key: item key carrying the gold reference. Copied into
            ``Sample.meta`` (and the synthetic ``Request.meta``) so the standard
            ``correctness_hook`` finds it. Mirrors ``EvalWorkloadType``.
        extra_meta_keys: additional item keys to copy verbatim into
            ``Sample.meta`` (e.g. ``task_id``).
        name: workload-type name (default ``"agent"``).
    """

    streaming = False
    # We split `reference` out of items ourselves, so apply_correctness should
    # NOT wrap us in EvalWorkloadType — install the post-hook directly.
    handles_reference = True

    def __init__(
        self,
        agent: Union[Agent, type, Callable[..., Any]],
        *,
        agent_kwargs: Optional[dict[str, Any]] = None,
        reference_key: str = "reference",
        extra_meta_keys: tuple[str, ...] = (),
        name: str = "agent",
    ):
        self.name = name
        self._reference_key = reference_key
        self._extra_meta_keys = tuple(extra_meta_keys)
        self._agent = _coerce_agent(agent, agent_kwargs or {})

    async def make_request(self, item: Any) -> Request:
        return Request(
            method="AGENT",
            url=f"agent://{self.name}",
            meta=self._split_meta(item),
        )

    async def run_ticket(self, ctx: TicketContext) -> Sample:
        req = await self.make_request(ctx.item)
        agent_ctx = AgentContext(
            item=ctx.item,
            workload_name=self.name,
            fire=ctx.fire,
            start_mono=ctx.start_mono,
        )
        start = ctx.start_mono

        try:
            result = await self._agent.run(agent_ctx)
        except Exception as e:
            latency = time.monotonic() - start
            error = f"{type(e).__name__}: {e}"
            sample = Sample(
                start_ts=start,
                latency_s=latency,
                status=0,
                ok=False,
                request_ok=False,
                error=error,
                workload=self.name,
                meta=dict(req.meta),
            )
            resp = Response(
                status=0, headers={}, body=b"",
                elapsed_s=latency, ok=False, error=error,
            )
            return await self._run_post_hooks(ctx, req, resp, sample)

        latency = time.monotonic() - start
        body = (result.output or "").encode("utf-8", errors="replace")
        # `resp.ok` gates the correctness post-hook. We want grading to run
        # whenever the agent finished its trajectory (even an unsuccessful one
        # — empty output correctly grades as "wrong"). Only skip grading on
        # actual infra failure (request_ok=False).
        resp = Response(
            status=200 if result.request_ok else 0,
            headers={},
            body=body,
            elapsed_s=latency,
            ok=result.request_ok,
            error=result.error,
        )

        merged_meta = dict(req.meta)
        merged_meta.update(result.meta or {})

        extra: dict[str, float] = {}
        for k, v in (result.metrics or {}).items():
            if isinstance(v, (int, float)):
                extra[k] = float(v)

        sample = Sample(
            start_ts=start,
            latency_s=latency,
            # When the agent ran to completion (request_ok=True) we record a
            # synthetic 200 even if `ok` is False — the trajectory was
            # *delivered*, it just didn't submit a successful answer. That keeps
            # the ok/wrong/failed bucketing meaningful.
            status=200 if result.request_ok else 0,
            ok=result.ok,
            request_ok=result.request_ok,
            bytes_sent=result.bytes_sent,
            bytes_recv=result.bytes_recv or len(body),
            error=result.error,
            workload=self.name,
            meta=merged_meta,
            extra=extra,
        )
        return await self._run_post_hooks(ctx, req, resp, sample)

    @staticmethod
    async def _run_post_hooks(ctx: TicketContext, req: Request, resp: Response,
                              sample: Sample) -> Sample:
        for hook in ctx.post_hooks:
            sample = await maybe_await(hook(req, resp, sample))
        return sample

    def _split_meta(self, item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {}
        keys = {self._reference_key, *self._extra_meta_keys}
        return {k: item[k] for k in keys if k in item}

    async def aclose(self) -> None:
        await self._agent.aclose()


def _coerce_agent(spec: Any, agent_kwargs: dict[str, Any]) -> Agent:
    if isinstance(spec, Agent):
        if agent_kwargs:
            raise ValueError(
                "agent_kwargs is only meaningful when 'agent' is a class or "
                "factory — got an Agent instance"
            )
        return spec
    if inspect.isclass(spec) and issubclass(spec, Agent):
        return spec(**agent_kwargs)
    if callable(spec):
        # Factory path: try calling it with agent_kwargs; if the result is an
        # Agent, use it. Otherwise treat the original callable as a per-task fn.
        if agent_kwargs:
            obj = spec(**agent_kwargs)
            if isinstance(obj, Agent):
                return obj
            if callable(obj):
                return CallableAgent(obj)
            raise TypeError(
                f"agent factory returned {type(obj).__name__}; "
                "expected Agent or callable"
            )
        return CallableAgent(spec)
    raise TypeError(
        f"agent must be Agent|Agent-subclass|callable, got {type(spec).__name__}"
    )

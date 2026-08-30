"""Core data types and hook protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional, Protocol


@dataclass
class Request:
    """An outgoing HTTP request.

    `meta` is opaque user-controlled state that travels with the request and is
    available to post-response hooks (useful for correlating prompts to results,
    tagging variants, etc.).
    """

    method: str = "GET"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    body: Optional[bytes] = None
    json: Any = None
    timeout_s: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    status: int
    headers: Mapping[str, str]
    body: bytes
    elapsed_s: float
    ok: bool
    error: Optional[str] = None
    # For streaming responses, the chunks captured (bytes) and per-chunk arrival times.
    stream_chunks: Optional[list[bytes]] = None
    stream_chunk_times: Optional[list[float]] = None  # seconds since request start


@dataclass
class Sample:
    """One observation. Workloads may attach extra numeric metrics via `extra`."""

    start_ts: float                # wall-clock start (time.monotonic)
    latency_s: float               # total request latency
    status: int                    # HTTP status (0 if connection failure)
    ok: bool                       # whether the workload considers it a success
    # Whether the request itself was delivered successfully at the transport/HTTP
    # layer. Stays True for responses that came back 2xx/3xx even if a post-hook
    # later flips `ok` to False (e.g. eval scorer judged the output wrong).
    # Use this to distinguish "request failed" from "request OK but output wrong".
    request_ok: bool = True
    bytes_sent: int = 0
    bytes_recv: int = 0
    error: Optional[str] = None
    workload: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)


# Hook signatures. Hooks may be sync or async; the runner awaits coroutines and
# calls plain callables directly.
PreRequestHook = Callable[[Request], "Request | Awaitable[Request]"]
PostResponseHook = Callable[
    [Request, Response, Sample],
    "Sample | Awaitable[Sample]",
]

# A `fire(req)` callable closes over the httpx2 async client, applies pre-hooks,
# times the call, and returns a Response. A WorkloadType.run_ticket implementation
# may call this 0..N times per ticket.
FireRequest = Callable[[Request], Awaitable[Response]]


@dataclass
class TicketContext:
    """Everything a WorkloadType needs to handle a single ticket.

    The runner builds one of these per ticket and hands it to
    `WorkloadType.run_ticket`. Multi-step workload-types (e.g. sandbox lifecycle)
    can call `fire(req)` repeatedly; the default base-class implementation calls
    it exactly once.
    """

    item: Any
    start_mono: float
    fire: FireRequest
    pre_hooks: tuple = ()
    post_hooks: tuple = ()
    workload_name: str = "workload-type"


async def maybe_await(value: Any) -> Any:
    """Await `value` if awaitable, else return it. Lets hooks be sync OR async."""
    if hasattr(value, "__await__"):
        return await value
    return value

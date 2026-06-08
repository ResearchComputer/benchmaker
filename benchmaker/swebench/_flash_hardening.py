"""Network hardening for harbor's flash-sandbox HTTP client.

Why this exists
---------------
harbor builds its flash-sandbox ``AsyncHTTPClient`` with aiohttp's *default*
connector and only ``retries=2`` (see
``harbor/environments/flash_sandbox.py`` -> ``start()``). Against a **remote**
orchestrator reached over Tailscale (e.g. ``http://100.71.204.79:8080``) with an
nginx proxy in front, two things go wrong:

1. aiohttp parks idle keep-alive sockets in its pool and reuses them. The proxy
   closes idle connections, so the next reuse lands on a half-closed socket and
   the peer replies RST -> ``aiohttp.ClientOSError: [Errno 104] Connection reset
   by peer``. That *is* a ``ClientConnectionError`` subclass, so harbor's
   ``_request`` already retries it (which is why trials still pass), but it is
   noisy and a short burst can outlast the 2 retries -- e.g. the recurring
   "Failed to upload agent logs back to environment" warning.

2. When such a reset lands mid-write on a pooled connection, aiohttp's transport
   stores the error on a write future that nobody awaits, and asyncio prints the
   alarming ``Future exception was never retrieved /
   ConnectionError('Connection lost')`` traceback at GC time. Purely cosmetic,
   but it dominates the logs.

What we do
----------
:func:`harden_flash_sandbox_client` monkeypatches
``flash_sandbox.AsyncHTTPClient`` (idempotently) so its internal
``aiohttp.ClientSession`` is built on a connector with ``force_close=True`` --
no keep-alive reuse, so the stale-socket RST class is gone -- and raises the
transient-retry budget for anything that still slips through. It also installs,
lazily on the running loop, an asyncio exception handler that drops *only* the
orphaned connection-reset noise and delegates everything else.

Both effects are safe to apply more than once and from either entrypoint:
``harbor run <config>`` loads us via the agent ``import_path`` (this module is
imported by ``benchmaker.swebench.pi_agent``); ``benchmaker swebench`` loads us
via ``harbor_eval``. Call :func:`harden_flash_sandbox_client` at import time,
*before* the first sandbox request, so the patched session is the one harbor
actually uses.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger("benchmaker.flash_hardening")

# --- tunables ---------------------------------------------------------------
# force_close=True disables aiohttp keep-alive entirely: every request gets a
# fresh connection, so a proxy that drops idle sockets can never hand us a
# half-closed one to reuse. Over plain HTTP on Tailscale with sparse, long
# requests the per-request connect cost is negligible. Flip to False to keep
# keep-alive and instead cap idle reuse at ``KEEPALIVE_TIMEOUT_SEC``.
FORCE_CLOSE = True
KEEPALIVE_TIMEOUT_SEC = 15.0
# harbor leaves AsyncHTTPClient at retries=2 / retry_backoff=0.25; bump both so
# a transient burst of resets is ridden out rather than surfacing as a failed
# log upload.
RETRIES = 5
RETRY_BACKOFF_SEC = 0.5

_PATCHED = False
_LOOP_FLAG = "_benchmaker_reset_noise_silenced"


def _build_connector() -> Any:
    import aiohttp

    if FORCE_CLOSE:
        return aiohttp.TCPConnector(force_close=True)
    return aiohttp.TCPConnector(keepalive_timeout=KEEPALIVE_TIMEOUT_SEC)


def silence_connection_reset_noise(loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Drop orphaned connection-reset chatter from the loop's exception handler.

    Suppresses only futures/transports whose error is a connection reset (the
    ``Future exception was never retrieved`` / ``Connection lost`` lines). Any
    other unhandled-exception context is passed to the previous handler so real
    bugs still surface. Idempotent per loop.
    """
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
    if getattr(loop, _LOOP_FLAG, False):
        return

    try:
        import aiohttp

        conn_excs: tuple[type[BaseException], ...] = (
            ConnectionError,
            aiohttp.ClientConnectionError,
        )
    except Exception:  # pragma: no cover - aiohttp always present here
        conn_excs = (ConnectionError,)

    prev = loop.get_exception_handler()

    def _handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        msg = context.get("message") or ""
        is_conn = isinstance(exc, conn_excs) or "Connection lost" in msg
        is_orphan = "Future exception was never retrieved" in msg
        if is_conn or (is_orphan and isinstance(exc, conn_excs)):
            log.debug("suppressed connection-reset noise: %s", msg or exc)
            return
        if prev is not None:
            prev(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)
    setattr(loop, _LOOP_FLAG, True)
    log.debug("installed connection-reset noise filter on event loop")


def harden_flash_sandbox_client() -> bool:
    """Patch ``flash_sandbox.AsyncHTTPClient`` for resilient remote I/O.

    Returns True if the patch was applied (or already in place), False if
    flash-sandbox is not importable. Idempotent.
    """
    global _PATCHED
    if _PATCHED:
        return True
    try:
        import aiohttp  # noqa: F401
        from flash_sandbox import http_client as _hc
    except Exception:
        log.debug("flash-sandbox not importable; client hardening skipped")
        return False

    client_cls = _hc.AsyncHTTPClient
    orig_init = client_cls.__init__

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("retries", RETRIES)
        kwargs.setdefault("retry_backoff", RETRY_BACKOFF_SEC)
        orig_init(self, *args, **kwargs)
        # Also lift values harbor may pass positionally/explicitly below ours.
        if getattr(self, "retries", 0) < RETRIES:
            self.retries = RETRIES
        if getattr(self, "retry_backoff", 0.0) < RETRY_BACKOFF_SEC:
            self.retry_backoff = RETRY_BACKOFF_SEC

    async def _get_session(self: Any) -> Any:
        import aiohttp

        # The first request runs on harbor's event loop; attach the noise
        # filter there before any reset can fire.
        silence_connection_reset_noise()
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={"X-Otela-Fallback": "2"},
                connector=_build_connector(),
            )
        return self._session

    client_cls.__init__ = __init__
    client_cls._get_session = _get_session
    _PATCHED = True
    log.debug(
        "flash-sandbox AsyncHTTPClient hardened (force_close=%s, retries=%s, backoff=%ss)",
        FORCE_CLOSE,
        RETRIES,
        RETRY_BACKOFF_SEC,
    )
    return True

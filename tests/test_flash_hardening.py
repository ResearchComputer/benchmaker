"""Tests for the flash-sandbox HTTP client hardening shim.

These exercise the monkeypatch and the asyncio noise filter directly; they do
not need a live orchestrator. flash-sandbox is an optional dep, so the
client-patch tests skip when it is absent, while the loop-handler test runs
unconditionally.
"""

from __future__ import annotations

import asyncio

import pytest

from benchmaker.swebench import _flash_hardening as fh

flash_sandbox = pytest.importorskip(
    "flash_sandbox", reason="flash-sandbox not installed"
)


@pytest.fixture()
def hardened():
    assert fh.harden_flash_sandbox_client() is True
    yield


def test_client_retries_are_lifted(hardened):
    from flash_sandbox import AsyncHTTPClient

    client = AsyncHTTPClient(address="http://127.0.0.1:1")
    assert client.retries >= fh.RETRIES
    assert client.retry_backoff >= fh.RETRY_BACKOFF_SEC


def test_session_uses_force_close_connector(hardened):
    from flash_sandbox import AsyncHTTPClient

    async def go():
        client = AsyncHTTPClient(address="http://127.0.0.1:1")
        try:
            session = await client._get_session()
            # No keep-alive reuse: the connector force-closes every connection.
            assert session.connector._force_close is fh.FORCE_CLOSE
            # The noise filter self-installed on this running loop.
            loop = asyncio.get_running_loop()
            assert getattr(loop, fh._LOOP_FLAG, False) is True
        finally:
            await client.close()

    asyncio.run(go())


def test_idempotent():
    assert fh.harden_flash_sandbox_client() is True
    assert fh.harden_flash_sandbox_client() is True


def test_noise_filter_suppresses_resets_but_passes_real_errors():
    """The loop handler drops connection-reset chatter and forwards the rest."""

    async def go():
        loop = asyncio.get_running_loop()
        seen: list[dict] = []
        loop.set_exception_handler(lambda _loop, ctx: seen.append(ctx))

        # Install our filter on top of the recording handler.
        fh.silence_connection_reset_noise(loop)

        # 1) raw connection reset -> suppressed
        loop.call_exception_handler(
            {"message": "x", "exception": ConnectionResetError(104, "reset")}
        )
        # 2) the "Connection lost" orphaned-future shape -> suppressed
        loop.call_exception_handler(
            {
                "message": "Future exception was never retrieved",
                "exception": ConnectionError("Connection lost"),
            }
        )
        # 3) an unrelated real error -> forwarded to the previous handler
        loop.call_exception_handler(
            {"message": "boom", "exception": ValueError("real bug")}
        )

        assert len(seen) == 1
        assert isinstance(seen[0]["exception"], ValueError)

    asyncio.run(go())

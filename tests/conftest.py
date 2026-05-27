"""Spin up a tiny aiohttp server on a free port for tests."""

import asyncio
import json
import socket
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio
from aiohttp import web


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _hello(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "path": request.path})


async def _echo(request: web.Request) -> web.Response:
    body = await request.read()
    return web.json_response({"received_bytes": len(body)})


async def _slow(request: web.Request) -> web.Response:
    await asyncio.sleep(0.05)
    return web.json_response({"ok": True})


async def _fail(request: web.Request) -> web.Response:
    return web.json_response({"err": "boom"}, status=500)


_metrics_counter = {"n": 0}


async def _prom_metrics(request: web.Request) -> web.Response:
    """Pretend to be vLLM /metrics. Counter monotonically increases on each scrape."""
    _metrics_counter["n"] += 1
    body = (
        "# HELP stub_requests_total Total requests served\n"
        "# TYPE stub_requests_total counter\n"
        f"stub_requests_total {_metrics_counter['n']}\n"
        "# HELP stub_gpu_util GPU utilization percentage\n"
        "# TYPE stub_gpu_util gauge\n"
        f'stub_gpu_util{{gpu="0"}} {0.5 + 0.1 * _metrics_counter["n"]}\n'
        f'stub_gpu_util{{gpu="1"}} {0.3 + 0.05 * _metrics_counter["n"]}\n'
        "stub_kv_cache_usage 0.42\n"
    )
    return web.Response(text=body, content_type="text/plain")


async def _sse(request: web.Request) -> web.StreamResponse:
    """Pretend to be an OpenAI streaming endpoint. 5 token chunks + usage + DONE."""
    resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for i in range(5):
        await asyncio.sleep(0.005)
        chunk = {
            "choices": [{
                "index": 0,
                "delta": {"content": f"tok{i} "},
                "finish_reason": None,
            }]
        }
        await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
    await resp.write(
        b"data: " + json.dumps({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        }).encode() + b"\n\n"
    )
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


_sandbox_state: dict[str, Any] = {
    "next_id": 0,
    "created": [],      # list of (sid, body)
    "deleted": [],      # list of sid
    "exec_calls": [],   # list of (sid, body, endpoint)
}


def _new_sid() -> str:
    _sandbox_state["next_id"] += 1
    return f"sb-test-{_sandbox_state['next_id']:04d}"


async def _sb_create(request: web.Request) -> web.Response:
    body = await request.json()
    sid = _new_sid()
    _sandbox_state["created"].append((sid, body))
    return web.json_response({"id": sid})


async def _sb_exec(request: web.Request) -> web.Response:
    sid = request.match_info["sid"]
    body = await request.json()
    _sandbox_state["exec_calls"].append((sid, body, "exec"))
    cmd = body.get("command") or []
    if isinstance(cmd, list) and any("fail" in str(x) for x in cmd):
        return web.json_response({
            "stdout": "", "stderr": "boom\n", "exit_code": 1, "duration": 0.001,
        })
    return web.json_response({
        "stdout": "hello\n", "stderr": "", "exit_code": 0, "duration": 0.002,
    })


async def _sb_pshell(request: web.Request) -> web.Response:
    sid = request.match_info["sid"]
    body = await request.json()
    _sandbox_state["exec_calls"].append((sid, body, "pshell"))
    return web.json_response({
        "stdout": "pshell-ok\n", "stderr": "", "exit_code": 0, "duration": 0.001,
    })


async def _sb_delete(request: web.Request) -> web.Response:
    sid = request.match_info["sid"]
    _sandbox_state["deleted"].append(sid)
    return web.json_response({"ok": True})


@pytest_asyncio.fixture
async def sandbox_state():
    """Reset and yield the stub sandbox state recorder."""
    _sandbox_state["next_id"] = 0
    _sandbox_state["created"].clear()
    _sandbox_state["deleted"].clear()
    _sandbox_state["exec_calls"].clear()
    yield _sandbox_state


@pytest_asyncio.fixture
async def stub_server() -> AsyncIterator[str]:
    """Yield base URL of a stub HTTP server."""
    app = web.Application()
    app.router.add_get("/hello", _hello)
    app.router.add_post("/echo", _echo)
    app.router.add_get("/slow", _slow)
    app.router.add_get("/fail", _fail)
    app.router.add_get("/metrics", _prom_metrics)
    app.router.add_post("/v1/chat/completions", _sse)
    # Flash Sandbox stubs (cluster + node prefixes share handlers).
    for prefix in ("/sandboxes", "/native/sandboxes"):
        app.router.add_post(prefix, _sb_create)
        app.router.add_post(prefix + "/{sid}/exec", _sb_exec)
        app.router.add_post(prefix + "/{sid}/pshell", _sb_pshell)
        app.router.add_delete(prefix + "/{sid}", _sb_delete)
    _metrics_counter["n"] = 0  # reset between tests
    _sandbox_state["next_id"] = 0
    _sandbox_state["created"].clear()
    _sandbox_state["deleted"].clear()
    _sandbox_state["exec_calls"].clear()

    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()

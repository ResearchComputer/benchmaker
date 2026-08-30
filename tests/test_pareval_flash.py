"""Tests for benchmaker.pareval.flash — the Flash Sandbox exec/write_file backend.

Pin the EXACT wire contract derived from the existing client
(benchmaker/workloads/sandbox.py + benchmaker/swebench/agent.py). Heavy coverage
lives in the pure request-building / response-parsing helpers; one mocked
round-trip proves the wiring (create id -> exec/delete URLs).
"""
from __future__ import annotations

import pytest

from benchmaker.pareval.flash import (
    FlashSandbox,
    _create_url,
    _create_body,
    _exec_url,
    _exec_body,
    _file_put_url,
    _file_put_params,
    _delete_url,
    _parse_exec_response,
)


# ---- URL builders ----------------------------------------------------------

def test_create_url():
    assert _create_url("http://h:8080/", "/sandboxes") == "http://h:8080/sandboxes"


def test_create_url_strips_trailing_slash():
    assert _create_url("http://h:8080", "/sandboxes") == "http://h:8080/sandboxes"


def test_exec_url_includes_id():
    assert _exec_url("http://h:8080", "/sandboxes", "abc") == "http://h:8080/sandboxes/abc/exec"


def test_file_put_url_includes_id():
    assert _file_put_url("http://h:8080", "/sandboxes", "abc") == "http://h:8080/sandboxes/abc/files"


def test_delete_url_includes_id():
    assert _delete_url("http://h:8080", "/sandboxes", "abc") == "http://h:8080/sandboxes/abc"


def test_node_prefix():
    assert _exec_url("http://h", "/native/sandboxes", "z") == "http://h/native/sandboxes/z/exec"


# ---- create body -----------------------------------------------------------

def test_create_body_sets_image_top_level():
    # Mirrors sandbox.py _create_sandbox / _DEFAULT_SPEC: image lives at the
    # top level of the create body, not nested under a "spec" key.
    body = _create_body("pareval-toolchain")
    assert body["image"] == "pareval-toolchain"


def test_create_body_overrides_merge():
    body = _create_body("img", {"ttl_seconds": 99, "image": "other"})
    assert body["image"] == "other"
    assert body["ttl_seconds"] == 99


# ---- exec body -------------------------------------------------------------

def test_exec_body_wraps_in_sh_c():
    # Mirrors agent.py _sandbox_exec: {"command": ["sh", "-c", cmd]}.
    assert _exec_body("echo hi") == {"command": ["sh", "-c", "echo hi"]}


# ---- file put --------------------------------------------------------------

def test_file_put_params_carries_path():
    # Mirrors sandbox.py _make_file_put_request: path travels as a query param,
    # the bytes go in the raw body (octet-stream), not base64 JSON.
    assert _file_put_params("/tmp/x.hpp") == {"path": "/tmp/x.hpp"}


# ---- exec response parsing -------------------------------------------------

def test_parse_exec_response_returns_rc_and_combined():
    rc, out = _parse_exec_response({"exit_code": 0, "stdout": "hello", "stderr": "warn"})
    assert rc == 0
    assert "hello" in out and "warn" in out


def test_parse_exec_response_nonzero_rc():
    rc, out = _parse_exec_response({"exit_code": 2, "stdout": "", "stderr": "boom"})
    assert rc == 2
    assert out == "boom"


def test_parse_exec_response_stdout_only():
    rc, out = _parse_exec_response({"exit_code": 0, "stdout": "only", "stderr": ""})
    assert (rc, out) == (0, "only")


def test_parse_exec_response_missing_exit_code_defaults_zero():
    # agent.py treats a missing/None exit_code as 0.
    rc, out = _parse_exec_response({"stdout": "x"})
    assert rc == 0


def test_parse_exec_response_unparseable_exit_code_defaults_zero():
    rc, out = _parse_exec_response({"exit_code": "weird", "stdout": "x"})
    assert rc == 0


def test_parse_exec_response_non_dict():
    rc, out = _parse_exec_response(None)
    assert rc == -1


# ---- mocked round-trip -----------------------------------------------------

class _FakeResp:
    def __init__(self, status, text="", payload=None):
        self.status_code = status
        self._text = text
        self._payload = payload

    @property
    def text(self):
        return self._text

    async def aread(self):
        return b""


class _FakeSession:
    """Records every request and replays canned responses by URL suffix."""

    def __init__(self):
        self.calls = []
        self.closed = False

    async def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        if url.endswith("/sandboxes"):
            return _FakeResp(200, text='{"id": "sb-123"}')
        # exec
        import json
        return _FakeResp(200, text=json.dumps({"exit_code": 7, "stdout": "OUT", "stderr": "ERR"}))

    async def put(self, url, **kw):
        self.calls.append(("PUT", url, kw))
        return _FakeResp(200, text="")

    async def delete(self, url, **kw):
        self.calls.append(("DELETE", url, kw))
        return _FakeResp(200, text="")

    async def aclose(self):
        self.closed = True


async def test_roundtrip_mocked(monkeypatch):
    fake = _FakeSession()
    monkeypatch.setattr("benchmaker.pareval.flash.httpx2.AsyncClient", lambda *a, **k: fake)

    async with FlashSandbox("http://h:8080", image="pareval-toolchain") as sb:
        await sb.write_file("/tmp/pareval/x.hpp", b"code")
        rc, out = await sb.exec("echo hi", timeout_s=5.0)

    assert rc == 7
    assert "OUT" in out and "ERR" in out

    methods_urls = [(m, u) for (m, u, _kw) in fake.calls]
    # create then file put then exec then delete
    assert ("POST", "http://h:8080/sandboxes") in methods_urls
    assert ("PUT", "http://h:8080/sandboxes/sb-123/files") in methods_urls
    assert ("POST", "http://h:8080/sandboxes/sb-123/exec") in methods_urls
    assert ("DELETE", "http://h:8080/sandboxes/sb-123") in methods_urls
    assert fake.closed is True


async def test_exec_http_error_returns_negative_rc(monkeypatch):
    class _ErrSession(_FakeSession):
        async def post(self, url, **kw):
            self.calls.append(("POST", url, kw))
            if url.endswith("/sandboxes"):
                return _FakeResp(200, text='{"id": "sb-1"}')
            return _FakeResp(500, text="kaboom")

    fake = _ErrSession()
    monkeypatch.setattr("benchmaker.pareval.flash.httpx2.AsyncClient", lambda *a, **k: fake)
    async with FlashSandbox("http://h", image="img") as sb:
        rc, out = await sb.exec("boom", timeout_s=1.0)
    assert rc == -1
    assert "500" in out

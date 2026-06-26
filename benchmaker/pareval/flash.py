"""Flash Sandbox HTTP exec backend for the pareval completion grader.

Supplies the ``exec_fn`` and ``write_file`` seams that
``benchmaker.pareval.sandbox_runner.CompletionGrader.grade`` expects:

    async exec(command: str, timeout_s: float) -> (rc, combined_stdout_stderr)
    async write_file(path: str, content: bytes) -> None

The wire protocol is NOT invented here — it mirrors the existing Flash Sandbox
clients, which are the source of truth:

  * create  POST  {base}{prefix}                 body = spec dict (``image`` at
            top level)  ->  response JSON with ``"id"``
            mirrors benchmaker/workloads/sandbox.py:585-606 (_create_sandbox)
            and benchmaker/swebench/agent.py:306-329 (_sandbox_create)
  * exec    POST  {base}{prefix}/{id}/exec       body = {"command": ["sh","-c",cmd]}
            response has ``exit_code`` / ``stdout`` / ``stderr``
            mirrors benchmaker/swebench/agent.py:331-360 (_sandbox_exec) and
            benchmaker/workloads/sandbox.py:716-742 (_apply_exec_metrics)
  * file    PUT   {base}{prefix}/{id}/files      ?path=<path>, raw octet-stream body
            mirrors benchmaker/workloads/sandbox.py:319-334 (_make_file_put_request)
            NOTE: the real client carries the path as a query param and the bytes
            as the raw request body (Content-Type application/octet-stream) — it
            does NOT base64-encode into a JSON body.
  * delete  DELETE {base}{prefix}/{id}
            mirrors benchmaker/workloads/sandbox.py:615-631 (aclose) and
            benchmaker/swebench/agent.py:362-372 (_sandbox_delete)
"""
from __future__ import annotations

import json
from typing import Any, Optional

import aiohttp

PREFIX_CLUSTER = "/sandboxes"
_EXCERPT_MAX = 500


def _excerpt(text: str) -> str:
    return text if len(text) <= _EXCERPT_MAX else text[:_EXCERPT_MAX] + "...[truncated]"


# ---- pure request builders / response parser -------------------------------

def _create_url(base_url: str, prefix: str) -> str:
    return f"{base_url.rstrip('/')}{prefix}"


def _create_body(image: str, spec_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    # Mirrors sandbox.py _DEFAULT_SPEC: image lives at the top level of the
    # create body. Overrides win (e.g. ttl_seconds, command, a different image).
    body: dict[str, Any] = {"image": image}
    if spec_overrides:
        body.update(spec_overrides)
    return body


def _exec_url(base_url: str, prefix: str, sandbox_id: str) -> str:
    return f"{base_url.rstrip('/')}{prefix}/{sandbox_id}/exec"


def _exec_body(command: str) -> dict[str, Any]:
    # Mirrors agent.py _sandbox_exec: wrap the shell string as argv.
    return {"command": ["sh", "-c", command]}


def _file_put_url(base_url: str, prefix: str, sandbox_id: str) -> str:
    return f"{base_url.rstrip('/')}{prefix}/{sandbox_id}/files"


def _file_put_params(path: str) -> dict[str, str]:
    # Mirrors sandbox.py _make_file_put_request: path as query param.
    return {"path": path}


def _delete_url(base_url: str, prefix: str, sandbox_id: str) -> str:
    return f"{base_url.rstrip('/')}{prefix}/{sandbox_id}"


def _parse_exec_response(obj: Any) -> tuple[int, str]:
    """Return (returncode, combined stdout+stderr) from an exec response body.

    Mirrors agent.py:352-360: a missing/None/unparseable exit_code becomes 0;
    stdout and stderr are joined with a newline when both are present. A
    non-dict body (e.g. unparseable) yields rc -1.
    """
    if not isinstance(obj, dict):
        return -1, ""
    rc = obj.get("exit_code")
    try:
        rc_i = int(rc) if rc is not None else 0
    except (TypeError, ValueError):
        rc_i = 0
    stdout = obj.get("stdout") or ""
    stderr = obj.get("stderr") or ""
    combined = stdout if not stderr else (stdout + ("\n" if stdout else "") + stderr)
    return rc_i, combined


# ---- async client ----------------------------------------------------------

class FlashSandbox:
    def __init__(
        self,
        base_url: str,
        *,
        image: str,
        endpoint_prefix: str = PREFIX_CLUSTER,
        headers: Optional[dict[str, str]] = None,
        create_timeout_s: float = 120.0,
    ):
        if not endpoint_prefix.startswith("/"):
            endpoint_prefix = "/" + endpoint_prefix
        endpoint_prefix = "/" + endpoint_prefix.strip("/")

        self._base_url = base_url.rstrip("/")
        self._prefix = endpoint_prefix
        self._image = image
        self._create_timeout_s = create_timeout_s

        hdrs = dict(headers or {})
        hdrs.setdefault("Content-Type", "application/json")
        self._headers = hdrs

        self._session: Optional[aiohttp.ClientSession] = None
        self._sandbox_id: Optional[str] = None

    async def __aenter__(self) -> "FlashSandbox":
        self._session = aiohttp.ClientSession()
        body = _create_body(self._image)
        url = _create_url(self._base_url, self._prefix)
        timeout = aiohttp.ClientTimeout(total=self._create_timeout_s)
        async with self._session.post(
            url, headers=self._headers, json=body, timeout=timeout
        ) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"sandbox create HTTP {resp.status}: {_excerpt(text)}")
            try:
                data = json.loads(text)
            except ValueError as e:
                raise RuntimeError(f"sandbox create returned non-JSON: {_excerpt(text)}") from e
        if not isinstance(data, dict) or "id" not in data:
            raise RuntimeError(f"sandbox create unexpected body: {data!r}")
        self._sandbox_id = str(data["id"])
        return self

    async def __aexit__(self, *exc) -> None:
        # Best-effort delete; never raise from teardown.
        if self._session is not None and self._sandbox_id is not None:
            url = _delete_url(self._base_url, self._prefix, self._sandbox_id)
            timeout = aiohttp.ClientTimeout(total=self._create_timeout_s)
            try:
                async with self._session.delete(
                    url, headers=self._headers, timeout=timeout
                ) as resp:
                    await resp.read()
            except Exception:
                pass
        if self._session is not None:
            try:
                await self._session.close()
            finally:
                self._session = None
        self._sandbox_id = None

    async def exec(self, command: str, timeout_s: float) -> tuple[int, str]:
        assert self._session is not None and self._sandbox_id is not None, (
            "FlashSandbox.exec called outside an active context"
        )
        url = _exec_url(self._base_url, self._prefix, self._sandbox_id)
        body = _exec_body(command)
        # Pad the client-side timeout so the server can return its own timeout
        # error rather than us tearing down the connection first (agent.py:338).
        timeout = aiohttp.ClientTimeout(total=timeout_s + 30.0)
        try:
            async with self._session.post(
                url, headers=self._headers, json=body, timeout=timeout
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    return -1, f"<sandbox exec HTTP {resp.status}: {_excerpt(text)}>"
                try:
                    data = json.loads(text)
                except ValueError:
                    return -1, f"<sandbox exec non-JSON: {_excerpt(text)}>"
        except (TimeoutError, aiohttp.ServerTimeoutError):
            return -1, f"<sandbox exec timed out after {timeout_s}s>"
        return _parse_exec_response(data)

    async def write_file(self, path: str, content: bytes) -> None:
        assert self._session is not None and self._sandbox_id is not None, (
            "FlashSandbox.write_file called outside an active context"
        )
        url = _file_put_url(self._base_url, self._prefix, self._sandbox_id)
        params = _file_put_params(path)
        headers = dict(self._headers)
        headers["Content-Type"] = "application/octet-stream"
        timeout = aiohttp.ClientTimeout(total=self._create_timeout_s)
        async with self._session.put(
            url, headers=headers, params=params, data=content, timeout=timeout
        ) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(
                    f"sandbox file put HTTP {resp.status} for {path!r}: {_excerpt(text)}"
                )
            await resp.read()

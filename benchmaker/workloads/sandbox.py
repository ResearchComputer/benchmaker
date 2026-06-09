"""Flash Sandbox workload-type.

Talks to the Flash Sandbox HTTP API (see project docs / ``docs/native-http-api``).
The wire protocol is the same for the cluster coordinator (``/sandboxes/...``)
and a single node (``/native/sandboxes/...``); the only difference is which
prefix you point at.

Operation modes
---------------
``exec`` (default)
    Each request execs a command in a shared sandbox. The sandbox is created
    lazily on the first call (under a lock) and deleted on ``aclose()``.
    Pass ``sandbox_id=`` to target an existing sandbox instead.

``create``
    Each request creates a fresh sandbox. Use ``ttl_seconds`` so the
    orchestrator reaps the sandboxes after the run (currently only enforced
    by the Kubernetes backend, but harmless on others).

``lifecycle``
    Each ticket fires the full ``POST /sandboxes`` → ``POST .../exec`` →
    ``DELETE /sandboxes/{id}`` sequence on its own sandbox. Per-step durations
    are captured in ``Sample.extra`` (``create_s``, ``exec_s``, ``delete_s``)
    so you can disentangle startup vs. execution vs. teardown latency.

``file``
    Each ticket performs ``PUT .../files`` then ``GET .../files`` against a
    shared sandbox and byte-compares the read-back content to what was written.
    Optionally, run an extra ``exec`` verification command (defaults to
    ``cat <path>``) and compare that output too.

Item interpretation (``exec`` mode)
-----------------------------------
- ``None``        → run the configured ``default_command``
- ``str``         → ``["sh", "-c", item]``
- ``list[str]``   → argv directly
- ``dict``        → ``{"command": ..., "env": {...}, "input": "...", "persistent": bool}``.
  ``persistent: True`` switches the request from ``/exec`` to ``/pshell``
  (preserves ``cd``/``export``/``source`` across calls).

Item interpretation (``create`` mode)
-------------------------------------
- ``None``        → use the configured ``spec`` verbatim
- ``dict``        → merged into the spec (e.g. to vary ``image`` per item)

Item interpretation (``lifecycle`` mode)
----------------------------------------
Same as ``exec`` mode (items name the command). The sandbox spec is taken from
the workload-type configuration.

Item interpretation (``file`` mode)
-----------------------------------
- ``None``        → write ``file_content`` to ``file_path``
- ``bytes``       → write bytes to ``file_path``
- ``str``         → UTF-8 encode and write to ``file_path``
- ``dict``        → ``{"path": "...", "content": <bytes|str>,
  "content_base64": "<b64>", "verify_exec": bool,
  "verify_command": <str|list[str]>}``. Use ``content_base64`` instead of
  ``content`` to carry arbitrary bytes through a text (e.g. YAML/JSON) config.

Captured metrics
----------------
``exec`` / ``pshell``:
    ``exit_code``         exit code from the exec (also flips ``ok`` to False on non-zero)
    ``server_duration_s`` server-reported execution duration
    ``stdout_bytes``      length of stdout in the response body
    ``stderr_bytes``      length of stderr in the response body

``create``:
    ``server_created``    ``1.0`` on success, with ``sandbox_id`` in ``Sample.meta``

``lifecycle``:
    ``create_s``          wall time of ``POST /sandboxes``
    ``exec_s``            wall time of ``POST /sandboxes/{id}/exec``
    ``delete_s``          wall time of ``DELETE /sandboxes/{id}`` (best-effort,
                          does not affect ``ok``)
    ``lifecycle_s``       wall time across all three steps
    plus the standard ``exec_code`` / ``server_duration_s`` / ``stdout_bytes`` /
    ``stderr_bytes`` from the exec step.

``file``:
    ``file_write_s``             wall time of ``PUT .../files``
    ``file_read_s``              wall time of ``GET .../files``
    ``file_write_bytes``         bytes written
    ``file_read_bytes``          bytes read back
    ``file_mismatch_count``      ``1.0`` when read-back differs, else ``0.0``
    ``file_mismatch_bytes``      number of mismatched bytes (including length diff)
    ``file_exec_s``              wall time of optional exec verifier
    ``file_exec_mismatch_count`` ``1.0`` when exec output differs, else ``0.0``
    ``file_exec_mismatch_bytes`` mismatched bytes for exec output
"""

from __future__ import annotations

import asyncio
import base64
import json
import shlex
import time
from typing import Any, Optional

import aiohttp

from benchmaker.core.types import Request, Response, Sample, TicketContext, maybe_await
from benchmaker.workloads.base import WorkloadType, _estimate_request_size


PREFIX_CLUSTER = "/sandboxes"
PREFIX_NODE = "/native/sandboxes"

_DEFAULT_SPEC: dict[str, Any] = {
    "image": "alpine:3.20",
    "command": ["sh", "-c", "sleep 3600"],
}


class SandboxWorkloadType(WorkloadType):
    """Flash Sandbox exec/create workload-type."""

    name = "sandbox"
    streaming = False

    def __init__(
        self,
        base_url: str,
        operation: str = "exec",
        spec: Optional[dict[str, Any]] = None,
        sandbox_id: Optional[str] = None,
        endpoint_prefix: str = PREFIX_CLUSTER,
        default_command: Any = None,
        persistent: bool = False,
        timeout_s: Optional[float] = 60.0,
        create_timeout_s: float = 60.0,
        ttl_seconds: Optional[int] = None,
        headers: Optional[dict[str, str]] = None,
        cleanup_on_close: bool = True,
        name: str = "sandbox",
        file_path: str = "/tmp/benchmaker.bin",
        file_content: Any = b"benchmaker",
        file_verify_with_exec: bool = False,
        file_verify_command: Any = None,
    ):
        op = operation.lower()
        if op not in ("exec", "create", "lifecycle", "file"):
            raise ValueError(
                "operation must be 'exec', 'create', 'lifecycle', or 'file', "
                f"got {operation!r}"
            )
        if op == "lifecycle" and sandbox_id is not None:
            raise ValueError(
                "operation='lifecycle' creates its own sandbox per ticket; do not pass sandbox_id"
            )
        if not endpoint_prefix.startswith("/"):
            endpoint_prefix = "/" + endpoint_prefix
        endpoint_prefix = "/" + endpoint_prefix.strip("/")

        self.name = name
        self._base_url = base_url.rstrip("/")
        self._prefix = endpoint_prefix
        self._operation = op
        self._spec = {**_DEFAULT_SPEC, **(spec or {})}
        self._sandbox_id = sandbox_id
        self._default_command = default_command
        self._persistent = persistent
        self._timeout_s = timeout_s
        self._create_timeout_s = create_timeout_s
        self._ttl_seconds = ttl_seconds
        self._cleanup_on_close = cleanup_on_close
        self._file_path = str(file_path)
        self._file_content = _coerce_file_content(file_content)
        self._file_verify_with_exec = bool(file_verify_with_exec)
        self._file_verify_command = (
            _coerce_command(file_verify_command)
            if file_verify_command is not None
            else None
        )

        hdrs = dict(headers or {})
        hdrs.setdefault("Content-Type", "application/json")
        self._headers = hdrs

        self._owned_sandbox = False
        self._create_lock = asyncio.Lock()
        self._aux_session: Optional[aiohttp.ClientSession] = None
        self._aux_session_lock = asyncio.Lock()

    async def make_request(self, item: Any) -> Request:
        if self._operation == "create":
            return self._make_create_request(item)
        if self._operation == "lifecycle":
            # Lifecycle's full flow lives in run_ticket; if anyone reaches
            # make_request directly, give them the first step (create).
            return self._make_create_request(None)
        if self._operation == "file":
            sid = await self._ensure_sandbox()
            path, content, _verify_exec, _verify_cmd = self._parse_file_item(item)
            return self._make_file_put_request(sid, path, content)
        sid = await self._ensure_sandbox()
        return self._make_exec_request(sid, item)

    async def run_ticket(self, ctx: TicketContext) -> Sample:
        if self._operation == "lifecycle":
            return await self._run_lifecycle(ctx)
        if self._operation == "file":
            return await self._run_file(ctx)
        return await super().run_ticket(ctx)

    def _make_create_request(self, item: Any) -> Request:
        body = dict(self._spec)
        if isinstance(item, dict):
            body.update(item)
        elif item is not None:
            raise TypeError(
                f"sandbox 'create' mode expects None or dict items, got {type(item).__name__}"
            )
        if self._ttl_seconds is not None and "ttl_seconds" not in body:
            body["ttl_seconds"] = self._ttl_seconds
        return Request(
            method="POST",
            url=f"{self._base_url}{self._prefix}",
            headers=dict(self._headers),
            json=body,
            timeout_s=self._timeout_s,
            meta={"sandbox_operation": "create"},
        )

    def _parse_exec_item(
        self, item: Any
    ) -> tuple[list[str], Optional[dict[str, str]], Optional[str], bool]:
        """Decode an item into (command, env, stdin, persistent)."""
        command: list[str]
        env: Optional[dict[str, str]] = None
        stdin: Optional[str] = None
        persistent = self._persistent

        if item is None:
            if self._default_command is None:
                raise ValueError(
                    "SandboxWorkloadType: item was None and no default_command was set"
                )
            command = _coerce_command(self._default_command)
        elif isinstance(item, str):
            command = ["sh", "-c", item]
        elif isinstance(item, list):
            if not all(isinstance(x, str) for x in item):
                raise TypeError("list items must be list[str] (argv)")
            command = list(item)
        elif isinstance(item, dict):
            if "command" not in item:
                raise ValueError("dict items must contain 'command'")
            command = _coerce_command(item["command"])
            env = item.get("env")
            stdin = item.get("input")
            persistent = bool(item.get("persistent", persistent))
        else:
            raise TypeError(
                f"SandboxWorkloadType cannot interpret item of type {type(item).__name__}"
            )

        return command, env, stdin, persistent

    def _make_exec_request(self, sid: str, item: Any) -> Request:
        command, env, stdin, persistent = self._parse_exec_item(item)

        endpoint = "pshell" if persistent else "exec"
        body: dict[str, Any] = {"command": command}
        if env:
            body["env"] = env
        if stdin is not None:
            body["input"] = stdin

        return Request(
            method="POST",
            url=f"{self._base_url}{self._prefix}/{sid}/{endpoint}",
            headers=dict(self._headers),
            json=body,
            timeout_s=self._timeout_s,
            meta={
                "sandbox_id": sid,
                "sandbox_operation": "pshell" if persistent else "exec",
                "command": command,
            },
        )

    def _parse_file_item(self, item: Any) -> tuple[str, bytes, bool, Optional[list[str]]]:
        path = self._file_path
        content = self._file_content
        verify_exec = self._file_verify_with_exec
        verify_command = self._file_verify_command

        if item is None:
            return path, content, verify_exec, verify_command
        if isinstance(item, (bytes, bytearray)):
            return path, bytes(item), verify_exec, verify_command
        if isinstance(item, str):
            return path, item.encode("utf-8"), verify_exec, verify_command
        if not isinstance(item, dict):
            raise TypeError(
                "sandbox 'file' mode expects None|bytes|str|dict items, "
                f"got {type(item).__name__}"
            )

        if "path" in item and item["path"] is not None:
            path = str(item["path"])
        if "content" in item:
            content = _coerce_file_content(item["content"])
        elif "content_base64" in item:
            raw = item["content_base64"]
            if not isinstance(raw, str):
                raise TypeError("content_base64 must be a base64 string")
            content = base64.b64decode(raw.encode("ascii"))
        if "verify_exec" in item:
            verify_exec = bool(item["verify_exec"])
        if "verify_command" in item and item["verify_command"] is not None:
            verify_command = _coerce_command(item["verify_command"])
        return path, content, verify_exec, verify_command

    def _make_file_put_request(self, sid: str, path: str, content: bytes) -> Request:
        headers = dict(self._headers)
        headers["Content-Type"] = "application/octet-stream"
        return Request(
            method="PUT",
            url=f"{self._base_url}{self._prefix}/{sid}/files",
            headers=headers,
            params={"path": path},
            body=content,
            timeout_s=self._timeout_s,
            meta={
                "sandbox_id": sid,
                "sandbox_operation": "file_put",
                "file_path": path,
            },
        )

    def _make_file_get_request(self, sid: str, path: str) -> Request:
        return Request(
            method="GET",
            url=f"{self._base_url}{self._prefix}/{sid}/files",
            headers=dict(self._headers),
            params={"path": path},
            timeout_s=self._timeout_s,
            meta={
                "sandbox_id": sid,
                "sandbox_operation": "file_get",
                "file_path": path,
            },
        )

    async def make_sample(self, item: Any, request: Request, response: Response,
                          start_ts: float) -> Sample:
        sample = await super().make_sample(item, request, response, start_ts)
        op = request.meta.get("sandbox_operation")
        if not response.ok:
            return sample

        obj = _try_json(response.body)
        if op in ("exec", "pshell"):
            if isinstance(obj, dict):
                ec = obj.get("exit_code")
                if ec is not None:
                    try:
                        ec_i = int(ec)
                        sample.extra["exit_code"] = float(ec_i)
                        if ec_i != 0:
                            sample.ok = False
                            sample.error = sample.error or f"exit_code={ec_i}"
                    except (TypeError, ValueError):
                        pass
                dur = obj.get("duration")
                if dur is not None:
                    try:
                        sample.extra["server_duration_s"] = float(dur)
                    except (TypeError, ValueError):
                        pass
                if isinstance(obj.get("stdout"), str):
                    sample.extra["stdout_bytes"] = float(len(obj["stdout"]))
                if isinstance(obj.get("stderr"), str):
                    sample.extra["stderr_bytes"] = float(len(obj["stderr"]))
        elif op == "create":
            if isinstance(obj, dict) and "id" in obj:
                sample.meta["sandbox_id"] = obj["id"]
            sample.extra["server_created"] = 1.0
        return sample

    async def _run_lifecycle(self, ctx: TicketContext) -> Sample:
        command, env, stdin, persistent = self._parse_exec_item(ctx.item)

        # Step 1: create
        create_req = self._make_create_request(None)
        create_resp = await ctx.fire(create_req)

        sample = Sample(
            start_ts=ctx.start_mono,
            latency_s=time.monotonic() - ctx.start_mono,
            status=create_resp.status,
            ok=False,
            request_ok=create_resp.ok,
            bytes_recv=len(create_resp.body or b""),
            bytes_sent=_estimate_request_size(create_req),
            error=None,
            workload=self.name,
            meta={"sandbox_operation": "lifecycle"},
        )
        sample.extra["create_s"] = create_resp.elapsed_s

        if not create_resp.ok:
            sample.error = f"create failed: {create_resp.error or f'HTTP {create_resp.status}'}"
            sample.latency_s = time.monotonic() - ctx.start_mono
            sample.extra["lifecycle_s"] = sample.latency_s
            return await self._run_post_hooks(ctx, create_req, create_resp, sample)

        create_obj = _try_json(create_resp.body)
        if not (isinstance(create_obj, dict) and "id" in create_obj):
            # Create returned 2xx but the body is unparseable — treat as a
            # transport-layer failure so it's bucketed under "failed", not "wrong".
            sample.request_ok = False
            sample.error = f"create returned unexpected body: {create_resp.body!r}"
            sample.latency_s = time.monotonic() - ctx.start_mono
            sample.extra["lifecycle_s"] = sample.latency_s
            return await self._run_post_hooks(ctx, create_req, create_resp, sample)
        sid = str(create_obj["id"])
        sample.meta["sandbox_id"] = sid

        # Step 2: exec
        endpoint = "pshell" if persistent else "exec"
        exec_body: dict[str, Any] = {"command": command}
        if env:
            exec_body["env"] = env
        if stdin is not None:
            exec_body["input"] = stdin
        exec_req = Request(
            method="POST",
            url=f"{self._base_url}{self._prefix}/{sid}/{endpoint}",
            headers=dict(self._headers),
            json=exec_body,
            timeout_s=self._timeout_s,
            meta={"sandbox_id": sid, "sandbox_operation": endpoint, "command": command},
        )
        exec_resp = await ctx.fire(exec_req)
        sample.extra["exec_s"] = exec_resp.elapsed_s
        sample.bytes_recv = len(exec_resp.body or b"")
        sample.bytes_sent += _estimate_request_size(exec_req)
        sample.status = exec_resp.status

        exit_code: Optional[int] = None
        exec_obj = _try_json(exec_resp.body) if exec_resp.ok else None
        if isinstance(exec_obj, dict):
            ec = exec_obj.get("exit_code")
            if ec is not None:
                try:
                    exit_code = int(ec)
                    sample.extra["exit_code"] = float(exit_code)
                except (TypeError, ValueError):
                    pass
            dur = exec_obj.get("duration")
            if dur is not None:
                try:
                    sample.extra["server_duration_s"] = float(dur)
                except (TypeError, ValueError):
                    pass
            if isinstance(exec_obj.get("stdout"), str):
                sample.extra["stdout_bytes"] = float(len(exec_obj["stdout"]))
            if isinstance(exec_obj.get("stderr"), str):
                sample.extra["stderr_bytes"] = float(len(exec_obj["stderr"]))

        # Step 3: delete (best-effort; always attempted so the sandbox doesn't leak)
        delete_req = Request(
            method="DELETE",
            url=f"{self._base_url}{self._prefix}/{sid}",
            headers=dict(self._headers),
            timeout_s=self._timeout_s,
            meta={"sandbox_id": sid, "sandbox_operation": "delete"},
        )
        delete_resp = await ctx.fire(delete_req)
        sample.extra["delete_s"] = delete_resp.elapsed_s

        sample.latency_s = time.monotonic() - ctx.start_mono
        sample.extra["lifecycle_s"] = sample.latency_s

        # Exec is the meaningful HTTP step — anchor request_ok on it.
        sample.request_ok = exec_resp.ok

        if not exec_resp.ok:
            sample.error = f"exec failed: {exec_resp.error or f'HTTP {exec_resp.status}'}"
        elif exit_code is not None and exit_code != 0:
            sample.error = f"exit_code={exit_code}"
        else:
            sample.ok = True

        if not delete_resp.ok:
            sample.meta["delete_error"] = delete_resp.error or f"HTTP {delete_resp.status}"

        return await self._run_post_hooks(ctx, exec_req, exec_resp, sample)

    async def _run_file(self, ctx: TicketContext) -> Sample:
        sid = await self._ensure_sandbox()
        path, content, verify_exec, verify_command = self._parse_file_item(ctx.item)

        put_req = self._make_file_put_request(sid, path, content)
        put_resp = await ctx.fire(put_req)

        sample = Sample(
            start_ts=ctx.start_mono,
            latency_s=time.monotonic() - ctx.start_mono,
            status=put_resp.status,
            ok=False,
            request_ok=put_resp.ok,
            bytes_recv=len(put_resp.body or b""),
            bytes_sent=_estimate_request_size(put_req),
            error=None,
            workload=self.name,
            meta={
                "sandbox_operation": "file",
                "sandbox_id": sid,
                "file_path": path,
            },
        )
        sample.extra["file_write_s"] = put_resp.elapsed_s
        sample.extra["file_write_bytes"] = float(len(content))
        last_req: Request = put_req
        last_resp: Response = put_resp

        if not put_resp.ok:
            sample.error = f"file put failed: {put_resp.error or f'HTTP {put_resp.status}'}"
            sample.latency_s = time.monotonic() - ctx.start_mono
            return await self._run_post_hooks(ctx, last_req, last_resp, sample)

        get_req = self._make_file_get_request(sid, path)
        get_resp = await ctx.fire(get_req)
        last_req, last_resp = get_req, get_resp
        sample.extra["file_read_s"] = get_resp.elapsed_s
        sample.extra["file_read_bytes"] = float(len(get_resp.body or b""))
        sample.bytes_sent += _estimate_request_size(get_req)
        sample.bytes_recv += len(get_resp.body or b"")
        sample.status = get_resp.status

        read_mismatch = 0
        if not get_resp.ok:
            sample.error = f"file get failed: {get_resp.error or f'HTTP {get_resp.status}'}"
        else:
            read_mismatch = _count_mismatched_bytes(content, get_resp.body or b"")
            sample.extra["file_mismatch_count"] = 1.0 if read_mismatch else 0.0
            sample.extra["file_mismatch_bytes"] = float(read_mismatch)
            if read_mismatch:
                sample.error = (
                    f"file content mismatch for {path!r}: {read_mismatch} mismatched bytes"
                )

        exec_mismatch = 0
        exec_resp: Optional[Response] = None
        if verify_exec and get_resp.ok:
            # Only the default `cat <path>` verifier is expected to echo the
            # file content back, so only then do we compare stdout to it. A
            # custom verify_command (e.g. a checksum) is graded on its exit
            # code alone — comparing its output to the bytes would always
            # register a spurious mismatch.
            compare_stdout = verify_command is None
            command = verify_command or ["sh", "-c", f"cat {shlex.quote(path)}"]
            exec_req = self._make_exec_request(sid, {"command": command})
            exec_resp = await ctx.fire(exec_req)
            last_req, last_resp = exec_req, exec_resp
            sample.extra["file_exec_s"] = exec_resp.elapsed_s
            sample.bytes_sent += _estimate_request_size(exec_req)
            sample.bytes_recv += len(exec_resp.body or b"")
            sample.status = exec_resp.status

            if not exec_resp.ok:
                if sample.error is None:
                    sample.error = (
                        f"file verify exec failed: "
                        f"{exec_resp.error or f'HTTP {exec_resp.status}'}"
                    )
            else:
                exec_obj = _try_json(exec_resp.body)
                if compare_stdout:
                    stdout_bytes = _extract_stdout_bytes(exec_obj)
                    if stdout_bytes is not None:
                        exec_mismatch = _count_mismatched_bytes(content, stdout_bytes)
                        sample.extra["file_exec_mismatch_count"] = 1.0 if exec_mismatch else 0.0
                        sample.extra["file_exec_mismatch_bytes"] = float(exec_mismatch)
                        if exec_mismatch and sample.error is None:
                            sample.error = (
                                "file exec verify mismatch: "
                                f"{exec_mismatch} mismatched bytes"
                            )
                ec: Optional[int] = None
                if isinstance(exec_obj, dict) and exec_obj.get("exit_code") is not None:
                    try:
                        ec = int(exec_obj.get("exit_code"))
                    except (TypeError, ValueError):
                        ec = None
                if ec is not None and ec != 0 and sample.error is None:
                    sample.error = f"file verify exec exit_code={ec}"

        exec_ok = exec_resp is None or exec_resp.ok
        sample.request_ok = put_resp.ok and get_resp.ok and exec_ok
        sample.latency_s = time.monotonic() - ctx.start_mono
        sample.ok = sample.error is None and read_mismatch == 0 and exec_mismatch == 0
        return await self._run_post_hooks(ctx, last_req, last_resp, sample)

    @staticmethod
    async def _run_post_hooks(ctx: TicketContext, req: Request, resp: Response,
                              sample: Sample) -> Sample:
        for hook in ctx.post_hooks:
            sample = await maybe_await(hook(req, resp, sample))
        return sample

    async def _ensure_sandbox(self) -> str:
        if self._sandbox_id:
            return self._sandbox_id
        async with self._create_lock:
            if self._sandbox_id:
                return self._sandbox_id
            sid = await self._create_sandbox()
            self._sandbox_id = sid
            self._owned_sandbox = True
            return sid

    async def _create_sandbox(self) -> str:
        body = dict(self._spec)
        if self._ttl_seconds is not None and "ttl_seconds" not in body:
            body["ttl_seconds"] = self._ttl_seconds
        session = await self._get_aux_session()
        timeout = aiohttp.ClientTimeout(total=self._create_timeout_s)
        async with session.post(
            f"{self._base_url}{self._prefix}",
            json=body,
            headers=self._headers,
            timeout=timeout,
        ) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"sandbox create failed: HTTP {resp.status}: {text}")
            try:
                data = json.loads(text)
            except ValueError as e:
                raise RuntimeError(f"sandbox create returned non-JSON: {text!r}") from e
            if not isinstance(data, dict) or "id" not in data:
                raise RuntimeError(f"sandbox create returned unexpected body: {data!r}")
            return str(data["id"])

    async def _get_aux_session(self) -> aiohttp.ClientSession:
        if self._aux_session is None:
            async with self._aux_session_lock:
                if self._aux_session is None:
                    self._aux_session = aiohttp.ClientSession()
        return self._aux_session

    async def aclose(self) -> None:
        if self._owned_sandbox and self._cleanup_on_close and self._sandbox_id:
            sid = self._sandbox_id
            try:
                session = await self._get_aux_session()
                timeout = aiohttp.ClientTimeout(total=self._create_timeout_s)
                async with session.delete(
                    f"{self._base_url}{self._prefix}/{sid}",
                    headers=self._headers,
                    timeout=timeout,
                ) as resp:
                    await resp.read()
            except Exception:
                pass
            finally:
                self._sandbox_id = None
                self._owned_sandbox = False
        if self._aux_session is not None:
            try:
                await self._aux_session.close()
            finally:
                self._aux_session = None


def _coerce_command(cmd: Any) -> list[str]:
    if isinstance(cmd, str):
        return ["sh", "-c", cmd]
    if isinstance(cmd, list) and all(isinstance(x, str) for x in cmd):
        return list(cmd)
    raise TypeError(f"command must be str or list[str], got {type(cmd).__name__}")


def _try_json(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None


def _coerce_file_content(content: Any) -> bytes:
    if isinstance(content, bytes):
        return content
    if isinstance(content, bytearray):
        return bytes(content)
    if isinstance(content, str):
        return content.encode("utf-8")
    raise TypeError(
        f"file content must be bytes|bytearray|str, got {type(content).__name__}"
    )


def _count_mismatched_bytes(expected: bytes, got: bytes) -> int:
    # Fast path for the overwhelmingly common (no-corruption) case: a C-level
    # bytes compare instead of an O(n) Python loop. This matters because the
    # comparison runs per ticket and is charged to the sample latency, so for
    # large file payloads the naive loop would skew the measured latency.
    if expected == got:
        return 0
    common = min(len(expected), len(got))
    mismatched = sum(1 for i in range(common) if expected[i] != got[i])
    mismatched += abs(len(expected) - len(got))
    return mismatched


def _extract_stdout_bytes(exec_obj: Any) -> Optional[bytes]:
    """Best-effort bytes view of exec stdout for verifier comparisons.

    Prefer latin-1 to preserve single-byte round-trips from test stubs; if the
    text contains code points outside latin-1, fall back to UTF-8 bytes.
    """
    if not isinstance(exec_obj, dict):
        return None
    out = exec_obj.get("stdout")
    if isinstance(out, str):
        try:
            return out.encode("latin-1")
        except UnicodeEncodeError:
            return out.encode("utf-8")
    if isinstance(out, bytes):
        return out
    return None

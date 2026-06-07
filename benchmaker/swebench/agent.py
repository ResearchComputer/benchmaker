"""Small SWE-style coding agent for benchmaker.

A faithful but compact port of mini-swe-agent's control flow
(https://github.com/SWE-agent/mini-swe-agent). The loop is:

  1. Render a system prompt + a task message (with the task baked in).
  2. Ask the model for one reply.
  3. Extract a single ```bash ...``` fence as the action.
  4. If the action's first line is `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`,
     the remaining lines are the submission and the agent halts.
  5. Otherwise run the action (locally or in a Flash Sandbox pod), capture
     stdout+stderr, and feed it back as the next user message. Repeat.

Halt conditions: submission, no fenced action in a reply, `step_limit`
exceeded, or `total_wall_s` exceeded.

The model side is intentionally pluggable. Pass either an OpenAI-compatible
chat endpoint (``url`` + ``model`` + ``api_key``) or a ``send_fn`` callable —
the latter is what the tests use to drive the agent without an LLM. Each call
is one HTTP round-trip; per-trajectory metrics (``steps``, ``elapsed_s``) are
surfaced via ``AgentResult.metrics``.

Execution backend: by default each action runs in a local subprocess under a
temp dir. Pass ``sandbox_url=`` (e.g. ``https://sandbox.swissai.cscs.ch``) to
allocate one Flash Sandbox pod per task instead; ``sandbox_spec`` controls the
pod shape (image, ``cpu_cores``, ``memory_mb``).

YAML wiring lives in ``examples/swebench/config.yaml``.
"""
import re
import aiohttp
import os
import asyncio
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional
from benchmaker import Agent, AgentContext, AgentResult


SUBMIT_TOKEN = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


# An executor runs one shell action and returns (returncode, combined-output).
# This is the seam that decouples the agent *loop* from *where* commands run — a
# local subprocess, a benchmaker Flash Sandbox pod, or a harbor environment.
# Plug a different executor in and the same loop drives a different backend.
Executor = Callable[[str, float], Awaitable[tuple[int, str]]]


@dataclass
class LoopResult:
    """Outcome of one ``run_loop`` trajectory (backend-agnostic)."""

    submission: Optional[str]
    exit_status: str
    n_calls: int
    n_actions: int
    elapsed_s: float
    messages: list[dict] = field(default_factory=list)

SYSTEM_PROMPT = """You are a senior software engineer working in a fresh shell.

Solve the user's task by emitting **exactly one** shell command per reply,
wrapped in a fenced bash block:

```bash
your-command-here
```

After each command you will see its output, then plan your next step.
When you are done, emit a fenced block whose **first line** is exactly:

    COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

…followed by the final answer (or output) on subsequent lines. Example:

```bash
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
42
```

Guidelines:
- One bash block per reply. No prose outside the block is required.
- Prefer `cat`, `grep`, `sed -n`, `python -c "..."` over heavy edits.
- Do not invoke interactive editors. Use `python -c` or heredocs to write files.
"""

INSTANCE_TEMPLATE = "# Task\n\n{task}\n\nBegin."

# Greedy across newlines; first fence wins. Accepts ```bash, ```sh, ```shell,
# or a bare ``` fence (some models drop the language tag).
_BASH_FENCE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)


SendFn = Callable[[list[dict]], Awaitable[str]]


_DEFAULT_SANDBOX_SPEC: dict[str, Any] = {
    "type": "docker",
    "image": "alpine:3.20",
    "command": ["sh", "-c", "sleep 3600"],
    "cpu_cores": 0.1,
}


class CodingAgent(Agent):
    """A minimal SWE-style coding agent.

    Args:
        url, model, api_key: OpenAI-compatible chat-completions endpoint.
            Unused if ``send_fn`` is provided.
        send_fn: async ``(messages) -> assistant_text`` callable. Lets the
            caller plug in any LLM client (or mock).
        system_prompt / instance_template: overridable prompt templates.
            ``instance_template`` is formatted with ``{task}``.
        step_limit: max model calls per task.
        timeout_per_step_s: per-command shell timeout.
        total_wall_s: hard wall-clock cap on the whole trajectory.
        cwd_template: working-dir template (formatted with the item dict).
            If unset, a fresh tmp dir is created per task. Ignored when
            ``sandbox_url`` is set (the sandbox pod is the workspace).
        keep_cwd: if False (default), tmp dirs created by this agent are
            removed after the task finishes. Ignored when ``cwd_template`` is set.
        max_obs_chars: head+tail truncation cap on each observation fed back
            to the model (keeps the context bounded on chatty commands).
        sandbox_url: if set (e.g. ``https://sandbox.swissai.cscs.ch``), run
            each task inside a fresh Flash Sandbox pod instead of a local
            subprocess. One pod is allocated per task and deleted on teardown.
        sandbox_spec: pod spec merged over the default
            ({image: alpine:3.20, cpu_cores: 0.1, ...}). Set ``cpu_cores`` /
            ``memory_mb`` / ``image`` here.
        sandbox_endpoint_prefix: ``/sandboxes`` (cluster, default) or
            ``/native/sandboxes`` (single node).
        sandbox_ttl_seconds: safety-net TTL so leaked pods get reaped server-side.
        sandbox_headers: extra HTTP headers for sandbox requests (e.g. auth).
        sandbox_persistent: when True (default), exec via ``/pshell`` so
            ``cd`` / ``export`` persist across steps; False uses ``/exec``.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        send_fn: Optional[SendFn] = None,
        system_prompt: str = SYSTEM_PROMPT,
        instance_template: str = INSTANCE_TEMPLATE,
        step_limit: int = 30,
        timeout_per_step_s: float = 60.0,
        total_wall_s: float = 600.0,
        cwd_template: Optional[str] = None,
        keep_cwd: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_obs_chars: int = 4000,
        sandbox_url: Optional[str] = None,
        sandbox_spec: Optional[dict[str, Any]] = None,
        sandbox_endpoint_prefix: str = "/sandboxes",
        sandbox_ttl_seconds: Optional[int] = None,
        sandbox_headers: Optional[dict[str, str]] = None,
        sandbox_persistent: bool = True,
        sandbox_create_timeout_s: float = 60.0,
    ):
        self._url = url
        self._model = model
        self._api_key = api_key
        self._send_fn = send_fn
        self._system_prompt = system_prompt
        self._instance_template = instance_template
        self._step_limit = step_limit
        self._timeout_per_step_s = timeout_per_step_s
        self._total_wall_s = total_wall_s
        self._cwd_template = cwd_template
        self._keep_cwd = keep_cwd
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_obs_chars = max_obs_chars
        self._session: Optional[aiohttp.ClientSession] = None

        self._sandbox_url = sandbox_url.rstrip("/") if sandbox_url else None
        self._sandbox_spec = {**_DEFAULT_SANDBOX_SPEC, **(sandbox_spec or {})}
        prefix = sandbox_endpoint_prefix or "/sandboxes"
        self._sandbox_prefix = "/" + prefix.strip("/")
        self._sandbox_ttl_seconds = sandbox_ttl_seconds
        self._sandbox_headers = {"Content-Type": "application/json",
                                 **(sandbox_headers or {})}
        self._sandbox_persistent = sandbox_persistent
        self._sandbox_create_timeout_s = sandbox_create_timeout_s

    # ---- model -------------------------------------------------------- #

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _send(self, messages: list[dict]) -> str:
        if self._send_fn is not None:
            return await self._send_fn(messages)
        if not (self._url and self._model):
            raise ValueError(
                "CodingAgent needs (url + model) or a send_fn to talk to a model"
            )
        sess = await self._ensure_session()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        async with sess.post(self._url, headers=headers, json=body) as resp:
            text = await resp.text()
            if resp.status >= 400:
                # Surface the upstream error verbatim (truncated) — the
                # exception bubbles up to AgentWorkloadType, which bucket-marks
                # the sample as a real infrastructure failure ("fail").
                excerpt = text if len(text) <= 500 else text[:500] + "...[truncated]"
                raise RuntimeError(
                    f"model endpoint HTTP {resp.status}: {excerpt}"
                )
            import json as _json
            try:
                data = _json.loads(text)
            except _json.JSONDecodeError as e:
                excerpt = text if len(text) <= 500 else text[:500] + "...[truncated]"
                raise RuntimeError(
                    f"model endpoint returned non-JSON: {excerpt}"
                ) from e
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"model endpoint returned no choices: {data!r}")
        return (choices[0].get("message") or {}).get("content") or ""

    # ---- action / shell ----------------------------------------------- #

    @staticmethod
    def _parse_action(reply: str) -> Optional[str]:
        m = _BASH_FENCE.search(reply or "")
        return m.group(1).strip() if m else None

    async def _bash(self, cmd: str, cwd: str) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout_per_step_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return -1, f"<command timed out after {self._timeout_per_step_s}s>"
        return (proc.returncode or 0), out.decode("utf-8", errors="replace")

    # ---- sandbox ------------------------------------------------------ #

    async def _sandbox_create(self) -> str:
        body = dict(self._sandbox_spec)
        if self._sandbox_ttl_seconds is not None and "ttl_seconds" not in body:
            body["ttl_seconds"] = self._sandbox_ttl_seconds
        sess = await self._ensure_session()
        url = f"{self._sandbox_url}{self._sandbox_prefix}"
        timeout = aiohttp.ClientTimeout(total=self._sandbox_create_timeout_s)
        async with sess.post(url, headers=self._sandbox_headers,
                             json=body, timeout=timeout) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(
                    f"sandbox create HTTP {resp.status}: {_excerpt(text)}"
                )
            import json as _json
            try:
                data = _json.loads(text)
            except _json.JSONDecodeError as e:
                raise RuntimeError(
                    f"sandbox create returned non-JSON: {_excerpt(text)}"
                ) from e
        if not isinstance(data, dict) or "id" not in data:
            raise RuntimeError(f"sandbox create unexpected body: {data!r}")
        return str(data["id"])

    async def _sandbox_exec(self, sid: str, cmd: str) -> tuple[int, str]:
        endpoint = "pshell" if self._sandbox_persistent else "exec"
        url = f"{self._sandbox_url}{self._sandbox_prefix}/{sid}/{endpoint}"
        body = {"command": ["sh", "-c", cmd]}
        sess = await self._ensure_session()
        # Pad the client-side timeout so the server can return its own
        # timeout error rather than us tearing down the connection first.
        timeout = aiohttp.ClientTimeout(total=self._timeout_per_step_s + 30.0)
        try:
            async with sess.post(url, headers=self._sandbox_headers,
                                 json=body, timeout=timeout) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    return -1, f"<sandbox exec HTTP {resp.status}: {_excerpt(text)}>"
                import json as _json
                try:
                    data = _json.loads(text)
                except _json.JSONDecodeError:
                    return -1, f"<sandbox exec non-JSON: {_excerpt(text)}>"
        except asyncio.TimeoutError:
            return -1, f"<sandbox exec timed out after {self._timeout_per_step_s}s>"
        rc = data.get("exit_code")
        try:
            rc_i = int(rc) if rc is not None else 0
        except (TypeError, ValueError):
            rc_i = 0
        stdout = data.get("stdout") or ""
        stderr = data.get("stderr") or ""
        combined = stdout if not stderr else (stdout + ("\n" if stdout else "") + stderr)
        return rc_i, combined

    async def _sandbox_delete(self, sid: str) -> None:
        sess = await self._ensure_session()
        url = f"{self._sandbox_url}{self._sandbox_prefix}/{sid}"
        timeout = aiohttp.ClientTimeout(total=self._sandbox_create_timeout_s)
        try:
            async with sess.delete(url, headers=self._sandbox_headers,
                                   timeout=timeout) as resp:
                await resp.read()
        except Exception:
            # Best-effort cleanup; never fail the trajectory over this.
            pass

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_obs_chars:
            return text
        half = self._max_obs_chars // 2
        return text[:half] + "\n...[truncated]...\n" + text[-half:]

    # ---- cwd ---------------------------------------------------------- #

    def _resolve_cwd(self, item: Any) -> tuple[str, bool]:
        """Return (cwd, owned). When `owned` is True, the agent created the dir
        and will remove it unless `keep_cwd=True`."""
        if self._cwd_template:
            tmpl = self._cwd_template
            cwd = tmpl.format(**item) if isinstance(item, dict) else tmpl
            os.makedirs(cwd, exist_ok=True)
            return cwd, False
        return tempfile.mkdtemp(prefix="bm-agent-"), True

    # ---- main loop ---------------------------------------------------- #

    async def run_loop(
        self,
        task: str,
        executor: Executor,
        *,
        system_prompt: Optional[str] = None,
        instance_template: Optional[str] = None,
    ) -> LoopResult:
        """Drive the model/observe loop against an injected ``executor``.

        This is the backend-agnostic core: it owns the conversation, action
        parsing, and halting (submission / no-action / step-limit / time-limit),
        but runs every shell action through ``executor`` instead of a hard-wired
        backend. ``CodingAgent.run`` wires in a local-subprocess or Flash Sandbox
        executor; a harbor ``BaseAgent`` can wire in ``environment.exec`` — same
        loop, different host. See ``harbor_agent.py``.

        ``executor`` is ``async (action, timeout_s) -> (returncode, output)``.
        Sandbox / workspace lifecycle is the *caller's* responsibility.
        """
        sys_p = system_prompt if system_prompt is not None else self._system_prompt
        inst_t = (instance_template if instance_template is not None
                  else self._instance_template)
        messages: list[dict] = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": inst_t.format(task=task)},
        ]
        start = time.monotonic()
        n_calls = 0
        n_actions = 0
        submission: Optional[str] = None
        exit_status = "step_limit"

        while True:
            if n_calls >= self._step_limit:
                exit_status = "step_limit"
                break
            if time.monotonic() - start > self._total_wall_s:
                exit_status = "time_limit"
                break

            reply = await self._send(messages)
            n_calls += 1
            messages.append({"role": "assistant", "content": reply})

            action = self._parse_action(reply)
            if action is None:
                exit_status = "no_action"
                break

            first_line = action.splitlines()[0].strip() if action.strip() else ""
            if first_line == SUBMIT_TOKEN:
                submission = "\n".join(action.splitlines()[1:]).strip()
                exit_status = "submitted"
                break

            rc, output = await executor(action, self._timeout_per_step_s)
            n_actions += 1
            obs = self._truncate(output)
            messages.append({
                "role": "user",
                "content": f"$ {action}\nreturncode: {rc}\n```\n{obs}\n```",
            })

        return LoopResult(
            submission=submission,
            exit_status=exit_status,
            n_calls=n_calls,
            n_actions=n_actions,
            elapsed_s=time.monotonic() - start,
            messages=messages,
        )

    async def run(self, ctx: AgentContext) -> AgentResult:
        item = ctx.item if isinstance(ctx.item, dict) else {}
        task = item.get("prompt") or item.get("instruction") or ""

        cwd: Optional[str] = None
        owned = False
        sandbox_id: Optional[str] = None
        if self._sandbox_url:
            sandbox_id = await self._sandbox_create()
            async def executor(action: str, _timeout: float) -> tuple[int, str]:
                return await self._sandbox_exec(sandbox_id, action)
        else:
            cwd, owned = self._resolve_cwd(ctx.item)
            async def executor(action: str, _timeout: float) -> tuple[int, str]:
                return await self._bash(action, cwd)

        try:
            result = await self.run_loop(task, executor)
        finally:
            if sandbox_id is not None:
                await self._sandbox_delete(sandbox_id)
            elif owned and not self._keep_cwd and cwd is not None:
                # Best-effort cleanup; never fail the trajectory over this.
                import shutil
                shutil.rmtree(cwd, ignore_errors=True)

        return AgentResult(
            output=(result.submission or "").strip(),
            ok=(result.exit_status == "submitted"),
            error=None if result.exit_status == "submitted" else result.exit_status,
            metrics={
                "steps": float(result.n_calls),
                "actions": float(result.n_actions),
                "elapsed_s": float(result.elapsed_s),
            },
            meta={
                "exit_status": result.exit_status,
                "cwd": cwd,
                "sandbox_id": sandbox_id,
                "task_id": item.get("task_id"),
            },
        )


def _excerpt(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else text[:limit] + "...[truncated]"

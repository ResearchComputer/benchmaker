"""Track B: generate *verified* SWE-bench trajectories with our own agent.

This is the "native runner" track. The agent rollout and the test verification
both live here; we reuse benchmaker's `Agent`/`AgentResult` abstraction and the
Flash Sandbox wire protocol rather than an external harness.

Pipeline, per SWE-bench instance:

  1. Rollout — create a Flash Sandbox from the instance's prebuilt SWE-bench
     image (repo checked out at `base_commit` in /testbed). Drive the model
     (`.env`) in an OpenAI tool-calling loop with a `bash` tool that execs in
     the sandbox and a `submit` tool to finish. The full conversation
     (assistant content + reasoning + tool_calls, tool results) is recorded in
     the canonical schema.
  2. Patch — `git add -A && git diff` in /testbed → the agent's `model_patch`.
  3. Verify — in a fresh sandbox, apply `model_patch` + the hidden `test_patch`,
     run the FAIL_TO_PASS / PASS_TO_PASS tests, and classify resolution using
     the `swebench` package's authoritative test spec + grading.
  4. Emit — one `WarmupRecord` with `verified = resolved` and a `verification`
     block carrying the per-test report.

The agent never sees `test_patch`, so it can't cheat; verification is the only
thing that flips `verified` to True.

Prerequisites (the live path needs all of these — the pure helpers don't):
  * Flash Sandbox orchestrator reachable at `FLASH_SANDBOX_URL` (Docker backend).
  * `pip install swebench` — for per-instance image keys, eval scripts, grading.
  * The SWE-bench per-instance Docker images pullable by the sandbox host.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
from typing import Any, Optional

from benchmaker.env import load_dotenv
from benchmaker.swebench.grading import grade, make_test_spec
from benchmaker.workloads.agent import Agent, AgentContext, AgentResult

from . import protocol as P

WORKDIR = "/testbed"
MAX_TOOL_OUTPUT_CHARS = 4000


# --------------------------------------------------------------------------- #
# Tool schemas exposed to the model
# --------------------------------------------------------------------------- #

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in the repository sandbox. Each call runs "
                "in a FRESH, non-interactive shell starting at /testbed — the "
                "working directory, environment variables, activated "
                "virtualenvs, and background processes do NOT persist between "
                "calls. Chain dependent steps in one command with '&&', use "
                "absolute paths or re-cd as needed, and supply any input "
                "non-interactively (heredocs, pipes, '-y'/'--yes' flags). "
                "Interactive programs needing a TTY (REPLs held open, editors, "
                "prompts) will not work. Output is truncated if very long."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "the bash command"},
                    "timeout": {"type": "integer",
                                "description": "max seconds (default 120)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Call when the fix is complete and ready to be tested.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_SYSTEM_PROMPT = (
    "You are an expert software engineer fixing a bug in a checked-out "
    "repository at /testbed. Investigate with the `bash` tool, implement a "
    "minimal correct fix, and verify it. Do not modify or write tests.\n\n"
    "Your `bash` tool runs each command in a fresh, non-interactive shell that "
    "starts at /testbed; the working directory, environment variables, and any "
    "activated virtualenv do NOT carry over between calls. Plan around this: "
    "combine dependent steps into one command, use absolute paths, set up the "
    "environment as part of each command if needed, and never launch a program "
    "that waits for interactive input. When the fix is complete, call `submit`."
)


def build_initial_messages(instance: dict[str, Any]) -> list[dict[str, Any]]:
    """The system + first user turn for a SWE-bench instance."""
    problem = instance.get("problem_statement", "")
    hints = instance.get("hints_text") or ""
    repo = instance.get("repo", "")
    user = f"Repository: {repo}\n\n## Issue\n{problem}"
    if hints.strip():
        user += f"\n\n## Hints\n{hints}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# Flash Sandbox HTTP client (mirrors benchmaker.workloads.sandbox wire protocol)
# --------------------------------------------------------------------------- #


class SandboxSession:
    """One sandbox: create -> exec* -> delete.

    The Flash Sandbox `exec` endpoint does not persist the working directory
    between calls (each command runs from `/`), so we anchor every command at
    `cwd` when set — the agent always operates relative to the repo root.
    """

    def __init__(self, base_url: str, spec: dict[str, Any],
                 prefix: str = "/sandboxes", timeout_s: float = 600.0,
                 cwd: Optional[str] = None):
        self._base = base_url.rstrip("/")
        self._prefix = "/" + prefix.strip("/")
        self._spec = spec
        self._timeout_s = timeout_s
        self._cwd = cwd
        self._id: Optional[str] = None
        self._session = None  # aiohttp.ClientSession

    async def __aenter__(self) -> "SandboxSession":
        import aiohttp
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout_s))
        async with self._session.post(f"{self._base}{self._prefix}",
                                       json=self._spec) as r:
            text = await r.text()
            if r.status >= 400:
                raise RuntimeError(f"sandbox create failed: HTTP {r.status}: {text}")
            data = json.loads(text)
        self._id = str(data["id"])
        return self

    async def exec(self, command: str, timeout: Optional[int] = None, *,
                   raw: bool = False) -> dict[str, Any]:
        """Run `bash -lc <command>` (anchored at cwd); return {exit_code, stdout, stderr}.

        `raw=True` skips the cwd anchor — needed for setup commands that create
        the working dir (e.g. cloning into it before it exists).
        """
        # Anchor at cwd without a subshell — wrapping in `(...)` breaks commands
        # that end in a here-document (the exec endpoint has no cwd field).
        full = command if (raw or not self._cwd) else f"cd {self._cwd} && {command}"
        body: dict[str, Any] = {"command": ["bash", "-lc", full]}
        if timeout:
            body["timeout"] = timeout
        url = f"{self._base}{self._prefix}/{self._id}/exec"
        async with self._session.post(url, json=body) as r:
            text = await r.text()
            if r.status >= 400:
                return {"exit_code": -1, "stdout": "", "stderr": f"HTTP {r.status}: {text}"}
            obj = json.loads(text) if text else {}
        return {
            "exit_code": obj.get("exit_code", -1),
            "stdout": obj.get("stdout", ""),
            "stderr": obj.get("stderr", ""),
        }

    async def write_file(self, path: str, content: str) -> None:
        """Write text to a file in the sandbox (base64 to dodge shell escaping)."""
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        await self.exec(f"echo {b64} | base64 -d > {path}")

    async def __aexit__(self, *exc: Any) -> None:
        try:
            if self._id is not None:
                async with self._session.delete(
                        f"{self._base}{self._prefix}/{self._id}") as r:
                    await r.read()
        except Exception:
            pass
        finally:
            if self._session is not None:
                await self._session.close()


# --------------------------------------------------------------------------- #
# OpenAI-compatible chat client (non-streaming, tool-calling)
# --------------------------------------------------------------------------- #


class ChatModel:
    def __init__(self, url: str, model: str, api_key: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 4096,
                 timeout_s: float = 600.0,
                 chat_template_kwargs: Optional[dict[str, Any]] = None):
        self.url = url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Extra kwargs forwarded to the server's chat-template render. On the CSCS
        # SGLang Kimi-K2.5 deployment we pass {"thinking": False}: with thinking on,
        # the template renders a tool-followup turn the model completes with an
        # immediate stop (empty assistant turn). The model emits no reasoning when
        # `tools` are present anyway, so disabling it loses nothing here.
        self.chat_template_kwargs = chat_template_kwargs
        # Learned from the server the first time a request overflows the context
        # window; once known we clamp `max_tokens` per request so input+completion
        # always fits (a too-large --max-tokens otherwise 400s every call).
        self.context_window: Optional[int] = None
        self._clamp_notified = False
        self._timeout_s = timeout_s
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._session = None

    @classmethod
    def from_env(cls, env_path: Optional[str], model_override: Optional[str],
                 **kw: Any) -> "ChatModel":
        load_dotenv(env_path or ".env")
        base = (os.environ.get("OPENAI_API_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_API_BASE"))
        if not base:
            raise SystemExit("OPENAI_API_BASE_URL not set (need it for the model).")
        model = (model_override or os.environ.get("OPENAI_COMPATIBLE_MODEL")
                 or os.environ.get("OPENAI_MODEL"))
        if not model:
            raise SystemExit("OPENAI_COMPATIBLE_MODEL not set (and no --model).")
        url = base.rstrip("/") + "/chat/completions"
        return cls(url=url, model=model, api_key=os.environ.get("OPENAI_API_KEY"), **kw)

    async def _ensure(self) -> None:
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout_s))

    async def complete(self, messages: list[dict[str, Any]],
                       tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the assistant `message` + `usage` + `finish_reason`.

        Robust to an over-large `max_tokens`: the request budget is clamped so
        input+completion fits the model's context window. The window is learned
        from the server's context-length 400 (then cached), and as a backstop any
        such 400 is retried with a budget computed from the error itself.
        """
        await self._ensure()
        while True:
            budget = self._completion_budget(messages, tools)
            body = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": self.temperature,
                "max_tokens": budget,
                "stream": False,
            }
            if self.chat_template_kwargs:
                body["chat_template_kwargs"] = self.chat_template_kwargs
            async with self._session.post(self.url, headers=self._headers,
                                          json=body) as r:
                text = await r.text()
                status = r.status
            if status < 400:
                data = json.loads(text)
                choice = (data.get("choices") or [{}])[0]
                return {"message": choice.get("message") or {},
                        "usage": data.get("usage") or {},
                        "finish_reason": choice.get("finish_reason")}
            over = _parse_context_overflow(text)
            if over is not None:
                ctx, n_input = over
                self.context_window = ctx  # learn it for future proactive clamps
                safe = ctx - n_input - _COMPLETION_MARGIN
                if safe < _MIN_COMPLETION:
                    raise RuntimeError(
                        f"prompt too long: {n_input} input tokens leave only "
                        f"{safe} of the {ctx}-token context for a completion")
                if safe < budget:
                    self._notify_clamp(ctx)
                    continue  # retry with the now-known window applied
            raise RuntimeError(f"chat completion failed: HTTP {status}: {text}")

    def _completion_budget(self, messages: list[dict[str, Any]],
                           tools: list[dict[str, Any]]) -> int:
        """`max_tokens` clamped so input+completion fits the context window."""
        if not self.context_window:
            return self.max_tokens
        est_input = _estimate_tokens(messages) + _estimate_tokens(tools)
        room = self.context_window - est_input - _COMPLETION_MARGIN
        return max(_MIN_COMPLETION, min(self.max_tokens, room))

    def _notify_clamp(self, ctx: int) -> None:
        if not self._clamp_notified:
            self._clamp_notified = True
            print(f"[generate] --max-tokens {self.max_tokens} exceeds the model's "
                  f"{ctx}-token context; clamping the completion budget per request "
                  f"to fit. Pass a smaller --max-tokens (e.g. 8192) to silence this.")

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


# --------------------------------------------------------------------------- #
# The agent (rollout only — verification happens in the runner)
# --------------------------------------------------------------------------- #


class CodingAgent(Agent):
    """Drives the model + sandbox tool loop for one SWE-bench instance."""

    def __init__(self, model: ChatModel, sandbox_base: str, image_key: str,
                 max_turns: int = 50, sandbox_type: str = "kubernetes",
                 setup_commands: Optional[list[str]] = None):
        self.model = model
        self.sandbox_base = sandbox_base
        self.image_key = image_key
        self.max_turns = max_turns
        self.sandbox_type = sandbox_type
        self.setup_commands = setup_commands or []

    async def run(self, ctx: AgentContext) -> AgentResult:
        instance = ctx.item
        spec = {"type": self.sandbox_type, "image": self.image_key,
                "command": ["sh", "-c", "sleep infinity"]}

        api_messages = build_initial_messages(instance)
        record_messages: list[dict[str, Any]] = list(api_messages)
        submitted = False
        turns = 0
        usage_total = {"prompt": 0, "completion": 0}
        request_error: Optional[str] = None
        nudged_this_turn = False

        async with SandboxSession(self.sandbox_base, spec, cwd=WORKDIR) as sbx:
            # Setup (e.g. clone the repo for non-swebench images). Run unanchored
            # since the working dir may not exist yet. Abort if setup fails.
            for cmd in self.setup_commands:
                r = await sbx.exec(cmd, timeout=600, raw=True)
                if r.get("exit_code") != 0:
                    return AgentResult(
                        output="", ok=False, request_ok=False,
                        error=f"setup failed: {_format_exec(r)[:300]}",
                        meta={"messages": record_messages, "model_patch": "",
                              "submitted": False, "n_turns": 0,
                              "model": self.model.model, "tokens": usage_total},
                    )
            # Baseline so the later diff captures only the agent's edits.
            await sbx.exec(f"cd {WORKDIR} && git config user.email a@b.c "
                           f"&& git config user.name a && git add -A "
                           f"&& git commit -q -m baseline --allow-empty")

            for turns in range(1, self.max_turns + 1):
                out = await self.model.complete(api_messages, TOOLS)
                msg, usage = out["message"], out["usage"]
                usage_total["prompt"] += int(usage.get("prompt_tokens") or 0)
                usage_total["completion"] += int(usage.get("completion_tokens") or 0)
                fr = out.get("finish_reason")

                # An assistant turn with no visible content AND no tool_calls is a
                # degenerate completion. Two distinct causes, handled differently:
                #
                #  * finish_reason="length": the model truncated mid-output (e.g. a
                #    reasoning model that spent the whole budget thinking). A retry
                #    would just truncate again — fail with an actionable error.
                #  * otherwise (typically "stop"): the thinking-mode chat-template
                #    bug on the CSCS Kimi-K2.5 SGLang deployment — a history ending
                #    in a `tool` message renders a prompt the model completes with an
                #    immediate stop. The primary fix is chat_template_kwargs
                #    {"thinking": False} (see ChatModel / --thinking); this nudge is a
                #    model-agnostic backstop: ANY further message unsticks it, so we
                #    retry once with a throwaway `user` nudge. The nudge goes into the
                #    prompt ONLY, never the recorded trajectory, so SFT data stays
                #    canonical.
                if _is_empty_turn(msg):
                    if fr != "length" and not nudged_this_turn:
                        api_messages.append({"role": "user", "content": _NUDGE})
                        nudged_this_turn = True
                        continue  # re-complete with the nudge appended
                    request_error = (
                        f"empty assistant turn at turn {turns} "
                        f"(finish_reason={fr!r}); "
                        + ("hit max_tokens mid-output — raise --max-tokens"
                           if fr == "length"
                           else "model returned no content or tool_call even after "
                                "a continuation nudge"))
                    break
                nudged_this_turn = False

                api_messages.append(_strip_for_api(msg))
                record_messages.append(_to_record_assistant(msg))

                tool_calls = msg.get("tool_calls") or []

                if not tool_calls:
                    break  # model answered with no tool call -> done

                for tc in tool_calls:
                    name, args = _parse_tool_call(tc)
                    if name == "submit":
                        submitted = True
                        result_text = "submitted"
                    elif name == "bash":
                        res = await sbx.exec(args.get("command", ""),
                                             timeout=args.get("timeout"))
                        result_text = _format_exec(res)
                    else:
                        result_text = f"error: unknown tool {name!r}"
                    tool_result = {"role": "tool", "tool_call_id": tc.get("id"),
                                   "name": name, "content": result_text}
                    api_messages.append(tool_result)
                    record_messages.append(dict(tool_result))
                if submitted:
                    break

            patch = await self._extract_patch(sbx)

        return AgentResult(
            output=patch,
            ok=submitted,
            request_ok=request_error is None,
            error=request_error,
            meta={
                "messages": record_messages,
                "model_patch": patch,
                "submitted": submitted,
                "n_turns": turns,
                "model": self.model.model,
                "tokens": usage_total,
            },
        )

    @staticmethod
    async def _extract_patch(sbx: SandboxSession) -> str:
        res = await sbx.exec(
            f"cd {WORKDIR} && git add -A && git diff --cached HEAD | base64 -w0")
        try:
            return base64.b64decode(res.get("stdout", "")).decode("utf-8", "replace")
        except Exception:
            return res.get("stdout", "")


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #


# Throwaway prompt-only message used to unstick servers whose chat template
# emits an empty assistant turn after a `tool` message (see the loop in `run`).
_NUDGE = "Continue."

# Context-budget guardrails (see ChatModel.complete). Keep a little headroom
# below the window, and never request a uselessly tiny completion.
_COMPLETION_MARGIN = 512
_MIN_COMPLETION = 256

# Matches the server's context-length 400 body, e.g. "... maximum context length
# of 262144 tokens. You requested a total of 263717 tokens: 3717 tokens from the
# input messages and 260000 tokens for the completion."
_CTX_OVERFLOW_RE = re.compile(
    r"maximum context length of (\d+) tokens.*?(\d+) tokens from the input",
    re.DOTALL)


def _parse_context_overflow(body: str) -> Optional[tuple[int, int]]:
    """(context_window, input_tokens) from a context-length 400 body, else None."""
    m = _CTX_OVERFLOW_RE.search(body)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _estimate_tokens(obj: Any) -> int:
    """Cheap upper-ish estimate of how many tokens `obj` serializes to.

    Only used to pre-clamp the completion budget so we rarely round-trip a 400;
    the server's error is the source of truth, so a rough ~3 chars/token (which
    tends to over-count, leaving extra headroom) is fine.
    """
    return len(json.dumps(obj, ensure_ascii=False)) // 3


def _is_empty_turn(msg: dict[str, Any]) -> bool:
    """True when an assistant message carries no usable content and no tool_calls."""
    return not (msg.get("content") or "").strip() and not (msg.get("tool_calls") or [])


def _parse_tool_call(tc: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """(name, parsed-arguments) from an OpenAI tool_call."""
    fn = tc.get("function") or {}
    name = fn.get("name") or ""
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        return name, raw
    try:
        parsed = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        parsed = {}
    return name, parsed if isinstance(parsed, dict) else {}


def _to_record_assistant(msg: dict[str, Any]) -> dict[str, Any]:
    """Map a raw OpenAI assistant message to the canonical record form.

    Reasoning may arrive as `reasoning_content` (vLLM/DeepSeek) or `reasoning`.
    """
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    tcs = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        tcs.append(P.make_tool_call(tc.get("id") or f"call_{i}",
                                    fn.get("name") or "", fn.get("arguments") or "{}"))
    return P.assistant_msg(msg.get("content"), reasoning=reasoning, tool_calls=tcs or None)


def _strip_for_api(msg: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields the chat API accepts when replaying the assistant turn."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def _format_exec(res: dict[str, Any]) -> str:
    parts = []
    if res.get("stdout"):
        parts.append(res["stdout"])
    if res.get("stderr"):
        parts.append("[stderr]\n" + res["stderr"])
    text = "\n".join(parts) or "(no output)"
    text = _truncate(text, MAX_TOOL_OUTPUT_CHARS)
    return f"exit_code={res.get('exit_code')}\n{text}"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2:]
    return f"{head}\n... [truncated {len(text) - limit} chars] ...\n{tail}"


# --------------------------------------------------------------------------- #
# Verification (uses the swebench package as the source of truth)
# --------------------------------------------------------------------------- #
#
# Spec-building (`make_test_spec`) and grading (`grade`) are shared with the
# eval path — see `benchmaker.swebench.grading`. `make_test_spec` uses the
# Docker Hub `swebench` namespace, so `spec.instance_image_key` is a pullable
# `swebench/sweb.eval.<arch>.<id>:latest` ref.


async def verify_patch(instance: dict[str, Any], model_patch: str,
                       sandbox_base: str,
                       sandbox_type: str = "kubernetes") -> dict[str, Any]:
    """Apply model_patch + test_patch in a fresh sandbox, run tests, grade.

    Returns a verification block: {resolved, method, fail_to_pass, pass_to_pass,
    error?}. On any failure the patch is treated as unverified (resolved False).
    """
    block: dict[str, Any] = {"method": "swebench:FAIL_TO_PASS+PASS_TO_PASS",
                             "resolved": False}
    if not model_patch.strip():
        block["error"] = "empty patch"
        return block
    try:
        spec = make_test_spec(instance)
        eval_script = spec.eval_script  # full bash: reset, apply test_patch, run tests
        image_key = spec.instance_image_key
    except Exception as e:
        block["error"] = f"swebench spec unavailable: {type(e).__name__}: {e}"
        return block

    sbx_spec = {"type": sandbox_type, "image": image_key,
                "command": ["sh", "-c", "sleep infinity"]}
    try:
        async with SandboxSession(sandbox_base, sbx_spec, cwd=WORKDIR) as sbx:
            await sbx.write_file("/tmp/patch.diff", model_patch)
            apply = await sbx.exec(
                f"cd {WORKDIR} && git apply -v /tmp/patch.diff "
                f"|| git apply --3way /tmp/patch.diff || patch -p1 < /tmp/patch.diff")
            if apply.get("exit_code") != 0:
                block["error"] = "model patch did not apply"
                return block
            await sbx.write_file("/tmp/eval.sh", eval_script)
            run = await sbx.exec("bash /tmp/eval.sh 2>&1", timeout=1800)
            block.update(grade(spec, run.get("stdout", "")))
    except Exception as e:
        block["error"] = f"verification run failed: {type(e).__name__}: {e}"
    return block


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _load_instances(dataset: str, split: str, num_tasks: Optional[int],
                    instance_ids: Optional[list[str]]) -> list[dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split)
    rows = [dict(r) for r in ds]
    if instance_ids:
        wanted = set(instance_ids)
        rows = [r for r in rows if r.get("instance_id") in wanted]
    if num_tasks is not None:
        rows = rows[:num_tasks]
    return rows


def _image_key_for(instance: dict[str, Any]) -> str:
    """Best-effort instance image key (swebench is the source of truth)."""
    return make_test_spec(instance).instance_image_key


def _resolve_image(instance: dict[str, Any],
                   fallback_image: str) -> tuple[str, Optional[list[str]]]:
    """Return `(image, setup_commands)` for an instance.

    swebench-supported repos use the prebuilt eval image (repo already at
    /testbed @ base_commit, deps installed) and need no setup. Other repos fall
    back to a generic image and clone the repo at base_commit — usable for
    *generation* only (no test env), so they can't be verified.
    """
    try:
        return make_test_spec(instance).instance_image_key, None
    except Exception:
        repo = instance.get("repo")
        base = instance.get("base_commit") or instance.get("environment_setup_commit")
        setup = [f"rm -rf {WORKDIR} && git clone https://github.com/{repo}.git "
                 f"{WORKDIR} && cd {WORKDIR} && git checkout -q {base}"]
        return fallback_image, setup


def _supported_repos() -> set[str]:
    """Repos swebench can build an eval image + test spec for.

    Track B can only verify (and even just check out) instances whose repo has a
    swebench harness spec; others have no Docker image. Used to skip e.g. the
    extra repos in the full SWE-bench train split (wagtail, …) that aren't in
    SWE-bench_Verified / _Lite.
    """
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
        return set(MAP_REPO_VERSION_TO_SPECS)
    except Exception:
        return set()


def _default_env() -> str:
    """Repo-root `.env` (the tool usually runs from the package subdir)."""
    return os.path.join(os.path.abspath(os.path.join(_HERE, "..", "..", "..")), ".env")


def _thinking_kwargs(args: argparse.Namespace) -> Optional[dict[str, Any]]:
    """chat_template_kwargs implied by --thinking (None = send nothing).

    `auto` disables thinking for Kimi models, where thinking-mode + tool calls
    makes the SGLang template emit empty assistant turns after a tool result
    (see ChatModel). Other models are left untouched.
    """
    mode = getattr(args, "thinking", "auto")
    if mode == "on":
        return None
    if mode == "off":
        return {"thinking": False}
    model = (args.model or os.environ.get("OPENAI_COMPATIBLE_MODEL")
             or os.environ.get("OPENAI_MODEL") or "")
    return {"thinking": False} if "kimi" in model.lower() else None


def _load_done_ids(out_path: str) -> set[str]:
    """Instance ids already present in an existing output file (for --resume).

    Tolerant of a truncated/corrupt trailing line (a killed run may leave one):
    unparseable lines are skipped rather than aborting the resume.
    """
    done: set[str] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = (obj.get("meta") or {}).get("instance_id")
            if iid:
                done.add(iid)
    return done


async def _run_async(args: argparse.Namespace) -> int:
    env_path = args.env or _default_env()
    load_dotenv(env_path)
    sandbox_base = os.environ.get("FLASH_SANDBOX_URL")
    if not sandbox_base:
        raise SystemExit("FLASH_SANDBOX_URL not set (need the sandbox service).")

    model = ChatModel.from_env(env_path, args.model, max_tokens=args.max_tokens,
                               chat_template_kwargs=_thinking_kwargs(args))
    instances = _load_instances(args.dataset, args.split, args.num_tasks,
                                args.instance_ids)

    # Repos the swebench harness can verify. When verification is required we
    # drop the rest (no eval image); with --skip-verification we keep them and
    # fall back to cloning into a generic image (generation only).
    skipped = 0
    supported = _supported_repos()
    if supported and not args.skip_verification:
        unsupported: dict[str, int] = {}
        kept = []
        for inst in instances:
            repo = inst.get("repo")
            if repo in supported:
                kept.append(inst)
            else:
                unsupported[repo] = unsupported.get(repo, 0) + 1
        if unsupported:
            skipped = sum(unsupported.values())
            print(f"[generate] skipping {skipped} task(s) — repos not supported by "
                  f"the swebench harness (use --skip-verification to generate them "
                  f"unverified): {dict(sorted(unsupported.items()))}")
        instances = kept

    # Resume: skip instances already in the output file and append to it, so a
    # re-run after an interrupt preserves prior rows instead of overwriting them.
    resumed = 0
    write_mode = "w"
    if args.resume:
        done = _load_done_ids(args.out)
        if done:
            before = len(instances)
            instances = [i for i in instances if i.get("instance_id") not in done]
            resumed = before - len(instances)
            write_mode = "a"
            print(f"[generate] resume: {len(done)} row(s) already in {args.out}; "
                  f"skipping {resumed} done task(s), {len(instances)} remaining")

    print(f"[generate] {len(instances)} tasks  model={model.model}  "
          f"dataset={args.dataset}:{args.split}")

    sem = asyncio.Semaphore(args.concurrency)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_fh = open(args.out, write_mode, encoding="utf-8")
    counts = {"written": 0, "verified": 0, "failed": 0, "skipped": skipped,
              "resumed": resumed}
    lock = asyncio.Lock()

    async def handle(inst: dict[str, Any]) -> None:
        iid = inst.get("instance_id", "?")
        async with sem:
            try:
                image_key, setup = _resolve_image(inst, args.fallback_image)
                agent = CodingAgent(model, sandbox_base, image_key, args.max_turns,
                                    sandbox_type=args.sandbox_type, setup_commands=setup)
                result = await agent.run(AgentContext(item=inst))
                if not result.request_ok:
                    print(f"  ! {iid}: {result.error}")
                    counts["failed"] += 1
                    return
                if args.skip_verification:
                    verification = {"method": "skipped", "resolved": False}
                else:
                    verification = await verify_patch(
                        inst, result.meta["model_patch"], sandbox_base,
                        sandbox_type=args.sandbox_type)
            except Exception as e:
                print(f"  ! {iid}: {type(e).__name__}: {e}")
                counts["failed"] += 1
                return
            rec = P.WarmupRecord(
                id=f"swebench:{iid}",
                source=args.dataset.split("/")[-1].lower(),
                messages=result.meta["messages"],
                tools=TOOLS,
                verified=bool(verification.get("resolved")),
                verification=verification,
                meta={"instance_id": iid, "repo": inst.get("repo"),
                      "base_commit": inst.get("base_commit"),
                      "model": result.meta["model"], "n_turns": result.meta["n_turns"],
                      "tokens": result.meta["tokens"]},
            )
            err = P.validate(rec)
            if err:
                print(f"  ! {iid}: invalid record: {err}")
                counts["failed"] += 1
                return
            if rec.verified:
                counts["verified"] += 1
            if rec.verified or args.keep_unverified or args.skip_verification:
                async with lock:
                    out_fh.write(rec.to_json() + "\n")
                    out_fh.flush()
                    counts["written"] += 1
            if args.skip_verification:
                tag = "generated (unverified)"
            else:
                tag = "RESOLVED" if rec.verified else "unresolved"
            print(f"  - {iid}: {tag} (turns={rec.meta['n_turns']})")

    try:
        await asyncio.gather(*(handle(i) for i in instances))
    finally:
        out_fh.close()
        await model.aclose()

    print(f"[done] wrote {counts['written']:,} rows to {args.out} "
          f"(verified {counts['verified']:,}, failed {counts['failed']:,}, "
          f"skipped {counts['skipped']:,}, resumed {counts['resumed']:,})")
    return 0


def run_generate(args: argparse.Namespace) -> int:
    return asyncio.run(_run_async(args))

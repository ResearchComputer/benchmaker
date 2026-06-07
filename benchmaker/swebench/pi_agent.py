"""Run **pi** (`@earendil-works/pi-coding-agent`) on SWE-bench tasks via harbor.

pi is a Node CLI coding agent that runs shell commands **locally in its working
directory** — unlike benchmaker's :class:`CodingAgent`, it has no Python-side
injectable executor. So there are exactly two ways to make pi edit a task's
``/testbed`` (harbor owns the per-instance environment; its verifier grades the
in-environment edits afterward), and this module exposes one harbor ``BaseAgent``
for each:

* :class:`PiContainerAgent` (``--agent pi``) — **install pi inside the
  environment** and run it there. pi's local shell *is* the container shell, so
  no command routing is needed. ``setup()`` installs Node + pi; ``run()`` writes
  a ``models.json`` pointing pi at our OpenAI-compatible endpoint and execs
  ``pi --mode json`` at ``/testbed``.

* :class:`PiHostAgent` (``--agent pi-host``) — **run pi on the host** (the harbor
  trial process) and route every shell/file action into the environment. A tiny
  localhost HTTP bridge forwards to ``environment.exec``; pi's built-in ``bash``
  tool is replaced (via the ``pi_ext/remote_exec.js`` extension) to POST to that
  bridge. This mirrors :class:`BenchmakerHostAgent`'s "host loop + remote exec"
  pattern, but for pi.

Model wiring (both modes): pi's built-in ``openai`` provider can't be pointed at
a custom base URL, so we write a ``models.json`` custom-provider entry
(``api: "openai-completions"``) and launch with ``--provider``/``--model``. The
endpoint + key come from harbor's ``AgentConfig.env`` (``OPENAI_API_BASE_URL`` /
``OPENAI_API_KEY``) or the host environment, exactly like the other agents here.

Runtime requirements / things to verify against your pi build (marked NOTE
below): the exact non-interactive/auto-approve flag for ``--mode json``, the
extension auto-load path, and that the environment pods have network egress to
npm + the model endpoint.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
from pathlib import Path
from typing import Any, Optional

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.utils.env import resolve_env_vars

PI_NPM_PACKAGE = "@earendil-works/pi-coding-agent"
PROVIDER_NAME = "bench"  # the models.json provider id we register for pi
WORKDIR = "/testbed"

# Explicit pi config dir, exported as PI_CODING_AGENT_DIR. We pin this instead of
# relying on the default ``$HOME/.pi/agent`` because the two ends disagree on
# what $HOME is inside the environment: the staging step expands ``$HOME`` with
# bash (empty when HOME is unset in the exec, → ``/.pi/agent``), while pi resolves
# its config dir via Node's ``os.homedir()`` (falls back to the passwd entry,
# e.g. ``/root``, when HOME is unset). That mismatch left pi unable to find the
# models.json registering our provider — "Unknown provider \"bench\"". An explicit
# PI_CODING_AGENT_DIR (config.getAgentDir honors it) makes write and read agree.
PI_AGENT_DIR = "/tmp/pi-agent"

# This file lives at benchmaker/swebench/pi_agent.py; the JS extension ships
# alongside it under pi_ext/.
_EXT_DIR = Path(__file__).resolve().parent / "pi_ext"
REMOTE_EXEC_EXT = _EXT_DIR / "remote_exec.js"

# Default Node install for a per-instance SWE-bench image (Debian-based Python,
# no Node). Uses the official static tarball (.tar.gz → no xz dependency) so we
# don't touch apt. Overridable via the ``install_script`` kwarg.
DEFAULT_NODE_VERSION = "v20.18.1"
_DEFAULT_INSTALL_SCRIPT = r"""
set -e
if command -v pi >/dev/null 2>&1; then echo "pi already present"; exit 0; fi
NODE_VERSION="${NODE_VERSION:-%(node_version)s}"
NODE_PREFIX=/opt/node
if ! command -v node >/dev/null 2>&1; then
  arch="$(uname -m)"; case "$arch" in x86_64) na=x64;; aarch64|arm64) na=arm64;; *) na=x64;; esac
  url="https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-${na}.tar.gz"
  echo "installing node from $url"
  mkdir -p "$NODE_PREFIX"
  curl -fsSL "$url" | tar -xz -C "$NODE_PREFIX" --strip-components=1
  export PATH="$NODE_PREFIX/bin:$PATH"
fi
export PATH="$NODE_PREFIX/bin:$PATH"
npm install -g --ignore-scripts %(pkg)s
pi --version || true
""" % {"node_version": DEFAULT_NODE_VERSION, "pkg": PI_NPM_PACKAGE}

# Prepend the static-node prefix so `pi` is found after a tarball install.
_PATH_PREFIX = "export PATH=/opt/node/bin:$PATH"

SWEBENCH_PROMPT_TEMPLATE = """Fix the bug described below in the repository \
checked out at `{workdir}` (already at the buggy commit, dependencies \
installed). Investigate, edit the SOURCE files in place to fix it, and do NOT \
modify the tests — a hidden test suite grades your work. When done, stop.

# Task: {instance_id}
Repository: {repo}

## Problem statement
{problem_statement}
"""


# --------------------------------------------------------------------------- #
# Shared helpers (pure — unit-tested in tests/test_pi_agent.py)
# --------------------------------------------------------------------------- #


def resolve_model(extra_env: dict[str, str], model_name: Optional[str]) -> tuple[str, str, str]:
    """Return ``(base_url, model_id, api_key)`` from env + harbor's model name.

    Mirrors :class:`BenchmakerHostAgent._resolve_model`: a leading ``openai/``
    litellm prefix is stripped (pi talks to the bare OpenAI-compatible id), and
    the base URL / key fall back across the common alias names.
    """
    def _env(*keys: str) -> Optional[str]:
        for k in keys:
            v = extra_env.get(k) or os.environ.get(k)
            if v:
                return v
        return None

    base_url = _env("OPENAI_API_BASE_URL", "OPENAI_API_BASE", "OPENAI_BASE_URL") or ""
    api_key = _env("OPENAI_API_KEY") or ""
    model = model_name or _env("OPENAI_COMPATIBLE_MODEL", "OPENAI_MODEL") or ""
    # harbor templatizes sensitive AgentConfig.env values for safe persistence
    # (``OPENAI_API_KEY`` -> the literal string ``"${OPENAI_API_KEY}"``); its own
    # consumers call resolve_env_vars to expand them from os.environ. We must too,
    # or pi sends the unexpanded template as its bearer token and the endpoint
    # 401s. Literal values (and empty strings) pass through unchanged.
    resolved = resolve_env_vars({"b": base_url, "k": api_key, "m": model})
    base_url, api_key, model = resolved["b"], resolved["k"], resolved["m"]
    if model.startswith("openai/"):
        model = model[len("openai/"):]
    if not base_url or not model:
        raise RuntimeError(
            "pi agent needs an OpenAI-compatible endpoint + model id "
            "(OPENAI_API_BASE_URL and --model / OPENAI_COMPATIBLE_MODEL)."
        )
    return base_url.rstrip("/"), model, api_key


def models_json(
    base_url: str,
    model: str,
    *,
    provider: str = PROVIDER_NAME,
    context_window: int = 128000,
    max_tokens: int = 8192,
    api_key_ref: str = "$OPENAI_API_KEY",
) -> str:
    """Render a pi ``models.json`` registering our OpenAI-compatible endpoint.

    ``api_key_ref`` is written verbatim — pi resolves a ``$VAR`` form from the
    environment, so we keep the secret out of the file by default.
    """
    return json.dumps(
        {
            "providers": {
                provider: {
                    "baseUrl": base_url,
                    "api": "openai-completions",
                    "apiKey": api_key_ref,
                    "models": [
                        {
                            "id": model,
                            "contextWindow": context_window,
                            "maxTokens": max_tokens,
                        }
                    ],
                }
            }
        },
        indent=2,
    )


def pi_command(
    model: str,
    prompt_path: str,
    *,
    provider: str = PROVIDER_NAME,
    extra_args: Optional[list[str]] = None,
) -> str:
    """The ``pi`` invocation (prompt passed via file to survive multiline text).

    NOTE: ``--mode json`` is the headless JSONL mode. If your pi build still
    gates tool execution behind a permission prompt, pass the appropriate
    auto-approve flag through ``pi_extra_args`` (e.g. ``--yolo``); it is appended
    here verbatim.
    """
    args = ["pi", "--mode", "json", "--provider", provider, "--model", model]
    args += list(extra_args or [])
    quoted = " ".join(shlex.quote(a) for a in args)
    return f'{quoted} "$(cat {shlex.quote(prompt_path)})"'


def build_prompt(item: dict[str, Any], *, workdir: str = WORKDIR) -> str:
    return SWEBENCH_PROMPT_TEMPLATE.format(
        workdir=workdir,
        instance_id=item.get("instance_id", "?"),
        repo=item.get("repo", ""),
        problem_statement=item.get("problem_statement", ""),
    )


def _b64_write(path: str, content: str) -> str:
    """A shell command that writes ``content`` to ``path`` (base64, escape-safe)."""
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    return f"mkdir -p {shlex.quote(str(Path(path).parent))} && echo {b64} | base64 -d > {shlex.quote(path)}"


# --------------------------------------------------------------------------- #
# Base agent
# --------------------------------------------------------------------------- #


class _PiAgentBase(BaseAgent):
    SUPPORTS_ATIF = False
    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: Optional[str] = None,
        *,
        extra_env: Optional[dict[str, str]] = None,
        cwd: str = WORKDIR,
        provider: str = PROVIDER_NAME,
        context_window: int = 128000,
        max_tokens: int = 8192,
        total_wall_s: float = 2400.0,
        pi_extra_args: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._extra_env = extra_env or {}
        self._cwd = cwd
        self._provider = provider
        self._context_window = int(context_window)
        self._max_tokens = int(max_tokens)
        self._total_wall_s = float(total_wall_s)
        # pi_extra_args may arrive as a space-joined string via --agent-kwarg.
        if isinstance(pi_extra_args, str):
            pi_extra_args = shlex.split(pi_extra_args)
        self._pi_extra_args = list(pi_extra_args or [])

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:  # default: nothing
        return

    def _resolved(self) -> tuple[str, str, str]:
        return resolve_model(self._extra_env, self.model_name)


# --------------------------------------------------------------------------- #
# Mode 1: pi runs INSIDE the environment
# --------------------------------------------------------------------------- #


class PiContainerAgent(_PiAgentBase):
    """Install pi in the per-instance environment and run it there at ``/testbed``."""

    def __init__(self, *args: Any, install_script: Optional[str] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._install_script = install_script or _DEFAULT_INSTALL_SCRIPT

    @staticmethod
    def name() -> str:
        return "pi-container"

    async def setup(self, environment: BaseEnvironment) -> None:
        res = await environment.exec(
            command=f"bash -lc {shlex.quote(self._install_script)}",
            timeout_sec=900,
        )
        if int(res.return_code) != 0:
            self.logger.warning(
                "pi install returned %s:\n%s", res.return_code,
                ((res.stdout or "") + (res.stderr or ""))[-2000:],
            )

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        item = _task_item(context, instruction)
        base_url, model, api_key = self._resolved()
        # Embed the resolved key directly in models.json rather than the
        # ``$OPENAI_API_KEY`` env-ref: pi's in-container env-var resolution does
        # not pick up the inline-exported key, causing it to send an empty/wrong
        # bearer (-> 401) even though the key reaches the pod fine.
        cfg = models_json(base_url, model, provider=self._provider,
                          context_window=self._context_window, max_tokens=self._max_tokens,
                          api_key_ref=api_key)
        prompt = build_prompt(item, workdir=self._cwd)

        # Stage config + prompt inside the environment. models.json goes in the
        # explicit PI_AGENT_DIR (see its docstring) so pi finds it regardless of
        # how $HOME resolves in the exec.
        setup_cmd = " && ".join([
            _b64_write(f"{PI_AGENT_DIR}/models.json", cfg),
            _b64_write("/tmp/pi_task.txt", prompt),
        ])
        await environment.exec(command=f"bash -lc {shlex.quote(setup_cmd)}", timeout_sec=120)

        run_cmd = (
            f"{_PATH_PREFIX} && cd {shlex.quote(self._cwd)} && "
            f"PI_CODING_AGENT_DIR={shlex.quote(PI_AGENT_DIR)} "
            f"OPENAI_API_KEY={shlex.quote(api_key)} "
            + pi_command(model, "/tmp/pi_task.txt", provider=self._provider,
                         extra_args=self._pi_extra_args)
        )
        try:
            res = await environment.exec(
                command=f"bash -lc {shlex.quote(run_cmd)}",
                cwd=self._cwd,
                timeout_sec=int(self._total_wall_s),
            )
            exit_status = "ok" if int(res.return_code) == 0 else f"exit_{res.return_code}"
            self._write_log("pi-container", (res.stdout or "") + (res.stderr or ""))
        except Exception as exc:  # noqa: BLE001 — surface, let harbor grade what exists
            self.logger.exception("pi-container run failed: %s", exc)
            exit_status = type(exc).__name__
        context.metadata = {"exit_status": exit_status, "mode": "container"}

    def _write_log(self, tag: str, text: str) -> None:
        try:
            (Path(self.logs_dir) / f"{tag}.log").write_text(text)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Mode 2: pi runs on the HOST, shell/file ops routed into the environment
# --------------------------------------------------------------------------- #


class PiHostAgent(_PiAgentBase):
    """Run pi on the host; route its ``bash`` tool into the environment via a bridge."""

    def __init__(self, *args: Any, bridge_host: str = "127.0.0.1",
                 exec_timeout_s: float = 600.0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._bridge_host = bridge_host
        self._exec_timeout_s = float(exec_timeout_s)

    @staticmethod
    def name() -> str:
        return "pi-host"

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        item = _task_item(context, instruction)
        base_url, model, api_key = self._resolved()

        bridge = _ExecBridge(environment, cwd=self._cwd,
                             exec_timeout_s=self._exec_timeout_s, host=self._bridge_host)
        await bridge.start()
        home = Path(self.logs_dir) / "pi-home"
        try:
            self._stage_host_config(home, base_url, model)
            prompt_path = home / "task.txt"
            prompt_path.write_text(build_prompt(item, workdir=self._cwd))

            cmd = pi_command(model, str(prompt_path), provider=self._provider,
                             extra_args=self._pi_extra_args)
            env = {
                **os.environ,
                "HOME": str(home),
                # Point pi at the exact dir we staged config into, rather than
                # trusting it to derive ~/.pi/agent from HOME (see PI_AGENT_DIR).
                "PI_CODING_AGENT_DIR": str(home / ".pi" / "agent"),
                "OPENAI_API_KEY": api_key,
                "PI_EXEC_BRIDGE": bridge.url,
                "PI_EXEC_CWD": self._cwd,
            }
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=str(home), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=self._total_wall_s)
                exit_status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
                self._write_log("pi-host", (out or b"").decode("utf-8", "replace"))
            except asyncio.TimeoutError:
                proc.kill()
                exit_status = "time_limit"
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("pi-host run failed: %s", exc)
            exit_status = type(exc).__name__
        finally:
            await bridge.stop()
        context.metadata = {"exit_status": exit_status, "mode": "host",
                            "exec_count": bridge.count}

    def _stage_host_config(self, home: Path, base_url: str, model: str) -> None:
        agent_dir = home / ".pi" / "agent"
        (agent_dir / "extensions").mkdir(parents=True, exist_ok=True)
        (agent_dir / "models.json").write_text(
            models_json(base_url, model, provider=self._provider,
                        context_window=self._context_window, max_tokens=self._max_tokens)
        )
        # Force every file/shell action through the (overridden) bash tool so it
        # lands in the environment, never the host fs.  NOTE: verify your pi
        # build honors a "tools" allowlist in settings.json.
        (agent_dir / "settings.json").write_text(json.dumps(
            {"defaultProvider": self._provider, "defaultModel": model,
             "tools": ["bash"]}, indent=2))
        # Auto-loaded extension: replaces `bash` with a remote executor.
        if REMOTE_EXEC_EXT.exists():
            (agent_dir / "extensions" / "remote_exec.js").write_text(
                REMOTE_EXEC_EXT.read_text())

    def _write_log(self, tag: str, text: str) -> None:
        try:
            (Path(self.logs_dir) / f"{tag}.log").write_text(text)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Localhost bridge: pi's bash override (JS) -> here -> environment.exec
# --------------------------------------------------------------------------- #


class _ExecBridge:
    """A 127.0.0.1 HTTP server: ``POST /exec {command,timeout}`` -> environment.exec."""

    def __init__(self, environment: BaseEnvironment, *, cwd: str,
                 exec_timeout_s: float, host: str = "127.0.0.1"):
        self._env = environment
        self._cwd = cwd
        self._exec_timeout_s = exec_timeout_s
        self._host = host
        self._runner: Any = None
        self._port: int = 0
        self.count = 0

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    async def start(self) -> None:
        from aiohttp import web

        async def handle(request: "web.Request") -> "web.Response":
            body = await request.json()
            command = body.get("command", "")
            timeout = int(body.get("timeout") or self._exec_timeout_s)
            self.count += 1
            # Anchor at cwd without a subshell (matches the other agents).
            full = f"cd {shlex.quote(self._cwd)} && {command}" if self._cwd else command
            try:
                res = await self._env.exec(command=full, cwd=self._cwd, timeout_sec=timeout)
                return web.json_response({
                    "return_code": int(res.return_code),
                    "stdout": res.stdout or "",
                    "stderr": res.stderr or "",
                })
            except Exception as e:  # noqa: BLE001
                return web.json_response(
                    {"return_code": -1, "stdout": "", "stderr": f"{type(e).__name__}: {e}"})

        app = web.Application()
        app.router.add_post("/exec", handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, 0)
        await site.start()
        # Resolve the ephemeral port.
        for sock in site._server.sockets:  # type: ignore[attr-defined]
            self._port = sock.getsockname()[1]
            break

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None


def _task_item(context: AgentContext, instruction: str) -> dict[str, Any]:
    """Best-effort SWE-bench row from harbor's context, falling back to the text."""
    item = getattr(context, "task", None) or getattr(context, "item", None)
    if isinstance(item, dict) and item.get("problem_statement"):
        return item
    return {"problem_statement": instruction, "instance_id": "?", "repo": ""}

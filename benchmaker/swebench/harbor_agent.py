"""Run benchmaker's coding-agent *loop* as a harbor agent.

This is the adapter that lets harbor (the eval engine) drive **our** agent loop,
alongside harbor's own built-ins (mini-swe-agent, claude-code, …). It mirrors
``flash-sandbox/examples/harbor/mini_swe_host_agent.py``: the model/observe loop
runs on the host (inside the harbor trial process) and each shell action is
pushed into the per-instance sandbox via harbor's ``environment.exec``. Harbor
owns the environment (prebuilt SWE-bench image, lifecycle) and the verifier
(reward.txt); we only contribute the loop.

The seam is :meth:`CodingAgent.run_loop`, which takes an injected ``executor``.
Here the executor is ``environment.exec`` anchored at ``cwd`` (default
``/testbed``) — no benchmaker sandbox client is involved. Because both
``run_loop`` and ``environment.exec`` are async on the same event loop, there's
no sync/async bridge to manage.

Plug it into harbor via an import-path agent (see ``harbor_eval.py``)::

    AgentConfig(
        import_path="benchmaker.swebench.harbor_agent:BenchmakerHostAgent",
        model_name="zai-org/GLM-4.7-Flash",
        env={"OPENAI_API_KEY": "...", "OPENAI_API_BASE_URL": "..."},
        kwargs={"step_limit": 40, "cwd": "/testbed"},
    )

The agent reads ``OPENAI_API_BASE_URL`` / ``OPENAI_API_KEY`` (and the model id,
from harbor's ``--model`` or ``OPENAI_COMPATIBLE_MODEL`` / ``OPENAI_MODEL``)
from ``env`` (harbor's ``config.env``) or the host environment.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from benchmaker.swebench.agent import CodingAgent, Executor

DEFAULT_LOOP_AGENT = "benchmaker.swebench.agent:CodingAgent"

# A SWE-bench-shaped default: the repo is already set up by harbor's environment;
# the agent just edits source in place and harbor's verifier grades the result.
_DEFAULT_SYSTEM_PROMPT = """You are a senior software engineer. The task's
repository is already cloned and fully set up at `{cwd}` — you do NOT need to
clone, install, or build anything.

Each turn, emit **exactly one** shell command in a fenced bash block:

```bash
your-command-here
```

You will see the command's combined stdout+stderr next. **Each command runs in a
fresh shell starting at `{cwd}`** — the working directory and environment
variables do NOT persist across turns, but filesystem edits DO. Use
non-interactive flags; avoid TTY editors. Use `python -c "..."` or heredocs
(`cat <<'EOF' > path ... EOF`) to edit files.

Investigate the codebase, then edit the **source** files in place to solve the
task. Do NOT modify the tests — a hidden test suite grades your work.

When the fix is complete, emit a fenced block whose **first line** is exactly:

    COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

Nothing else is needed after it — your in-tree edits under `{cwd}` are what gets
graded, so you do NOT need to print or paste a diff.
"""

_DEFAULT_INSTANCE_TEMPLATE = "# Task\n\n{task}\n\nBegin."


def _load_loop_agent_class(ref: str) -> type:
    module_path, _, class_name = ref.partition(":")
    if not class_name:
        raise ValueError(f"loop agent ref must be 'module:Class', got {ref!r}")
    return getattr(importlib.import_module(module_path), class_name)


class BenchmakerHostAgent(BaseAgent):
    """Harbor agent that delegates to a benchmaker ``CodingAgent``-style loop."""

    SUPPORTS_ATIF = False
    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: Optional[str] = None,
        *,
        extra_env: Optional[dict[str, str]] = None,
        loop_agent: str = DEFAULT_LOOP_AGENT,
        cwd: str = "/testbed",
        step_limit: int = 40,
        per_command_timeout_sec: int = 120,
        total_wall_s: float = 2400.0,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_obs_chars: int = 4000,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        instance_template: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._extra_env = extra_env or {}
        self._loop_agent_ref = loop_agent
        self._cwd = cwd
        self._step_limit = int(step_limit)
        self._per_command_timeout = int(per_command_timeout_sec)
        self._total_wall_s = float(total_wall_s)
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._max_obs_chars = int(max_obs_chars)
        self._api_base = api_base
        self._api_key = api_key
        self._model = model
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT.format(cwd=cwd)
        self._instance_template = instance_template or _DEFAULT_INSTANCE_TEMPLATE
        self._traj_path = Path(logs_dir) / "benchmaker-host.trajectory.json"

    @staticmethod
    def name() -> str:
        return "benchmaker-host"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        # Nothing to install in-sandbox; the loop lives on the host.
        return

    # ---- config resolution ------------------------------------------- #

    def _env(self, key: str) -> Optional[str]:
        return self._extra_env.get(key) or os.environ.get(key)

    @staticmethod
    def _chat_url(base: str) -> str:
        base = base.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def _resolve_model(self) -> tuple[str, str, str]:
        api_base = (self._api_base or self._env("OPENAI_API_BASE_URL")
                    or self._env("OPENAI_API_BASE") or self._env("OPENAI_BASE_URL"))
        api_key = self._api_key or self._env("OPENAI_API_KEY")
        # Strip a leading litellm-style provider prefix (e.g. "openai/") — the
        # CodingAgent talks to an OpenAI-compatible endpoint with the bare id.
        model = self._model or self.model_name
        if model and model.startswith("openai/"):
            model = model[len("openai/"):]
        model = model or self._env("OPENAI_COMPATIBLE_MODEL") or self._env("OPENAI_MODEL")
        if not (api_base and model):
            raise RuntimeError(
                "BenchmakerHostAgent needs OPENAI_API_BASE_URL and a model id "
                "(harbor --model / OPENAI_COMPATIBLE_MODEL / OPENAI_MODEL)."
            )
        return self._chat_url(api_base), model, (api_key or "")

    def _build_loop_agent(self) -> CodingAgent:
        url, model, api_key = self._resolve_model()
        cls = _load_loop_agent_class(self._loop_agent_ref)
        return cls(
            url=url,
            model=model,
            api_key=api_key,
            step_limit=self._step_limit,
            timeout_per_step_s=self._per_command_timeout,
            total_wall_s=self._total_wall_s,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            max_obs_chars=self._max_obs_chars,
        )

    def _make_executor(self, environment: BaseEnvironment) -> Executor:
        cwd = self._cwd

        async def executor(action: str, timeout_s: float) -> tuple[int, str]:
            # Anchor at cwd without a subshell — `(...)` breaks trailing
            # heredocs, and several env backends ignore the exec `cwd` field.
            full = f"cd {cwd} && {action}" if cwd else action
            try:
                res = await environment.exec(
                    command=full, cwd=cwd, timeout_sec=int(timeout_s),
                )
            except Exception as e:  # surface as a failed command, keep the loop alive
                return -1, f"<exec error: {type(e).__name__}: {e}>"
            out = (res.stdout or "") + (res.stderr or "")
            return int(res.return_code), out

        return executor

    # ---- run --------------------------------------------------------- #

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        loop_agent = self._build_loop_agent()
        executor = self._make_executor(environment)
        try:
            result = await loop_agent.run_loop(
                instruction,
                executor,
                system_prompt=self._system_prompt,
                instance_template=self._instance_template,
            )
        except Exception as exc:
            self.logger.exception("benchmaker-host agent crashed: %s", exc)
            context.metadata = {"exit_status": type(exc).__name__, "submission": ""}
            return
        finally:
            await loop_agent.aclose()

        try:
            self._traj_path.write_text(json.dumps({
                "exit_status": result.exit_status,
                "submission": result.submission,
                "messages": result.messages,
            }, indent=2))
        except Exception:
            pass

        context.metadata = {
            "exit_status": result.exit_status,
            "submission": result.submission,
            "n_calls": result.n_calls,
            "n_actions": result.n_actions,
            "trajectory_path": str(self._traj_path),
        }

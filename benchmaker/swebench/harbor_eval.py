"""Evaluate a coding agent on SWE-bench via **harbor**, driven from benchmaker.

This is the "use harbor for the evaluation" path: harbor owns the per-instance
environment (prebuilt SWE-bench image on Flash Sandbox), the agent execution,
and the verifier (reward.txt). benchmaker supplies the entrypoint, ``.env``
wiring, a pluggable **agent registry**, and result summarisation.

The agent is pluggable — harbor's ``BaseAgent`` is the interface, and any of
these slot in via ``--agent``:

  * a registry key (``mini-swe-agent``, ``coding-agent``, ``claude-code``,
    ``pi`` / ``pi-container`` / ``pi-host``);
  * a bare harbor built-in name (``openhands``, ``swe-agent``, ``oracle``, …);
  * a custom ``module.path:ClassName`` (a harbor ``BaseAgent`` subclass).

``coding-agent`` wraps benchmaker's own loop (``harbor_agent.py``); the loop is
backend-agnostic (``CodingAgent.run_loop``), so the same code runs under harbor
here and under benchmaker's own Flash Sandbox executor (``examples/swebench/
config.yaml``).

``pi`` / ``pi-host`` run **pi** (``@earendil-works/pi-coding-agent``) — see
``benchmaker.swebench.pi_agent``: ``pi`` installs pi inside the environment and
runs it at ``/testbed``; ``pi-host`` runs pi on the host and routes its shell
into the environment via a localhost bridge.

Usage::

    # Boot the Flash Sandbox cluster first (see ../../../flash-sandbox/examples/harbor).
    FLASH_SANDBOX_URL=http://localhost:8080 \\
        python -m benchmaker.swebench.harbor_eval \\
            --dataset swebench-verified \\
            --agent mini-swe-agent \\
            --model "$OPENAI_COMPATIBLE_MODEL" \\
            --n-tasks 5

    # Evaluate *our* wrapped CodingAgent loop through harbor:
    python -m benchmaker.swebench.harbor_eval --agent coding-agent --n-tasks 5

    python -m benchmaker.swebench.harbor_eval --list-agents

``--api-base`` / ``--api-key`` / ``--model`` default to ``OPENAI_API_BASE_URL`` /
``OPENAI_API_KEY`` / ``OPENAI_COMPATIBLE_MODEL`` (loaded from a repo-root
``.env``). Exit code 0 iff every trial passed (reward >= 1).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

# Repo root, used only to locate the ``.env`` for default credentials. The
# custom agent (`benchmaker.swebench.harbor_agent`) is an installed package, so
# harbor's AgentFactory resolves its import path without any sys.path surgery.
_REPO_ROOT = Path(__file__).resolve().parents[2]

from benchmaker.env import load_dotenv  # noqa: E402

from harbor.job import Job  # noqa: E402
from harbor.models.environment_type import EnvironmentType  # noqa: E402
from harbor.models.job.config import DatasetConfig, JobConfig  # noqa: E402
from harbor.models.trial.config import AgentConfig, EnvironmentConfig  # noqa: E402

from benchmaker.swebench._flash_hardening import (  # noqa: E402
    harden_flash_sandbox_client,
)

# Harden the flash-sandbox HTTP client before any Job runs (the `benchmaker
# swebench` entrypoint). The `harbor run <config>` path is covered separately by
# the agent modules' import-time call. Idempotent.
harden_flash_sandbox_client()


# Friendly key -> harbor agent spec. A spec is either {"name": <built-in>} or
# {"import_path": "module:Class"} (a harbor BaseAgent subclass). Extend this to
# register more loops (the import-path form is the general plug-in point).
AGENT_REGISTRY: dict[str, dict[str, str]] = {
    "mini-swe-agent": {"name": "mini-swe-agent"},
    "coding-agent": {
        "import_path": "benchmaker.swebench.harbor_agent:BenchmakerHostAgent",
    },
    "claude-code": {"name": "claude-code"},
    # pi (pi-coding-agent), two execution modes — see benchmaker.swebench.pi_agent.
    "pi": {"import_path": "benchmaker.swebench.pi_agent:PiContainerAgent"},
    "pi-container": {"import_path": "benchmaker.swebench.pi_agent:PiContainerAgent"},
    "pi-host": {"import_path": "benchmaker.swebench.pi_agent:PiHostAgent"},
}


def _resolve_agent_spec(agent: str) -> dict[str, str]:
    """Map ``--agent`` to an AgentConfig field dict.

    ``module:Class`` -> import_path; a registry key -> its spec; anything else is
    passed through as a harbor built-in ``name``.
    """
    if ":" in agent:
        return {"import_path": agent}
    if agent in AGENT_REGISTRY:
        return dict(AGENT_REGISTRY[agent])
    return {"name": agent}


# litellm provider prefixes we must NOT clobber when normalising a model id.
# Anything else (e.g. a bare server id like ``zai-org/GLM-4.7-Flash``) gets an
# ``openai/`` prefix so litellm routes to the OpenAI-compatible provider.
_LITELLM_PROVIDER_PREFIXES = (
    "openai/", "anthropic/", "gemini/", "groq/", "mistral/", "deepseek/",
    "together_ai/", "openrouter/", "vertex_ai/", "bedrock/", "azure/",
    "xai/", "cohere/", "zai/", "ollama/", "litellm_proxy/",
)


def _litellm_model_name(model: str) -> str:
    """Prefix a bare model id with ``openai/`` for harbor's litellm built-ins.

    harbor's mini-swe-agent / swe-agent resolve the API key via
    ``litellm.get_llm_provider(model)``. A bare id such as
    ``zai-org/GLM-4.7-Flash`` has no recognised provider, so litellm raises
    "Unknown model" and harbor reports it can't determine the API key. Routing
    through the OpenAI-compatible provider (``openai/<id>``) makes it look up
    ``OPENAI_API_KEY`` + ``OPENAI_API_BASE``. Our own BenchmakerHostAgent strips
    the prefix back off, so prefixing is safe across agents.
    """
    if not model or model.startswith(_LITELLM_PROVIDER_PREFIXES):
        return model
    return f"openai/{model}"


def _normalize_openai_env() -> None:
    """Mirror the OpenAI-compatible endpoint across the var names harbor reads.

    harbor's built-in litellm agents read the endpoint from ``os.environ`` under
    ``OPENAI_API_BASE`` (mini_swe_agent only forwards that exact name into the
    sandbox) — not ``AgentConfig.env`` and not the ``OPENAI_API_BASE_URL`` name
    used in benchmaker's ``.env``. Without this, litellm falls back to
    api.openai.com. Mirror whichever alias is set across all three.
    """
    base = (os.environ.get("OPENAI_API_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL"))
    if base:
        for alias in ("OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENAI_API_BASE_URL"):
            os.environ.setdefault(alias, base)


def _parse_kv(pairs: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` flags; ints/floats/bools are coerced."""
    out: dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--agent-kwarg must be key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = _coerce(v.strip())
    return out


def _coerce(v: str) -> Any:
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _build_job_config(args: argparse.Namespace) -> JobConfig:
    environment = EnvironmentConfig(
        type=EnvironmentType.FLASH_SANDBOX,
        force_build=args.force_build,
        delete=True,
        kwargs={
            "backend_type": args.backend_type,
            "request_timeout_sec": args.request_timeout_sec,
            # Large SWE-bench images + high concurrency need well over harbor's
            # old 30s default for the in-sandbox agent to come up.
            "agent_ready_timeout_sec": args.agent_ready_timeout_sec,
        },
    )

    agent_env: dict[str, str] = {}
    if args.api_key:
        agent_env["OPENAI_API_KEY"] = args.api_key
    if args.api_base:
        # Cover the various names different agents read.
        agent_env["OPENAI_API_BASE_URL"] = args.api_base
        agent_env["OPENAI_API_BASE"] = args.api_base
        agent_env["OPENAI_BASE_URL"] = args.api_base

    agent_kwargs = _parse_kv(args.agent_kwarg)
    if args.agent_config_file:
        agent_kwargs.setdefault("config_file", str(Path(args.agent_config_file).resolve()))

    spec = _resolve_agent_spec(args.agent)
    # harbor built-ins (spec has "name") drive litellm and need a provider
    # prefix; our import-path host agent reads the bare id (and strips a stray
    # "openai/" itself), so leave it untouched there.
    model_name = _litellm_model_name(args.model) if spec.get("name") else args.model
    agent = AgentConfig(
        name=spec.get("name"),
        import_path=spec.get("import_path"),
        model_name=model_name,
        env=agent_env,
        kwargs=agent_kwargs,
    )

    # exclude_task is only set by the swebench-replay recipe's namespace; the
    # swebench recipe and harbor_eval's own CLI omit it, so read defensively
    # (matching the getattr() guards for jobs_dir / qos_enabled below).
    dataset = DatasetConfig(name=args.dataset, n_tasks=args.n_tasks,
                            task_names=args.task or None,
                            exclude_task_names=getattr(args, "exclude_task", None) or None)

    # Parent directory for the run bundle (harbor writes to <jobs_dir>/<job_name>).
    # Omit when unset so harbor keeps its own default of "jobs".
    job_kwargs: dict[str, Any] = {}
    jobs_dir = getattr(args, "jobs_dir", None)
    if jobs_dir:
        job_kwargs["jobs_dir"] = Path(jobs_dir)

    # QoS: when enabled, splat the cpu.weight knobs into the environment kwargs
    # (consumed by FlashSandboxEnvironment.__init__) and couple the verifier
    # timeout — QoS demotes verifier-phase CPU, so the verifier needs more
    # wall-clock time. Left untouched (None) when QoS is off.
    verifier_timeout_multiplier = None
    # QoS is wired only via the swebench-replay recipe CLI; harbor_eval's own
    # _parse_args does not expose these flags, so this guard no-ops there.
    if getattr(args, "qos_enabled", False):
        environment.kwargs.update(
            qos_enabled=True,
            on_demand_cpu_weight=args.on_demand_cpu_weight,
            best_effort_cpu_weight=args.best_effort_cpu_weight,
        )
        verifier_timeout_multiplier = args.qos_verifier_timeout_multiplier

    return JobConfig(
        job_name=args.job_name or "",
        n_attempts=args.n_attempts,
        n_concurrent_trials=args.concurrency,
        quiet=False,
        timeout_multiplier=args.timeout_multiplier,
        verifier_timeout_multiplier=verifier_timeout_multiplier,
        environment=environment,
        agents=[agent],
        datasets=[dataset],
        **job_kwargs,
    )


def _summarise(job_result: Any) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    n_pass = 0
    for tr in list(job_result.trial_results or []):
        rewards = tr.verifier_result.rewards if tr.verifier_result is not None else None
        reward: float | None = None
        if rewards:
            for key in ("reward", "default"):
                if key in rewards:
                    reward = float(rewards[key])
                    break
            else:
                reward = float(next(iter(rewards.values())))
        passed = reward is not None and reward >= 1.0
        n_pass += int(passed)
        info = getattr(tr, "exception_info", None)
        err = None if info is None else (
            getattr(info, "exception_type", None) or getattr(info, "exception_message", None)
        )
        rows.append({
            "trial": tr.trial_name,
            "reward": reward,
            "passed": passed,
            "error": err,
        })
    accuracy = n_pass / len(rows) if rows else 0.0
    return rows, accuracy


def _print_summary(rows: list[dict[str, Any]], accuracy: float) -> None:
    if not rows:
        print("\nNo trials completed.")
        return
    w = max(len(r["trial"]) for r in rows)
    print(f"\n{'TRIAL'.ljust(w)}  REWARD  STATUS  ERROR")
    print("-" * (w + 32))
    for r in rows:
        reward = "—" if r["reward"] is None else f"{r['reward']:.2f}"
        status = "PASS" if r["passed"] else "FAIL"
        err = "" if r["error"] is None else str(r["error"])[:60]
        print(f"{r['trial'].ljust(w)}  {reward:>6}  {status:>4}    {err}")
    n_pass = sum(1 for r in rows if r["passed"])
    print(f"\naccuracy: {accuracy:.1%}  ({n_pass}/{len(rows)})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list-agents", action="store_true",
                   help="List registry agent keys and exit.")
    p.add_argument("--dataset", default="swebench-verified",
                   help="Harbor dataset slug (default: swebench-verified).")
    p.add_argument("--agent", default="mini-swe-agent",
                   help="Registry key, harbor built-in name, or 'module:Class'.")
    p.add_argument("--model", default=os.environ.get("OPENAI_COMPATIBLE_MODEL")
                   or os.environ.get("OPENAI_MODEL"),
                   help="Model id. Default: $OPENAI_COMPATIBLE_MODEL / $OPENAI_MODEL.")
    p.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE_URL")
                   or os.environ.get("OPENAI_API_BASE"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--agent-kwarg", action="append", default=[],
                   help="Extra agent kwarg key=value (repeatable).")
    p.add_argument("--agent-config-file", default=None,
                   help="YAML forwarded to the agent's config_file kwarg "
                        "(e.g. mini-swe-agent step-limit config).")
    p.add_argument("--n-tasks", type=int, default=None,
                   help="Cap the number of dataset tasks.")
    p.add_argument("--task", action="append", default=[],
                   help="Restrict to specific task name(s)/glob(s) (repeatable).")
    p.add_argument("--exclude-task", action="append", default=[],
                   help="Skip specific task name(s)/glob(s) (repeatable). Applied "
                        "after --task and before the --n-tasks cap, so the cap "
                        "selects the first N tasks that remain after exclusion.")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--n-attempts", type=int, default=1)
    p.add_argument("--timeout-multiplier", type=float, default=4.0,
                   help="Multiplier on harbor's timeouts (SWE-bench cold-start "
                        "needs 4-6x).")
    p.add_argument("--force-build", action="store_true")
    p.add_argument("--backend-type", default="docker")
    p.add_argument("--request-timeout-sec", type=float, default=120.0)
    p.add_argument("--agent-ready-timeout-sec", type=float, default=600.0,
                   help="How long to wait for the in-sandbox agent to come up "
                        "(harbor default was 30s — too low for big SWE-bench "
                        "images + high concurrency).")
    p.add_argument("--job-name", default=None)
    p.add_argument("--jobs-dir", default=None,
                   help="Parent directory for the run bundle "
                        "(harbor writes to <jobs-dir>/<job-name>; default 'jobs').")
    p.add_argument("--timeline", dest="timeline", action="store_true", default=True,
                   help="Capture timeline + utilization + tokens (default on).")
    p.add_argument("--no-timeline", dest="timeline", action="store_false",
                   help="Disable observability capture.")
    p.add_argument("--utilization-interval-sec", dest="utilization_interval_sec",
                   type=float, default=5.0,
                   help="Seconds between /status utilization polls (default 5).")
    return p.parse_args()


async def _amain(args: argparse.Namespace) -> int:
    if args.list_agents:
        print("Registry agents (also accepts a bare harbor name or module:Class):")
        for k, v in AGENT_REGISTRY.items():
            tgt = v.get("name") or v.get("import_path")
            print(f"  {k:16s} -> {tgt}")
        return 0

    if not (os.environ.get("FLASH_SANDBOX_URL") or os.environ.get("FLASH_SANDBOX_HOST")):
        print("ERROR: set FLASH_SANDBOX_URL (e.g. http://localhost:8080).",
              file=sys.stderr)
        return 2
    if not args.model:
        print("ERROR: --model required (or set OPENAI_COMPATIBLE_MODEL).",
              file=sys.stderr)
        return 2

    job_config = _build_job_config(args)
    print(f"harbor job: dataset={args.dataset} agent={args.agent} "
          f"model={args.model} flash={os.environ.get('FLASH_SANDBOX_URL', '—')}")

    from benchmaker.swebench.observability import run_job_with_observability

    job, job_result, summary_text = await run_job_with_observability(
        job_config,
        flash_url=os.environ.get("FLASH_SANDBOX_URL"),
        util_interval=args.utilization_interval_sec,
        enabled=args.timeline,
    )

    rows, accuracy = _summarise(job_result)
    _print_summary(rows, accuracy)
    if summary_text:
        print(summary_text)
    print(f"\njob dir: {job.job_dir}")
    return 0 if rows and all(r["passed"] for r in rows) else 1


def main() -> int:
    load_dotenv(str(_REPO_ROOT / ".env"))
    _normalize_openai_env()
    return asyncio.run(_amain(_parse_args()))


if __name__ == "__main__":
    sys.exit(main())

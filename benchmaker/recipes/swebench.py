"""``swebench`` recipe — evaluate a coding agent on SWE-bench via **harbor**.

Unlike the other recipes (which flow through ``BenchRunner``), this is a
*self-driving* recipe: harbor owns the per-instance environment (a prebuilt
SWE-bench image on Flash Sandbox), the agent execution, and the verifier. The
recipe just builds a harbor ``Job`` from the CLI flags, runs it, and prints
harbor's accuracy summary + job dir. There is no benchmaker run-bundle here.

The agent is pluggable via ``--agent``:
  * a registry key — ``pi`` (default), ``pi-container``, ``pi-host``,
    ``coding-agent`` (benchmaker's own loop), ``mini-swe-agent``, ``claude-code``;
  * a bare harbor built-in name (``openhands``, ``swe-agent``, ``oracle``, …);
  * a custom ``module.path:ClassName`` (a harbor ``BaseAgent`` subclass).

Model URL / model / key fall back to ``OPENAI_API_BASE_URL`` /
``OPENAI_COMPATIBLE_MODEL`` / ``OPENAI_API_KEY`` (loaded from ``.env``); the
sandbox URL comes from ``FLASH_SANDBOX_URL``.
"""

from __future__ import annotations

import os
from typing import Optional

import click

from benchmaker.env import load_dotenv
from benchmaker.recipes import register
from benchmaker.recipes.base import Recipe, SharedOpts


class SWEBenchRecipe(Recipe):
    name = "swebench"
    help = ("Evaluate a coding agent on SWE-bench via harbor (per-instance "
            "Flash Sandbox env + agent + verifier). Self-driving: prints "
            "harbor's accuracy summary, not a benchmaker run-bundle.")
    # harbor is its own runner; the load/timeout/bundle flags don't apply.
    wants_load_options = False

    def options(self) -> list:
        return [
            click.option("--agent", default="pi", show_default=True,
                         help="Agent: registry key (pi, pi-container, pi-host, "
                              "coding-agent, mini-swe-agent, claude-code), a bare "
                              "harbor built-in, or 'module:Class'."),
            click.option("--dataset", default="swebench-verified", show_default=True,
                         help="Harbor dataset slug."),
            click.option("--model", default=None,
                         help="Model id. Falls back to "
                              "$OPENAI_COMPATIBLE_MODEL/$OPENAI_MODEL."),
            click.option("--api-base", "api_base", default=None,
                         help="API base. Falls back to "
                              "$OPENAI_API_BASE_URL/$OPENAI_API_BASE."),
            click.option("--api-key", "api_key", default=None,
                         help="API key. Falls back to $OPENAI_API_KEY."),
            click.option("--n-tasks", "n_tasks", type=int, default=None,
                         help="Cap the number of dataset tasks."),
            click.option("--task", multiple=True,
                         help="Restrict to specific task name(s)/glob(s). Repeatable."),
            click.option("--concurrency", type=int, default=4, show_default=True,
                         help="Concurrent trials (harbor n_concurrent_trials)."),
            click.option("--n-attempts", "n_attempts", type=int, default=1,
                         show_default=True, help="Attempts per task."),
            click.option("--timeout-multiplier", "timeout_multiplier", type=float,
                         default=4.0, show_default=True,
                         help="Multiplier on harbor timeouts (SWE-bench cold-start "
                              "needs 4-6x)."),
            click.option("--force-build", "force_build", is_flag=True,
                         help="Force-rebuild the environment image."),
            click.option("--backend-type", "backend_type", default="docker",
                         show_default=True,
                         help="Flash Sandbox backend (docker, kubernetes)."),
            click.option("--request-timeout-sec", "request_timeout_sec", type=float,
                         default=120.0, show_default=True),
            click.option("--agent-ready-timeout-sec", "agent_ready_timeout_sec",
                         type=float, default=600.0, show_default=True,
                         help="How long to wait for the in-sandbox agent to come up."),
            click.option("--agent-kwarg", "agent_kwarg", multiple=True,
                         help="Extra agent kwarg key=value (JSON-ish coerced). "
                              "Repeatable."),
            click.option("--agent-config-file", "agent_config_file", default=None,
                         type=click.Path(exists=True, dir_okay=False),
                         help="YAML forwarded to the agent's config_file kwarg."),
            click.option("--job-name", "job_name", default=None,
                         help="Harbor job name (defaults to a timestamp)."),
            click.option("--list-agents", "list_agents", is_flag=True,
                         help="List registry agent keys and exit."),
        ]

    def run(self, shared: SharedOpts, *, agent, dataset, model, api_base, api_key,
            n_tasks, task, concurrency, n_attempts, timeout_multiplier,
            force_build, backend_type, request_timeout_sec,
            agent_ready_timeout_sec, agent_kwarg, agent_config_file, job_name,
            list_agents) -> Optional[int]:
        import argparse
        import asyncio

        # harbor_eval imports the (now required) `harbor` package at module top;
        # import it lazily so the recipe registry doesn't pull harbor in eagerly.
        from benchmaker.swebench import harbor_eval as he

        if list_agents:
            click.echo("Registry agents (also accepts a bare harbor name or "
                       "module:Class):")
            for k, v in he.AGENT_REGISTRY.items():
                click.echo(f"  {k:16s} -> {v.get('name') or v.get('import_path')}")
            return 0

        if shared.dotenv:
            load_dotenv(shared.dotenv)
        he._normalize_openai_env()

        model = (model or os.environ.get("OPENAI_COMPATIBLE_MODEL")
                 or os.environ.get("OPENAI_MODEL"))
        if not (os.environ.get("FLASH_SANDBOX_URL")
                or os.environ.get("FLASH_SANDBOX_HOST")):
            raise click.UsageError(
                "set FLASH_SANDBOX_URL (e.g. http://localhost:8080).")
        if not model:
            raise click.UsageError(
                "--model required (or set OPENAI_COMPATIBLE_MODEL).")

        # Reuse harbor_eval's JobConfig builder by handing it an argparse-shaped
        # namespace (the same field names its CLI produces).
        ns = argparse.Namespace(
            dataset=dataset,
            agent=agent,
            model=model,
            api_base=(api_base or os.environ.get("OPENAI_API_BASE_URL")
                      or os.environ.get("OPENAI_API_BASE")),
            api_key=(api_key or os.environ.get("OPENAI_API_KEY")),
            agent_kwarg=list(agent_kwarg),
            agent_config_file=agent_config_file,
            n_tasks=n_tasks,
            task=list(task),
            concurrency=concurrency,
            n_attempts=n_attempts,
            timeout_multiplier=timeout_multiplier,
            force_build=force_build,
            backend_type=backend_type,
            request_timeout_sec=request_timeout_sec,
            agent_ready_timeout_sec=agent_ready_timeout_sec,
            job_name=job_name,
        )
        job_config = he._build_job_config(ns)

        click.echo(f"harbor job: dataset={dataset} agent={agent} model={model} "
                   f"flash={os.environ.get('FLASH_SANDBOX_URL', '—')}")

        async def _go():
            from harbor.job import Job
            job = await Job.create(job_config)
            return job, await job.run()

        job, job_result = asyncio.run(_go())
        rows, accuracy = he._summarise(job_result)
        he._print_summary(rows, accuracy)
        click.echo(f"\njob dir: {job.job_dir}")
        return None


register(SWEBenchRecipe())

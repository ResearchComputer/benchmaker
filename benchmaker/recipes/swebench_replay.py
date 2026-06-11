# benchmaker/recipes/swebench_replay.py
"""``swebench-replay`` recipe — deterministic SWE-bench re-evaluation.

Builds a replay store from recorded pi logs (or loads a prebuilt
`replay-trajectories.jsonl`), starts the stateless replay server in-process, and
runs the *real* harbor SWE-bench pipeline (pi + sandbox + verifier) with the
model endpoint pointed at the replay server — at one ``--concurrency`` or a
``--sweep`` of them. The LLM is the only thing mocked; everything else runs for
real, so re-runs are deterministic and free of model cost/variance.

Still requires ``FLASH_SANDBOX_URL`` (the sandbox + verifier are real). For
``--mode pi-container`` the server must be reachable from inside the sandbox:
bind ``--host 0.0.0.0`` and pass ``--reachable-host <ip-or-dns>``. See
`docs/superpowers/specs/2026-06-11-swebench-trajectory-replay-design.md`.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from benchmaker.env import load_dotenv
from benchmaker.recipes import register
from benchmaker.recipes.base import Recipe, SharedOpts


def _replay_url(host: str, port: int, reachable_host: Optional[str]) -> str:
    """Base URL the agent should dial. A 0.0.0.0/:: bind is not itself dialable,
    so fall back to loopback unless an explicit reachable host is given."""
    h = reachable_host or (host if host not in ("0.0.0.0", "::") else "127.0.0.1")
    return f"http://{h}:{port}/v1"


def _parse_concurrencies(sweep: Optional[str], concurrency: int) -> list[int]:
    if not sweep:
        return [concurrency]
    return [int(x.strip()) for x in sweep.split(",") if x.strip()]


class SWEBenchReplayRecipe(Recipe):
    name = "swebench-replay"
    help = (
        "Replay recorded SWE-bench trajectories deterministically: mock the LLM "
        "with recorded outputs, run the real pi+sandbox+verifier pipeline at one "
        "--concurrency or a --sweep. Requires FLASH_SANDBOX_URL."
    )
    wants_load_options = False

    def options(self) -> list:
        return [
            click.option("--job", "job", default=None, type=click.Path(file_okay=False),
                         help="Harbor job dir to convert (its pi logs)."),
            click.option("--trajectories", "trajectories", default=None,
                         type=click.Path(dir_okay=False),
                         help="Prebuilt replay-trajectories.jsonl (instead of --job)."),
            click.option("--concurrency", type=int, default=4, show_default=True,
                         help="Concurrent trials (harbor n_concurrent_trials)."),
            click.option("--sweep", default=None,
                         help="Comma list of concurrencies to run in sequence, "
                              "e.g. '1,5,25' (overrides --concurrency)."),
            click.option("--mode", type=click.Choice(["pi-host", "pi-container"]),
                         default="pi-host", show_default=True,
                         help="pi run mode (the harbor agent key)."),
            click.option("--host", default="127.0.0.1", show_default=True,
                         help="Replay server bind host (use 0.0.0.0 for container mode)."),
            click.option("--port", type=int, default=9100, show_default=True,
                         help="Replay server bind port (0 = ephemeral; good for "
                              "pi-host sweeps)."),
            click.option("--reachable-host", "reachable_host", default=None,
                         help="Host/IP the sandbox dials to reach the replay server "
                              "(container mode)."),
            click.option("--model", default=None,
                         help="Model id sent to the agent. Default: the recorded "
                              "trajectory's model."),
            click.option("--dataset", default="swebench-verified", show_default=True,
                         help="Harbor dataset slug."),
            click.option("--n-tasks", "n_tasks", type=int, default=None,
                         help="Cap the number of dataset tasks."),
            click.option("--task", multiple=True,
                         help="Restrict to specific task name(s)/glob(s). Repeatable."),
            click.option("--n-attempts", "n_attempts", type=int, default=1,
                         show_default=True, help="Attempts per task."),
            click.option("--timeout-multiplier", "timeout_multiplier", type=float,
                         default=4.0, show_default=True,
                         help="Multiplier on harbor timeouts."),
            click.option("--backend-type", "backend_type", default="docker",
                         show_default=True, help="Flash Sandbox backend."),
            click.option("--request-timeout-sec", "request_timeout_sec", type=float,
                         default=120.0, show_default=True),
            click.option("--agent-ready-timeout-sec", "agent_ready_timeout_sec",
                         type=float, default=600.0, show_default=True),
            click.option("--jobs-dir", "jobs_dir", default=None,
                         type=click.Path(file_okay=False),
                         help="Parent dir for run bundles (default 'jobs')."),
            click.option("--timeline/--no-timeline", "timeline", default=True,
                         show_default=True,
                         help="Capture timeline/utilization/tokens into the job dir."),
            click.option("--utilization-interval-sec", "utilization_interval_sec",
                         type=float, default=5.0, show_default=True),
        ]

    def run(self, shared: SharedOpts, *, job, trajectories, concurrency, sweep, mode,
            host, port, reachable_host, model, dataset, n_tasks, task, n_attempts,
            timeout_multiplier, backend_type, request_timeout_sec,
            agent_ready_timeout_sec, jobs_dir, timeline,
            utilization_interval_sec) -> Optional[int]:
        from benchmaker.swebench import harbor_eval as he
        from benchmaker.swebench import trajectory as T

        if (job is None) == (trajectories is None):
            raise click.UsageError("provide exactly one of --job or --trajectories.")
        if shared.dotenv:
            load_dotenv(shared.dotenv)
        he._normalize_openai_env()
        if not (os.environ.get("FLASH_SANDBOX_URL") or os.environ.get("FLASH_SANDBOX_HOST")):
            raise click.UsageError("set FLASH_SANDBOX_URL (the sandbox + verifier are real).")

        # 1) Resolve the trajectory store path.
        tmpdir: Optional[tempfile.TemporaryDirectory] = None
        if job:
            tmpdir = tempfile.TemporaryDirectory()
            traj_path = Path(tmpdir.name) / "replay-trajectories.jsonl"
            n = T.convert_job(job, traj_path)
            click.echo(f"converted {n} trajectories from {job}")
        else:
            traj_path = Path(trajectories)
        store = T.load_store(traj_path)
        if not store:
            raise click.UsageError(f"no trajectories loaded from {traj_path}.")

        # 2) Model: explicit flag, else the recorded model.
        recorded_model = next((t.model for t in store.values() if t.model), "")
        run_model = model or recorded_model
        if not run_model:
            raise click.UsageError("--model required (no model recorded in trajectories).")

        replay_url = _replay_url(host, port, reachable_host)
        concurrencies = _parse_concurrencies(sweep, concurrency)
        click.echo(f"replay: {len(store)} trajectories, model={run_model}, "
                   f"agent={mode}, url={replay_url}, concurrencies={concurrencies}")

        # Static harbor config shared by every sweep iteration; only `concurrency`
        # and `job_name` vary per run (set inside `_run_one`).
        base_ns = argparse.Namespace(
            dataset=dataset, agent=mode, model=run_model,
            api_key="replay",
            agent_kwarg=[], agent_config_file=None,
            n_tasks=n_tasks, task=list(task),
            n_attempts=n_attempts, timeout_multiplier=timeout_multiplier,
            force_build=False, backend_type=backend_type,
            request_timeout_sec=request_timeout_sec,
            agent_ready_timeout_sec=agent_ready_timeout_sec,
            jobs_dir=jobs_dir,
        )

        results: list[tuple] = []
        try:
            for c in concurrencies:
                results.append(asyncio.run(self._run_one(
                    store, base_ns, c, run_model, host, port, reachable_host,
                    timeline, utilization_interval_sec)))
        finally:
            if tmpdir is not None:
                tmpdir.cleanup()

        # Comparison table.
        click.echo("\nCONCURRENCY  ACCURACY  PASS/TOTAL  MISSES  JOB_DIR")
        for c, accuracy, n_pass, n_total, misses, job_dir in results:
            click.echo(f"{c:>11}  {accuracy:>7.1%}  {n_pass:>4}/{n_total:<5}  "
                       f"{misses:>6}  {job_dir}")
        return None

    async def _run_one(self, store, base_ns, concurrency, run_model, host, port,
                       reachable_host, timeline, utilization_interval_sec) -> tuple:
        """Serve `store` on host:port and run one harbor job at `concurrency`.

        Binds a fresh listener per call (pass --port 0 for an ephemeral port,
        which sidesteps rebind contention across a sweep); the agent endpoint URL
        is built from the *actually bound* port. `base_ns` carries the static
        harbor config; this deep-copies it and stamps the per-run fields."""
        import copy

        from aiohttp import web

        from benchmaker.swebench import harbor_eval as he
        from benchmaker.swebench.observability import run_job_with_observability
        from benchmaker.swebench.replay_server import as_app, get_misses

        app = as_app(store, model_fallback=run_model)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        try:
            bound_port = site._server.sockets[0].getsockname()[1]
            ns = copy.deepcopy(base_ns)
            ns.api_base = _replay_url(host, bound_port, reachable_host)
            ns.concurrency = concurrency
            ns.job_name = (f"replay_{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"
                           f"_c{concurrency}_{secrets.token_hex(2)}")
            job_config = he._build_job_config(ns)
            job, job_result, summary_text = await run_job_with_observability(
                job_config, flash_url=os.environ.get("FLASH_SANDBOX_URL"),
                util_interval=utilization_interval_sec, enabled=timeline)
            rows, accuracy = he._summarise(job_result)
            n_pass = sum(1 for r in rows if r["passed"])
            return (concurrency, accuracy, n_pass, len(rows), get_misses(app),
                    str(job.job_dir))
        finally:
            await runner.cleanup()


register(SWEBenchReplayRecipe())

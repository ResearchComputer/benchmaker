# benchmaker/recipes/swebench_replay.py
"""``swebench-replay`` recipe — deterministic SWE-bench re-evaluation.

Builds a replay store from recorded pi logs (or loads a prebuilt
`replay-trajectories.jsonl`), starts the stateless replay server in-process, and
runs the *real* harbor SWE-bench pipeline (pi + sandbox + verifier) with the
model endpoint pointed at the replay server — at one ``--concurrency`` or a
``--concurrency-sweep`` of them. The LLM is the only thing mocked; everything else runs for
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


# Hosts the agent cannot reach from *inside* a sandbox container (they resolve
# to the container's own loopback, not this host).
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def _resolve_url_host(host: str, reachable_host: Optional[str]) -> str:
    """The host the agent will actually dial. A 0.0.0.0/:: bind is not itself
    dialable, so fall back to loopback unless an explicit reachable host is
    given."""
    return reachable_host or (host if host not in ("0.0.0.0", "::") else "127.0.0.1")


def _replay_url(host: str, port: int, reachable_host: Optional[str]) -> str:
    """Base URL the agent should dial."""
    return f"http://{_resolve_url_host(host, reachable_host)}:{port}/v1"


def _parse_concurrencies(sweep: Optional[str], concurrency: int) -> list[int]:
    if not sweep:
        return [concurrency]
    return [int(x.strip()) for x in sweep.split(",") if x.strip()]


def _resolve_task_filter(task, exclude_task, store) -> tuple[list[str], int]:
    """Which dataset tasks to run, and how many trajectories can't be targeted.

    Default to exactly the recorded tasks (each trajectory's instance_id) so
    harbor replays only what we have trajectories for — otherwise it would run
    the whole ``--dataset`` and every task without a recording becomes a replay
    miss. An explicit ``--task`` wins (the user is narrowing on purpose).
    ``--exclude-task`` drops the named id(s) from the resolved set.
    Returns ``(task_ids, n_missing_instance_id)``."""
    excluded = set(exclude_task)
    explicit = [t for t in task if t not in excluded]
    if explicit:
        return explicit, 0
    ids = sorted({t.instance_id for t in store.values()
                  if t.instance_id and t.instance_id not in excluded})
    missing = sum(1 for t in store.values() if not t.instance_id)
    return ids, missing

class SWEBenchReplayRecipe(Recipe):
    name = "swebench-replay"
    help = (
        "Replay recorded SWE-bench trajectories deterministically: mock the LLM "
        "with recorded outputs, run the real pi+sandbox+verifier pipeline at one "
        "--concurrency or a --concurrency-sweep. Requires FLASH_SANDBOX_URL."
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
            click.option("--concurrency-sweep", "concurrency_sweep", default=None,
                         help="Comma list of concurrencies to run in sequence, "
                              "e.g. '1,5,25' (overrides --concurrency)."),
            click.option("--mode", type=click.Choice(["pi-host", "pi-container"]),
                         default="pi-host", show_default=True,
                         help="pi run mode (the harbor agent key)."),
            click.option("--route-tools", "route_tools",
                         type=click.Choice(["all", "bash"]),
                         default="all", show_default=True,
                         help="pi-host: which tools to route into the sandbox. "
                              "'all' routes bash+read+write+edit (matches how "
                              "trajectories are recorded); 'bash' routes only bash "
                              "(file edits hit the host fs and are lost on replay). "
                              "Ignored for pi-container (pi runs in the sandbox)."),
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
            click.option("--exec-timeout-sec", "exec_timeout_sec", type=float,
                         default=None,
                         help="pi-host: real per-command timeout (seconds) passed "
                              "to environment.exec for every routed tool call "
                              "(default 600). Lower it to surface real sandbox "
                              "slowness/hangs under load. Ignored for pi-container "
                              "(pi runs as one process with no per-command timeout)."),
            click.option("--n-tasks", "n_tasks", type=int, default=None,
                         help="Cap the number of recorded tasks to replay "
                              "(applied on top of the recorded-task filter)."),
            click.option("--task", multiple=True,
                         help="Restrict to specific task name(s)/glob(s). Repeatable."),
            click.option("--exclude-task", "exclude_task", multiple=True,
                         help="Drop specific task id(s) from the replay set. "
                              "Repeatable."),
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
            click.option("--validate-observations/--no-validate-observations",
                         "validate_observations", default=False, show_default=True,
                         help="Fail-fast on environment divergence: compare each "
                              "step's tool-result status against the recording and "
                              "stop the agent at the first mismatch. Requires a "
                              "trajectory store recorded with tool_results."),
            click.option("--utilization-interval-sec", "utilization_interval_sec",
                         type=float, default=5.0, show_default=True),
        ]

    def run(self, shared: SharedOpts, *, job, trajectories, concurrency,
            concurrency_sweep, mode, route_tools, host, port, reachable_host, model,
            dataset, exec_timeout_sec, n_tasks, task, exclude_task, n_attempts,
            timeout_multiplier, backend_type, request_timeout_sec,
            agent_ready_timeout_sec, jobs_dir, timeline,
            utilization_interval_sec, validate_observations) -> Optional[int]:
        from benchmaker.swebench import harbor_eval as he
        from benchmaker.swebench import trajectory as T

        if (job is None) == (trajectories is None):
            raise click.UsageError("provide exactly one of --job or --trajectories.")
        # pi-container runs inside the sandbox; a loopback replay URL resolves to
        # the container's own loopback, not this host, so the agent can never
        # reach the mock LLM. Fail fast with the fix instead of a cryptic
        # connection error mid-run.
        if mode == "pi-container" and _resolve_url_host(host, reachable_host) in _LOOPBACK_HOSTS:
            raise click.UsageError(
                "--mode pi-container runs inside the sandbox and cannot reach a "
                f"loopback replay URL ({_resolve_url_host(host, reachable_host)}). "
                "Either pass --reachable-host <ip/dns the sandbox can reach this "
                "host at> (with --host 0.0.0.0), or use --mode pi-host (pi runs "
                "locally and reaches 127.0.0.1 directly).")
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

        # Run exactly the recorded tasks, not the whole dataset (see helper).
        task_filter, n_missing = _resolve_task_filter(task, exclude_task, store)
        if n_missing:
            click.echo(f"warning: {n_missing} trajectories have no instance_id "
                       f"and cannot be targeted; they will be skipped.")
        if not task_filter:
            raise click.UsageError(
                "no task ids resolved from the trajectories (and no --task given); "
                "cannot select which tasks to replay.")

        replay_url = _replay_url(host, port, reachable_host)
        concurrencies = _parse_concurrencies(concurrency_sweep, concurrency)
        click.echo(f"replay: {len(store)} trajectories, {len(task_filter)} tasks, "
                   f"model={run_model}, agent={mode}, url={replay_url}, "
                   f"concurrencies={concurrencies}")

        # pi-host edits the sandbox over a bridge; the file tools (read/write/edit)
        # only land in the sandbox when routed (route_tools=all), which is how the
        # trajectories were recorded. With the agent default (bash-only) those
        # recorded edits replay against the host fs and silently no-op. pi-container
        # runs pi inside the sandbox, so the kwarg does not apply.
        agent_kwargs = [f"route_tools={route_tools}"] if mode == "pi-host" else []
        # Real per-command sandbox timeout. Only pi-host routes each tool call
        # through environment.exec(timeout_sec=...); pi-container runs as one
        # process with no per-command budget, so the flag is a no-op there.
        if exec_timeout_sec is not None:
            if mode == "pi-host":
                agent_kwargs.append(f"exec_timeout_s={exec_timeout_sec}")
            else:
                click.echo("warning: --exec-timeout-sec is ignored for "
                           "pi-container (no per-command timeout).")

        # Static harbor config shared by every sweep iteration; only `concurrency`
        # and `job_name` vary per run (set inside `_run_one`).
        base_ns = argparse.Namespace(
            dataset=dataset, agent=mode, model=run_model,
            api_key="replay",
            agent_kwarg=agent_kwargs, agent_config_file=None,
            n_tasks=n_tasks, task=task_filter, exclude_task=None,
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
                    timeline, utilization_interval_sec, validate_observations)))
        finally:
            if tmpdir is not None:
                tmpdir.cleanup()

        # Comparison table.
        click.echo("\nCONCURRENCY  ACCURACY  PASS/TOTAL  MISSES  DIVERG  JOB_DIR")
        for c, accuracy, n_pass, n_total, misses, diverg, job_dir in results:
            click.echo(f"{c:>11}  {accuracy:>7.1%}  {n_pass:>4}/{n_total:<5}  "
                       f"{misses:>6}  {diverg:>6}  {job_dir}")
        return None

    async def _run_one(self, store, base_ns, concurrency, run_model, host, port,
                       reachable_host, timeline, utilization_interval_sec,
                       validate_observations) -> tuple:
        """Serve `store` on host:port and run one harbor job at `concurrency`.

        Binds a fresh listener per call (pass --port 0 for an ephemeral port,
        which sidesteps rebind contention across a sweep); the agent endpoint URL
        is built from the *actually bound* port. `base_ns` carries the static
        harbor config; this deep-copies it and stamps the per-run fields."""
        import copy

        from aiohttp import web

        from benchmaker.swebench import harbor_eval as he
        from benchmaker.swebench.observability import run_job_with_observability
        from benchmaker.swebench.replay_server import as_app, get_divergences, get_misses

        app = as_app(store, model_fallback=run_model, validate=validate_observations)
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
                    get_divergences(app), str(job.job_dir))
        finally:
            await runner.cleanup()


register(SWEBenchReplayRecipe())

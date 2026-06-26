"""``pareval`` recipe — evaluate parallel-code generation via the ParEval
3-stage orchestrator (generate -> sandbox grade -> aggregate).

Like ``swebench``, this is a *self-driving* recipe: it owns its own execution
(``benchmaker.pareval.run.run_pareval``) and prints a pass@k summary rather than
flowing through ``BenchRunner``. There is no benchmaker run-bundle here.

Generation source is exactly one of: a live model (``--model`` / env) OR a
precomputed completions file (``--completions``). Grading always requires a
Flash Sandbox URL (``--sandbox-url`` / ``FLASH_SANDBOX_URL``).
"""

from __future__ import annotations

import os
import asyncio
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from benchmaker.env import load_dotenv
from benchmaker.recipes import register
from benchmaker.recipes.base import Recipe, SharedOpts


class ParEvalRecipe(Recipe):
    name = "pareval"
    help = (
        "Evaluate parallel-code generation via the ParEval orchestrator "
        "(generate -> sandbox grade -> aggregate). Self-driving: prints a "
        "pass@k summary, not a benchmaker run-bundle. Generation source is "
        "exactly one of --model (live) or --completions <file>."
    )
    # ParEval owns its own runner; the load/timeout/bundle flags don't apply.
    wants_load_options = False

    def options(self) -> list:
        return [
            click.option(
                "--model",
                default=None,
                help="Model id for live generation. Falls back to "
                "$OPENAI_COMPATIBLE_MODEL/$OPENAI_MODEL.",
            ),
            click.option(
                "--api-base",
                "api_base",
                default=None,
                help="API base. Falls back to "
                "$OPENAI_API_BASE_URL/$OPENAI_API_BASE.",
            ),
            click.option(
                "--api-key",
                "api_key",
                default=None,
                help="API key. Falls back to $OPENAI_API_KEY.",
            ),
            click.option(
                "--completions",
                default=None,
                type=click.Path(exists=True, dir_okay=False),
                help="Precomputed completions JSONL (skip live generation).",
            ),
            click.option(
                "--sandbox-url",
                "sandbox_url",
                default=None,
                help="Flash Sandbox URL for grading. Falls back to "
                "$FLASH_SANDBOX_URL/$FLASH_SANDBOX_HOST.",
            ),
            click.option(
                "--parallelism-models",
                "parallelism_models",
                default="serial,omp,mpi,kokkos",
                show_default=True,
                help="Comma-separated parallelism models to evaluate.",
            ),
            click.option(
                "--problem-type",
                "problem_type",
                multiple=True,
                help="Restrict to specific problem type(s). Repeatable.",
            ),
            click.option(
                "--problem",
                multiple=True,
                help="Restrict to specific problem name(s). Repeatable.",
            ),
            click.option(
                "--num-samples",
                "num_samples",
                type=int,
                default=1,
                show_default=True,
                help="Samples generated per problem.",
            ),
            click.option(
                "--k",
                default="1",
                show_default=True,
                help="Comma-separated k values for pass@k.",
            ),
            click.option(
                "--temperature",
                type=float,
                default=0.2,
                show_default=True,
            ),
            click.option(
                "--max-threads",
                "max_threads",
                type=int,
                default=8,
                show_default=True,
                help="Max threads for OMP/Kokkos runs.",
            ),
            click.option(
                "--max-procs",
                "max_procs",
                type=int,
                default=8,
                show_default=True,
                help="Max processes for MPI runs.",
            ),
            click.option(
                "--run-reps",
                "run_reps",
                type=int,
                default=3,
                show_default=True,
                help="Timed repetitions per run.",
            ),
            click.option(
                "--build-timeout",
                "build_timeout",
                type=float,
                default=30.0,
                show_default=True,
            ),
            click.option(
                "--run-timeout",
                "run_timeout",
                type=float,
                default=120.0,
                show_default=True,
            ),
            click.option(
                "--concurrency",
                type=int,
                default=4,
                show_default=True,
                help="Concurrent grading sandboxes.",
            ),
            click.option(
                "--exclusive-cpus/--no-exclusive-cpus",
                "exclusive_cpus",
                default=False,
                show_default=True,
                help="Pin runs to exclusive CPUs (required for trusted "
                "speedup numbers).",
            ),
            click.option(
                "--cpuset",
                default=None,
                help="Explicit cpuset (e.g. '0-7'). Default derived from "
                "max-threads/max-procs.",
            ),
            click.option(
                "--image",
                default="pareval-toolchain",
                show_default=True,
                help="Sandbox toolchain image.",
            ),
            click.option(
                "--sandbox-drivers-cpp",
                "sandbox_drivers_cpp",
                default="/opt/pareval/drivers/cpp",
                show_default=True,
                help="In-sandbox path to the C++ drivers.",
            ),
            click.option(
                "--kokkos-root",
                "kokkos_root",
                default="/opt/kokkos",
                show_default=True,
                help="In-sandbox Kokkos install root.",
            ),
            click.option(
                "--endpoint-prefix",
                "endpoint_prefix",
                default="/sandboxes",
                show_default=True,
                help="Flash Sandbox endpoint prefix.",
            ),
            click.option(
                "--regenerate",
                is_flag=True,
                help="Force live regeneration even if a cached "
                "completions.jsonl exists.",
            ),
            click.option(
                "--out-dir",
                "out_dir_opt",
                default=None,
                type=click.Path(file_okay=False),
                help="Output dir for completions/runs/metrics (default "
                "'pareval-runs/<datetime>_<randhex>').",
            ),
        ]

    def run(
        self,
        shared: SharedOpts,
        *,
        model,
        api_base,
        api_key,
        completions,
        sandbox_url,
        parallelism_models,
        problem_type,
        problem,
        num_samples,
        k,
        temperature,
        max_threads,
        max_procs,
        run_reps,
        build_timeout,
        run_timeout,
        concurrency,
        exclusive_cpus,
        cpuset,
        image,
        sandbox_drivers_cpp,
        kokkos_root,
        endpoint_prefix,
        regenerate,
        out_dir_opt,
    ) -> Optional[int]:
        # Import lazily so the recipe registry doesn't pull heavy deps eagerly.
        from benchmaker.pareval.run import ParEvalConfig, run_pareval

        if shared.dotenv:
            load_dotenv(shared.dotenv)

        sandbox_url = (
            sandbox_url
            or os.environ.get("FLASH_SANDBOX_URL")
            or os.environ.get("FLASH_SANDBOX_HOST")
        )
        api_base = (
            api_base
            or os.environ.get("OPENAI_API_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        model = (
            model
            or os.environ.get("OPENAI_COMPATIBLE_MODEL")
            or os.environ.get("OPENAI_MODEL")
        )

        # Exactly one generation source: live model XOR a completions file.
        has_model = bool(model)
        has_completions = bool(completions)
        if has_model == has_completions:
            raise click.UsageError(
                "pass --model (live generation) OR --completions <file>, "
                "not both/neither."
            )
        if not sandbox_url:
            raise click.UsageError(
                "set --sandbox-url or FLASH_SANDBOX_URL (grading needs it)."
            )

        parallelism = tuple(
            p.strip() for p in parallelism_models.split(",") if p.strip()
        )
        k_values = tuple(int(x.strip()) for x in k.split(",") if x.strip())

        if out_dir_opt is None:
            stamp = (
                f"{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"
                f"_{secrets.token_hex(3)}"
            )
            out_dir_opt = os.path.join("pareval-runs", stamp)
        out_path = Path(out_dir_opt)
        out_path.mkdir(parents=True, exist_ok=True)

        cfg = ParEvalConfig(
            out_dir=out_path,
            parallelism_models=parallelism,
            problem_types=tuple(problem_type) or None,
            names=tuple(problem) or None,
            num_samples=num_samples,
            k=k_values,
            completions_path=(Path(completions) if completions else None),
            model=(model if has_model else None),
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            regenerate=regenerate,
            sandbox_url=sandbox_url,
            image=image,
            endpoint_prefix=endpoint_prefix,
            sandbox_drivers_cpp=sandbox_drivers_cpp,
            kokkos_root=kokkos_root,
            max_threads=max_threads,
            max_procs=max_procs,
            run_reps=run_reps,
            build_timeout=build_timeout,
            run_timeout=run_timeout,
            concurrency=concurrency,
            exclusive_cpus=exclusive_cpus,
            cpuset=cpuset,
        )

        source = f"model={model}" if has_model else f"completions={completions}"
        click.echo(
            f"pareval: {source} parallelism={','.join(parallelism)} "
            f"sandbox={sandbox_url} out={out_path}"
        )

        metrics = asyncio.run(run_pareval(cfg))
        _print_summary(metrics, out_path)
        return None


def _print_summary(metrics: dict, out_dir: Path) -> None:
    """Echo overall pass@k / build / correct rates + the trusted flag."""
    overall = metrics.get("overall", {})
    click.echo("\npareval summary:")
    pass_block = overall.get("pass@k", {}) or {}
    for k in sorted(pass_block, key=lambda x: int(x)):
        click.echo(f"  pass@{k}: {pass_block[k]:.4f}")
    if "build_rate" in overall:
        click.echo(f"  build_rate:   {overall['build_rate']:.4f}")
    if "correct_rate" in overall:
        click.echo(f"  correct_rate: {overall['correct_rate']:.4f}")

    trusted = bool(metrics.get("trusted"))
    click.echo(f"  trusted: {trusted}")
    if not trusted:
        click.echo(
            "  note: speedup numbers are UNTRUSTED (no exclusive CPU pinning; "
            "pass --exclusive-cpus)."
        )
    click.echo(f"\nout dir: {out_dir}")


register(ParEvalRecipe())

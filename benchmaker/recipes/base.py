"""Recipe abstractions: the base class, the parsed-option containers, and the
centralized run loop shared by every recipe subcommand.

A *recipe* is a named, self-contained benchmark scenario (``http``, ``llm``,
``sandbox``, ``swebench``, ...). Each recipe declares its own CLI options and
knows how to turn the parsed values into a workload-type + workload (+ optional
hooks). The shared options (load/timing/output/dotenv/quiet) are owned by the
CLI command factory and handed to ``build`` via :class:`SharedOpts`.
"""

from __future__ import annotations

import asyncio
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from benchmaker.core.load import parse_duration, parse_rate_spec
from benchmaker.core.runner import BenchConfig, BenchRunner
from benchmaker.recipes._cli_shared import write_bundle_if_requested
from benchmaker.workloads.base import WorkloadType
from benchmaker.workloads.datasets import StaticWorkload, Workload


# CLI-owned shared option defaults. The recipe-supplied ``default_*`` overrides
# in :class:`BuildResult` only take effect when the user left the flag at these
# values (i.e. did not pass the flag).
DEFAULT_RATE = "10"
DEFAULT_DURATION = "10s"
DEFAULT_TIMEOUT_S = 600.0


@dataclass
class SharedOpts:
    """Parsed values of the CLI-owned options common to every recipe."""

    rate: str
    duration: str
    max_requests: Optional[int]
    timeout_s: float
    connection_limit: int
    dotenv: Optional[str]
    quiet: bool
    # output bundle
    out_dir: Optional[str]
    run_id: Optional[str]
    labels: tuple[str, ...]
    notes: str


@dataclass
class BuildResult:
    """Everything a recipe's ``build`` produces.

    ``source_config`` is the reproducible config dict recorded in the run
    bundle (mirrors the YAML shape). The ``default_*`` fields let a recipe
    supply sensible load/timeout defaults (e.g. ``swebench`` → ``closed:16``);
    they apply only when the user did not override the corresponding CLI flag.
    """

    workload_type: WorkloadType
    workload: Workload = field(default_factory=StaticWorkload)
    pre_hooks: list = field(default_factory=list)
    post_hooks: list = field(default_factory=list)
    source_config: dict = field(default_factory=dict)
    default_rate: Optional[str] = None
    default_duration: Optional[str] = None
    default_timeout_s: Optional[float] = None


class Recipe(ABC):
    """A named bundle mapping CLI options to a benchmark run.

    Most recipes implement :meth:`build` and inherit the default :meth:`run`,
    which drives the result through :class:`BenchRunner` (metrics + run-bundle,
    like every other recipe). A *self-driving* recipe (one whose engine is not
    ``BenchRunner`` — e.g. the harbor-backed ``swebench`` recipe) instead sets
    ``wants_load_options = False`` and overrides :meth:`run` to own its own
    execution and output; it need not implement :meth:`build`.
    """

    name: str = ""
    help: str = ""
    # When True (default) the CLI factory attaches the full shared option block
    # (load/timeout/output) and the default run() drives BenchRunner. Set False
    # for a self-driving recipe that only wants `--dotenv` and runs itself.
    wants_load_options: bool = True

    @abstractmethod
    def options(self) -> list[Callable]:
        """Return the recipe-specific click option/argument decorators.

        List order is the order they render in ``--help``; the CLI factory
        appends the shared options below them. None of the option dest names may
        collide with the shared dest set (see ``recipes._factory``).
        """

    def build(self, shared: SharedOpts, **params: Any) -> BuildResult:
        """Build a :class:`BuildResult` from the parsed recipe params.

        ``params`` are keyed by the recipe option dest names; ``shared`` carries
        the CLI-owned options. Required unless :meth:`run` is overridden.
        """
        raise NotImplementedError(
            f"recipe {self.name!r} must implement build() or override run()"
        )

    def run(self, shared: SharedOpts, **params: Any) -> Optional[int]:
        """Execute the recipe.

        Default: ``build()`` → resolve load → ``BenchRunner`` → render → bundle.
        Return ``None`` (or ``0``) on success, or a non-zero exit code. Override
        for a self-driving recipe.
        """
        result = self.build(shared, **params)

        # Recipe defaults win only when the user left the shared flag untouched.
        rate = shared.rate
        if result.default_rate is not None and shared.rate == DEFAULT_RATE:
            rate = result.default_rate
        duration = shared.duration
        if result.default_duration is not None and shared.duration == DEFAULT_DURATION:
            duration = result.default_duration
        timeout_s = shared.timeout_s
        if result.default_timeout_s is not None and shared.timeout_s == DEFAULT_TIMEOUT_S:
            timeout_s = result.default_timeout_s

        dur_s = parse_duration(duration) if isinstance(duration, str) else duration
        load = parse_rate_spec(rate, duration_s=dur_s, max_requests=shared.max_requests)

        cfg = BenchConfig(
            workload_type=result.workload_type,
            workload=result.workload,
            load=load,
            pre_hooks=result.pre_hooks,
            post_hooks=result.post_hooks,
            timeout_s=timeout_s,
            connection_limit=shared.connection_limit,
        )
        if shared.quiet:
            cfg.progress_every_s = 0.0

        runner = BenchRunner(cfg)
        asyncio.run(runner.run())
        runner.metrics.render(sys.stdout)

        source_config = {
            **result.source_config,
            "load": rate,
            "duration": duration,
            "max_requests": shared.max_requests,
            "timeout_s": timeout_s,
            "connection_limit": shared.connection_limit,
        }
        write_bundle_if_requested(
            runner, source_config,
            shared.out_dir, shared.run_id, shared.labels, shared.notes,
        )
        return None


def run_recipe(recipe: Recipe, shared: SharedOpts, **params: Any) -> Optional[int]:
    """Dispatch to ``recipe.run`` (kept as a stable entry point for the factory)."""
    return recipe.run(shared, **params)

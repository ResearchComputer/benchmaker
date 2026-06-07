"""Build a click command per recipe and compose shared + recipe options."""

from __future__ import annotations

import functools
import sys

import click

from benchmaker.recipes._cli_shared import output_options
from benchmaker.recipes.base import (
    DEFAULT_DURATION,
    DEFAULT_RATE,
    DEFAULT_TIMEOUT_S,
    Recipe,
    SharedOpts,
    run_recipe,
)


# Dest names the CLI factory owns. Recipe options MUST NOT reuse any of these.
SHARED_DESTS = frozenset({
    "rate", "duration", "max_requests", "timeout_s", "connection_limit",
    "dotenv", "quiet", "out_dir", "run_id", "labels", "notes",
})


def shared_options(f):
    """CLI-owned options common to every recipe.

    Applied *after* the recipe options so they render below them in ``--help``
    (matching the old ``quick``/``llm`` layout). Each ``click.option`` mutates
    ``f.__click_params__``; click reverses that list at command-build time, so
    the call order here reads bottom-to-top in ``--help``.
    """
    f = click.option("--quiet", is_flag=True, help="Suppress progress output.")(f)
    f = output_options(f)  # --out-dir / --run-id / --label / --notes
    f = click.option("--dotenv", type=click.Path(), default=".env",
                     help="Path to .env (default: .env). Use --dotenv '' to disable.")(f)
    f = click.option("--connection-limit", "connection_limit", default=1000, type=int,
                     help="Max concurrent connections.")(f)
    f = click.option("--timeout", "timeout_s", default=DEFAULT_TIMEOUT_S, type=float,
                     help="Per-request timeout in seconds.")(f)
    f = click.option("--max-requests", "max_requests", type=int, default=None,
                     help="Cap on total requests.")(f)
    f = click.option("--duration", default=DEFAULT_DURATION,
                     help="Run duration (e.g. '30s', '2m').")(f)
    f = click.option("--rate", default=DEFAULT_RATE,
                     help="Load spec: '100', 'poisson:100', 'closed:32', "
                          "'ramp:10..500:30s'.")(f)
    return f


def minimal_shared_options(f):
    """Shared options for a self-driving recipe — just ``--dotenv``.

    Self-driving recipes (``wants_load_options = False``) own their own run loop
    and output, so the load/timeout/bundle flags don't apply; they still benefit
    from `.env` loading.
    """
    return click.option("--dotenv", type=click.Path(), default=".env",
                        help="Path to .env (default: .env). Use --dotenv '' to "
                             "disable.")(f)


def make_command(recipe: Recipe) -> click.Command:
    """Construct a click ``Command`` for ``recipe`` (recipe + shared options)."""

    def callback(**kwargs):
        # pop() with defaults so this works for both the full shared block and
        # the minimal one (where most shared options are absent).
        shared = SharedOpts(
            rate=kwargs.pop("rate", DEFAULT_RATE),
            duration=kwargs.pop("duration", DEFAULT_DURATION),
            max_requests=kwargs.pop("max_requests", None),
            timeout_s=kwargs.pop("timeout_s", DEFAULT_TIMEOUT_S),
            connection_limit=kwargs.pop("connection_limit", 1000),
            dotenv=(kwargs.pop("dotenv", ".env") or None),
            quiet=kwargs.pop("quiet", False),
            out_dir=kwargs.pop("out_dir", None),
            run_id=kwargs.pop("run_id", None),
            labels=kwargs.pop("labels", ()),
            notes=kwargs.pop("notes", ""),
        )
        # Everything left in kwargs is recipe-specific.
        code = run_recipe(recipe, shared, **kwargs)
        if isinstance(code, int) and code != 0:
            sys.exit(code)

    callback.__name__ = recipe.name or "recipe"
    callback = functools.wraps(callback)(callback)

    recipe_opts = recipe.options()

    # Defensive guard: a recipe option dest colliding with a shared dest would
    # be silently swallowed by the SharedOpts pop()s in the callback. Detect it
    # by inspecting the params the recipe options attach to a throwaway target.
    _assert_no_clash(recipe, recipe_opts)

    # Click reverses __click_params__ at command-build time, so the LAST-applied
    # decorator renders FIRST. Apply the shared block first (renders at the
    # bottom), then the recipe options (render on top). Within the recipe list,
    # reversed() makes list order == display order.
    fn = callback
    fn = shared_options(fn) if recipe.wants_load_options else minimal_shared_options(fn)
    for opt in reversed(recipe_opts):
        fn = opt(fn)

    return click.command(name=recipe.name, help=recipe.help or None)(fn)


def _assert_no_clash(recipe: Recipe, recipe_opts: list) -> None:
    def _probe():
        pass
    for opt in recipe_opts:
        opt(_probe)
    dests = {p.name for p in getattr(_probe, "__click_params__", [])}
    bad = dests & SHARED_DESTS
    if bad:
        raise ValueError(
            f"recipe {recipe.name!r} declares option dest(s) {sorted(bad)} "
            f"that collide with the shared CLI options {sorted(SHARED_DESTS)}"
        )

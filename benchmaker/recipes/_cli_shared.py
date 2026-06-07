"""Shared CLI helpers used by both the recipe framework and ``benchmaker.cli``.

These used to live in ``benchmaker.cli``; they were moved here so the recipe
modules can reuse them without importing ``cli`` (which would create an import
cycle: ``cli`` imports ``recipes``, ``recipes`` would import ``cli``).
"""

from __future__ import annotations

import sys

import click


def output_options(f):
    """Attach --out-dir / --run-id / --label / --notes to a command."""
    f = click.option("--out-dir", type=click.Path(file_okay=False), default=None,
                     help="Parent directory for the run bundle. The bundle is written "
                          "to <out-dir>/<run-id>/.")(f)
    f = click.option("--run-id", default=None,
                     help="Explicit run id. Defaults to a UTC timestamp.")(f)
    f = click.option("--label", "labels", multiple=True,
                     help="Free-form 'key=value' tag stored in meta.json. Repeatable.")(f)
    f = click.option("--notes", default="", help="Free-form notes stored in meta.json.")(f)
    return f


def parse_labels(items: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise click.BadParameter(f"--label must be 'key=value', got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_headers(items: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if ":" not in it:
            raise click.BadParameter(f"Header must be 'Name: value', got {it!r}")
        k, v = it.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def write_bundle_if_requested(runner, source_config: dict,
                              out_dir: str | None, run_id: str | None,
                              labels: tuple[str, ...], notes: str) -> None:
    if not out_dir:
        return
    path = runner.write_bundle(
        out_dir,
        run_id=run_id,
        source_config=source_config,
        labels=parse_labels(labels),
        notes=notes,
    )
    sys.stderr.write(f"[benchmaker] wrote bundle to {path}\n")

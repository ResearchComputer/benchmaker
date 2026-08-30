"""benchmaker CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import sys

import click
import yaml

from benchmaker.config import build_config
from benchmaker.core.runner import BenchRunner
from benchmaker.recipes import all_recipes
from benchmaker.recipes._cli_shared import (
    output_options as _output_options,
    parse_headers as _parse_headers,
    write_bundle_if_requested as _write_bundle_if_requested,
)
from benchmaker.recipes._factory import make_command


# ---------------------------------------------------------------- main


@click.group()
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              help="Logging level (default: INFO).")
def main(log_level: str) -> None:
    """[benchmaker]: async HTTP benchmarking with pluggable workloads."""
    level = log_level.upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Chatty third-party loggers (one INFO line per HTTP request / hub fetch)
    # drown out our own output. Pin them to WARNING unless DEBUG was requested.
    if level != "DEBUG":
        for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "urllib3",
                      "huggingface_hub", "filelock", "fsspec", "datasets",
                      "aiohttp"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@_output_options
@click.option("--dotenv", type=click.Path(), default=".env",
              help="Path to .env file to load (default: .env). "
                   "Use --dotenv '' to disable.")
@click.option("--record", "record_path", type=click.Path(), default=None,
              help="Write a JSONL request trace (with relative timestamps) to "
                   "this path. A later run can replay it deterministically via "
                   "a 'replay:' config block. Overrides any 'record:' in YAML.")
@click.option("--replay", "replay_path", type=click.Path(exists=True, dir_okay=False),
              default=None,
              help="Replay a previously recorded trace at the same relative "
                   "timings. Overrides 'workload_type' / 'workload' / 'load' "
                   "(and any 'replay:' in YAML).")
@click.option("--replay-speed", type=float, default=None,
              help="Speed multiplier for --replay (default 1.0).")
@click.option("--quiet", is_flag=True, help="Suppress progress output.")
def run(config_path: str, out_dir: str | None, run_id: str | None,
        labels: tuple[str, ...], notes: str, dotenv: str,
        record_path: str | None, replay_path: str | None,
        replay_speed: float | None, quiet: bool) -> None:
    """Run a benchmark from a YAML config file.

    Environment variables (loaded from `.env` by default) are interpolated
    into the YAML using `${VAR}` or `${VAR:-default}` syntax.
    """
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)

    if record_path is not None:
        raw_cfg = {**raw_cfg, "record": {"path": record_path}}
    if replay_path is not None:
        replay_cfg: dict = {"path": replay_path}
        if replay_speed is not None:
            replay_cfg["speed"] = replay_speed
        raw_cfg = {**raw_cfg, "replay": replay_cfg}

    bench_cfg = build_config(raw_cfg, dotenv_path=(dotenv or None))
    if quiet:
        bench_cfg.progress_every_s = 0.0

    runner = BenchRunner(bench_cfg)
    asyncio.run(runner.run())
    runner.metrics.render(sys.stdout)
    _write_bundle_if_requested(runner, raw_cfg, out_dir, run_id, labels, notes)


@main.command()
@click.option("--url", required=True, help="Target URL.")
@click.option("--method", default="GET")
@click.option("--header", "-H", multiple=True, help="Header 'Name: value'.")
@click.option("--json-body", default=None, help="JSON body string.")
@click.option("--data", default=None, help="Raw body string.")
@click.option("--rate", default="10", help="Load spec, e.g. '100', 'poisson:100', "
              "'closed:32', 'ramp:10..500:30s'.")
@click.option("--duration", default="10s", help="Run duration (e.g. '30s', '2m').")
@click.option("--max-requests", type=int, default=None)
@click.option("--timeout", "timeout_s", default=60.0, type=float)
@click.option("--connection-limit", default=1000, type=int)
@_output_options
@click.option("--quiet", is_flag=True)
def quick(url: str, method: str, header: tuple[str, ...], json_body: str | None,
          data: str | None, rate: str, duration: str, max_requests: int | None,
          timeout_s: float, connection_limit: int,
          out_dir: str | None, run_id: str | None,
          labels: tuple[str, ...], notes: str, quiet: bool) -> None:
    """[deprecated] One-liner HTTP benchmark — use `benchmaker http` instead."""
    sys.stderr.write(
        "[benchmaker] 'quick' is deprecated; use 'benchmaker http'.\n"
    )
    cfg: dict = {
        "workload_type": {
            "type": "http",
            "url": url,
            "method": method,
            "headers": _parse_headers(header),
            "timeout_s": timeout_s,
        },
        "load": rate,
        "duration": duration,
        "max_requests": max_requests,
        "timeout_s": timeout_s,
        "connection_limit": connection_limit,
    }
    if json_body is not None:
        cfg["workload"] = {"type": "static", "items": [json.loads(json_body)]}
    elif data is not None:
        cfg["workload"] = {"type": "static", "items": [data.encode("utf-8")]}

    bench_cfg = build_config(cfg)
    if quiet:
        bench_cfg.progress_every_s = 0.0

    runner = BenchRunner(bench_cfg)
    asyncio.run(runner.run())
    runner.metrics.render(sys.stdout)
    _write_bundle_if_requested(runner, cfg, out_dir, run_id, labels, notes)


# ---------------------------------------------------------------- collect


@main.command()
@click.argument("paths", nargs=-1, required=True,
                type=click.Path(exists=True, file_okay=False))
@click.option("--format", "fmt", type=click.Choice(["md", "csv", "json"]),
              default="md", show_default=True,
              help="Output format. 'md' is a Markdown table, 'csv' is comma-separated, "
                   "'json' is a JSON array of row dicts.")
@click.option("--metric", "metrics", multiple=True,
              help="Extra dotted-path metric to add as a column "
                   "(e.g. 'workload_metrics.ttft_s.p50'). Repeatable.")
@click.option("--columns", default=None,
              help="Comma-separated list of column names to keep (after metrics are added). "
                   "Overrides the default column set.")
@click.option("--sort-by", default=None,
              help="Column name to sort rows by (ascending).")
@click.option("--label", "label_keys", multiple=True,
              help="Promote a meta.labels[<key>] entry into its own column. Repeatable.")
@click.option("--recursive/--no-recursive", default=True,
              help="When a path is a directory of run-dirs, descend one level to find them.")
def collect(paths: tuple[str, ...], fmt: str, metrics: tuple[str, ...],
            columns: str | None, sort_by: str | None,
            label_keys: tuple[str, ...], recursive: bool) -> None:
    """Collect summaries from one or more run-dirs into a table.

    Each PATH may be a run directory (containing meta.json + summary.json) or a
    directory of such run-dirs. With --recursive (default), a non-bundle
    directory is scanned for immediate subdirectories that are bundles.
    """
    from benchmaker.io.collect import collect_table, format_table, find_bundles

    bundle_dirs: list[str] = []
    for p in paths:
        bundle_dirs.extend(find_bundles(p, recursive=recursive))
    if not bundle_dirs:
        raise click.UsageError(
            f"No run bundles found under: {', '.join(paths)}. "
            "Run bundles must contain meta.json and summary.json."
        )

    rows, column_names = collect_table(
        bundle_dirs,
        extra_metrics=list(metrics),
        label_keys=list(label_keys),
    )
    if columns:
        column_names = [c.strip() for c in columns.split(",") if c.strip()]
    if sort_by:
        rows.sort(key=lambda r: (r.get(sort_by) is None, r.get(sort_by)))

    sys.stdout.write(format_table(rows, column_names, fmt))
    if fmt != "json":
        sys.stdout.write("\n")


# ---------------------------------------------------------------- recipes
#
# Each registered recipe (http, llm, sandbox, swebench, ...) is exposed as a
# `benchmaker <recipe> --args` subcommand, built from the recipe's options plus
# the shared load/output options. See benchmaker/recipes/.

for _recipe in all_recipes():
    main.add_command(make_command(_recipe))


if __name__ == "__main__":
    main()

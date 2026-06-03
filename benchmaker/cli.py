"""benchmaker CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

import click
import yaml

from benchmaker.config import build_config
from benchmaker.core.runner import BenchRunner


# ---------------------------------------------------------------- shared bits


def _output_options(f):
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


def _parse_labels(items: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise click.BadParameter(f"--label must be 'key=value', got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _write_bundle_if_requested(runner: BenchRunner, source_config: dict,
                               out_dir: str | None, run_id: str | None,
                               labels: tuple[str, ...], notes: str) -> None:
    if not out_dir:
        return
    path = runner.write_bundle(
        out_dir,
        run_id=run_id,
        source_config=source_config,
        labels=_parse_labels(labels),
        notes=notes,
    )
    sys.stderr.write(f"[bench-maker] wrote bundle to {path}\n")


# ---------------------------------------------------------------- main


@click.group()
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              help="Logging level (default: INFO).")
def main(log_level: str) -> None:
    """[benchmaker]: async HTTP benchmarking with pluggable workloads."""
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


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
    """One-liner benchmark of a single endpoint (no config file)."""
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


@main.command()
@click.option("--url", default=None,
              help="Endpoint URL (e.g. http://host:8000/v1/chat/completions). "
                   "Falls back to $OPENAI_API_BASE_URL/$OPENAI_BASE_URL.")
@click.option("--model", default=None,
              help="Model name. Falls back to $OPENAI_COMPATIBLE_MODEL/$OPENAI_MODEL.")
@click.option("--api-key", default=None,
              help="API key. Falls back to $OPENAI_API_KEY.")
@click.option("--header", "-H", multiple=True, help="Extra header 'Name: value'.")
@click.option("--prompt", "prompts", multiple=True,
              help="Prompt text (repeatable). Mutually exclusive with --prompts-jsonl.")
@click.option("--prompts-jsonl", type=click.Path(exists=True, dir_okay=False), default=None,
              help="JSONL file of prompts.")
@click.option("--prompt-field", default="prompt",
              help="Field to extract from each JSONL row (default: 'prompt').")
@click.option("--shuffle/--no-shuffle", default=True, help="Shuffle prompts (static only).")
@click.option("--seed", type=int, default=0)
@click.option("--max-tokens", type=int, default=128)
@click.option("--min-tokens", type=int, default=None,
              help="vLLM/SGLang extension: minimum tokens before EOS is honored.")
@click.option("--ignore-eos/--no-ignore-eos", default=None,
              help="vLLM/SGLang extension: keep generating past EOS until max_tokens.")
@click.option("--temperature", type=float, default=0.0)
@click.option("--top-p", type=float, default=None)
@click.option("--top-k", type=int, default=None)
@click.option("--stop", multiple=True, help="Stop string (repeatable).")
@click.option("--extra", "extras", multiple=True,
              help="Extra sampling param 'key=value' (value parsed as JSON, else string). "
                   "Repeatable.")
@click.option("--rate", default="10",
              help="Load spec, e.g. '100', 'poisson:100', 'closed:32', 'ramp:10..500:30s'.")
@click.option("--duration", default="10s")
@click.option("--max-requests", type=int, default=None)
@click.option("--timeout", "timeout_s", default=600.0, type=float)
@click.option("--connection-limit", default=1000, type=int)
@click.option("--dotenv", type=click.Path(), default=".env",
              help="Path to .env file (default: .env). Use --dotenv '' to disable.")
@_output_options
@click.option("--quiet", is_flag=True)
def llm(url: str | None, model: str | None, api_key: str | None,
        header: tuple[str, ...],
        prompts: tuple[str, ...], prompts_jsonl: str | None, prompt_field: str,
        shuffle: bool, seed: int,
        max_tokens: int, min_tokens: int | None, ignore_eos: bool | None,
        temperature: float, top_p: float | None, top_k: int | None,
        stop: tuple[str, ...], extras: tuple[str, ...],
        rate: str, duration: str, max_requests: int | None,
        timeout_s: float, connection_limit: int,
        dotenv: str,
        out_dir: str | None, run_id: str | None,
        labels: tuple[str, ...], notes: str,
        quiet: bool) -> None:
    """Benchmark an OpenAI-compatible chat-completions endpoint."""
    if prompts and prompts_jsonl:
        raise click.UsageError("--prompt and --prompts-jsonl are mutually exclusive.")
    if not prompts and not prompts_jsonl:
        raise click.UsageError("Provide at least one --prompt or --prompts-jsonl.")

    from benchmaker.config import build_workload
    from benchmaker.core.load import parse_duration, parse_rate_spec
    from benchmaker.core.runner import BenchConfig
    from benchmaker.workloads.llm import OpenAIChatWorkloadType

    wt_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout_s": timeout_s,
        "headers": _parse_headers(header),
    }
    if min_tokens is not None:
        wt_kwargs["min_tokens"] = min_tokens
    if ignore_eos is not None:
        wt_kwargs["ignore_eos"] = ignore_eos
    if top_p is not None:
        wt_kwargs["top_p"] = top_p
    if top_k is not None:
        wt_kwargs["top_k"] = top_k
    if stop:
        wt_kwargs["stop"] = list(stop)
    for item in extras:
        if "=" not in item:
            raise click.BadParameter(f"--extra must be 'key=value', got {item!r}")
        k, v = item.split("=", 1)
        try:
            parsed: Any = json.loads(v)
        except json.JSONDecodeError:
            parsed = v
        wt_kwargs[k.strip()] = parsed

    wt = OpenAIChatWorkloadType.from_env(
        url=url, model=model, api_key=api_key,
        dotenv_path=(dotenv or None),
        **wt_kwargs,
    )

    if prompts:
        workload_spec: Any = {
            "type": "static",
            "items": list(prompts),
            "shuffle": shuffle,
            "seed": seed,
        }
    else:
        workload_spec = {
            "type": "jsonl",
            "path": prompts_jsonl,
            "field": prompt_field,
        }
    workload = build_workload(workload_spec)

    dur = parse_duration(duration)
    load_model = parse_rate_spec(rate, duration_s=dur, max_requests=max_requests)
    bench_cfg = BenchConfig(
        workload_type=wt,
        workload=workload,
        load=load_model,
        timeout_s=timeout_s,
        connection_limit=connection_limit,
    )

    if quiet:
        bench_cfg.progress_every_s = 0.0

    runner = BenchRunner(bench_cfg)
    asyncio.run(runner.run())
    runner.metrics.render(sys.stdout)

    source_config = {
        "workload_type": {
            "type": "openai-chat", "url": wt._url, "model": wt._model,
            **{k: v for k, v in wt_kwargs.items() if k != "headers"},
        },
        "workload": workload_spec,
        "load": rate,
        "duration": duration,
        "max_requests": max_requests,
        "timeout_s": timeout_s,
        "connection_limit": connection_limit,
    }
    _write_bundle_if_requested(runner, source_config, out_dir, run_id, labels, notes)


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


def _parse_headers(items: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if ":" not in it:
            raise click.BadParameter(f"Header must be 'Name: value', got {it!r}")
        k, v = it.split(":", 1)
        out[k.strip()] = v.strip()
    return out


if __name__ == "__main__":
    main()

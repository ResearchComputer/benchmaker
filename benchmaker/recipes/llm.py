"""``llm`` recipe — benchmark an OpenAI-compatible chat-completions endpoint."""

from __future__ import annotations

import json
from typing import Any

import click

from benchmaker.recipes import register
from benchmaker.recipes._cli_shared import parse_headers
from benchmaker.recipes.base import BuildResult, Recipe, SharedOpts


class LlmRecipe(Recipe):
    name = "llm"
    help = "Benchmark an OpenAI-compatible chat-completions endpoint."

    def options(self) -> list:
        return [
            click.option("--url", default=None,
                         help="Endpoint URL (e.g. http://host:8000/v1/chat/completions). "
                              "Falls back to $OPENAI_API_BASE_URL/$OPENAI_BASE_URL."),
            click.option("--model", default=None,
                         help="Model name. Falls back to "
                              "$OPENAI_COMPATIBLE_MODEL/$OPENAI_MODEL."),
            click.option("--api-key", "api_key", default=None,
                         help="API key. Falls back to $OPENAI_API_KEY."),
            click.option("--header", "-H", "header", multiple=True,
                         help="Extra header 'Name: value'. Repeatable."),
            click.option("--prompt", "prompts", multiple=True,
                         help="Prompt text (repeatable). Mutually exclusive with "
                              "--prompts-jsonl."),
            click.option("--prompts-jsonl", "prompts_jsonl",
                         type=click.Path(exists=True, dir_okay=False), default=None,
                         help="JSONL file of prompts."),
            click.option("--prompt-field", "prompt_field", default="prompt",
                         help="Field to extract from each JSONL row (default: 'prompt')."),
            click.option("--full-jsonl-row/--no-full-jsonl-row", "full_jsonl_row",
                         default=False,
                         help="Yield each full JSONL object to the workload type "
                              "(preserve row metadata into samples.jsonl) instead "
                              "of extracting one field. Equivalent: "
                              "--prompt-field '' / none."),
            click.option("--shuffle/--no-shuffle", default=True,
                         help="Shuffle prompts (static only)."),
            click.option("--seed", type=int, default=0),
            click.option("--max-tokens", "max_tokens", type=int, default=128),
            click.option("--min-tokens", "min_tokens", type=int, default=None,
                         help="vLLM/SGLang extension: minimum tokens before EOS."),
            click.option("--ignore-eos/--no-ignore-eos", "ignore_eos", default=None,
                         help="vLLM/SGLang extension: keep generating past EOS."),
            click.option("--temperature", type=float, default=0.0),
            click.option("--top-p", "top_p", type=float, default=None),
            click.option("--top-k", "top_k", type=int, default=None),
            click.option("--stop", multiple=True, help="Stop string (repeatable)."),
            click.option("--extra", "extras", multiple=True,
                         help="Extra sampling param 'key=value' (value parsed as JSON, "
                              "else string). Repeatable."),
        ]

    def build(self, shared: SharedOpts, *, url, model, api_key, header, prompts,
              prompts_jsonl, prompt_field, full_jsonl_row, shuffle, seed,
              max_tokens, min_tokens, ignore_eos, temperature, top_p, top_k,
              stop, extras) -> BuildResult:
        from benchmaker.config import build_workload
        from benchmaker.workloads.llm import OpenAIChatWorkloadType

        if prompts and prompts_jsonl:
            raise click.UsageError("--prompt and --prompts-jsonl are mutually exclusive.")
        if not prompts and not prompts_jsonl:
            raise click.UsageError("Provide at least one --prompt or --prompts-jsonl.")

        wt_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout_s": shared.timeout_s,
            "headers": parse_headers(header),
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

        full_row = full_jsonl_row or (prompt_field or "").strip().lower() in (
            "", "none", "null")
        if full_row and prompts:
            raise click.UsageError(
                "--full-jsonl-row (or an empty --prompt-field) requires "
                "--prompts-jsonl; it has no effect with static --prompt.")
        if full_row:
            wt_kwargs["passthrough_meta"] = True

        wt = OpenAIChatWorkloadType.from_env(
            url=url, model=model, api_key=api_key,
            dotenv_path=shared.dotenv,
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
                "field": None if full_row else prompt_field,
            }
        workload = build_workload(workload_spec)

        source_config = {
            "workload_type": {
                "type": "openai-chat", "url": wt._url, "model": wt._model,
                **{k: v for k, v in wt_kwargs.items() if k != "headers"},
            },
            "workload": workload_spec,
        }
        return BuildResult(
            workload_type=wt,
            workload=workload,
            source_config=source_config,
        )


register(LlmRecipe())

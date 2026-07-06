"""``tracelab`` recipe — benchmark an OpenAI-compatible endpoint against the
real coding-agent workload distribution from the TraceLab dataset.

Each request reproduces a real Claude/Codex round's token shape: a synthesized
prompt sized to the round's recorded ``input_tokens_total`` and a
``max_tokens`` set to its ``output_tokens``. With ``--prefix-cache`` the rounds
of each session are replayed with byte-exact growing prefixes so the server's
prefix cache is exercised the way it is on a real coding agent.

Prepare the dataset once with ``python tools/tracelab/prepare.py``.
"""

from __future__ import annotations

import json
from typing import Any

import click

from benchmaker.recipes import register
from benchmaker.recipes._cli_shared import parse_headers
from benchmaker.recipes.base import BuildResult, Recipe, SharedOpts


class TraceLabRecipe(Recipe):
    name = "tracelab"
    help = ("Benchmark an OpenAI-compatible endpoint with the TraceLab "
            "coding-agent trace (token-faithful synthetic replay).")

    def options(self) -> list:
        return [
            # --- endpoint ---
            click.option("--url", default=None,
                         help="Endpoint URL (e.g. http://host:8000/v1/chat/completions). "
                              "Falls back to $OPENAI_API_BASE_URL/$OPENAI_BASE_URL."),
            click.option("--model", default=None,
                         help="Model name sent in the request body. Falls back to "
                              "$OPENAI_COMPATIBLE_MODEL/$OPENAI_MODEL. (Note: this is the "
                              "target model under test, NOT the model recorded in the trace.)"),
            click.option("--api-key", "api_key", default=None,
                         help="API key. Falls back to $OPENAI_API_KEY."),
            click.option("--header", "-H", "header", multiple=True,
                         help="Extra header 'Name: value'. Repeatable."),
            click.option("--temperature", type=float, default=0.0),
            click.option("--top-p", "top_p", type=float, default=None),
            click.option("--top-k", "top_k", type=int, default=None),
            click.option("--extra", "extras", multiple=True,
                         help="Extra sampling param 'key=value' (JSON or string)."),

            # --- dataset ---
            click.option("--trace", "trace",
                         type=click.Path(exists=True, dir_okay=False), required=True,
                         help="TraceLab JSONL (.jsonl / .jsonl.gz / .gz) trace. "
                              "Get it with `python tools/tracelab/prepare.py`."),
            click.option("--prefix-cache/--flat", "prefix_cache", default=False,
                         help="Replay each session with byte-exact growing prefixes "
                              "(exercises the server's prefix cache) vs. independent "
                              "flat requests (reproduces the token distribution)."),
            click.option("--match-output-tokens/--no-match-output-tokens",
                         "match_output_tokens", default=False,
                         help="Force the server to decode exactly the recorded "
                              "output_tokens (sets min_tokens + ignore_eos). Best on "
                              "vLLM/SGLang; reproduces the true decode-length load."),
            click.option("--max-tokens-cap", "max_tokens_cap", type=int, default=None,
                         help="Ceiling on the per-request max_tokens derived from the "
                              "trace (guards against a few pathologically long rounds)."),
            click.option("--provider", default=None,
                         help="Keep only rows from this provider ('claude' or 'codex')."),
            click.option("--model-filter", "model_filter", default=None,
                         help="Keep only rows recorded under this model (e.g. "
                              "'claude-opus-4-8'). Distinct from --model (the target)."),
            click.option("--min-input-tokens", "min_input_tokens", type=int, default=None),
            click.option("--max-input-tokens", "max_input_tokens", type=int, default=None),
            click.option("--min-output-tokens", "min_output_tokens", type=int, default=None),
            click.option("--max-output-tokens", "max_output_tokens", type=int, default=None),
            click.option("--max-items", "max_items", type=int, default=None,
                         help="Cap on the number of rows replayed (after filtering)."),
            click.option("--max-sessions", "max_sessions", type=int, default=None,
                         help="Cap on the number of sessions kept (--prefix-cache only)."),

            # --- prompt sizing ---
            click.option("--chars-per-token", "chars_per_token", type=float, default=4.0,
                         help="Char-mode token-size approximation (default 4.0). Ignored "
                              "when --tokenizer is set."),
            click.option("--tokenizer", "tokenizer", default=None,
                         help="HuggingFace tokenizer id for exact token-count prompt "
                              "sizing (needs `pip install -e .[tokenizer]`)."),

            # --- ordering / looping ---
            click.option("--shuffle/--no-shuffle", default=True,
                         help="Shuffle rows (flat) or sessions (prefix-cache). Rounds "
                              "within a session always stay ordered."),
            click.option("--seed", type=int, default=0),
            click.option("--no-loop", "loop", flag_value=False, default=True,
                         help="Stop when the filtered trace is exhausted instead of "
                              "cycling it."),
        ]

    def build(self, shared: SharedOpts, *, url, model, api_key, header,
              temperature, top_p, top_k, extras, trace, prefix_cache,
              match_output_tokens, max_tokens_cap, provider, model_filter,
              min_input_tokens, max_input_tokens, min_output_tokens,
              max_output_tokens, max_items, max_sessions, chars_per_token,
              tokenizer, shuffle, seed, loop) -> BuildResult:
        from benchmaker.workloads.llm import OpenAIChatWorkloadType
        from benchmaker.workloads.tracelab import TraceLabWorkload

        wt_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "timeout_s": shared.timeout_s,
            "headers": parse_headers(header),
            "passthrough_meta": True,
        }
        if top_p is not None:
            wt_kwargs["top_p"] = top_p
        if top_k is not None:
            wt_kwargs["top_k"] = top_k
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
            dotenv_path=shared.dotenv,
            **wt_kwargs,
        )

        workload = TraceLabWorkload(
            trace,
            prefix_cache=prefix_cache,
            match_output_tokens=match_output_tokens,
            max_tokens_cap=max_tokens_cap,
            provider=provider,
            model_filter=model_filter,
            min_input_tokens=min_input_tokens,
            max_input_tokens=max_input_tokens,
            min_output_tokens=min_output_tokens,
            max_output_tokens=max_output_tokens,
            max_items=max_items,
            max_sessions=max_sessions,
            chars_per_token=chars_per_token,
            tokenizer=tokenizer,
            shuffle=shuffle,
            seed=seed,
            loop=loop,
        )

        workload_spec: dict[str, Any] = {
            "type": "tracelab",
            "path": trace,
            "prefix_cache": prefix_cache,
            "match_output_tokens": match_output_tokens,
            "provider": provider,
            "model_filter": model_filter,
            "min_input_tokens": min_input_tokens,
            "max_input_tokens": max_input_tokens,
            "min_output_tokens": min_output_tokens,
            "max_output_tokens": max_output_tokens,
            "max_items": max_items,
            "max_sessions": max_sessions,
            "chars_per_token": chars_per_token,
            "tokenizer": tokenizer,
            "shuffle": shuffle,
            "seed": seed,
            "loop": loop,
        }
        if max_tokens_cap is not None:
            workload_spec["max_tokens_cap"] = max_tokens_cap
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


register(TraceLabRecipe())

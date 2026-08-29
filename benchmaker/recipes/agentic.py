"""``agentic`` recipe — prefix-replay multi-turn agent trajectory datasets.

Expands each trajectory into one chat request per assistant turn (growing shared
prefix) against an OpenAI-compatible endpoint, recording the prefix-cache parity
pair: meta.expected_prefix_tokens (tokenizer upper bound) vs extra.cached_tokens
(server actual).

Two scheduling regimes:

* **Contiguous (default)** — all of trajectory A's turns, then all of B's. Turn
  k+1 is served within a few requests of turn k, so its history is reused while
  still hot in the local cache. Use ``--rate closed:N`` for clean prefix-cache
  locality (best case: locality preserved).
* **Interleaved** (``--concurrent-sessions N``) — keep up to N sessions active
  and round-robin their turns, gating each session's turn k+1 on turn k's
  completion (+ an optional ``--inter-turn-gap`` think time). Concurrent session
  histories overflow the device KV pool, so a session's history is evicted
  before its next turn — the multi-turn *reuse-after-eviction* regime that
  stresses hierarchical / shared KV tiers. The in-flight ceiling defaults to
  ``closed:N`` to match the active session count.
"""

from __future__ import annotations

from typing import Any

import click

from benchmaker.recipes import register
from benchmaker.recipes._cli_shared import parse_headers
from benchmaker.recipes.base import BuildResult, Recipe, SharedOpts


# Dataset presets: name -> (dataset id, split).
_PRESETS: dict[str, dict[str, Any]] = {
    "swe-smith": {"dataset": "SWE-bench/SWE-smith-trajectories", "split": "tool"},
}


class AgenticRecipe(Recipe):
    name = "agentic"
    help = ("Prefix-replay a multi-turn trajectory dataset (e.g. SWE-smith) "
            "against an OpenAI-compatible endpoint; one request per assistant "
            "turn, recording expected vs actual cached prefix tokens.")

    def options(self) -> list:
        return [
            click.option("--url", default=None,
                         help="Endpoint URL. Falls back to "
                              "$OPENAI_API_BASE_URL/$OPENAI_BASE_URL."),
            click.option("--model", default=None,
                         help="Target model. Falls back to "
                              "$OPENAI_COMPATIBLE_MODEL/$OPENAI_MODEL."),
            click.option("--api-key", "api_key", default=None,
                         help="API key. Falls back to $OPENAI_API_KEY."),
            click.option("--header", "-H", "header", multiple=True,
                         help="Extra header 'Name: value'. Repeatable."),
            click.option("--dataset", default=None,
                         help="HuggingFace dataset id (needs `datasets`). "
                              "Mutually exclusive with --prompts-jsonl."),
            click.option("--prompts-jsonl", "prompts_jsonl",
                         type=click.Path(exists=True, dir_okay=False), default=None,
                         help="Local JSONL of trajectory rows."),
            click.option("--split", default="tool", show_default=True,
                         help="Dataset split (HF source only)."),
            click.option("--preset", default=None,
                         help=f"Dataset preset: {', '.join(sorted(_PRESETS))}."),
            click.option("--tokenizer", default=None,
                         help="HF tokenizer id; enables exact "
                              "expected_prefix_tokens (theoretical upper bound)."),
            click.option("--messages-field", "messages_field", default="messages",
                         show_default=True),
            click.option("--id-field", "id_field", default="instance_id",
                         show_default=True),
            click.option("--model-field", "model_field", default="model",
                         show_default=True),
            click.option("--max-tokens", "max_tokens", type=int, default=1024,
                         show_default=True, help="Per-request generation cap."),
            click.option("--max-turns-per-trajectory", "max_turns_per_trajectory",
                         type=int, default=None,
                         help="Cap assistant turns replayed per trajectory."),
            click.option("--max-trajectories", "max_trajectories", type=int,
                         default=None, help="Cap number of trajectories replayed."),
            click.option("--concurrent-sessions", "concurrent_sessions", type=int,
                         default=None,
                         help="Interleave turns across up to N concurrent "
                              "sessions (round-robin, each session's turn k+1 "
                              "gated on turn k completing) instead of replaying "
                              "each trajectory contiguously. Enables the "
                              "reuse-after-eviction regime; defaults the rate to "
                              "closed:N."),
            click.option("--inter-turn-gap", "inter_turn_gap", default=None,
                         help="Per-session think time between consecutive turns "
                              "(interleaved mode). E.g. 'const:2s', 'exp:1.5', "
                              "'uniform:1s..3s'. Default: no gap."),
        ]

    def build(self, shared: SharedOpts, *, url, model, api_key, header, dataset,
              prompts_jsonl, split, preset, tokenizer, messages_field, id_field,
              model_field, max_tokens, max_turns_per_trajectory, max_trajectories,
              concurrent_sessions=None, inter_turn_gap=None) -> BuildResult:
        from benchmaker.workloads.llm import OpenAIChatWorkloadType
        from benchmaker.workloads.agentic import AgenticWorkload

        if preset:
            if preset not in _PRESETS:
                raise click.BadParameter(
                    f"unknown --preset {preset!r}; known: {sorted(_PRESETS)}")
            spec = _PRESETS[preset]
            dataset = dataset or spec["dataset"]
            if split == "tool":  # only override when left at default
                split = spec["split"]

        if bool(dataset) == bool(prompts_jsonl):
            raise click.UsageError(
                "provide exactly one of --dataset/--preset or --prompts-jsonl.")

        wt = OpenAIChatWorkloadType.from_env(
            url=url, model=model, api_key=api_key, dotenv_path=shared.dotenv,
            max_tokens=max_tokens, timeout_s=shared.timeout_s,
            headers=parse_headers(header), passthrough_meta=True)

        workload = AgenticWorkload(
            dataset=dataset, split=split, path=prompts_jsonl,
            messages_field=messages_field, id_field=id_field,
            model_field=model_field, max_tokens=max_tokens,
            max_turns_per_trajectory=max_turns_per_trajectory,
            max_trajectories=max_trajectories, tokenizer=tokenizer,
            concurrent_sessions=concurrent_sessions, inter_turn_gap=inter_turn_gap)

        source_config = {
            "workload_type": {"type": "openai-chat", "url": wt._url,
                              "model": wt._model, "passthrough_meta": True,
                              "max_tokens": max_tokens},
            "workload": {"type": "agentic", "dataset": dataset,
                         "split": split, "path": prompts_jsonl,
                         "messages_field": messages_field, "id_field": id_field,
                         "model_field": model_field, "tokenizer": tokenizer,
                         "max_tokens": max_tokens,
                         "max_trajectories": max_trajectories,
                         "max_turns_per_trajectory": max_turns_per_trajectory,
                         "concurrent_sessions": concurrent_sessions,
                         "inter_turn_gap": inter_turn_gap},
        }

        # Interleaved mode needs a per-turn completion signal to gate each
        # session's next turn; wire the workload's post-hook and default the
        # in-flight ceiling to the active session count.
        hook = workload.completion_hook()
        post_hooks: list = [hook] if hook is not None else []
        default_rate = ("closed:8" if concurrent_sessions is None
                        else f"closed:{concurrent_sessions}")

        # Finite dataset: replay once. The workload raises StopAsyncIteration when
        # exhausted, which halts the run; default to closed-loop with a long
        # nominal duration so exhaustion (not the clock) ends it.
        return BuildResult(
            workload_type=wt, workload=workload, source_config=source_config,
            post_hooks=post_hooks,
            default_rate=default_rate, default_duration="24h")


register(AgenticRecipe())

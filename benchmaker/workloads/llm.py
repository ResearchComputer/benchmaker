"""OpenAI-compatible chat-completions workload-type (with SSE streaming).

Item interpretation:
  * `str`            → wrap as `[{"role": "user", "content": item}]`
  * `list[dict]`     → use as `messages`
  * `dict`           → merged with the request body. May contain any of:
                        messages, prompt (alias for a user-message str), model,
                        max_tokens, temperature, top_p, stop, …
                        plus any extra OpenAI-compat sampling params.

Captured metrics (on top of base latency/throughput):
    ttft_s              first-token latency
    itl_ms_mean/p50/p99 inter-token latency (ms)
    tokens_out          completion tokens (from usage, else counted)
    prompt_tokens       from usage block when available
    tokens_per_s        completion_tokens / generation_time
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Any, Optional, Union

from benchmaker.env import load_dotenv
from benchmaker.types import Request, Response, Sample
from benchmaker.workloads.base import WorkloadType


# Env vars recognised by `OpenAIChatWorkloadType.from_env`.
# `OPENAI_BASE_URL` is the OpenAI SDK convention; `OPENAI_API_BASE_URL` and
# `OPENAI_API_BASE` are common alternates.
_ENV_BASE_URL_KEYS = ("OPENAI_API_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE")
_ENV_MODEL_KEYS = ("OPENAI_COMPATIBLE_MODEL", "OPENAI_MODEL")
_ENV_KEY_KEYS = ("OPENAI_API_KEY",)


class OpenAIChatWorkloadType(WorkloadType):
    streaming = True
    name = "openai-chat"

    def __init__(
        self,
        url: str,
        model: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        extra_body: Optional[dict] = None,
        headers: Optional[dict[str, str]] = None,
        api_key: Optional[str] = None,
        timeout_s: Optional[float] = 600.0,
        name: str = "openai-chat",
        **sampling: Any,
    ):
        """Any extra keyword argument (e.g. ``min_tokens``, ``ignore_eos``,
        ``top_p``, ``stop``, ``repetition_penalty``, ``guided_json``, ...) is
        forwarded into the request body. ``extra_body`` is still accepted as
        an explicit dict; ``**sampling`` overrides it on key conflict.
        """
        self.name = name
        self._url = url
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        merged: dict[str, Any] = dict(extra_body or {})
        merged.update(sampling)
        self._extra_body = merged
        self._timeout_s = timeout_s
        hdrs = dict(headers or {})
        hdrs.setdefault("Content-Type", "application/json")
        hdrs.setdefault("Accept", "text/event-stream")
        if api_key:
            hdrs.setdefault("Authorization", f"Bearer {api_key}")
        self._headers = hdrs

    @classmethod
    def from_env(
        cls,
        url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        dotenv_path: Optional[str] = ".env",
        endpoint_path: str = "chat/completions",
        **kwargs: Any,
    ) -> "OpenAIChatWorkloadType":
        """Construct from environment variables.

        Reads (in order of preference):
          base URL : OPENAI_API_BASE_URL, OPENAI_BASE_URL, OPENAI_API_BASE
          model    : OPENAI_COMPATIBLE_MODEL, OPENAI_MODEL
          api key  : OPENAI_API_KEY

        `dotenv_path` is loaded into `os.environ` first (without overriding
        existing vars). Explicit kwargs win over env vars.

        The full URL becomes `<base>/<endpoint_path>`, with at most one slash
        between them — so both `https://x/v1` and `https://x/v1/` work.
        """
        if dotenv_path:
            load_dotenv(dotenv_path)

        if url is None:
            base = _first_env(_ENV_BASE_URL_KEYS)
            if not base:
                raise ValueError(
                    "OPENAI_API_BASE_URL (or OPENAI_BASE_URL) is not set; "
                    "pass url=... or set the env var."
                )
            url = base.rstrip("/") + "/" + endpoint_path.lstrip("/")

        if model is None:
            model = _first_env(_ENV_MODEL_KEYS)
            if not model:
                raise ValueError(
                    "OPENAI_COMPATIBLE_MODEL (or OPENAI_MODEL) is not set; "
                    "pass model=... or set the env var."
                )

        if api_key is None:
            api_key = _first_env(_ENV_KEY_KEYS)  # optional; may stay None

        return cls(url=url, model=model, api_key=api_key, **kwargs)

    async def make_request(self, item: Any) -> Request:
        body: dict[str, Any] = {
            "model": self._model,
            "stream": True,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            **self._extra_body,
        }
        body.setdefault("stream_options", {"include_usage": True})

        if item is None:
            body["messages"] = [{"role": "user", "content": ""}]
        elif isinstance(item, str):
            body["messages"] = [{"role": "user", "content": item}]
        elif isinstance(item, list):
            body["messages"] = item
        elif isinstance(item, dict):
            # Promote "prompt" -> messages if no messages provided.
            if "messages" not in item and "prompt" in item:
                item = dict(item)
                item["messages"] = [{"role": "user", "content": item.pop("prompt")}]
            body.update(item)
            body.setdefault("messages", [{"role": "user", "content": ""}])
        else:
            raise TypeError(f"OpenAIChatWorkloadType cannot interpret item {type(item).__name__}")

        return Request(
            method="POST",
            url=self._url,
            headers=dict(self._headers),
            json=body,
            timeout_s=self._timeout_s,
            meta={
                "prompt_messages": body["messages"],
                "max_tokens": body.get("max_tokens"),
            },
        )

    async def make_sample(self, item: Any, request: Request, response: Response,
                          start_ts: float) -> Sample:
        sample = await super().make_sample(item, request, response, start_ts)

        chunks = response.stream_chunks or []
        chunk_times = response.stream_chunk_times or []
        ttft: Optional[float] = None
        token_arrival_times: list[float] = []
        usage_completion_tokens: Optional[int] = None
        prompt_tokens: Optional[int] = None
        finish_reason: Optional[str] = None

        for raw, t in zip(chunks, chunk_times):
            for line in raw.splitlines():
                if not line:
                    continue
                line = line.strip()
                if line.startswith(b"data:"):
                    line = line[5:].strip()
                if line == b"[DONE]":
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("usage"):
                    u = obj["usage"]
                    if u.get("completion_tokens") is not None:
                        usage_completion_tokens = int(u["completion_tokens"])
                    if u.get("prompt_tokens") is not None:
                        prompt_tokens = int(u["prompt_tokens"])
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        if ttft is None:
                            ttft = t
                        token_arrival_times.append(t)
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]

        completion_tokens = (
            usage_completion_tokens
            if usage_completion_tokens is not None
            else len(token_arrival_times)
        )

        if ttft is not None:
            sample.extra["ttft_s"] = ttft

        if len(token_arrival_times) >= 2:
            itl = [
                (token_arrival_times[i] - token_arrival_times[i - 1]) * 1000.0
                for i in range(1, len(token_arrival_times))
            ]
            sample.extra["itl_ms_mean"] = statistics.mean(itl)
            sample.extra["itl_ms_p50"] = _pct(itl, 50)
            sample.extra["itl_ms_p99"] = _pct(itl, 99)

        sample.extra["tokens_out"] = float(completion_tokens)
        if prompt_tokens is not None:
            sample.extra["prompt_tokens"] = float(prompt_tokens)

        if completion_tokens > 0 and ttft is not None and response.elapsed_s > ttft:
            sample.extra["tokens_per_s"] = completion_tokens / (response.elapsed_s - ttft)
        elif completion_tokens > 0 and response.elapsed_s > 0:
            sample.extra["tokens_per_s"] = completion_tokens / response.elapsed_s

        if finish_reason:
            sample.meta["finish_reason"] = finish_reason
        if response.ok and completion_tokens == 0:
            sample.ok = False
            sample.error = sample.error or "no tokens received"

        return sample


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _first_env(keys: tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return None

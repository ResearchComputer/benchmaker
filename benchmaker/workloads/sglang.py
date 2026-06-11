"""SGLang native ``/generate`` workload-type (SSE streaming).

SGLang's /generate is not OpenAI-compatible: the body is
``{"text": ..., "sampling_params": {...}, "stream": true}`` and the streamed
events carry *cumulative* ``text`` plus a ``meta_info`` object with
``prompt_tokens``, ``completion_tokens``, ``cached_tokens`` and ``finish_reason``.
Prefer the OpenAI chat path where possible; this type exists for raw-text
parity checks where /generate is the only endpoint.

Item interpretation:
  * ``str``   -> ``{"text": item}``
  * ``dict``  -> ``text``/``input_ids`` pulled out; remaining sampling keys merged
                into ``sampling_params``. With ``passthrough_meta`` (default on
                via the recipe), non-sampling keys are recorded into
                ``Request.meta`` instead of sent.

Captured metrics: ttft_s, itl_ms_mean/p50/p99, tokens_out, prompt_tokens,
cached_tokens, tokens_per_s, meta.finish_reason.
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Any, Optional

from benchmaker.env import load_dotenv
from benchmaker.core.types import Request, Response, Sample
from benchmaker.workloads.base import WorkloadType
from benchmaker.workloads.llm import _pct


_ENV_BASE_URL_KEYS = ("SGLANG_API_BASE_URL", "SGLANG_BASE_URL")

# Keys that belong in sampling_params (everything else in a dict item, in
# passthrough mode, is recorded into Request.meta).
_SAMPLING_KEYS = frozenset({
    "temperature", "max_new_tokens", "top_p", "top_k", "min_p", "stop",
    "stop_token_ids", "frequency_penalty", "presence_penalty",
    "repetition_penalty", "ignore_eos", "skip_special_tokens", "n", "seed",
    "min_new_tokens", "regex", "json_schema", "ebnf",
})


class SGLangGenerateWorkloadType(WorkloadType):
    streaming = True
    name = "sglang-generate"

    def __init__(
        self,
        url: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        extra_body: Optional[dict] = None,
        headers: Optional[dict[str, str]] = None,
        timeout_s: Optional[float] = 600.0,
        name: str = "sglang-generate",
        passthrough_meta: bool = False,
        **sampling: Any,
    ):
        self.name = name
        self._url = url
        self._max_tokens = max_tokens
        self._temperature = temperature
        merged: dict[str, Any] = dict(extra_body or {})
        merged.update(sampling)
        self._extra_sampling = merged
        self._timeout_s = timeout_s
        self._passthrough_meta = passthrough_meta
        hdrs = dict(headers or {})
        hdrs.setdefault("Content-Type", "application/json")
        hdrs.setdefault("Accept", "text/event-stream")
        self._headers = hdrs

    @classmethod
    def from_env(
        cls,
        url: Optional[str] = None,
        dotenv_path: Optional[str] = ".env",
        endpoint_path: str = "generate",
        **kwargs: Any,
    ) -> "SGLangGenerateWorkloadType":
        if dotenv_path:
            load_dotenv(dotenv_path)
        if url is None:
            base = None
            for k in _ENV_BASE_URL_KEYS:
                base = os.environ.get(k)
                if base:
                    break
            if not base:
                raise ValueError(
                    "SGLANG_API_BASE_URL (or SGLANG_BASE_URL) is not set; "
                    "pass url=... or set the env var.")
            url = base.rstrip("/") + "/" + endpoint_path.lstrip("/")
        return cls(url=url, **kwargs)

    async def make_request(self, item: Any) -> Request:
        sampling: dict[str, Any] = {
            "temperature": self._temperature,
            "max_new_tokens": self._max_tokens,
            **self._extra_sampling,
        }
        body: dict[str, Any] = {"stream": True}
        meta_extra: dict[str, Any] = {}

        if item is None:
            body["text"] = ""
        elif isinstance(item, str):
            body["text"] = item
        elif isinstance(item, dict):
            item = dict(item)
            if "text" in item:
                body["text"] = item.pop("text")
            elif "input_ids" in item:
                body["input_ids"] = item.pop("input_ids")
            if "max_tokens" in item and "max_new_tokens" not in item:
                item["max_new_tokens"] = item.pop("max_tokens")
            inner = item.pop("sampling_params", None)
            if isinstance(inner, dict):
                sampling.update(inner)
            for k, v in item.items():
                if k in _SAMPLING_KEYS:
                    sampling[k] = v
                elif self._passthrough_meta:
                    if k == "meta" and isinstance(v, dict):
                        meta_extra.update(v)
                    elif k == "model":
                        meta_extra["model_label"] = v
                    else:
                        meta_extra[k] = v
                else:
                    body[k] = v
            if "input_ids" not in body:
                body.setdefault("text", "")
        else:
            raise TypeError(
                f"SGLangGenerateWorkloadType cannot interpret {type(item).__name__}")

        body["sampling_params"] = sampling
        # Row metadata first, then canonical keys last so they always win over
        # any same-named row field (mirrors OpenAIChatWorkloadType).
        meta: dict[str, Any] = dict(meta_extra)
        meta["prompt_text"] = body.get("text")
        meta["max_tokens"] = sampling.get("max_new_tokens")
        return Request(method="POST", url=self._url, headers=dict(self._headers),
                       json=body, timeout_s=self._timeout_s, meta=meta)

    async def make_sample(self, item: Any, request: Request, response: Response,
                          start_ts: float) -> Sample:
        sample = await super().make_sample(item, request, response, start_ts)
        chunks = response.stream_chunks or []
        chunk_times = response.stream_chunk_times or []
        ttft: Optional[float] = None
        arrivals: list[float] = []
        last_text = ""
        completion_tokens: Optional[int] = None
        prompt_tokens: Optional[int] = None
        cached_tokens: Optional[int] = None
        finish_reason: Any = None

        for raw, t in zip(chunks, chunk_times):
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith(b"data:"):
                    line = line[5:].strip()
                if not line or line == b"[DONE]":
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                text = obj.get("text")
                if isinstance(text, str) and text != last_text:
                    if ttft is None:
                        ttft = t
                    arrivals.append(t)
                    last_text = text
                mi = obj.get("meta_info")
                if isinstance(mi, dict):
                    if mi.get("completion_tokens") is not None:
                        completion_tokens = int(mi["completion_tokens"])
                    if mi.get("prompt_tokens") is not None:
                        prompt_tokens = int(mi["prompt_tokens"])
                    if mi.get("cached_tokens") is not None:
                        cached_tokens = int(mi["cached_tokens"])
                    fr = mi.get("finish_reason")
                    if fr is not None:
                        finish_reason = fr

        out_tokens = completion_tokens if completion_tokens is not None else len(arrivals)

        if ttft is not None:
            sample.extra["ttft_s"] = ttft
        if len(arrivals) >= 2:
            itl = [(arrivals[i] - arrivals[i - 1]) * 1000.0
                   for i in range(1, len(arrivals))]
            sample.extra["itl_ms_mean"] = statistics.mean(itl)
            sample.extra["itl_ms_p50"] = _pct(itl, 50)
            sample.extra["itl_ms_p99"] = _pct(itl, 99)
        sample.extra["tokens_out"] = float(out_tokens)
        if prompt_tokens is not None:
            sample.extra["prompt_tokens"] = float(prompt_tokens)
        if cached_tokens is not None:
            sample.extra["cached_tokens"] = float(cached_tokens)
        if out_tokens > 0 and ttft is not None and response.elapsed_s > ttft:
            sample.extra["tokens_per_s"] = out_tokens / (response.elapsed_s - ttft)
        elif out_tokens > 0 and response.elapsed_s > 0:
            sample.extra["tokens_per_s"] = out_tokens / response.elapsed_s

        if finish_reason is not None:
            sample.meta["finish_reason"] = (
                finish_reason.get("type") if isinstance(finish_reason, dict)
                else finish_reason)
        if response.ok and out_tokens == 0:
            sample.ok = False
            sample.error = sample.error or "no tokens received"
        return sample

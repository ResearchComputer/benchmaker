"""Correctness / accuracy evaluation as a composable plugin.

Layered design (matches the rest of benchmaker):

  * `EvalWorkloadType` wraps any base WorkloadType. It strips an
    eval-only reference field (default `"reference"`) out of each dict item
    before delegating to the base, and stashes the reference on `Request.meta`
    so post-hooks can read it.

  * `correctness_hook(scorer, ...)` returns a `PostResponseHook` that extracts
    the model output from the `Response`, calls a `scorer(reference, prediction)`
    function, merges the returned scores into `Sample.extra`, and (optionally)
    fails the sample when a `gate_key` score is <= 0.

  * Stock scorers — `exact_match`, `contains`, `regex_match`, `json_valid`,
    `multiple_choice`, `judge_llm` — cover the usual graders. Custom scorers
    are just `(reference, prediction) -> dict[str, float]` (sync or async).

Because everything lands in `Sample.extra`, the `MetricsAggregator` summarises
accuracy (`correct.mean`, `judge_score.p50`, ...) generically — no changes to
the core required.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Optional, Union

from benchmaker.core.types import (
    PostResponseHook,
    Request,
    Response,
    Sample,
    TicketContext,
    maybe_await,
)
from benchmaker.workloads._sse import reassemble_sse_lines
from benchmaker.workloads.base import WorkloadType


# --------------------------------------------------------------------------- #
# Reference plumbing: EvalWorkloadType wrapper
# --------------------------------------------------------------------------- #


class EvalWorkloadType(WorkloadType):
    """Wrap a base WorkloadType to carry eval references through to post-hooks.

    Items are typically dicts like `{"prompt": "...", "reference": "..."}`.
    The named `reference_key` (and any `extra_meta_keys`) are stripped from the
    item before it reaches the base workload-type — so the base can't accidentally
    forward them to the service — and copied onto `Request.meta` so a downstream
    post-hook (e.g. `correctness_hook`) can score the response against them.

    The wrapper relies on `WorkloadType.run_ticket`'s default flow
    (one `make_request` -> one fire -> one `make_sample`). Workload-types with
    a custom `run_ticket` (e.g. sandbox lifecycle) should add a bespoke
    post-hook instead of wrapping.
    """

    def __init__(
        self,
        base: WorkloadType,
        reference_key: str = "reference",
        extra_meta_keys: tuple[str, ...] = (),
        name: Optional[str] = None,
    ):
        self._base = base
        self._reference_key = reference_key
        self._extra_meta_keys = tuple(extra_meta_keys)
        self.name = name or base.name
        self.streaming = base.streaming

    def _split(self, item: Any) -> tuple[Any, dict[str, Any]]:
        """Return (clean_item, eval_meta). For non-dict items, no split."""
        if not isinstance(item, dict):
            return item, {}
        keys = {self._reference_key, *self._extra_meta_keys}
        eval_meta = {k: item[k] for k in keys if k in item}
        if not eval_meta:
            return item, {}
        clean = {k: v for k, v in item.items() if k not in keys}
        return clean, eval_meta

    async def make_request(self, item: Any) -> Request:
        clean, eval_meta = self._split(item)
        req = await self._base.make_request(clean)
        for k, v in eval_meta.items():
            req.meta.setdefault(k, v)
        return req

    async def make_sample(self, item: Any, request: Request, response: Response,
                          start_ts: float) -> Sample:
        clean, _ = self._split(item)
        return await self._base.make_sample(clean, request, response, start_ts)

    async def aclose(self) -> None:
        await self._base.aclose()


# --------------------------------------------------------------------------- #
# Response -> text extraction
# --------------------------------------------------------------------------- #


def extract_openai_text(response: Response) -> str:
    """Concatenate assistant text from an OpenAI chat-completions response.

    Handles both streaming (`stream_chunks` of SSE bytes) and non-streaming
    (`body` containing a single JSON object) forms. Returns "" if nothing
    parses out — callers can fall back to `extract_raw_text` if they prefer
    the raw body.
    """
    parts: list[str] = []
    chunks = response.stream_chunks
    if chunks:
        # Reassemble across chunk boundaries: a content delta split by an
        # arbitrary byte boundary would otherwise be dropped, silently
        # truncating the extracted answer used for correctness scoring (#13).
        for line, _ in reassemble_sse_lines(chunks, response.stream_chunk_times):
            line = line.strip()
            if line.startswith(b"data:"):
                line = line[5:].strip()
            if not line or line == b"[DONE]":
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            _collect_openai_text(obj, parts)
        return "".join(parts)

    if not response.body:
        return ""
    try:
        obj = json.loads(response.body)
    except Exception:
        return ""
    _collect_openai_text(obj, parts)
    return "".join(parts)


def _collect_openai_text(obj: Any, parts: list[str]) -> None:
    if not isinstance(obj, dict):
        return
    for ch in obj.get("choices") or []:
        delta = ch.get("delta") or {}
        if isinstance(delta.get("content"), str):
            parts.append(delta["content"])
        msg = ch.get("message") or {}
        if isinstance(msg.get("content"), str):
            parts.append(msg["content"])


def extract_raw_text(response: Response) -> str:
    """Decode `response.body` as UTF-8 (best effort)."""
    if not response.body:
        return ""
    return response.body.decode("utf-8", errors="replace")


def extract_text(response: Response) -> str:
    """Default extractor: try OpenAI-chat shape, fall back to raw body."""
    text = extract_openai_text(response)
    if text:
        return text
    return extract_raw_text(response)


Extractor = Callable[[Response], str]


# --------------------------------------------------------------------------- #
# Scorer protocol + correctness hook
# --------------------------------------------------------------------------- #


# A scorer takes (reference, prediction) and returns a dict of numeric scores,
# either sync or async. Reference may be None when the workload didn't supply one.
Scorer = Callable[
    [Any, str],
    Union[dict[str, float], Awaitable[dict[str, float]]],
]


def correctness_hook(
    scorer: Scorer,
    *,
    reference_key: str = "reference",
    extractor: Optional[Extractor] = None,
    gate_key: Optional[str] = "correct",
    prefix: str = "",
    require_reference: bool = True,
    max_prediction_chars: Optional[int] = 2048,
) -> PostResponseHook:
    """Build a post-response hook that scores each request's output.

    Args:
        scorer: `(reference, prediction) -> dict[str, float]` (sync or async).
            The returned scores are merged into `Sample.extra` (with `prefix`
            prepended). Reference comes from `Request.meta[reference_key]`,
            populated by `EvalWorkloadType`.
        reference_key: meta key carrying the gold reference.
        extractor: maps `Response` -> prediction string. Defaults to
            `extract_text` (OpenAI chat first, raw body fallback).
        gate_key: if set and present in the scorer's output, the sample is
            marked `ok=False` when that score is <= 0. Set to None to disable
            gating (correctness still recorded, but doesn't affect goodput).
        prefix: prepended to every extra key (e.g. `"eval_"`).
        require_reference: if True and no reference is on the request, the
            sample is left untouched and a `<prefix>missing_reference=1` flag is
            added. If False, the scorer is still called with reference=None.
        max_prediction_chars: cap on the prediction copy stored in
            `Sample.meta["<prefix>prediction"]` (saved into `samples.jsonl`).
            Default 2048 chars to keep bundles small; set to `None` (or 0) to
            store the full output, or a smaller integer to truncate further.
    """
    extractor = extractor or extract_text

    async def hook(req: Request, resp: Response, sample: Sample) -> Sample:
        if not resp.ok:
            # Don't grade a failed request — keep the failure visible.
            return sample
        reference = req.meta.get(reference_key)
        if reference is None and require_reference:
            sample.extra[f"{prefix}missing_reference"] = 1.0
            return sample
        try:
            prediction = extractor(resp)
        except Exception as e:
            sample.extra[f"{prefix}extractor_error"] = 1.0
            sample.meta[f"{prefix}extractor_error_msg"] = f"{type(e).__name__}: {e}"
            return sample
        try:
            result = scorer(reference, prediction)
            result = await maybe_await(result)
        except Exception as e:
            sample.extra[f"{prefix}score_error"] = 1.0
            sample.meta[f"{prefix}score_error_msg"] = f"{type(e).__name__}: {e}"
            return sample
        if not isinstance(result, dict):
            sample.extra[f"{prefix}score_error"] = 1.0
            sample.meta[f"{prefix}score_error_msg"] = (
                f"scorer returned {type(result).__name__}, expected dict"
            )
            return sample
        for k, v in result.items():
            try:
                sample.extra[f"{prefix}{k}"] = float(v)
            except (TypeError, ValueError):
                sample.meta[f"{prefix}{k}"] = v
        if gate_key is not None and gate_key in result:
            try:
                if float(result[gate_key]) <= 0.0:
                    sample.ok = False
                    sample.error = sample.error or f"failed-{gate_key}"
            except (TypeError, ValueError):
                pass
        # Stash the prediction for offline inspection (capped to keep bundles small).
        if max_prediction_chars in (None, 0):
            saved = prediction
        else:
            saved = prediction[:max_prediction_chars]
        sample.meta.setdefault(f"{prefix}prediction", saved)
        return sample

    return hook


# --------------------------------------------------------------------------- #
# Stock scorers
# --------------------------------------------------------------------------- #


def _normalize(s: str, *, strip: bool, case_insensitive: bool) -> str:
    if strip:
        s = s.strip()
    if case_insensitive:
        s = s.lower()
    return s


def exact_match(*, strip: bool = True, case_insensitive: bool = False) -> Scorer:
    """`correct=1` iff prediction == reference (after optional strip/lower)."""

    def _score(reference: Any, prediction: str) -> dict[str, float]:
        a = _normalize(str(prediction), strip=strip, case_insensitive=case_insensitive)
        b = _normalize(str(reference), strip=strip, case_insensitive=case_insensitive)
        return {"correct": 1.0 if a == b else 0.0}

    return _score


def contains(*, strip: bool = True, case_insensitive: bool = True) -> Scorer:
    """`correct=1` iff reference substring appears in prediction."""

    def _score(reference: Any, prediction: str) -> dict[str, float]:
        a = _normalize(str(prediction), strip=strip, case_insensitive=case_insensitive)
        b = _normalize(str(reference), strip=strip, case_insensitive=case_insensitive)
        return {"correct": 1.0 if b and b in a else 0.0}

    return _score


def regex_match(pattern: str, *, group: int = 0,
                case_insensitive: bool = False) -> Scorer:
    """Match `pattern` against the prediction.

    If `reference` is None, `correct=1` whenever the pattern matches at all.
    Otherwise `correct=1` only when the captured `group` equals `str(reference)`
    (after stripping). The captured string is also recorded in `extra` as a
    side-channel for debugging.
    """
    flags = re.IGNORECASE if case_insensitive else 0
    rx = re.compile(pattern, flags)

    def _score(reference: Any, prediction: str) -> dict[str, float]:
        m = rx.search(prediction or "")
        if not m:
            return {"correct": 0.0, "matched": 0.0}
        try:
            captured = m.group(group)
        except IndexError:
            captured = ""
        if reference is None:
            return {"correct": 1.0, "matched": 1.0}
        ok = 1.0 if str(reference).strip() == (captured or "").strip() else 0.0
        return {"correct": ok, "matched": 1.0}

    return _score


def json_valid(*, required_keys: Optional[tuple[str, ...]] = None) -> Scorer:
    """`valid_json=1` if the prediction parses as JSON.

    If `required_keys` is given and the parsed value is a dict, `correct=1`
    only when every key is present. When `required_keys` is None, `correct`
    mirrors `valid_json`.
    """

    def _score(reference: Any, prediction: str) -> dict[str, float]:
        try:
            obj = json.loads(prediction)
        except Exception:
            return {"valid_json": 0.0, "correct": 0.0}
        if required_keys is None:
            return {"valid_json": 1.0, "correct": 1.0}
        if not isinstance(obj, dict):
            return {"valid_json": 1.0, "correct": 0.0}
        for k in required_keys:
            if k not in obj:
                return {"valid_json": 1.0, "correct": 0.0}
        return {"valid_json": 1.0, "correct": 1.0}

    return _score


def multiple_choice(*, choices: tuple[str, ...] = ("A", "B", "C", "D"),
                    case_insensitive: bool = True) -> Scorer:
    """Find the first choice letter mentioned in the prediction; match vs reference.

    Looks for one of `choices` as a word boundary token. Reference should be
    one of the choices.
    """
    flags = re.IGNORECASE if case_insensitive else 0
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(c) for c in choices) + r")\b", flags
    )

    def _score(reference: Any, prediction: str) -> dict[str, float]:
        m = pattern.search(prediction or "")
        if not m:
            return {"correct": 0.0, "answered": 0.0}
        chosen = m.group(1)
        ref = str(reference).strip()
        if case_insensitive:
            ok = 1.0 if chosen.lower() == ref.lower() else 0.0
        else:
            ok = 1.0 if chosen == ref else 0.0
        return {"correct": ok, "answered": 1.0}

    return _score


# --------------------------------------------------------------------------- #
# LLM-as-judge scorer
# --------------------------------------------------------------------------- #


JudgeSend = Callable[[str], Awaitable[str]]
JudgeTemplate = Union[str, Callable[[Any, str], str]]
JudgeParse = Callable[[str], dict[str, float]]


_DEFAULT_JUDGE_TEMPLATE = (
    "You are grading a model's answer.\n\n"
    "Question reference / expected answer:\n{reference}\n\n"
    "Model answer:\n{prediction}\n\n"
    "Reply with a single integer from 0 to 10 measuring how correct the model "
    "answer is. Output ONLY the integer."
)


def _default_judge_parse(text: str, *, pass_threshold: int = 7) -> dict[str, float]:
    m = re.search(r"-?\d+", text or "")
    if not m:
        return {"judge_score": 0.0, "correct": 0.0, "judge_parsed": 0.0}
    score = max(0, min(10, int(m.group(0))))
    return {
        "judge_score": float(score),
        "correct": 1.0 if score >= pass_threshold else 0.0,
        "judge_parsed": 1.0,
    }


def judge_llm(
    send: JudgeSend,
    *,
    template: JudgeTemplate = _DEFAULT_JUDGE_TEMPLATE,
    parse: Optional[JudgeParse] = None,
    max_concurrency: int = 4,
) -> Scorer:
    """LLM-as-judge scorer.

    `send(prompt) -> awaitable[str]` is the user's hook into a judge endpoint —
    the caller is responsible for opening / closing any HTTP client. Use the
    convenience constructor below if you just want to talk to an OpenAI-compat
    chat endpoint.

    `template` is either a format string (with `{reference}` and `{prediction}`)
    or a callable `(reference, prediction) -> str`.

    `parse(text) -> dict[str, float]` extracts numeric scores from the judge's
    reply. Default: parse a 0..10 integer, set `correct=1` when >= 7.
    """
    parse = parse or _default_judge_parse
    sem = asyncio.Semaphore(max_concurrency)

    if isinstance(template, str):
        tmpl_str = template

        def _render(ref: Any, pred: str) -> str:
            return tmpl_str.format(reference=ref, prediction=pred)
    else:
        _render = template  # type: ignore[assignment]

    async def _score(reference: Any, prediction: str) -> dict[str, float]:
        prompt = _render(reference, prediction)
        async with sem:
            reply = await send(prompt)
        return parse(reply)

    return _score


def openai_chat_judge(
    url: str,
    model: str,
    *,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8,
    timeout_s: float = 60.0,
) -> tuple[JudgeSend, Callable[[], Awaitable[None]]]:
    """Convenience: returns `(send, aclose)` for an OpenAI-compat chat endpoint.

    Opens a dedicated aiohttp session on first call. The caller should
    `await aclose()` at the end of the run (or wire it into a workload-type's
    `aclose` chain).
    """
    import aiohttp

    state: dict[str, Any] = {"session": None}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def _send(prompt: str) -> str:
        if state["session"] is None:
            state["session"] = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout_s)
            )
        sess: aiohttp.ClientSession = state["session"]
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        async with sess.post(url, headers=headers, json=body) as resp:
            data = await resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return msg.get("content") or ""

    async def _aclose() -> None:
        sess = state["session"]
        if sess is not None:
            await sess.close()
            state["session"] = None

    return _send, _aclose

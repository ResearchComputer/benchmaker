"""Streaming-SSE parsing must survive chunk boundaries that split an event.

Regression for issue #13: the HTTP client (httpx2's ``aiter_raw()``, formerly
aiohttp's ``iter_any()``) yields bytes at arbitrary boundaries, so a single
SSE ``data:`` event — most often the final, largest ``usage`` line — can be split across two ``stream_chunks``. Parsing each chunk
independently silently dropped the split event, losing ``prompt_tokens`` /
``cached_tokens`` (making prefix-cache ``hit`` read as 0 for every request).
"""
from __future__ import annotations

import json
import logging

import pytest

from benchmaker.core.types import Request, Response
from benchmaker.workloads.llm import OpenAIChatWorkloadType
from benchmaker.workloads.sglang import SGLangGenerateWorkloadType


def _usage_sse() -> bytes:
    events = [
        'data: ' + json.dumps({"choices": [{"index": 0, "delta": {"content": "w0"}}]}),
        'data: ' + json.dumps({"choices": [{"index": 0, "delta": {"content": "w1"}}]}),
        'data: ' + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        'data: ' + json.dumps({"choices": [], "usage": {
            "prompt_tokens": 18130, "completion_tokens": 2, "total_tokens": 18132,
            "prompt_tokens_details": {"cached_tokens": 18112}}}),
        'data: [DONE]',
    ]
    return "".join(e + "\n\n" for e in events).encode()


def _split_at(raw: bytes, marker: bytes, offset: int) -> list[bytes]:
    cut = raw.index(marker) + offset
    return [raw[:cut], raw[cut:]]


async def test_openai_usage_survives_split_across_chunks():
    raw = _usage_sse()
    # Cut in the middle of the usage JSON, as a byte-boundary split would.
    chunks = _split_at(raw, b'"usage"', 8)
    resp = Response(status=200, headers={}, body=raw, elapsed_s=0.6, ok=True,
                    stream_chunks=chunks, stream_chunk_times=[0.40, 0.50])
    wt = OpenAIChatWorkloadType(url="http://x", model="m", passthrough_meta=True)
    req = await wt.make_request({"messages": [{"role": "user", "content": "hi"}]})
    sample = await wt.make_sample("hi", req, resp, 0.0)
    assert sample.extra.get("prompt_tokens") == 18130.0
    assert sample.extra.get("cached_tokens") == 18112.0
    assert sample.extra.get("tokens_out") == 2.0


async def test_openai_usage_clean_chunks_still_parses():
    raw = _usage_sse()
    chunks = [e + b"\n\n" for e in raw.split(b"\n\n") if e]
    times = [0.40 + 0.01 * i for i in range(len(chunks))]
    resp = Response(status=200, headers={}, body=raw, elapsed_s=0.6, ok=True,
                    stream_chunks=chunks, stream_chunk_times=times)
    wt = OpenAIChatWorkloadType(url="http://x", model="m")
    req = await wt.make_request("hi")
    sample = await wt.make_sample("hi", req, resp, 0.0)
    assert sample.extra["prompt_tokens"] == 18130.0
    assert sample.extra["cached_tokens"] == 18112.0


async def test_openai_content_delta_split_across_chunks_counts_once():
    # A content event split across chunks must still count exactly one token
    # (not zero, not two) and preserve TTFT.
    one = 'data: ' + json.dumps({"choices": [{"index": 0, "delta": {"content": "hello"}}]})
    raw = (one + "\n\n" + "data: [DONE]\n\n").encode()
    cut = 20
    chunks = [raw[:cut], raw[cut:]]
    resp = Response(status=200, headers={}, body=raw, elapsed_s=0.3, ok=True,
                    stream_chunks=chunks, stream_chunk_times=[0.10, 0.20])
    wt = OpenAIChatWorkloadType(url="http://x", model="m")
    sample = await wt.make_sample("hi", await wt.make_request("hi"), resp, 0.0)
    assert sample.extra["tokens_out"] == 1.0
    # The token's arrival is the chunk that completed its line (the second one).
    assert sample.extra["ttft_s"] == 0.20


async def test_openai_warns_once_when_usage_requested_but_absent(caplog):
    # The one-time guard is process-global by design; reset it so this test
    # measures the warning behaviour in isolation from test ordering.
    import benchmaker.workloads.llm as llm_mod
    llm_mod._warned_missing_usage = False

    # Server omits usage entirely though include_usage was requested.
    raw = ("data: " + json.dumps({"choices": [{"delta": {"content": "hi"}}]})
           + "\n\ndata: [DONE]\n\n").encode()
    resp = Response(status=200, headers={}, body=raw, elapsed_s=0.2, ok=True,
                    stream_chunks=[raw], stream_chunk_times=[0.1])
    wt = OpenAIChatWorkloadType(url="http://x", model="m")
    req = await wt.make_request("hi")  # sets stream_options.include_usage=True
    with caplog.at_level(logging.WARNING):
        await wt.make_sample("hi", req, resp, 0.0)
        await wt.make_sample("hi", req, resp, 0.0)
    warnings = [r for r in caplog.records if "include_usage" in r.getMessage()]
    assert len(warnings) == 1  # one-time, not per-request spam


def test_extract_openai_text_survives_split_content_delta():
    from benchmaker.workloads.eval import extract_openai_text

    events = [
        'data: ' + json.dumps({"choices": [{"delta": {"content": "Hello "}}]}),
        'data: ' + json.dumps({"choices": [{"delta": {"content": "world"}}]}),
        'data: [DONE]',
    ]
    raw = "".join(e + "\n\n" for e in events).encode()
    # Split in the middle of the second content delta's JSON.
    cut = raw.index(b"world") - 3
    chunks = [raw[:cut], raw[cut:]]
    resp = Response(status=200, headers={}, body=raw, elapsed_s=0.2, ok=True,
                    stream_chunks=chunks, stream_chunk_times=[0.1, 0.2])
    assert extract_openai_text(resp) == "Hello world"


async def test_sglang_meta_info_survives_split_across_chunks():
    payload = json.dumps({"text": "foo bar", "meta_info": {
        "prompt_tokens": 50, "completion_tokens": 2, "cached_tokens": 40,
        "finish_reason": {"type": "stop"}}})
    raw = ("data: " + payload + "\n\ndata: [DONE]\n\n").encode()
    chunks = _split_at(raw, b'"meta_info"', 12)
    resp = Response(status=200, headers={}, body=raw, elapsed_s=0.2, ok=True,
                    stream_chunks=chunks, stream_chunk_times=[0.02, 0.06])
    wt = SGLangGenerateWorkloadType(url="http://x/generate")
    sample = await wt.make_sample("hi", Request(), resp, 0.0)
    assert sample.extra["prompt_tokens"] == 50.0
    assert sample.extra["cached_tokens"] == 40.0
    assert sample.extra["tokens_out"] == 2.0


# --------------------------------------------------------------------------- #
# Reasoning models: delta.reasoning_content must be counted (#14)              #
# --------------------------------------------------------------------------- #

def _reasoning_events(*, with_usage=True):
    """A thinking-model stream: reasoning_content deltas, then content deltas.

    Returns (raw, chunks, times) with one SSE event per chunk and explicit
    per-token arrival times (seconds since request start). Reasoning tokens
    arrive first (0.10, 0.11), then content (0.50, 0.51).
    """
    evs = [
        ('data: ' + json.dumps({"choices": [
            {"index": 0, "delta": {"reasoning_content": "Hmm"}}]}), 0.10),
        ('data: ' + json.dumps({"choices": [
            {"index": 0, "delta": {"reasoning_content": " let me think"}}]}), 0.11),
        ('data: ' + json.dumps({"choices": [
            {"index": 0, "delta": {"content": "The answer"}}]}), 0.50),
        ('data: ' + json.dumps({"choices": [
            {"index": 0, "delta": {"content": " is 42"}}]}), 0.51),
        ('data: ' + json.dumps({"choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}]}), 0.52),
    ]
    if with_usage:
        evs.append(('data: ' + json.dumps({
            "choices": [], "usage": {
                "prompt_tokens": 10, "completion_tokens": 4,
                "completion_tokens_details": {"reasoning_tokens": 2}}}), 0.53))
    evs.append(('data: [DONE]', 0.54))
    raw = b"".join((e + "\n\n").encode() for e, _ in evs)
    chunks = [(e + "\n\n").encode() for e, _ in evs]
    times = [t for _, t in evs]
    return raw, chunks, times


async def test_reasoning_tokens_counted_in_ttft_and_tokens():
    """A thinking model streams reasoning_content before content; the parser
    must count those tokens so ttft_s is the true first-byte latency and
    tokens_out is not undercounted (#14)."""
    raw, chunks, times = _reasoning_events()
    resp = Response(status=200, headers={}, body=raw, elapsed_s=0.6, ok=True,
                    stream_chunks=chunks, stream_chunk_times=times)
    wt = OpenAIChatWorkloadType(url="http://x", model="m")
    req = await wt.make_request("hi")
    sample = await wt.make_sample("hi", req, resp, 0.0)
    # TTFT is the first *any* token (the reasoning delta at 0.10s), not the
    # first content delta at 0.50s.
    assert sample.extra["ttft_s"] == 0.10
    # The first visible (content) token is surfaced separately.
    assert sample.extra["content_ttft_s"] == 0.50
    # tokens_out comes from usage.completion_tokens (4), not the old
    # content-only count of 2.
    assert sample.extra["tokens_out"] == 4.0
    assert sample.extra["reasoning_tokens"] == 2.0
    assert sample.extra["content_tokens"] == 2.0
    # ITL spans the whole generation (reasoning + content arrivals).
    assert "itl_ms_mean" in sample.extra
    assert sample.extra["itl_ms_mean"] > 0


async def test_ttft_token_content_uses_first_visible_token():
    """With ttft_token='content', the headline ttft_s is the first visible
    token; content_ttft_s is not duplicated onto it."""
    raw, chunks, times = _reasoning_events()
    resp = Response(status=200, headers={}, body=raw, elapsed_s=0.6, ok=True,
                    stream_chunks=chunks, stream_chunk_times=times)
    wt = OpenAIChatWorkloadType(url="http://x", model="m", ttft_token="content")
    req = await wt.make_request("hi")
    sample = await wt.make_sample("hi", req, resp, 0.0)
    assert sample.extra["ttft_s"] == 0.50
    assert "content_ttft_s" not in sample.extra
    assert sample.extra["reasoning_tokens"] == 2.0


async def test_reasoning_tokens_counted_when_usage_omitted():
    """Without a usage block, tokens_out falls back to the count of streamed
    tokens — which must include reasoning tokens, not just content (#14)."""
    raw, chunks, times = _reasoning_events(with_usage=False)
    resp = Response(status=200, headers={}, body=raw, elapsed_s=0.6, ok=True,
                    stream_chunks=chunks, stream_chunk_times=times)
    wt = OpenAIChatWorkloadType(url="http://x", model="m")
    req = await wt.make_request("hi")
    sample = await wt.make_sample("hi", req, resp, 0.0)
    # 2 reasoning + 2 content = 4 streamed tokens (was 2 before the fix).
    assert sample.extra["tokens_out"] == 4.0
    assert sample.extra["ttft_s"] == 0.10
    assert "reasoning_tokens" not in sample.extra


async def test_non_reasoning_response_unchanged_no_content_ttft():
    """A plain (non-reasoning) stream has no reasoning_content, so ttft_s is
    the first content token and content_ttft_s is not redundantly emitted."""
    evs = [
        ('data: ' + json.dumps({"choices": [
            {"index": 0, "delta": {"content": "w0"}}]}), 0.40),
        ('data: ' + json.dumps({"choices": [
            {"index": 0, "delta": {"content": "w1"}}]}), 0.41),
        ('data: ' + json.dumps({"choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}]}), 0.42),
        ('data: ' + json.dumps({"choices": [], "usage": {
            "prompt_tokens": 10, "completion_tokens": 2}}), 0.43),
        ('data: [DONE]', 0.44),
    ]
    raw = b"".join((e + "\n\n").encode() for e, _ in evs)
    chunks = [(e + "\n\n").encode() for e, _ in evs]
    times = [t for _, t in evs]
    resp = Response(status=200, headers={}, body=raw, elapsed_s=0.6, ok=True,
                    stream_chunks=chunks, stream_chunk_times=times)
    wt = OpenAIChatWorkloadType(url="http://x", model="m")
    req = await wt.make_request("hi")
    sample = await wt.make_sample("hi", req, resp, 0.0)
    assert sample.extra["ttft_s"] == 0.40
    assert "content_ttft_s" not in sample.extra
    assert "reasoning_tokens" not in sample.extra
    assert sample.extra["tokens_out"] == 2.0


def test_ttft_token_invalid_raises():
    with pytest.raises(ValueError):
        OpenAIChatWorkloadType(url="http://x", model="m", ttft_token="bogus")

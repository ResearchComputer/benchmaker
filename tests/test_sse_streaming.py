"""Streaming-SSE parsing must survive chunk boundaries that split an event.

Regression for issue #13: aiohttp's ``iter_any()`` yields bytes at arbitrary
boundaries, so a single SSE ``data:`` event — most often the final, largest
``usage`` line — can be split across two ``stream_chunks``. Parsing each chunk
independently silently dropped the split event, losing ``prompt_tokens`` /
``cached_tokens`` (making prefix-cache ``hit`` read as 0 for every request).
"""
from __future__ import annotations

import json
import logging

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

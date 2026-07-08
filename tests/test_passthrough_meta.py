"""passthrough_meta: record-not-send row metadata + cached-token capture."""
from __future__ import annotations

from benchmaker.core.types import Request, Response
from benchmaker.workloads.llm import OpenAIChatWorkloadType


def _wt(**kw):
    return OpenAIChatWorkloadType(
        url="http://x/v1/chat/completions", model="target-model", **kw)


async def test_passthrough_splits_body_and_meta():
    wt = _wt(passthrough_meta=True)
    req = await wt.make_request({
        "messages": [{"role": "user", "content": "hi"}],
        "max_new_tokens": 256,                 # aliased to max_tokens
        "temperature": 0.7,                    # allowlisted body param
        "model": "claude-3-7-sonnet",          # a LABEL, not the target
        "conversation_id": "c1",               # metadata
        "expected_prefix_tokens": 1024,        # metadata
        "meta": {"nested": "kept"},            # explicit envelope
    })
    body = req.json
    # body carries only request params
    assert body["model"] == "target-model"     # row 'model' did NOT override
    assert body["max_tokens"] == 256           # max_new_tokens aliased
    assert body["temperature"] == 0.7
    assert "conversation_id" not in body
    assert "expected_prefix_tokens" not in body
    assert "model_label" not in body
    # metadata recorded for samples.jsonl
    assert req.meta["model_label"] == "claude-3-7-sonnet"
    assert req.meta["conversation_id"] == "c1"
    assert req.meta["expected_prefix_tokens"] == 1024
    assert req.meta["nested"] == "kept"


async def test_passthrough_off_keeps_legacy_merge():
    wt = _wt()  # passthrough_meta defaults False
    req = await wt.make_request({"messages": [{"role": "user", "content": "x"}],
                                 "frequency_penalty": 0.1})
    assert req.json["frequency_penalty"] == 0.1  # merged into body as before


async def test_cached_tokens_from_prompt_tokens_details():
    wt = _wt()
    chunks = [
        b'data: {"choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null}]}\n\n',
        (b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
         b'"usage":{"prompt_tokens":100,"completion_tokens":1,'
         b'"prompt_tokens_details":{"cached_tokens":80}}}\n\n'),
        b'data: [DONE]\n\n',
    ]
    resp = Response(status=200, headers={}, body=b"", elapsed_s=0.1, ok=True,
                    stream_chunks=chunks, stream_chunk_times=[0.01, 0.05, 0.06])
    sample = await wt.make_sample({"x": 1}, Request(), resp, 0.0)
    assert sample.extra["cached_tokens"] == 80.0
    assert sample.extra["prompt_tokens"] == 100.0


async def test_cached_tokens_absent_not_recorded():
    wt = _wt()
    chunks = [
        b'data: {"choices":[{"index":0,"delta":{"content":"a"},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n',
        b'data: [DONE]\n\n',
    ]
    resp = Response(status=200, headers={}, body=b"", elapsed_s=0.1, ok=True,
                    stream_chunks=chunks, stream_chunk_times=[0.01, 0.02])
    sample = await wt.make_sample(None, Request(), resp, 0.0)
    assert "cached_tokens" not in sample.extra


async def test_passthrough_canonical_meta_keys_win_over_row_fields():
    wt = _wt(passthrough_meta=True)
    req = await wt.make_request({
        "messages": [{"role": "user", "content": "real"}],
        "prompt_messages": "row-annotation-not-the-real-prompt",
    })
    # canonical ground-truth must not be clobbered by a same-named row field
    assert req.meta["prompt_messages"] == [{"role": "user", "content": "real"}]


async def test_passthrough_max_tokens_wins_over_max_new_tokens():
    wt = _wt(passthrough_meta=True)
    req = await wt.make_request({
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 512,
        "max_new_tokens": 256,
    })
    assert req.json["max_tokens"] == 512          # explicit max_tokens wins in body
    assert "max_new_tokens" not in req.json       # not sent
    assert req.meta["max_new_tokens"] == 256      # stray value recorded to meta


# --------------------------------------------------------- llm recipe wiring

import json

from benchmaker.recipes import get
from benchmaker.recipes.base import SharedOpts


def _shared():
    return SharedOpts(
        rate="5", duration="0.3s", max_requests=None, timeout_s=30.0,
        connection_limit=100, dotenv="", quiet=True,
        out_dir=None, run_id=None, labels=(), notes="")


def test_llm_full_jsonl_row_build_enables_passthrough(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}],
                             "conversation_id": "c1"}) + "\n")
    built = get("llm").build(
        _shared(), url="http://x/v1/chat/completions", model="m", api_key=None,
        header=(), prompts=(), prompts_jsonl=str(p), prompt_field="prompt",
        full_jsonl_row=True, shuffle=False, seed=0, max_tokens=128,
        min_tokens=None, ignore_eos=None, temperature=0.0, top_p=None,
        top_k=None, stop=(), ttft_token="any", extras=())
    assert built.workload_type._passthrough_meta is True
    assert built.workload._field is None  # JsonlWorkload yields whole rows


def test_llm_empty_prompt_field_implies_full_row(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps({"prompt": "hi", "tag": "t"}) + "\n")
    built = get("llm").build(
        _shared(), url="http://x/v1/chat/completions", model="m", api_key=None,
        header=(), prompts=(), prompts_jsonl=str(p), prompt_field="",
        full_jsonl_row=False, shuffle=False, seed=0, max_tokens=128,
        min_tokens=None, ignore_eos=None, temperature=0.0, top_p=None,
        top_k=None, stop=(), ttft_token="any", extras=())
    assert built.workload_type._passthrough_meta is True
    assert built.workload._field is None


def test_llm_normal_jsonl_does_not_enable_passthrough(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps({"prompt": "hi"}) + "\n")
    built = get("llm").build(
        _shared(), url="http://x/v1/chat/completions", model="m", api_key=None,
        header=(), prompts=(), prompts_jsonl=str(p), prompt_field="prompt",
        full_jsonl_row=False, shuffle=False, seed=0, max_tokens=128,
        min_tokens=None, ignore_eos=None, temperature=0.0, top_p=None,
        top_k=None, stop=(), ttft_token="any", extras=())
    assert built.workload_type._passthrough_meta is False
    assert built.workload._field == "prompt"


def test_llm_full_row_with_static_prompt_errors():
    import click
    import pytest
    with pytest.raises(click.UsageError):
        get("llm").build(
            _shared(), url="http://x/v1/chat/completions", model="m",
            api_key=None, header=(), prompts=("hi",), prompts_jsonl=None,
            prompt_field="prompt", full_jsonl_row=True, shuffle=False, seed=0,
            max_tokens=128, min_tokens=None, ignore_eos=None, temperature=0.0,
            top_p=None, top_k=None, stop=(), ttft_token="any", extras=())

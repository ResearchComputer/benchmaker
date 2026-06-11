"""SGLang /generate workload type: request shape + meta_info parsing."""
from __future__ import annotations

import json

from benchmaker.core.types import Request, Response
from benchmaker.workloads.sglang import SGLangGenerateWorkloadType


def _wt(**kw):
    return SGLangGenerateWorkloadType(url="http://x/generate", **kw)


async def test_make_request_text_and_sampling():
    wt = _wt(max_tokens=64, temperature=0.0, top_p=0.9)
    req = await wt.make_request("hello")
    assert req.json["text"] == "hello"
    sp = req.json["sampling_params"]
    assert sp["max_new_tokens"] == 64
    assert sp["temperature"] == 0.0
    assert sp["top_p"] == 0.9
    assert req.json["stream"] is True


async def test_make_request_dict_passthrough_meta():
    wt = _wt(passthrough_meta=True)
    req = await wt.make_request({
        "text": "hi", "max_new_tokens": 32, "conversation_id": "c1"})
    assert req.json["text"] == "hi"
    assert req.json["sampling_params"]["max_new_tokens"] == 32
    assert "conversation_id" not in req.json
    assert req.meta["conversation_id"] == "c1"


async def test_make_sample_parses_meta_info():
    wt = _wt()
    # SGLang streams cumulative text + a meta_info object each chunk.
    chunks = [
        b'data: {"text":"foo","meta_info":{"prompt_tokens":50,"completion_tokens":1,"cached_tokens":40,"finish_reason":null}}\n\n',
        b'data: {"text":"foo bar","meta_info":{"prompt_tokens":50,"completion_tokens":2,"cached_tokens":40,"finish_reason":{"type":"stop"}}}\n\n',
        b'data: [DONE]\n\n',
    ]
    resp = Response(status=200, headers={}, body=b"", elapsed_s=0.2, ok=True,
                    stream_chunks=chunks, stream_chunk_times=[0.02, 0.06, 0.07])
    sample = await wt.make_sample("hi", Request(), resp, 0.0)
    assert sample.extra["prompt_tokens"] == 50.0
    assert sample.extra["cached_tokens"] == 40.0
    assert sample.extra["tokens_out"] == 2.0
    assert sample.extra["ttft_s"] == 0.02
    assert sample.meta["finish_reason"] == "stop"


async def test_make_request_input_ids_no_empty_text():
    wt = _wt(passthrough_meta=True)
    req = await wt.make_request({"input_ids": [1, 2, 3], "conversation_id": "c1"})
    assert req.json["input_ids"] == [1, 2, 3]
    assert "text" not in req.json              # no empty text injected
    assert req.meta["conversation_id"] == "c1"


async def test_make_sample_zero_tokens_marks_failure():
    wt = _wt()
    resp = Response(status=200, headers={}, body=b"", elapsed_s=0.1, ok=True,
                    stream_chunks=[], stream_chunk_times=[])
    sample = await wt.make_sample("hi", Request(), resp, 0.0)
    assert sample.ok is False
    assert sample.error == "no tokens received"
    assert sample.extra["tokens_out"] == 0.0


async def test_passthrough_canonical_meta_keys_win_over_row_fields():
    wt = _wt(passthrough_meta=True)
    req = await wt.make_request({
        "text": "real-prompt", "max_new_tokens": 32,
        "prompt_text": "row-annotation", "max_tokens": "row-annotation"})
    # canonical provenance must not be clobbered by same-named row fields
    assert req.meta["prompt_text"] == "real-prompt"
    assert req.meta["max_tokens"] == 32


import asyncio
import socket
import threading

import pytest
from aiohttp import web
from click.testing import CliRunner

from benchmaker.cli import main
from benchmaker.recipes import all_recipes, get
from benchmaker.recipes.base import SharedOpts


def test_sglang_recipe_registered():
    assert "sglang" in {r.name for r in all_recipes()}
    assert "sglang" in main.commands


def test_sglang_build_sets_url_and_passthrough(tmp_path):
    import json as _json
    p = tmp_path / "p.jsonl"
    p.write_text(_json.dumps({"text": "hi", "conversation_id": "c1"}) + "\n")
    shared = SharedOpts(rate="5", duration="0.3s", max_requests=None,
                        timeout_s=30.0, connection_limit=100, dotenv="",
                        quiet=True, out_dir=None, run_id=None, labels=(), notes="")
    built = get("sglang").build(
        shared, url="http://h:30000/generate", header=(), prompts=(),
        prompts_jsonl=str(p), prompt_field="text", full_jsonl_row=True,
        shuffle=False, seed=0, max_tokens=64, temperature=0.0, top_p=None,
        top_k=None, extras=())
    assert built.workload_type._url == "http://h:30000/generate"
    assert built.workload_type._passthrough_meta is True


def _generate_sse_server():
    async def _gen(request):
        resp = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i in range(2):
            payload = {"text": "x " * (i + 1),
                       "meta_info": {"prompt_tokens": 5, "completion_tokens": i + 1,
                                     "cached_tokens": 3, "finish_reason": None}}
            await resp.write(b"data: " + __import__("json").dumps(payload).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/generate", _gen)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    loop = asyncio.new_event_loop(); ready = threading.Event()

    def _serve():
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app); loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", port)
        loop.run_until_complete(site.start()); ready.set(); loop.run_forever()

    t = threading.Thread(target=_serve, daemon=True); t.start(); ready.wait(timeout=5)
    return f"http://127.0.0.1:{port}", loop, t


def test_sglang_recipe_runs():
    url, loop, t = _generate_sse_server()
    try:
        res = CliRunner().invoke(main, [
            "sglang", "--url", f"{url}/generate", "--prompt", "hi",
            "--rate", "5", "--duration", "0.3s", "--dotenv", "", "--quiet"])
        assert res.exit_code == 0, res.output
    finally:
        loop.call_soon_threadsafe(loop.stop); t.join(timeout=5)

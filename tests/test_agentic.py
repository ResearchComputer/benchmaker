"""Agentic workload: message sanitize, parse, per-turn prefix expansion."""
from __future__ import annotations

import json

from benchmaker.workloads.agentic import (
    AgenticWorkload,
    expand_trajectory,
    parse_messages,
    sanitize_message,
)


def test_sanitize_drops_nonstandard_keeps_tool_keys():
    msg = {"role": "assistant", "content": "ok", "agent": "main",
           "message_type": "action",
           "tool_calls": [{"id": "t1", "type": "function",
                           "function": {"name": "bash", "arguments": "{}"}}]}
    out = sanitize_message(msg)
    assert out == {"role": "assistant", "content": "ok",
                   "tool_calls": msg["tool_calls"]}
    tool = sanitize_message({"role": "tool", "content": "result",
                             "tool_call_id": "t1", "message_type": "observation"})
    assert tool == {"role": "tool", "content": "result", "tool_call_id": "t1"}


def test_parse_messages_from_json_string():
    raw = json.dumps([{"role": "system", "content": "s", "agent": "x"},
                      {"role": "user", "content": "u"}])
    msgs = parse_messages(raw)
    assert msgs == [{"role": "system", "content": "s"},
                    {"role": "user", "content": "u"}]


def test_expand_one_item_per_assistant_turn():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t", "tool_call_id": "1"},
        {"role": "assistant", "content": "a2"},
    ]
    items = expand_trajectory(messages, meta_base={"conversation_id": "c"},
                              max_tokens=99, max_turns=None, count_tokens=None)
    assert len(items) == 2
    # turn 0: prefix is [system, user]; turn 1: prefix adds assistant1 + tool
    assert [m["role"] for m in items[0]["messages"]] == ["system", "user"]
    assert [m["role"] for m in items[1]["messages"]] == [
        "system", "user", "assistant", "tool"]
    assert items[0]["max_tokens"] == 99
    assert items[0]["meta"]["turn_index"] == 0
    assert items[1]["meta"]["turn_index"] == 1
    assert items[1]["meta"]["prefix_messages"] == 4


def test_expected_prefix_tokens_is_previous_turn_count():
    messages = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    # stub counter: token count == number of messages in the prefix
    items = expand_trajectory(messages, meta_base={}, max_tokens=10,
                              max_turns=None, count_tokens=lambda msgs: len(msgs))
    assert items[0]["meta"]["expected_prefix_tokens"] == 0       # no previous
    assert items[1]["meta"]["expected_prefix_tokens"] == 1       # prev prefix had 1 msg


async def test_workload_over_jsonl_yields_then_stops(tmp_path):
    p = tmp_path / "traj.jsonl"
    rows = [
        {"instance_id": "i1", "model": "m",
         "messages": json.dumps([{"role": "user", "content": "u"},
                                 {"role": "assistant", "content": "a"}])},
        {"instance_id": "i2", "model": "m",
         "messages": json.dumps([{"role": "user", "content": "u"},
                                 {"role": "assistant", "content": "a1"},
                                 {"role": "user", "content": "u2"},
                                 {"role": "assistant", "content": "a2"}])},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    wl = AgenticWorkload(path=str(p), max_tokens=16)
    got = []
    try:
        while True:
            got.append(await wl.next_item())
    except StopAsyncIteration:
        pass
    assert len(got) == 3  # 1 turn + 2 turns
    assert got[0]["meta"]["instance_id"] == "i1"
    assert got[0]["meta"]["conversation_id"] == "i1"
    assert got[2]["meta"]["instance_id"] == "i2"


async def test_workload_caps(tmp_path):
    p = tmp_path / "traj.jsonl"
    rows = [{"instance_id": f"i{n}", "model": "m",
             "messages": json.dumps([{"role": "user", "content": "u"},
                                     {"role": "assistant", "content": "a1"},
                                     {"role": "user", "content": "u2"},
                                     {"role": "assistant", "content": "a2"}])}
            for n in range(5)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    wl = AgenticWorkload(path=str(p), max_trajectories=2,
                                  max_turns_per_trajectory=1)
    got = []
    try:
        while True:
            got.append(await wl.next_item())
    except StopAsyncIteration:
        pass
    assert len(got) == 2  # 2 trajectories x 1 turn cap


def test_sanitize_none_role_defaults_to_user():
    out = sanitize_message({"role": None, "content": "hi"})
    assert out["role"] == "user"
    assert out["content"] == "hi"


async def test_max_trajectories_skips_empty_trajectories(tmp_path):
    p = tmp_path / "traj.jsonl"
    rows = [
        {"instance_id": "a", "model": "m",
         "messages": json.dumps([{"role": "user", "content": "u"},
                                 {"role": "assistant", "content": "a"}])},
        # degenerate: no assistant turn -> produces no items, must not count
        {"instance_id": "empty", "model": "m",
         "messages": json.dumps([{"role": "user", "content": "u"},
                                 {"role": "user", "content": "u2"}])},
        {"instance_id": "b", "model": "m",
         "messages": json.dumps([{"role": "user", "content": "u"},
                                 {"role": "assistant", "content": "a"}])},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    wl = AgenticWorkload(path=str(p), max_trajectories=2)
    got = []
    try:
        while True:
            got.append(await wl.next_item())
    except StopAsyncIteration:
        pass
    # the empty trajectory is skipped; the 2 producing trajectories both replay
    assert [it["meta"]["instance_id"] for it in got] == ["a", "b"]


async def test_aclose_after_partial_iteration(tmp_path):
    p = tmp_path / "traj.jsonl"
    p.write_text(json.dumps({"instance_id": "a", "model": "m",
        "messages": json.dumps([{"role": "user", "content": "u"},
                                {"role": "assistant", "content": "a1"},
                                {"role": "user", "content": "u2"},
                                {"role": "assistant", "content": "a2"}])}) + "\n")
    wl = AgenticWorkload(path=str(p))
    first = await wl.next_item()
    assert first["meta"]["turn_index"] == 0
    await wl.aclose()                 # must not raise; releases the generator/file
    assert wl._gen is None


# --------------------------------------------------------------------------
# agentic recipe (CLI subcommand) tests.
# --------------------------------------------------------------------------

import asyncio
import glob
import os
import socket
import threading

from aiohttp import web
from click.testing import CliRunner

from benchmaker.cli import main
from benchmaker.recipes import all_recipes, get
from benchmaker.recipes.base import SharedOpts


def _shared():
    return SharedOpts(rate="closed:2", duration="8s", max_requests=None,
                      timeout_s=30.0, connection_limit=100, dotenv="", quiet=True,
                      out_dir=None, run_id=None, labels=(), notes="")


def test_agentic_recipe_registered():
    assert "agentic" in {r.name for r in all_recipes()}
    assert "agentic" in main.commands


def test_agentic_build_passthrough_and_workload(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({
        "instance_id": "i1", "model": "m",
        "messages": json.dumps([{"role": "user", "content": "u"},
                                {"role": "assistant", "content": "a"}])}) + "\n")
    built = get("agentic").build(
        _shared(), url="http://x/v1/chat/completions", model="target",
        api_key=None, header=(), dataset=None, prompts_jsonl=str(p),
        split="tool", preset=None, tokenizer=None, messages_field="messages",
        id_field="instance_id", model_field="model", max_tokens=256,
        max_turns_per_trajectory=None, max_trajectories=None)
    from benchmaker.workloads.agentic import AgenticWorkload
    assert built.workload_type._passthrough_meta is True
    assert isinstance(built.workload, AgenticWorkload)


def test_swe_smith_preset_sets_dataset_defaults():
    built = get("agentic").build(
        _shared(), url="http://x/v1/chat/completions", model="target",
        api_key=None, header=(), dataset=None, prompts_jsonl=None,
        split="tool", preset="swe-smith", tokenizer=None,
        messages_field="messages", id_field="instance_id", model_field="model",
        max_tokens=256, max_turns_per_trajectory=None, max_trajectories=1)
    assert built.workload._dataset == "SWE-bench/SWE-smith-trajectories"
    assert built.workload._split == "tool"


def _chat_sse_server():
    async def _sse(request):
        resp = web.StreamResponse(status=200,
                                  headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i in range(2):
            chunk = {"choices": [{"index": 0, "delta": {"content": f"t{i} "},
                                  "finish_reason": None}]}
            await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        await resp.write(b"data: " + json.dumps({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2,
                      "prompt_tokens_details": {"cached_tokens": 4}}}).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat/completions", _sse)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    loop = asyncio.new_event_loop(); ready = threading.Event()

    def _serve():
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app); loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", port)
        loop.run_until_complete(site.start()); ready.set(); loop.run_forever()

    t = threading.Thread(target=_serve, daemon=True); t.start(); ready.wait(timeout=5)
    return f"http://127.0.0.1:{port}", loop, t


def test_agentic_e2e_records_metadata(tmp_path):
    url, loop, t = _chat_sse_server()
    try:
        traj = tmp_path / "t.jsonl"
        rows = [
            {"instance_id": "i1", "model": "m",
             "messages": json.dumps([{"role": "user", "content": "u"},
                                     {"role": "assistant", "content": "a"}])},
            {"instance_id": "i2", "model": "m",
             "messages": json.dumps([{"role": "user", "content": "u"},
                                     {"role": "assistant", "content": "a1"},
                                     {"role": "user", "content": "u2"},
                                     {"role": "assistant", "content": "a2"}])},
        ]
        traj.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        out = tmp_path / "runs"
        res = CliRunner().invoke(main, [
            "agentic", "--url", f"{url}/v1/chat/completions",
            "--model", "target", "--prompts-jsonl", str(traj),
            "--rate", "closed:2", "--duration", "8s",
            "--out-dir", str(out), "--dotenv", "", "--quiet"])
        assert res.exit_code == 0, res.output
        samples = glob.glob(str(out / "**" / "samples.jsonl"), recursive=True)
        assert samples
        recs = [json.loads(l) for l in open(samples[0]) if l.strip()]
        # 1 + 2 = 3 turns total; all carry conversation_id + cached_tokens.
        assert len(recs) == 3
        assert {r["meta"]["conversation_id"] for r in recs} == {"i1", "i2"}
        assert all(r["extra"].get("cached_tokens") == 4.0 for r in recs)
    finally:
        loop.call_soon_threadsafe(loop.stop); t.join(timeout=5)


def test_agentic_interleaved_e2e_completes_all_turns(tmp_path):
    # Interleaved mode drives turns through the runner: the completion post-hook
    # gates each session's next turn, so the run must still emit every turn and
    # terminate on exhaustion (no hang, no dropped turns).
    url, loop, t = _chat_sse_server()
    try:
        traj = tmp_path / "t.jsonl"
        rows = [
            {"instance_id": "i1", "model": "m",
             "messages": json.dumps([{"role": "user", "content": "u"},
                                     {"role": "assistant", "content": "a"}])},
            {"instance_id": "i2", "model": "m",
             "messages": json.dumps([{"role": "user", "content": "u"},
                                     {"role": "assistant", "content": "a1"},
                                     {"role": "user", "content": "u2"},
                                     {"role": "assistant", "content": "a2"}])},
        ]
        traj.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        out = tmp_path / "runs"
        res = CliRunner().invoke(main, [
            "agentic", "--url", f"{url}/v1/chat/completions",
            "--model", "target", "--prompts-jsonl", str(traj),
            "--concurrent-sessions", "2", "--duration", "8s",
            "--out-dir", str(out), "--dotenv", "", "--quiet"])
        assert res.exit_code == 0, res.output
        samples = glob.glob(str(out / "**" / "samples.jsonl"), recursive=True)
        assert samples
        recs = [json.loads(l) for l in open(samples[0]) if l.strip()]
        assert len(recs) == 3
        assert {r["meta"]["conversation_id"] for r in recs} == {"i1", "i2"}
        # i2's two turns stayed causally ordered (turn 0 before turn 1).
        i2_turns = [r["meta"]["turn_index"] for r in recs
                    if r["meta"]["conversation_id"] == "i2"]
        assert i2_turns == sorted(i2_turns)
        # The run bundle records the interleave settings for reproducibility.
        meta = glob.glob(str(out / "**" / "meta.json"), recursive=True)
        assert meta
        run_cfg = json.load(open(meta[0]))["source_config"]
        assert run_cfg["workload"]["concurrent_sessions"] == 2
    finally:
        loop.call_soon_threadsafe(loop.stop); t.join(timeout=5)


def test_agentic_source_config_records_field_names(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({
        "iid": "i1", "mdl": "m",
        "msgs": json.dumps([{"role": "user", "content": "u"},
                            {"role": "assistant", "content": "a"}])}) + "\n")
    built = get("agentic").build(
        _shared(), url="http://x/v1/chat/completions", model="target",
        api_key=None, header=(), dataset=None, prompts_jsonl=str(p),
        split="tool", preset=None, tokenizer=None, messages_field="msgs",
        id_field="iid", model_field="mdl", max_tokens=256,
        max_turns_per_trajectory=None, max_trajectories=None)
    wl_cfg = built.source_config["workload"]
    assert wl_cfg["messages_field"] == "msgs"
    assert wl_cfg["id_field"] == "iid"
    assert wl_cfg["model_field"] == "mdl"
    assert built.source_config["workload_type"]["max_tokens"] == 256

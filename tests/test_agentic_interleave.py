"""Interleaved / concurrent-session scheduling for agentic replay.

Covers the inter-turn gap distribution parser and the round-robin session
scheduler that gates each session's turn k+1 on turn k's completion (+ gap).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from benchmaker.workloads.agentic import (
    AgenticWorkload,
    parse_gap_spec,
)


# --------------------------------------------------------------------------
# parse_gap_spec — inter-turn think-time distribution
# --------------------------------------------------------------------------

def test_gap_none_and_zero_are_no_delay():
    assert parse_gap_spec(None)() == 0.0
    assert parse_gap_spec("")() == 0.0
    assert parse_gap_spec("0")() == 0.0
    assert parse_gap_spec("none")() == 0.0


def test_gap_bare_duration_is_constant():
    assert parse_gap_spec("2s")() == 2.0
    assert parse_gap_spec("500ms")() == 0.5


def test_gap_const_prefix():
    gap = parse_gap_spec("const:1500ms")
    assert gap() == 1.5
    assert gap() == 1.5  # constant: same every draw


def test_gap_exp_has_positive_mean_and_is_seeded():
    g1 = parse_gap_spec("exp:2.0", seed=0)
    g2 = parse_gap_spec("exp:2.0", seed=0)
    draws1 = [g1() for _ in range(200)]
    draws2 = [g2() for _ in range(200)]
    assert draws1 == draws2                      # same seed -> reproducible
    assert all(d >= 0.0 for d in draws1)
    assert len(set(draws1)) > 1                  # actually random, not constant
    mean = sum(draws1) / len(draws1)
    assert 1.0 < mean < 3.0                      # mean ~2.0


def test_gap_uniform_within_bounds_and_seeded():
    g = parse_gap_spec("uniform:1s..3s", seed=1)
    draws = [g() for _ in range(200)]
    assert all(1.0 <= d <= 3.0 for d in draws)
    assert len(set(draws)) > 1
    assert parse_gap_spec("uniform:1s..3s", seed=1)() == draws[0]  # reproducible


def test_gap_invalid_spec_raises():
    with pytest.raises(ValueError):
        parse_gap_spec("wat:5")


# --------------------------------------------------------------------------
# Interleaved session scheduler
# --------------------------------------------------------------------------

def _traj(instance_id: str, n_turns: int) -> dict:
    msgs = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return {"instance_id": instance_id, "model": "m", "messages": json.dumps(msgs)}


def _write_trajs(tmp_path, specs) -> str:
    p = tmp_path / "traj.jsonl"
    p.write_text("\n".join(json.dumps(_traj(iid, n)) for iid, n in specs) + "\n")
    return str(p)


def _key(item) -> tuple:
    return (item["meta"]["conversation_id"], item["meta"]["turn_index"])


async def test_interleave_round_robins_turns_across_sessions(tmp_path):
    # Two sessions, two turns each. Turn k of every session goes out before any
    # session's turn k+1; each session's turn k+1 waits for turn k to complete.
    path = _write_trajs(tmp_path, [("i1", 2), ("i2", 2)])
    wl = AgenticWorkload(path=path, concurrent_sessions=2)

    a = await wl.next_item()                    # i1 turn0
    b = await wl.next_item()                     # i2 turn0 (i1 in flight)
    wl.notify_turn_complete("i1")
    c = await wl.next_item()                     # i1 turn1
    wl.notify_turn_complete("i2")
    d = await wl.next_item()                     # i2 turn1
    wl.notify_turn_complete("i1")
    wl.notify_turn_complete("i2")
    with pytest.raises(StopAsyncIteration):
        await wl.next_item()

    assert [_key(x) for x in (a, b, c, d)] == [
        ("i1", 0), ("i2", 0), ("i1", 1), ("i2", 1)]


async def test_interleave_gates_next_turn_until_prior_completes(tmp_path):
    # One session, concurrency 1: turn1 must not be emitted until turn0 completes.
    path = _write_trajs(tmp_path, [("i1", 2)])
    wl = AgenticWorkload(path=path, concurrent_sessions=1)

    first = await wl.next_item()
    assert _key(first) == ("i1", 0)

    # turn0 still in flight -> next_item has nothing to hand out and must block.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wl.next_item(), timeout=0.1)

    wl.notify_turn_complete("i1")
    second = await wl.next_item()
    assert _key(second) == ("i1", 1)

    wl.notify_turn_complete("i1")
    with pytest.raises(StopAsyncIteration):
        await wl.next_item()


async def test_interleave_inter_turn_gap_defers_next_turn(tmp_path):
    # With a long gap, the session is not eligible immediately after completion.
    path = _write_trajs(tmp_path, [("i1", 2)])
    wl = AgenticWorkload(
        path=path, concurrent_sessions=1, inter_turn_gap="const:30s")

    await wl.next_item()                         # i1 turn0
    wl.notify_turn_complete("i1")                # eligible only after +30s
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wl.next_item(), timeout=0.1)


async def test_completion_post_hook_extracts_conversation_id(tmp_path):
    # The wired post-hook reads conversation_id off Request.meta and advances
    # that session — this is the real runner integration point.
    from benchmaker.core.types import Request, Response, Sample

    path = _write_trajs(tmp_path, [("i1", 2)])
    wl = AgenticWorkload(path=path, concurrent_sessions=1)

    item = await wl.next_item()
    req = Request(meta={"conversation_id": "i1", "turn_index": 0})
    resp = Response(status=200, headers={}, body=b"", elapsed_s=0.1, ok=True)
    sample = Sample(start_ts=0.0, latency_s=0.1, status=200, ok=True)
    out = wl.note_complete_hook(req, resp, sample)
    assert out is sample                          # hook is pass-through

    nxt = await wl.next_item()
    assert _key(nxt) == ("i1", 1)


async def test_interleave_respects_max_turns_and_skips_empty(tmp_path):
    p = tmp_path / "traj.jsonl"
    rows = [
        _traj("i1", 3),
        {"instance_id": "empty", "model": "m",                 # no assistant turn
         "messages": json.dumps([{"role": "user", "content": "u"}])},
        _traj("i2", 3),
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    wl = AgenticWorkload(
        path=str(p), concurrent_sessions=4, max_turns_per_trajectory=1,
        max_trajectories=2)

    got = []
    # Drain: complete each turn as soon as it is handed out.
    while True:
        try:
            item = await asyncio.wait_for(wl.next_item(), timeout=1.0)
        except StopAsyncIteration:
            break
        got.append(_key(item))
        wl.notify_turn_complete(item["meta"]["conversation_id"])

    # max_trajectories=2 keeps i1 + i2 (empty skipped, not counted);
    # max_turns_per_trajectory=1 keeps only turn0 of each.
    assert sorted(got) == [("i1", 0), ("i2", 0)]


# --------------------------------------------------------------------------
# agentic recipe wiring
# --------------------------------------------------------------------------

from benchmaker.recipes import get
from benchmaker.recipes.base import SharedOpts


def _shared():
    return SharedOpts(rate="10", duration="10s", max_requests=None,
                      timeout_s=600.0, connection_limit=100, dotenv="", quiet=True,
                      out_dir=None, run_id=None, labels=(), notes="")


def _build(tmp_path, **overrides):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(_traj("i1", 2)) + "\n")
    params = dict(
        url="http://x/v1/chat/completions", model="target", api_key=None,
        header=(), dataset=None, prompts_jsonl=str(p), split="tool", preset=None,
        tokenizer=None, messages_field="messages", id_field="instance_id",
        model_field="model", max_tokens=256, max_turns_per_trajectory=None,
        max_trajectories=None, concurrent_sessions=None, inter_turn_gap=None)
    params.update(overrides)
    return get("agentic").build(_shared(), **params)


def test_recipe_interleave_installs_hook_and_defaults_rate(tmp_path):
    built = _build(tmp_path, concurrent_sessions=4, inter_turn_gap="const:2s")
    assert built.workload._interleave is True
    assert built.workload._concurrency == 4
    # The completion post-hook is wired so the scheduler is notified per turn.
    assert built.post_hooks == [built.workload.note_complete_hook]
    # In-flight ceiling matches the active session count by default.
    assert built.default_rate == "closed:4"


def test_recipe_contiguous_mode_has_no_hook(tmp_path):
    built = _build(tmp_path)  # concurrent_sessions=None
    assert built.workload._interleave is False
    assert built.post_hooks == []
    assert built.default_rate == "closed:8"


def test_recipe_source_config_records_interleave_fields(tmp_path):
    built = _build(tmp_path, concurrent_sessions=8, inter_turn_gap="exp:1.5")
    wl_cfg = built.source_config["workload"]
    assert wl_cfg["concurrent_sessions"] == 8
    assert wl_cfg["inter_turn_gap"] == "exp:1.5"


# --------------------------------------------------------------------------
# completion_hook() protocol + YAML config auto-wiring
# --------------------------------------------------------------------------

def test_completion_hook_none_in_contiguous_mode(tmp_path):
    path = _write_trajs(tmp_path, [("i1", 2)])
    wl = AgenticWorkload(path=path)               # contiguous
    assert wl.completion_hook() is None


def test_completion_hook_present_in_interleaved_mode(tmp_path):
    path = _write_trajs(tmp_path, [("i1", 2)])
    wl = AgenticWorkload(path=path, concurrent_sessions=2)
    assert wl.completion_hook() == wl.note_complete_hook


def test_build_config_autowires_completion_hook(tmp_path):
    # A YAML config that turns on interleaving must not hang for lack of the
    # completion signal: build_config auto-installs the workload's hook.
    from benchmaker.config import build_config

    path = _write_trajs(tmp_path, [("i1", 2)])
    cfg = {
        "workload_type": {"type": "openai-chat",
                          "url": "http://x/v1/chat/completions", "model": "m"},
        "workload": {"type": "agentic", "path": path,
                     "concurrent_sessions": 2},
        "load": "closed:2",
    }
    bc = build_config(cfg, dotenv_path=None)
    assert bc.workload.completion_hook() in bc.post_hooks


def test_build_config_no_hook_for_contiguous_agentic(tmp_path):
    from benchmaker.config import build_config

    path = _write_trajs(tmp_path, [("i1", 2)])
    cfg = {
        "workload_type": {"type": "openai-chat",
                          "url": "http://x/v1/chat/completions", "model": "m"},
        "workload": {"type": "agentic", "path": path},
        "load": "closed:2",
    }
    bc = build_config(cfg, dotenv_path=None)
    assert bc.post_hooks == []

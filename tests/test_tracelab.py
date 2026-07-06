"""TraceLab coding-agent trace workload + preparation tooling tests."""

from __future__ import annotations

import gzip
import json
import os
import zipfile
from pathlib import Path

import pytest

from benchmaker import OpenAIChatWorkloadType, Response
from benchmaker.config import build_workload
from benchmaker.workloads.tracelab import TraceLabWorkload
from tools.tracelab.prepare import subset as prepare_subset


def _row(session: str, rnd: int, *, provider="claude", model="claude-opus-4-8",
         input_total=1000, prefix=800, newly=200, output=64, tools=0) -> dict:
    return {
        "provider": provider,
        "model": model,
        "session_id": session,
        "round_index": rnd,
        "round_id": f"{session}-r{rnd}",
        "trace_key": f"{session}:r{rnd}",
        "input_tokens_total": input_total,
        "prefix_tokens": prefix,
        "newly_append_tokens": newly,
        "output_tokens": output,
        "tools": [{"tool_name": "Bash"}] * tools,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


# ----------------------------------------------------------------- flat mode


@pytest.mark.asyncio
async def test_flat_mode_emits_one_item_per_row_with_token_faithful_size(tmp_path):
    path = _write_jsonl(tmp_path / "t.jsonl", [_row("s", 0), _row("s", 1)])
    wl = TraceLabWorkload(path, prefix_cache=False, shuffle=False, loop=False,
                          chars_per_token=4.0)

    item = await wl.next_item()

    assert item["max_tokens"] == 64
    assert item["messages"] == [{"role": "user", "content": item["messages"][0]["content"]}]
    # 1000 target tokens * 4 chars/token -> 4000 chars.
    assert len(item["messages"][0]["content"]) == 4000
    assert item["meta"]["prompt_tokens_hint"] == 1000
    assert "prefix_tokens_hint" not in item["meta"]  # flat mode: no cacheable prefix
    assert item["meta"]["tracelab_provider"] == "claude"
    assert item["meta"]["tracelab_prefix_tokens"] == 800
    assert item["meta"]["tracelab_newly_append_tokens"] == 200
    assert item["meta"]["tracelab_output_tokens_target"] == 64
    assert item["meta"]["tracelab_tool_count"] == 0


@pytest.mark.asyncio
async def test_flat_mode_independent_prompts_differ_per_row(tmp_path):
    path = _write_jsonl(tmp_path / "t.jsonl", [_row("s", 0), _row("s", 1)])
    wl = TraceLabWorkload(path, prefix_cache=False, shuffle=False, loop=False)

    a = await wl.next_item()
    b = await wl.next_item()
    # Different trace_keys -> different filler seeds -> different text.
    assert a["messages"][0]["content"] != b["messages"][0]["content"]


# ---------------------------------------------------------- prefix-cache mode


@pytest.mark.asyncio
async def test_prefix_cache_mode_replays_one_request_per_round(tmp_path):
    rows = [_row("sess", i) for i in range(3)]
    path = _write_jsonl(tmp_path / "t.jsonl", rows)
    wl = TraceLabWorkload(path, prefix_cache=True, shuffle=False, loop=False)

    assert len(wl._plan) == 3  # one record per round, not one per session

    items = [await wl.next_item() for _ in range(3)]
    prompts = [it["messages"][0]["content"] for it in items]
    # Round i's prompt is a byte-exact prefix of round i+1's.
    assert prompts[0].startswith(prompts[0])
    assert prompts[1].startswith(prompts[0])
    assert prompts[2].startswith(prompts[1])
    assert len(prompts[0]) < len(prompts[1]) < len(prompts[2])
    # prefix_tokens_hint tracks the recorded cacheable prefix per round.
    assert [it["meta"]["prefix_tokens_hint"] for it in items] == [800, 800, 800]
    # Each item carries the session's round count.
    assert all(it["meta"]["tracelab_session_rounds"] == 3 for it in items)


@pytest.mark.asyncio
async def test_prefix_cache_mode_seeds_base_prefix_from_first_round(tmp_path):
    # First round already carries a large recorded prefix (system prompt + ctx).
    rows = [_row("sess", 0, prefix=5000, newly=200, input_total=5200),
            _row("sess", 1, prefix=5200, newly=300, input_total=5500)]
    path = _write_jsonl(tmp_path / "t.jsonl", rows)
    wl = TraceLabWorkload(path, prefix_cache=True, shuffle=False, loop=False)

    items = [await wl.next_item() for _ in range(2)]
    # Round 0 prompt size = base_prefix(5000) + append(200) = 5200 tokens.
    # Round 1 prompt size = 5200 + 300 = 5500 tokens. Both at 4 chars/token.
    assert len(items[0]["messages"][0]["content"]) == 5200 * 4
    assert len(items[1]["messages"][0]["content"]) == 5500 * 4
    assert items[1]["messages"][0]["content"].startswith(items[0]["messages"][0]["content"])


@pytest.mark.asyncio
async def test_prefix_cache_mode_interleaves_sessions_round_robin(tmp_path):
    # Two sessions, two rounds each. Round-robin keeps prefix locality strong
    # while mixing sessions: order is a0, b0, a1, b1.
    rows = [_row("a", 0), _row("b", 0), _row("a", 1), _row("b", 1)]
    path = _write_jsonl(tmp_path / "t.jsonl", rows)
    wl = TraceLabWorkload(path, prefix_cache=True, shuffle=False, loop=False)

    sessions = [(await wl.next_item())["meta"]["tracelab_session_id"] for _ in range(4)]
    assert sessions == ["a", "b", "a", "b"]


@pytest.mark.asyncio
async def test_prefix_cache_shuffle_keeps_rounds_ordered_within_session(tmp_path):
    rows = [_row("a", 0), _row("a", 1), _row("a", 2),
            _row("b", 0), _row("b", 1), _row("b", 2)]
    path = _write_jsonl(tmp_path / "t.jsonl", rows)
    wl = TraceLabWorkload(path, prefix_cache=True, shuffle=True, seed=1, loop=False)

    items = [await wl.next_item() for _ in range(6)]
    # Within each session the round_index must be non-decreasing.
    for sess in ("a", "b"):
        idxs = [it["meta"]["tracelab_round_index"]
                for it in items if it["meta"]["tracelab_session_id"] == sess]
        assert idxs == sorted(idxs)


# ------------------------------------------------------------------- filters


@pytest.mark.asyncio
async def test_provider_filter(tmp_path):
    rows = [_row("s", 0, provider="claude"), _row("s", 1, provider="codex")]
    path = _write_jsonl(tmp_path / "t.jsonl", rows)
    wl = TraceLabWorkload(path, provider="codex", loop=False)
    assert len(wl._plan) == 1


@pytest.mark.asyncio
async def test_model_filter(tmp_path):
    rows = [_row("s", 0, model="claude-opus-4-8"),
            _row("s", 1, model="gpt-5")]
    path = _write_jsonl(tmp_path / "t.jsonl", rows)
    wl = TraceLabWorkload(path, model_filter="gpt-5", loop=False)
    assert len(wl._plan) == 1


@pytest.mark.asyncio
async def test_token_range_filters(tmp_path):
    rows = [
        _row("s", 0, input_total=100, output=5),
        _row("s", 1, input_total=1000, output=64),
        _row("s", 2, input_total=50000, output=2000),
    ]
    path = _write_jsonl(tmp_path / "t.jsonl", rows)
    wl = TraceLabWorkload(path, min_input_tokens=500, max_input_tokens=10000,
                          min_output_tokens=10, max_output_tokens=500, loop=False)
    assert len(wl._plan) == 1


@pytest.mark.asyncio
async def test_max_items_caps_filtered_rows(tmp_path):
    rows = [_row("s", i) for i in range(10)]
    path = _write_jsonl(tmp_path / "t.jsonl", rows)
    wl = TraceLabWorkload(path, max_items=3, loop=False)
    assert len(wl._plan) == 3


@pytest.mark.asyncio
async def test_max_sessions_caps_sessions_in_prefix_cache_mode(tmp_path):
    rows = []
    for s in ("a", "b", "c", "d"):
        rows += [_row(s, 0), _row(s, 1)]
    path = _write_jsonl(tmp_path / "t.jsonl", rows)
    wl = TraceLabWorkload(path, prefix_cache=True, max_sessions=2,
                          shuffle=False, loop=False)
    # 2 sessions * 2 rounds = 4 plan records.
    assert len(wl._plan) == 4


# ------------------------------------------------------------- decode control


@pytest.mark.asyncio
async def test_match_output_tokens_sets_min_tokens_and_ignore_eos(tmp_path):
    path = _write_jsonl(tmp_path / "t.jsonl", [_row("s", 0, output=77)])
    wl = TraceLabWorkload(path, match_output_tokens=True, loop=False)
    item = await wl.next_item()
    assert item["max_tokens"] == 77
    assert item["min_tokens"] == 77
    assert item["ignore_eos"] is True


@pytest.mark.asyncio
async def test_max_tokens_cap_clamps(tmp_path):
    path = _write_jsonl(tmp_path / "t.jsonl", [_row("s", 0, output=4096)])
    wl = TraceLabWorkload(path, max_tokens_cap=512, match_output_tokens=True,
                          loop=False)
    item = await wl.next_item()
    assert item["max_tokens"] == 512
    assert item["min_tokens"] == 512


@pytest.mark.asyncio
async def test_missing_output_tokens_uses_default(tmp_path):
    row = _row("s", 0)
    del row["output_tokens"]
    path = _write_jsonl(tmp_path / "t.jsonl", [row])
    wl = TraceLabWorkload(path, default_output_tokens=99, loop=False)
    item = await wl.next_item()
    assert item["max_tokens"] == 99


# -------------------------------------------------------------------- file io


@pytest.mark.asyncio
async def test_gz_input_is_supported(tmp_path):
    p = tmp_path / "t.jsonl.gz"
    with gzip.open(p, "wt") as f:
        f.write(json.dumps(_row("s", 0)) + "\n")
    wl = TraceLabWorkload(str(p), loop=False)
    item = await wl.next_item()
    assert item["meta"]["prompt_tokens_hint"] == 1000


@pytest.mark.asyncio
async def test_zip_input_is_supported(tmp_path):
    p = tmp_path / "t.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("trace.jsonl", json.dumps(_row("s", 0)) + "\n")
    wl = TraceLabWorkload(str(p), loop=False)
    item = await wl.next_item()
    assert item["meta"]["prompt_tokens_hint"] == 1000


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        TraceLabWorkload("/nonexistent/trace.jsonl")


# ---------------------------------------------------------------------- loop


@pytest.mark.asyncio
async def test_loops_by_default_and_stops_when_disabled(tmp_path):
    path = _write_jsonl(tmp_path / "t.jsonl", [_row("s", 0)])
    looping = TraceLabWorkload(path, loop=True)
    a = await looping.next_item()
    b = await looping.next_item()  # restarts
    assert a["meta"]["tracelab_session_id"] == b["meta"]["tracelab_session_id"]

    finite = TraceLabWorkload(path, loop=False)
    await finite.next_item()
    with pytest.raises(StopAsyncIteration):
        await finite.next_item()


# ----------------------------------------------- integration w/ workload-type


@pytest.mark.asyncio
async def test_metadata_is_not_sent_and_hints_are_promoted_to_extra(tmp_path):
    path = _write_jsonl(tmp_path / "t.jsonl", [_row("s", 0)])
    wl = TraceLabWorkload(path, prefix_cache=True, loop=False)
    item = await wl.next_item()
    wt = OpenAIChatWorkloadType(
        url="http://example/v1/chat/completions", model="test",
        passthrough_meta=True,
    )

    request = await wt.make_request(item)
    body = request.json
    # Body must be clean of provenance.
    for forbidden in ("tracelab_provider", "tracelab_session_id",
                      "prompt_tokens_hint", "prefix_tokens_hint", "reference"):
        assert forbidden not in body
    assert "messages" in body and body["max_tokens"] == 64
    assert request.meta["prompt_tokens_hint"] == 1000
    assert request.meta["prefix_tokens_hint"] == 800
    assert request.meta["tracelab_provider"] == "claude"

    chunks = [
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":1012,"completion_tokens":64,'
        b'"prompt_tokens_details":{"cached_tokens":820}}}\n\n',
        b"data: [DONE]\n\n",
    ]
    response = Response(
        status=200, headers={}, body=b"".join(chunks), elapsed_s=0.2, ok=True,
        stream_chunks=chunks, stream_chunk_times=[0.05, 0.06],
    )
    sample = await wt.make_sample(item, request, response, 0.0)
    # Target vs realized, all promoted to Sample.extra for aggregation.
    assert sample.extra["prompt_tokens_hint"] == 1000.0   # target input
    assert sample.extra["prefix_tokens_hint"] == 800.0     # target cacheable
    assert sample.extra["prompt_tokens"] == 1012.0         # realized input
    assert sample.extra["cached_tokens"] == 820.0          # realized cache hit
    assert sample.extra["tokens_out"] == 64.0


# --------------------------------------------------------------- YAML build


def test_yaml_build_workload(tmp_path):
    path = _write_jsonl(tmp_path / "t.jsonl", [_row("s", 0)])
    wl = build_workload({"type": "tracelab", "path": path,
                         "prefix_cache": True, "loop": False})
    assert isinstance(wl, TraceLabWorkload)
    assert wl._prefix_cache is True


# --------------------------------------------------------------- prepare tool


def test_prepare_subset_filters_and_caps(tmp_path):
    src = tmp_path / "src.jsonl"
    rows = [
        _row("s", 0, provider="claude", input_total=100, output=5),
        _row("s", 1, provider="codex", input_total=5000, output=64),
        _row("s", 2, provider="claude", input_total=2000, output=128),
        _row("s", 3, provider="claude", input_total=3000, output=2000),
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "out.jsonl"

    written, skipped = prepare_subset(
        str(src), str(out), provider="claude",
        model_filter=None, min_input=500, max_input=10000,
        min_output=10, max_output=500, max_items=None,
    )
    assert written == 1  # only row 2 (claude, 2000 in, 128 out)
    assert skipped == 3
    kept = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(kept) == 1
    assert kept[0]["provider"] == "claude"
    assert kept[0]["input_tokens_total"] == 2000


def test_prepare_subset_max_items_caps(tmp_path):
    src = tmp_path / "src.jsonl"
    rows = [_row("s", i) for i in range(5)]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "out.jsonl"
    written, _ = prepare_subset(
        str(src), str(out), provider=None, model_filter=None,
        min_input=None, max_input=None, min_output=None, max_output=None,
        max_items=2,
    )
    assert written == 2

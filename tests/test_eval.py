"""Tests for benchmaker.workloads.eval: EvalWorkloadType + correctness_hook + scorers.

The SSE stub server in conftest emits five token deltas — `"tok0 tok1 tok2 tok3 tok4 "`
— so we use that string as the canonical "prediction" for end-to-end tests.
"""

import asyncio
import json

import pytest

from benchmaker import (
    BenchConfig,
    BenchRunner,
    ConstantRPS,
    EvalWorkloadType,
    OpenAIChatWorkloadType,
    Response,
    StaticWorkload,
    contains,
    correctness_hook,
    exact_match,
    extract_openai_text,
    extract_text,
    json_valid,
    judge_llm,
    multiple_choice,
    regex_match,
)


# ----------------------- pure scorer unit tests ----------------------- #


def test_exact_match_basic():
    f = exact_match()
    assert f("hello", "hello") == {"correct": 1.0}
    assert f("hello", "  hello  ") == {"correct": 1.0}      # strip default
    assert f("HELLO", "hello") == {"correct": 0.0}          # case-sensitive default
    assert exact_match(case_insensitive=True)("HELLO", "hello") == {"correct": 1.0}


def test_contains_basic():
    f = contains()
    assert f("tok2", "tok0 tok1 tok2 tok3")["correct"] == 1.0
    assert f("Tok2", "tok0 tok1 tok2 tok3")["correct"] == 1.0   # ci default
    assert f("zzz", "abc")["correct"] == 0.0


def test_regex_match_capture():
    f = regex_match(r"answer:\s*(\d+)", group=1, case_insensitive=True)
    assert f("42", "Answer: 42")["correct"] == 1.0
    assert f("7", "Answer: 42")["correct"] == 0.0
    out = f("42", "no answer here")
    assert out == {"correct": 0.0, "matched": 0.0}


def test_regex_match_no_reference_means_match_only():
    f = regex_match(r"^OK$")
    assert f(None, "OK")["correct"] == 1.0
    assert f(None, "NOPE")["correct"] == 0.0


def test_json_valid_basic():
    f = json_valid()
    assert f(None, '{"a": 1}') == {"valid_json": 1.0, "correct": 1.0}
    assert f(None, "not json") == {"valid_json": 0.0, "correct": 0.0}


def test_json_valid_required_keys():
    f = json_valid(required_keys=("a", "b"))
    assert f(None, '{"a": 1, "b": 2}')["correct"] == 1.0
    assert f(None, '{"a": 1}')["correct"] == 0.0
    assert f(None, '{"a": 1}')["valid_json"] == 1.0


def test_multiple_choice_basic():
    f = multiple_choice()
    assert f("B", "The answer is B.")["correct"] == 1.0
    assert f("B", "I think A is correct")["correct"] == 0.0
    assert f("B", "no letter here") == {"correct": 0.0, "answered": 0.0}


# ----------------------- extract_openai_text ----------------------- #


def test_extract_openai_text_from_streaming_chunks():
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hi "}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"there"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    resp = Response(status=200, headers={}, body=b"".join(chunks),
                    elapsed_s=0.01, ok=True, stream_chunks=chunks,
                    stream_chunk_times=[0.0, 0.001, 0.002])
    assert extract_openai_text(resp) == "hi there"


def test_extract_openai_text_from_non_streaming_json():
    body = json.dumps({"choices": [{"message": {"content": "yo"}}]}).encode()
    resp = Response(status=200, headers={}, body=body, elapsed_s=0.01, ok=True)
    assert extract_openai_text(resp) == "yo"


def test_extract_text_falls_back_to_raw_body():
    resp = Response(status=200, headers={}, body=b"plain text",
                    elapsed_s=0.01, ok=True)
    assert extract_text(resp) == "plain text"


# ----------------------- EvalWorkloadType ----------------------- #


@pytest.mark.asyncio
async def test_eval_workload_type_strips_reference_into_meta():
    base = OpenAIChatWorkloadType(url="http://x/v1/chat/completions", model="m")
    wt = EvalWorkloadType(base, reference_key="expected",
                          extra_meta_keys=("qid",))
    req = await wt.make_request({
        "prompt": "What is 2+2?",
        "expected": "4",
        "qid": "q42",
    })
    # Eval fields lifted to meta:
    assert req.meta["expected"] == "4"
    assert req.meta["qid"] == "q42"
    # Eval fields NOT in the body sent to the LLM:
    assert "expected" not in req.json
    assert "qid" not in req.json
    # The prompt did make it through:
    assert req.json["messages"][0]["content"] == "What is 2+2?"


@pytest.mark.asyncio
async def test_eval_workload_type_passes_through_non_dict_items():
    base = OpenAIChatWorkloadType(url="http://x/v1/chat/completions", model="m")
    wt = EvalWorkloadType(base)
    req = await wt.make_request("just a string")
    assert req.json["messages"][0]["content"] == "just a string"
    assert "reference" not in req.meta


# ----------------------- end-to-end with stub server ----------------------- #


@pytest.mark.asyncio
async def test_correctness_hook_exact_match_pass(stub_server: str):
    """The SSE stub emits 'tok0 tok1 tok2 tok3 tok4 '. exact_match should pass."""
    base = OpenAIChatWorkloadType(
        url=f"{stub_server}/v1/chat/completions", model="stub", max_tokens=8,
    )
    wt = EvalWorkloadType(base)
    workload = StaticWorkload(items=[
        {"prompt": "hi", "reference": "tok0 tok1 tok2 tok3 tok4"},
    ])
    hook = correctness_hook(exact_match())  # strip=True by default
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=5, duration_s=0.4),
        post_hooks=[hook], progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["success"] == result.summary["total_requests"]
    wm = result.summary["workload_metrics"]
    assert wm["correct"]["mean"] == 1.0


@pytest.mark.asyncio
async def test_correctness_hook_gate_marks_sample_failed(stub_server: str):
    """A wrong reference should drop ok→False via the default `correct` gate."""
    base = OpenAIChatWorkloadType(
        url=f"{stub_server}/v1/chat/completions", model="stub", max_tokens=8,
    )
    wt = EvalWorkloadType(base)
    workload = StaticWorkload(items=[
        {"prompt": "hi", "reference": "definitely not what the stub returns"},
    ])
    hook = correctness_hook(exact_match())
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=5, duration_s=0.4),
        post_hooks=[hook], progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["total_requests"] > 0
    assert result.summary["success"] == 0
    assert result.summary["failed"] == result.summary["total_requests"]
    assert all(s.error == "failed-correct" for s in result.samples)


@pytest.mark.asyncio
async def test_correctness_hook_regex_capture(stub_server: str):
    """regex_match should pick the digit out of 'tokN' and compare to reference."""
    base = OpenAIChatWorkloadType(
        url=f"{stub_server}/v1/chat/completions", model="stub", max_tokens=8,
    )
    wt = EvalWorkloadType(base)
    workload = StaticWorkload(items=[{"prompt": "hi", "reference": "0"}])
    hook = correctness_hook(regex_match(r"tok(\d+)", group=1))
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=5, duration_s=0.4),
        post_hooks=[hook], progress_every_s=0,
    ))
    result = await runner.run()
    wm = result.summary["workload_metrics"]
    assert wm["correct"]["mean"] == 1.0
    assert wm["matched"]["mean"] == 1.0
    # base LLM metrics still present alongside correctness:
    assert "ttft_s" in wm and "tokens_out" in wm


@pytest.mark.asyncio
async def test_correctness_hook_missing_reference_flag(stub_server: str):
    """If the item has no reference, the hook records `missing_reference` without grading."""
    base = OpenAIChatWorkloadType(
        url=f"{stub_server}/v1/chat/completions", model="stub", max_tokens=8,
    )
    wt = EvalWorkloadType(base)
    # No "reference" key in the item → hook should flag, not grade.
    workload = StaticWorkload(items=[{"prompt": "hi"}])
    hook = correctness_hook(exact_match())
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=5, duration_s=0.4),
        post_hooks=[hook], progress_every_s=0,
    ))
    result = await runner.run()
    wm = result.summary["workload_metrics"]
    assert "missing_reference" in wm
    assert wm["missing_reference"]["mean"] == 1.0
    assert "correct" not in wm  # never produced because we bailed early


@pytest.mark.asyncio
async def test_correctness_hook_with_prefix_and_no_gate(stub_server: str):
    """`prefix` namespaces extra keys; gate_key=None means failures don't drop ok."""
    base = OpenAIChatWorkloadType(
        url=f"{stub_server}/v1/chat/completions", model="stub", max_tokens=8,
    )
    wt = EvalWorkloadType(base)
    workload = StaticWorkload(items=[{"prompt": "hi", "reference": "wrong"}])
    hook = correctness_hook(exact_match(), prefix="eval_", gate_key=None)
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=5, duration_s=0.4),
        post_hooks=[hook], progress_every_s=0,
    ))
    result = await runner.run()
    # All requests still considered successful because gating disabled.
    assert result.summary["failed"] == 0
    wm = result.summary["workload_metrics"]
    assert wm["eval_correct"]["mean"] == 0.0


@pytest.mark.asyncio
async def test_judge_llm_with_stub_send():
    """judge_llm should call `send`, parse the reply, and produce numeric scores."""
    calls: list[str] = []

    async def fake_send(prompt: str) -> str:
        calls.append(prompt)
        # Default parser expects an integer; return 9.
        return "9"

    scorer = judge_llm(fake_send)
    out = await scorer("reference text", "model prediction")
    assert calls and "reference text" in calls[0] and "model prediction" in calls[0]
    assert out == {"judge_score": 9.0, "correct": 1.0, "judge_parsed": 1.0}


# ----------------------- YAML loader integration ----------------------- #


def test_build_config_with_correctness_block(stub_server: str):
    from benchmaker.config import build_config

    cfg = build_config({
        "workload_type": {
            "type": "openai",
            "url": f"{stub_server}/v1/chat/completions",
            "model": "stub",
            "max_tokens": 8,
        },
        "workload": {
            "type": "static",
            "items": [{"prompt": "hi", "expected": "anything"}],
        },
        "load": "5",
        "duration": "0.3s",
        "correctness": {
            "reference_key": "expected",
            "scorer": {"type": "exact_match", "case_insensitive": True},
        },
    })
    # workload_type was wrapped:
    assert isinstance(cfg.workload_type, EvalWorkloadType)
    # post_hooks gained the correctness hook:
    assert len(cfg.post_hooks) == 1
    assert callable(cfg.post_hooks[0])


def test_build_config_scorer_shorthand_string(stub_server: str):
    """`scorer: exact_match` (bare string) should be equivalent to `{type: ...}`."""
    from benchmaker.config import build_config

    cfg = build_config({
        "workload_type": {"type": "http", "url": "http://x"},
        "workload": [{"reference": "a"}],
        "load": "5",
        "duration": "0.1s",
        "correctness": {"scorer": "exact_match"},
    })
    assert isinstance(cfg.workload_type, EvalWorkloadType)


def test_build_correctness_factory_scorer():
    """User-defined scorers come through `factory:`."""
    from benchmaker.config import build_scorer
    scorer, aclose = build_scorer({
        "factory": "benchmaker.workloads.eval:exact_match",
        "case_insensitive": True,
    })
    assert aclose is None
    out = scorer("HELLO", "hello")
    assert out == {"correct": 1.0}


def test_build_correctness_unknown_scorer_raises():
    from benchmaker.config import build_scorer
    with pytest.raises(ValueError):
        build_scorer({"type": "does_not_exist"})


def test_build_correctness_disabled_gate_via_null():
    """gate_key: null should disable the failure gate."""
    from benchmaker.config import apply_correctness

    base = OpenAIChatWorkloadType(url="http://x", model="m")
    wrapped, hooks = apply_correctness(base, {
        "scorer": {"type": "exact_match"},
        "gate_key": "null",
    })
    assert isinstance(wrapped, EvalWorkloadType)
    assert len(hooks) == 1


@pytest.mark.asyncio
async def test_yaml_correctness_end_to_end(stub_server: str):
    """A correctness-wired BenchConfig should produce `correct.mean` in the summary."""
    from benchmaker.config import build_config

    cfg = build_config({
        "workload_type": {
            "type": "openai",
            "url": f"{stub_server}/v1/chat/completions",
            "model": "stub",
            "max_tokens": 8,
        },
        "workload": {
            "type": "static",
            "items": [{"prompt": "hi", "reference": "tok0 tok1 tok2 tok3 tok4"}],
        },
        "load": "5",
        "duration": "0.4s",
        "progress_every_s": 0,
        "correctness": {"scorer": {"type": "exact_match"}},
    })
    result = await BenchRunner(cfg).run()
    wm = result.summary["workload_metrics"]
    assert wm["correct"]["mean"] == 1.0


@pytest.mark.asyncio
async def test_yaml_correctness_chains_judge_aclose(stub_server: str):
    """openai_chat judge's aclose should run on workload_type.aclose."""
    from benchmaker.config import apply_correctness

    base = OpenAIChatWorkloadType(url="http://x", model="m")
    wrapped, hooks = apply_correctness(base, {
        "scorer": {
            "type": "judge_llm",
            "openai_chat": {
                "url": f"{stub_server}/v1/chat/completions",
                "model": "judge",
            },
        },
    })
    # Force the session open by calling the scorer once.
    # We need to invoke through the hook; simplest is to call the inner send.
    # The send is closed over inside judge_llm — but openai_chat_judge tracks
    # state.  Just verify aclose runs without error (no session opened yet).
    await wrapped.aclose()  # should not raise


# ----------------------- raw-output persistence ----------------------- #


@pytest.mark.asyncio
async def test_prediction_default_truncation_is_2048(stub_server: str):
    """By default, sample.meta['prediction'] is capped at 2048 chars."""
    base = OpenAIChatWorkloadType(
        url=f"{stub_server}/v1/chat/completions", model="stub", max_tokens=8,
    )
    wt = EvalWorkloadType(base)
    workload = StaticWorkload(items=[{"prompt": "hi", "reference": "x"}])
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=5, duration_s=0.3),
        post_hooks=[correctness_hook(exact_match())],
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.samples
    for s in result.samples:
        assert "prediction" in s.meta
        assert len(s.meta["prediction"]) <= 2048
        # The stub emits ~30 chars, so we should see the full string here:
        assert "tok0" in s.meta["prediction"]


@pytest.mark.asyncio
async def test_prediction_max_chars_none_stores_full(stub_server: str):
    base = OpenAIChatWorkloadType(
        url=f"{stub_server}/v1/chat/completions", model="stub", max_tokens=8,
    )
    wt = EvalWorkloadType(base)
    workload = StaticWorkload(items=[{"prompt": "hi", "reference": "x"}])
    hook = correctness_hook(exact_match(), max_prediction_chars=None)
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=5, duration_s=0.3),
        post_hooks=[hook], progress_every_s=0,
    ))
    result = await runner.run()
    for s in result.samples:
        # The full SSE-decoded text from the stub:
        assert s.meta["prediction"] == "tok0 tok1 tok2 tok3 tok4 "


@pytest.mark.asyncio
async def test_prediction_max_chars_truncates(stub_server: str):
    base = OpenAIChatWorkloadType(
        url=f"{stub_server}/v1/chat/completions", model="stub", max_tokens=8,
    )
    wt = EvalWorkloadType(base)
    workload = StaticWorkload(items=[{"prompt": "hi", "reference": "x"}])
    hook = correctness_hook(exact_match(), max_prediction_chars=5)
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=5, duration_s=0.3),
        post_hooks=[hook], progress_every_s=0,
    ))
    result = await runner.run()
    for s in result.samples:
        assert s.meta["prediction"] == "tok0 "  # exactly 5 chars


@pytest.mark.asyncio
async def test_yaml_max_prediction_chars_full(stub_server: str):
    """YAML `max_prediction_chars: full` should disable truncation."""
    from benchmaker.config import build_config

    cfg = build_config({
        "workload_type": {
            "type": "openai",
            "url": f"{stub_server}/v1/chat/completions",
            "model": "stub", "max_tokens": 8,
        },
        "workload": {"type": "static",
                     "items": [{"prompt": "hi", "reference": "x"}]},
        "load": "5", "duration": "0.3s", "progress_every_s": 0,
        "correctness": {
            "max_prediction_chars": "full",
            "scorer": {"type": "exact_match"},
        },
    })
    result = await BenchRunner(cfg).run()
    for s in result.samples:
        assert s.meta["prediction"].endswith("tok4 ")


@pytest.mark.asyncio
async def test_samples_jsonl_contains_prediction_and_reference(stub_server, tmp_path):
    """End-to-end: write a bundle and verify samples.jsonl carries the raw output."""
    import json as _json

    base = OpenAIChatWorkloadType(
        url=f"{stub_server}/v1/chat/completions", model="stub", max_tokens=8,
    )
    wt = EvalWorkloadType(base)
    workload = StaticWorkload(items=[{"prompt": "p", "reference": "ref-val"}])
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=5, duration_s=0.3),
        post_hooks=[correctness_hook(exact_match(), max_prediction_chars=None)],
        progress_every_s=0,
    ))
    await runner.run()
    bundle = runner.write_bundle(str(tmp_path))
    samples_path = f"{bundle}/samples.jsonl"
    rows = [_json.loads(line) for line in open(samples_path)]
    assert rows
    for row in rows:
        assert row["meta"]["reference"] == "ref-val"
        assert "tok0" in row["meta"]["prediction"]
        assert row["extra"]["correct"] in (0.0, 1.0)


@pytest.mark.asyncio
async def test_correctness_hook_async_scorer(stub_server: str):
    """Async scorers (e.g. judge_llm) should be awaited transparently."""
    async def fake_send(prompt: str) -> str:
        # Always 8/10 → above threshold → correct=1
        return "8"

    base = OpenAIChatWorkloadType(
        url=f"{stub_server}/v1/chat/completions", model="stub", max_tokens=8,
    )
    wt = EvalWorkloadType(base)
    workload = StaticWorkload(items=[{"prompt": "hi", "reference": "anything"}])
    hook = correctness_hook(judge_llm(fake_send))
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=5, duration_s=0.4),
        post_hooks=[hook], progress_every_s=0,
    ))
    result = await runner.run()
    wm = result.summary["workload_metrics"]
    assert wm["judge_score"]["mean"] == 8.0
    assert wm["correct"]["mean"] == 1.0

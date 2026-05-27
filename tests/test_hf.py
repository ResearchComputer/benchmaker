"""Tests for HFDatasetWorkload.

The `datasets` library is mocked via sys.modules so these tests run without
HF installed and without network. We pass a fake `load_dataset` that returns
either a list (non-streaming) or an iterator (streaming) of dict rows.
"""

import sys
import types

import pytest

from benchmaker import (
    BenchConfig,
    BenchRunner,
    ConstantRPS,
    EvalWorkloadType,
    HFDatasetWorkload,
    HttpWorkloadType,
    OpenAIChatWorkloadType,
    correctness_hook,
    exact_match,
)
from benchmaker.workloads.hf import (
    _PRESETS,
    get_preset,
    get_transform,
    list_presets,
)


# --------------------------------------------------------------------------- #
# Fake `datasets` module
# --------------------------------------------------------------------------- #


class _FakeNonStreamingDataset:
    """List-of-rows that supports len() and indexing — like a HF Dataset."""

    def __init__(self, rows):
        self._rows = list(rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


def _make_fake_datasets(rows_by_split):
    """Install a fake `datasets` module whose load_dataset returns our rows.

    `rows_by_split` is either:
        list[dict]           — same rows for every split request
        {split_name: rows}   — per-split rows
        callable(**kwargs)   — full control
    """
    if callable(rows_by_split):
        loader = rows_by_split
    else:
        def loader(path, **kwargs):
            split = kwargs.get("split", "test")
            if isinstance(rows_by_split, dict):
                rows = rows_by_split.get(split, [])
            else:
                rows = rows_by_split
            if kwargs.get("streaming"):
                return iter(rows)
            return _FakeNonStreamingDataset(rows)

    fake = types.ModuleType("datasets")
    fake.load_dataset = loader
    sys.modules["datasets"] = fake
    return fake


@pytest.fixture
def fake_datasets():
    """Fixture: install fake datasets module, restore previous on teardown."""
    original = sys.modules.get("datasets")
    yield _make_fake_datasets
    if original is None:
        sys.modules.pop("datasets", None)
    else:
        sys.modules["datasets"] = original


# --------------------------------------------------------------------------- #
# Pure unit tests
# --------------------------------------------------------------------------- #


def test_gsm8k_answer_transform():
    f = get_transform("gsm8k_answer")
    assert f("Some reasoning here.\n#### 42") == "42"
    assert f("Plain 7 with no marker") == "7"
    # commas in numbers should be stripped:
    assert f("Reasoning #### 1,234") == "1234"


def test_strip_and_first_line_transforms():
    assert get_transform("strip")("  hi  \n") == "hi"
    assert get_transform("first_line")("first\nsecond\nthird") == "first"
    assert get_transform("lower_strip")("  HELLO  ") == "hello"


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        get_transform("bogus")


def test_presets_known():
    assert "gsm8k" in list_presets()
    spec = get_preset("gsm8k")
    assert spec["path"] == "gsm8k"
    assert spec["reference_transform"] == "gsm8k_answer"


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        get_preset("not-a-preset")


def test_constructor_validates():
    # No path AND no preset:
    with pytest.raises(ValueError):
        HFDatasetWorkload()
    # Neither prompt_field nor prompt_template:
    with pytest.raises(ValueError):
        HFDatasetWorkload(path="x", prompt_field=None, prompt_template=None)
    # Template without field map:
    with pytest.raises(ValueError):
        HFDatasetWorkload(path="x", prompt_template="{q}",
                          prompt_template_fields=None)


# --------------------------------------------------------------------------- #
# Field mapping / shaping
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_basic_field_mapping(fake_datasets):
    rows = [
        {"prompt": "p1", "reference": "r1"},
        {"prompt": "p2", "reference": "r2"},
    ]
    fake_datasets(rows)
    wl = HFDatasetWorkload(path="x")  # defaults match the row shape
    items = [await wl.next_item(), await wl.next_item()]
    assert items == rows
    with pytest.raises(StopAsyncIteration):
        await wl.next_item()


@pytest.mark.asyncio
async def test_rename_fields(fake_datasets):
    rows = [{"question": "What is 2+2?", "answer": "#### 4"}]
    fake_datasets(rows)
    wl = HFDatasetWorkload(
        path="x", prompt_field="question",
        reference_field="answer", reference_transform="gsm8k_answer",
    )
    item = await wl.next_item()
    assert item == {"prompt": "What is 2+2?", "reference": "4"}


@pytest.mark.asyncio
async def test_extra_fields_passthrough(fake_datasets):
    rows = [{"prompt": "p", "reference": "r", "qid": "q1", "split": "test"}]
    fake_datasets(rows)
    wl = HFDatasetWorkload(path="x", extra_fields=("qid", "split"))
    item = await wl.next_item()
    assert item["qid"] == "q1"
    assert item["split"] == "test"


@pytest.mark.asyncio
async def test_prompt_template_with_indexed_lookup(fake_datasets):
    rows = [{
        "question": "Pick:",
        "choices": ["alpha", "beta", "gamma", "delta"],
        "answer": 2,
    }]
    fake_datasets(rows)
    wl = HFDatasetWorkload(
        path="x",
        prompt_field=None,
        prompt_template="{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}",
        prompt_template_fields={
            "q": "question",
            "a": ("choices", 0),
            "b": ("choices", 1),
            "c": ("choices", 2),
            "d": ("choices", 3),
        },
        reference_field="answer",
        reference_transform="mmlu_letter",
    )
    item = await wl.next_item()
    assert item["prompt"] == "Pick:\nA. alpha\nB. beta\nC. gamma\nD. delta"
    assert item["reference"] == "C"  # idx 2 -> "C"


# --------------------------------------------------------------------------- #
# Preset wiring
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_gsm8k_preset(fake_datasets):
    rows = [
        {"question": "How many apples?",
         "answer": "Reasoning... #### 7"},
        {"question": "What is 100/4?",
         "answer": "Compute: #### 25"},
    ]
    seen_kwargs = {}

    def loader(path, **kwargs):
        seen_kwargs["path"] = path
        seen_kwargs.update(kwargs)
        return _FakeNonStreamingDataset(rows)

    fake_datasets(loader)
    wl = HFDatasetWorkload(preset="gsm8k")
    items = [await wl.next_item(), await wl.next_item()]
    assert items[0] == {"prompt": "How many apples?", "reference": "7"}
    assert items[1] == {"prompt": "What is 100/4?", "reference": "25"}
    # Preset should have populated path/name/split:
    assert seen_kwargs["path"] == "gsm8k"
    assert seen_kwargs["name"] == "main"
    assert seen_kwargs["split"] == "test"


@pytest.mark.asyncio
async def test_preset_overridden_by_explicit_kwargs(fake_datasets):
    """Explicit `split=` should beat the preset's split."""
    seen = {}

    def loader(path, **kwargs):
        seen.update(kwargs)
        return _FakeNonStreamingDataset([{"question": "q", "answer": "#### 1"}])

    fake_datasets(loader)
    wl = HFDatasetWorkload(preset="gsm8k", split="train")
    await wl.next_item()
    assert seen["split"] == "train"


# --------------------------------------------------------------------------- #
# max_items / shuffle / loop / streaming
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_max_items_halts(fake_datasets):
    fake_datasets([{"prompt": f"p{i}", "reference": str(i)} for i in range(10)])
    wl = HFDatasetWorkload(path="x", max_items=3)
    items = []
    while True:
        try:
            items.append(await wl.next_item())
        except StopAsyncIteration:
            break
    assert len(items) == 3


@pytest.mark.asyncio
async def test_loop_restarts(fake_datasets):
    fake_datasets([{"prompt": "a", "reference": "1"},
                   {"prompt": "b", "reference": "2"}])
    wl = HFDatasetWorkload(path="x", loop=True, max_items=5)
    items = []
    while True:
        try:
            items.append(await wl.next_item())
        except StopAsyncIteration:
            break
    # 5 items from a 2-row dataset = a, b, a, b, a
    assert [it["prompt"] for it in items] == ["a", "b", "a", "b", "a"]


@pytest.mark.asyncio
async def test_shuffle_deterministic(fake_datasets):
    fake_datasets([{"prompt": str(i), "reference": str(i)} for i in range(10)])
    wl1 = HFDatasetWorkload(path="x", shuffle=True, seed=42, max_items=10)
    wl2 = HFDatasetWorkload(path="x", shuffle=True, seed=42, max_items=10)
    seq1 = [(await wl1.next_item())["prompt"] for _ in range(10)]
    seq2 = [(await wl2.next_item())["prompt"] for _ in range(10)]
    assert seq1 == seq2
    # Not in the original order (probabilistic but seed-stable):
    assert seq1 != [str(i) for i in range(10)]


@pytest.mark.asyncio
async def test_streaming_mode(fake_datasets):
    fake_datasets([{"prompt": "p1", "reference": "r1"},
                   {"prompt": "p2", "reference": "r2"}])
    wl = HFDatasetWorkload(path="x", streaming=True)
    items = [await wl.next_item(), await wl.next_item()]
    assert items[0]["prompt"] == "p1"
    with pytest.raises(StopAsyncIteration):
        await wl.next_item()


# --------------------------------------------------------------------------- #
# YAML loader integration
# --------------------------------------------------------------------------- #


def test_build_workload_hf_type(fake_datasets):
    from benchmaker.config import build_workload

    fake_datasets([{"prompt": "p", "reference": "r"}])
    wl = build_workload({"type": "hf", "path": "any-id"})
    assert isinstance(wl, HFDatasetWorkload)


def test_build_workload_hf_preset(fake_datasets):
    from benchmaker.config import build_workload
    fake_datasets([{"question": "q", "answer": "#### 1"}])
    wl = build_workload({"type": "huggingface", "preset": "gsm8k"})
    assert isinstance(wl, HFDatasetWorkload)


# --------------------------------------------------------------------------- #
# End-to-end with stub LLM server: HF + Eval + correctness
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_hf_with_eval_and_correctness(stub_server, fake_datasets):
    """HF dataset → EvalWorkloadType → SSE LLM → correctness post-hook.

    The SSE stub emits "tok0 tok1 tok2 tok3 tok4 ". The fake HF dataset
    encodes that string in the gsm8k-style `#### <answer>` slot, so
    `reference_transform="gsm8k_answer"` extracts exactly the expected text.
    """
    fake_datasets([
        {"question": f"row {i}",
         "answer": "Reasoning #### tok0 tok1 tok2 tok3 tok4"}
        for i in range(3)
    ])
    workload = HFDatasetWorkload(
        path="my-ds",
        prompt_field="question",
        reference_field="answer",
        reference_transform="gsm8k_answer",
    )
    base = OpenAIChatWorkloadType(
        url=f"{stub_server}/v1/chat/completions", model="stub", max_tokens=8,
    )
    wt = EvalWorkloadType(base)

    hook = correctness_hook(exact_match())
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=10, duration_s=1.0),
        post_hooks=[hook], progress_every_s=0,
        stop_on_exhausted=True,
    ))
    result = await runner.run()
    assert result.summary["total_requests"] == 3
    wm = result.summary["workload_metrics"]
    assert wm["correct"]["mean"] == 1.0

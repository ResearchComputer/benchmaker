"""DeepRAG workload and preparation-tool tests."""

import json

import pytest

from benchmaker import OpenAIChatWorkloadType, Response
from benchmaker.workloads.rag import DeepRAGWorkload
from tools.rag.prepare import normalise_hotpot_row


@pytest.mark.asyncio
async def test_deeprag_packs_requested_depth_and_emits_eval_metadata(tmp_path):
    path = tmp_path / "rag.jsonl"
    path.write_text(json.dumps({
        "id": "q-1",
        "question": "Who wrote the paper?",
        "answer": "Ada",
        "passages": ["First retrieved passage.", "Second retrieved passage.", "Third."],
    }) + "\n")
    workload = DeepRAGWorkload(path=str(path), depth=2, max_tokens=32, loop=False)

    item = await workload.next_item()

    assert item["reference"] == "Ada"
    assert item["source_id"] == "q-1"
    assert item["rag_depth"] == 2
    assert item["max_tokens"] == 32
    assert item["prompt_tokens_hint"] > 0
    assert item["messages"][0]["content"] == "Answer the question using only the context below."
    user = item["messages"][1]["content"]
    assert "First retrieved passage." in user
    assert "Second retrieved passage." in user
    assert "Third." not in user


@pytest.mark.asyncio
async def test_deeprag_context_target_truncates_packed_passages(tmp_path):
    path = tmp_path / "rag.jsonl"
    path.write_text(json.dumps({
        "question": "Q?", "answer": "A",
        "passages": ["one two three", "four five six"],
    }) + "\n")
    workload = DeepRAGWorkload(
        path=str(path), depth=2, context_tokens_target=4, loop=False,
    )

    item = await workload.next_item()

    user = item["messages"][1]["content"]
    assert item["rag_depth"] == 2
    assert "one two three" in user
    assert "four" in user
    assert "five six" not in user


@pytest.mark.asyncio
async def test_deeprag_metadata_is_not_sent_and_is_reported_as_metrics(tmp_path):
    path = tmp_path / "rag.jsonl"
    path.write_text(json.dumps({
        "question": "Q?", "answer": "A", "passages": ["context words"],
    }) + "\n")
    item = await DeepRAGWorkload(path=str(path), loop=False).next_item()
    workload_type = OpenAIChatWorkloadType(
        url="http://example/v1/chat/completions", model="test", passthrough_meta=True,
    )

    request = await workload_type.make_request(item)
    assert "reference" not in request.json
    assert "prompt_tokens_hint" not in request.json
    assert request.meta["reference"] == "A"
    assert request.meta["rag_depth"] == 1

    chunks = [
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":99,"completion_tokens":1}}\n\n',
        b"data: [DONE]\n\n",
    ]
    response = Response(
        status=200, headers={}, body=b"".join(chunks), elapsed_s=0.1, ok=True,
        stream_chunks=chunks, stream_chunk_times=[0.01, 0.02],
    )
    sample = await workload_type.make_sample(item, request, response, 0.0)
    assert sample.extra["prompt_tokens"] == 99.0
    assert sample.extra["prompt_tokens_hint"] == item["prompt_tokens_hint"]
    assert sample.extra["rag_depth"] == 1.0


def test_deeprag_yaml_workload_type(tmp_path):
    from benchmaker.config import build_workload

    path = tmp_path / "rag.jsonl"
    path.write_text(json.dumps({
        "question": "Q?", "answer": "A", "passages": ["context"],
    }) + "\n")
    workload = build_workload({"type": "deeprag", "path": str(path), "depth": 1})
    assert isinstance(workload, DeepRAGWorkload)


def test_hotpot_normalizer_retains_multi_passage_context():
    row = {
        "id": "hotpot-id",
        "question": "Where?",
        "answer": "Here",
        "context": {
            "title": ["First", "Second"],
            "sentences": [["A first sentence.", "Another."], ["Second body."]],
        },
    }

    prepared = normalise_hotpot_row(row, max_passage_chars=None)

    assert prepared == {
        "id": "hotpot-id",
        "question": "Where?",
        "answer": "Here",
        "passages": [
            {"title": "First", "text": "A first sentence. Another."},
            {"title": "Second", "text": "Second body."},
        ],
    }

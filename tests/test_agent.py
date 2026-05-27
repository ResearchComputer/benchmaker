"""Agent workload-type tests."""

import pytest

from benchmaker import (
    Agent,
    AgentContext,
    AgentResult,
    AgentWorkloadType,
    BenchConfig,
    BenchRunner,
    ClosedLoop,
    StaticWorkload,
)
from benchmaker.config import build_config
from benchmaker.workloads.eval import correctness_hook, exact_match


class ReverseAgent(Agent):
    """Toy agent: output = reverse of the last word of the prompt.

    Lets us mix correct/wrong references in a single bench to exercise the
    ok / wrong / failed buckets.
    """

    async def run(self, ctx: AgentContext) -> AgentResult:
        item = ctx.item or {}
        prompt: str = item.get("prompt", "")
        last = prompt.strip().split()[-1] if prompt.strip() else ""
        return AgentResult(
            output=last[::-1],
            ok=True,
            metrics={"steps": 3, "tokens": float(len(prompt))},
            meta={"agent_label": "reverse"},
        )


@pytest.mark.asyncio
async def test_agent_runs_and_grading_distinguishes_wrong():
    items = [
        {"prompt": "hello world", "reference": "dlrow"},  # correct (reverse of 'world')
        {"prompt": "foo bar", "reference": "nope"},        # wrong
    ]
    cfg = BenchConfig(
        workload_type=AgentWorkloadType(ReverseAgent()),
        workload=StaticWorkload(items=items, max_items=2),
        load=ClosedLoop(concurrency=1, max_requests=2),
        post_hooks=[correctness_hook(exact_match())],
        progress_every_s=0,
    )
    summary = (await BenchRunner(cfg).run()).summary
    assert summary["total_requests"] == 2
    assert summary["success"] == 1
    assert summary["wrong_output"] == 1
    assert summary["request_failed"] == 0
    # Per-trajectory metric surfaces in workload_metrics.
    assert "steps" in summary["workload_metrics"]
    assert summary["workload_metrics"]["steps"]["mean"] == 3.0


@pytest.mark.asyncio
async def test_agent_exception_classified_as_request_failed():
    class BadAgent(Agent):
        async def run(self, ctx: AgentContext) -> AgentResult:
            raise RuntimeError("kaboom")

    cfg = BenchConfig(
        workload_type=AgentWorkloadType(BadAgent()),
        workload=StaticWorkload(items=[{"prompt": "anything"}], max_items=1),
        load=ClosedLoop(concurrency=1, max_requests=1),
        progress_every_s=0,
    )
    summary = (await BenchRunner(cfg).run()).summary
    assert summary["success"] == 0
    assert summary["request_failed"] == 1
    assert summary["wrong_output"] == 0


@pytest.mark.asyncio
async def test_agent_unsuccessful_trajectory_is_wrong_not_failed():
    """An agent that runs to completion but reports ok=False (e.g. step_limit
    in CodingAgent) should bucket as "wrong" — the request succeeded, the
    output is just unsuccessful. Reserve "failed" for real infra failures."""

    class IncompleteAgent(Agent):
        async def run(self, ctx: AgentContext) -> AgentResult:
            return AgentResult(
                output="",
                ok=False,         # didn't submit
                request_ok=True,  # but ran cleanly
                error="step_limit",
                metrics={"steps": 5},
            )

    cfg = BenchConfig(
        workload_type=AgentWorkloadType(IncompleteAgent()),
        workload=StaticWorkload(items=[{"prompt": "tricky"}], max_items=1),
        load=ClosedLoop(concurrency=1, max_requests=1),
        progress_every_s=0,
    )
    summary = (await BenchRunner(cfg).run()).summary
    assert summary["success"] == 0
    assert summary["request_failed"] == 0
    assert summary["wrong_output"] == 1


@pytest.mark.asyncio
async def test_agent_unsuccessful_trajectory_is_graded_by_correctness_hook():
    """When the trajectory completes without submission (output=""), the
    correctness hook should still run and mark it wrong — proves resp.ok
    tracks request_ok, not ok."""

    class EmptySubmissionAgent(Agent):
        async def run(self, ctx: AgentContext) -> AgentResult:
            return AgentResult(output="", ok=False, request_ok=True,
                               error="step_limit")

    cfg = BenchConfig(
        workload_type=AgentWorkloadType(EmptySubmissionAgent()),
        workload=StaticWorkload(
            items=[{"prompt": "x", "reference": "expected"}], max_items=1,
        ),
        load=ClosedLoop(concurrency=1, max_requests=1),
        post_hooks=[correctness_hook(exact_match())],
        progress_every_s=0,
    )
    result = await BenchRunner(cfg).run()
    sample = result.samples[0]
    # Grader ran (correct=0.0 in extras) — proof that resp.ok was True.
    assert sample.extra.get("correct") == 0.0
    assert not sample.ok
    assert sample.request_ok


@pytest.mark.asyncio
async def test_callable_agent_string_return_shortcut():
    """A bare async function returning a str works via CallableAgent."""

    async def fn(ctx: AgentContext) -> str:
        return f"hi-{ctx.item['prompt']}"

    cfg = BenchConfig(
        workload_type=AgentWorkloadType(fn),
        workload=StaticWorkload(items=[{"prompt": "p"}], max_items=1),
        load=ClosedLoop(concurrency=1, max_requests=1),
        progress_every_s=0,
    )
    result = await BenchRunner(cfg).run()
    assert result.summary["success"] == 1
    # The output (returned string) is stored as the response body so the
    # default extractor sees it.
    sample = result.samples[0]
    assert sample.workload == "agent"
    # bytes_recv == len of the encoded output
    assert sample.bytes_recv > 0


@pytest.mark.asyncio
async def test_yaml_agent_workload_with_correctness():
    """End-to-end via build_config: type=agent + correctness should not
    double-wrap (AgentWorkloadType handles reference itself)."""
    cfg_dict = {
        "workload_type": {
            "type": "agent",
            "agent": "tests.test_agent:ReverseAgent",
        },
        "workload": {
            "type": "static",
            "items": [
                {"prompt": "say hi", "reference": "ih"},   # correct
                {"prompt": "say bye", "reference": "x"},   # wrong
            ],
            "max_items": 2,
        },
        "load": "closed:1",
        "max_requests": 2,
        "correctness": {"scorer": "exact_match"},
    }
    bench_cfg = build_config(cfg_dict, dotenv_path=None, interpolate_env=False)
    # Workload-type should still be the AgentWorkloadType (NOT wrapped).
    assert isinstance(bench_cfg.workload_type, AgentWorkloadType)
    summary = (await BenchRunner(bench_cfg).run()).summary
    assert summary["total_requests"] == 2
    assert summary["success"] == 1
    assert summary["wrong_output"] == 1

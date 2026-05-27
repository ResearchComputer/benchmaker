"""Example: drive a custom Agent over the AgentTrove dataset.

Pair with `examples/configs/config_agent_trove.yaml`.

The agent here is intentionally minimal — it just echoes the task instruction
back as the "answer" — so the file is self-contained and runnable without an
LLM endpoint. In a real benchmark you'd replace `MyAgent.run` with whatever
multi-turn pipeline you want to measure: tool-using loops, sandbox exec,
graph-of-thoughts, etc. The runner records end-to-end latency per task and
any numeric per-trajectory metrics you report via `AgentResult.metrics`.

Reference field: the dataset's `expected_response` (or similar) becomes
`reference`, which `correctness_hook(exact_match())` then compares against
`AgentResult.output`.
"""

from __future__ import annotations

from benchmaker import Agent, AgentContext, AgentResult


class MyAgent(Agent):
    """Toy agent. Replace `.run` with your real pipeline."""

    def __init__(self, model: str = "stub", max_steps: int = 1):
        self.model = model
        self.max_steps = max_steps

    async def run(self, ctx: AgentContext) -> AgentResult:
        item = ctx.item or {}
        task = item.get("prompt") or item.get("instruction") or ""
        # ----- your agent loop goes here -----
        steps = 0
        answer = task.strip().split("\n")[0]
        steps += 1
        # -------------------------------------
        return AgentResult(
            output=answer,
            ok=True,
            metrics={
                "steps": float(steps),
                "prompt_chars": float(len(task)),
            },
            meta={"model": self.model},
        )

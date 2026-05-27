"""Tests for the example CodingAgent.

Drive the agent with a canned `send_fn` so no real LLM is needed.
"""

import asyncio
import socket
import pytest
from aiohttp import web

from benchmaker import (
    AgentContext,
    AgentWorkloadType,
    BenchConfig,
    BenchRunner,
    ClosedLoop,
    StaticWorkload,
)
from benchmaker.workloads.eval import correctness_hook, exact_match
from examples.coding_agent.coding_agent import SUBMIT_TOKEN, CodingAgent


def _canned_send(replies):
    """Return an async `send_fn` that yields one canned reply per call."""
    it = iter(replies)

    async def send(messages):
        try:
            return next(it)
        except StopIteration:
            return "no more"

    return send


def _fence(cmd: str) -> str:
    return f"```bash\n{cmd}\n```"


@pytest.mark.asyncio
async def test_agent_executes_action_and_submits():
    """Two-turn trajectory: list dir, then submit."""
    replies = [
        _fence("echo hello"),
        _fence(f"{SUBMIT_TOKEN}\nhello"),
    ]
    agent = CodingAgent(send_fn=_canned_send(replies), step_limit=5)
    result = await agent.run(AgentContext(item={"prompt": "say hi"}))
    assert result.ok
    assert result.output == "hello"
    assert result.metrics["steps"] == 2
    assert result.metrics["actions"] == 1
    assert result.meta["exit_status"] == "submitted"


@pytest.mark.asyncio
async def test_agent_uses_real_shell_output():
    """Feed back actual `pwd`/`ls` output so the agent's submission depends on
    the shell having actually run the command."""
    # Two-turn: model asks for `echo 42` then submits its (real) output.
    captured: list[str] = []

    async def send(messages):
        if len(captured) == 0:
            captured.append("first")
            return _fence("echo 42")
        # Echo the latest user message back as evidence we got the observation.
        last_user = messages[-1]["content"]
        assert "returncode: 0" in last_user and "42" in last_user
        return _fence(f"{SUBMIT_TOKEN}\n42")

    agent = CodingAgent(send_fn=send, step_limit=5)
    result = await agent.run(AgentContext(item={"prompt": "print 42"}))
    assert result.ok and result.output == "42"


@pytest.mark.asyncio
async def test_agent_no_action_halts_with_diagnostic():
    """If the model emits no fenced block, the agent stops with exit_status=no_action.

    The trajectory ran cleanly though — request_ok stays True (i.e. this
    should bucket as "wrong", not "fail")."""
    agent = CodingAgent(
        send_fn=_canned_send(["I don't know how to help."]),
        step_limit=5,
    )
    result = await agent.run(AgentContext(item={"prompt": "?"}))
    assert not result.ok
    assert result.request_ok          # infra fine
    assert result.meta["exit_status"] == "no_action"
    assert result.error == "no_action"


@pytest.mark.asyncio
async def test_agent_step_limit_caps_trajectory():
    """An infinite loop of useless commands is capped by step_limit."""
    replies = [_fence("true")] * 50
    agent = CodingAgent(send_fn=_canned_send(replies), step_limit=3)
    result = await agent.run(AgentContext(item={"prompt": "loop"}))
    assert not result.ok
    assert result.request_ok          # ran fine, just didn't submit
    assert result.meta["exit_status"] == "step_limit"
    assert result.metrics["steps"] == 3


@pytest.mark.asyncio
async def test_agent_send_http_error_propagates_to_request_failed():
    """LLM endpoint returning 5xx should surface as a clear error message and
    bucket as a real failure ("fail"), not as a silent "no_action"."""

    async def bad_send(messages):
        raise RuntimeError("model endpoint HTTP 503: upstream is overloaded")

    cfg = BenchConfig(
        workload_type=AgentWorkloadType(CodingAgent(send_fn=bad_send)),
        workload=StaticWorkload(items=[{"prompt": "x"}], max_items=1),
        load=ClosedLoop(concurrency=1, max_requests=1),
        progress_every_s=0,
    )
    result = await BenchRunner(cfg).run()
    sample = result.samples[0]
    assert not sample.ok
    assert not sample.request_ok
    assert "HTTP 503" in (sample.error or "")
    summary = result.summary
    assert summary["request_failed"] == 1
    assert summary["wrong_output"] == 0


async def _start_sandbox_stub():
    """Spin up a stub sandbox server that records every request and returns
    canned ``/exec`` responses. Returns (base_url, state, stop)."""
    state: dict = {
        "next_id": 0,
        "created": [],   # list[dict] of pod-spec bodies
        "execs": [],     # list[(sid, body)]
        "deleted": [],   # list[sid]
    }

    async def create(request: web.Request) -> web.Response:
        body = await request.json()
        state["next_id"] += 1
        sid = f"sb-test-{state['next_id']:04d}"
        state["created"].append(body)
        return web.json_response({"id": sid})

    async def exec_(request: web.Request) -> web.Response:
        sid = request.match_info["sid"]
        body = await request.json()
        state["execs"].append((sid, body))
        # Echo the command back in stdout so tests can assert routing.
        cmd = body.get("command") or []
        rendered = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        return web.json_response({
            "stdout": f"ran: {rendered}\n",
            "stderr": "",
            "exit_code": 0,
            "duration": 0.001,
        })

    async def delete(request: web.Request) -> web.Response:
        state["deleted"].append(request.match_info["sid"])
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/sandboxes", create)
    app.router.add_post("/sandboxes/{sid}/exec", exec_)
    app.router.add_post("/sandboxes/{sid}/pshell", exec_)
    app.router.add_delete("/sandboxes/{sid}", delete)
    runner = web.AppRunner(app)
    await runner.setup()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    async def stop():
        await runner.cleanup()

    return f"http://127.0.0.1:{port}", state, stop


@pytest.mark.asyncio
async def test_agent_routes_actions_to_sandbox_and_cleans_up():
    """When `sandbox_url` is set, the agent creates one pod per task, sends
    each action to `/pshell`, and deletes the pod on teardown."""
    base_url, state, stop = await _start_sandbox_stub()
    try:
        replies = [
            _fence("echo hello"),
            _fence(f"{SUBMIT_TOKEN}\ndone"),
        ]
        agent = CodingAgent(
            send_fn=_canned_send(replies),
            step_limit=5,
            sandbox_url=base_url,
            sandbox_spec={"cpu_cores": 0.1, "memory_mb": 256},
        )
        try:
            result = await agent.run(AgentContext(item={"prompt": "x"}))
        finally:
            await agent.aclose()
    finally:
        await stop()

    assert result.ok
    # One create / one exec / one delete for one task.
    assert len(state["created"]) == 1
    created_spec = state["created"][0]
    assert created_spec["cpu_cores"] == 0.1
    assert created_spec["memory_mb"] == 256
    assert created_spec["type"] == "kubernetes"   # default merged in
    assert len(state["execs"]) == 1
    sid, exec_body = state["execs"][0]
    assert exec_body["command"] == ["sh", "-c", "echo hello"]
    assert state["deleted"] == [sid]
    assert result.meta["sandbox_id"] == sid
    assert result.meta["cwd"] is None


@pytest.mark.asyncio
async def test_agent_sandbox_persistent_flag_routes_to_exec_endpoint():
    """`sandbox_persistent=False` switches the endpoint from /pshell to /exec.
    Distinguish by patching the stub to record which path was hit."""
    base_url, _, stop = await _start_sandbox_stub()
    hits: list[str] = []

    # Add a sniffing middleware on the same stub by re-running with a fresh app.
    app = web.Application()

    async def create(request: web.Request) -> web.Response:
        return web.json_response({"id": "sb-1"})

    async def hit(endpoint: str):
        async def _h(request: web.Request) -> web.Response:
            hits.append(endpoint)
            return web.json_response({
                "stdout": "", "stderr": "", "exit_code": 0, "duration": 0.0,
            })
        return _h

    app.router.add_post("/sandboxes", create)
    app.router.add_post("/sandboxes/{sid}/exec", await hit("exec"))
    app.router.add_post("/sandboxes/{sid}/pshell", await hit("pshell"))
    app.router.add_delete("/sandboxes/{sid}",
                          lambda r: web.json_response({"ok": True}))
    runner = web.AppRunner(app)
    await runner.setup()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        await stop()   # tear down the earlier stub; we use this one instead
        url = f"http://127.0.0.1:{port}"

        async def two_step():
            agent = CodingAgent(
                send_fn=_canned_send([_fence("true"), _fence(f"{SUBMIT_TOKEN}\n")]),
                step_limit=3,
                sandbox_url=url,
                sandbox_persistent=False,
            )
            try:
                await agent.run(AgentContext(item={"prompt": "x"}))
            finally:
                await agent.aclose()

        await two_step()
        assert hits == ["exec"]
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_agent_sandbox_exec_nonzero_exit_is_surfaced():
    """A non-zero `exit_code` from the sandbox surfaces in the observation."""
    app = web.Application()

    async def create(request: web.Request) -> web.Response:
        return web.json_response({"id": "sb-1"})

    async def exec_(request: web.Request) -> web.Response:
        return web.json_response({
            "stdout": "", "stderr": "boom\n", "exit_code": 1, "duration": 0.0,
        })

    app.router.add_post("/sandboxes", create)
    app.router.add_post("/sandboxes/{sid}/pshell", exec_)
    app.router.add_post("/sandboxes/{sid}/exec", exec_)
    app.router.add_delete("/sandboxes/{sid}",
                          lambda r: web.json_response({"ok": True}))
    runner = web.AppRunner(app)
    await runner.setup()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        observed: list[str] = []

        async def send(messages):
            if not observed:
                observed.append("first")
                return _fence("false")   # stub will return exit_code=1
            # Capture the observation message to assert on it.
            observed.append(messages[-1]["content"])
            return _fence(f"{SUBMIT_TOKEN}\n")

        agent = CodingAgent(
            send_fn=send, step_limit=3,
            sandbox_url=f"http://127.0.0.1:{port}",
        )
        try:
            result = await agent.run(AgentContext(item={"prompt": "x"}))
        finally:
            await agent.aclose()

        assert result.ok   # submitted final answer
        # Second send-call observed the failed action.
        obs_msg = observed[-1]
        assert "returncode: 1" in obs_msg
        assert "boom" in obs_msg
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_agent_grades_via_correctness_hook_in_bench():
    """End-to-end through BenchRunner + AgentWorkloadType + correctness_hook."""

    async def send(messages):
        # Inspect the task in the user message to pick an answer.
        task = messages[1]["content"]
        if "17 * 23" in task:
            return _fence(f"{SUBMIT_TOKEN}\n391")
        return _fence(f"{SUBMIT_TOKEN}\nwrong")

    items = [
        {"prompt": "Compute 17 * 23 and print the result.", "reference": "391"},
        {"prompt": "Compute 2 + 2.", "reference": "4"},  # agent will say 'wrong'
    ]
    cfg = BenchConfig(
        workload_type=AgentWorkloadType(CodingAgent(send_fn=send, step_limit=3)),
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
    assert "steps" in summary["workload_metrics"]

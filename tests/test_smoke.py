"""End-to-end smoke tests against a localhost aiohttp stub."""

import json
import os
import tempfile

import pytest

from benchmaker import (
    BenchConfig,
    BenchRunner,
    ClosedLoop,
    ConstantRPS,
    FunctionMonitor,
    HttpWorkloadType,
    JsonlWorkload,
    OpenAIChatWorkloadType,
    PoissonRPS,
    PrometheusMonitor,
    Ramp,
    SandboxWorkloadType,
    StaticWorkload,
    Sweep,
    parse_prometheus,
    parse_rate_spec,
)
from benchmaker.core.load import ConstantRPS as _C


@pytest.mark.asyncio
async def test_constant_rps_hits_target(stub_server: str):
    wt = HttpWorkloadType(url=f"{stub_server}/hello")
    runner = BenchRunner(BenchConfig(
        workload_type=wt, load=ConstantRPS(rps=50, duration_s=1.0),
        progress_every_s=0,
    ))
    result = await runner.run()
    assert 30 <= result.summary["total_requests"] <= 70
    assert result.summary["success"] == result.summary["total_requests"]


@pytest.mark.asyncio
async def test_closed_loop_concurrency(stub_server: str):
    wt = HttpWorkloadType(url=f"{stub_server}/slow")
    runner = BenchRunner(BenchConfig(
        workload_type=wt, load=ClosedLoop(concurrency=4, duration_s=1.0),
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["total_requests"] > 30


@pytest.mark.asyncio
async def test_poisson_runs(stub_server: str):
    wt = HttpWorkloadType(url=f"{stub_server}/hello")
    runner = BenchRunner(BenchConfig(
        workload_type=wt, load=PoissonRPS(rps=30, duration_s=1.0),
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["total_requests"] > 5


@pytest.mark.asyncio
async def test_ramp(stub_server: str):
    wt = HttpWorkloadType(url=f"{stub_server}/hello")
    runner = BenchRunner(BenchConfig(
        workload_type=wt, load=Ramp(start_rps=5, end_rps=50, duration_s=1.0),
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["total_requests"] > 5


@pytest.mark.asyncio
async def test_sweep(stub_server: str):
    wt = HttpWorkloadType(url=f"{stub_server}/hello")
    sweep = Sweep([
        ConstantRPS(rps=20, duration_s=0.5),
        ConstantRPS(rps=50, duration_s=0.5),
    ], labels=["low", "high"])
    runner = BenchRunner(BenchConfig(
        workload_type=wt, load=sweep, progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["total_requests"] > 10


@pytest.mark.asyncio
async def test_error_status(stub_server: str):
    wt = HttpWorkloadType(url=f"{stub_server}/fail")
    runner = BenchRunner(BenchConfig(
        workload_type=wt, load=ConstantRPS(rps=20, duration_s=0.5),
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["failed"] == result.summary["total_requests"]
    assert result.summary["status_codes"].get(500, 0) > 0


@pytest.mark.asyncio
async def test_hooks_are_called(stub_server: str):
    counter = {"pre": 0, "post": 0}

    def pre(req):
        counter["pre"] += 1
        req.headers["X-Probe"] = "1"
        return req

    def post(req, resp, sample):
        counter["post"] += 1
        sample.extra["my_metric"] = 42.0
        return sample

    wt = HttpWorkloadType(url=f"{stub_server}/hello")
    runner = BenchRunner(BenchConfig(
        workload_type=wt, load=ConstantRPS(rps=10, duration_s=0.5),
        pre_hooks=[pre], post_hooks=[post], progress_every_s=0,
    ))
    result = await runner.run()
    n = result.summary["total_requests"]
    assert counter["pre"] == n and counter["post"] == n
    assert "my_metric" in result.summary["workload_metrics"]
    assert result.summary["workload_metrics"]["my_metric"]["mean"] == 42.0


@pytest.mark.asyncio
async def test_llm_streaming_metrics(stub_server: str):
    wt = OpenAIChatWorkloadType(
        url=f"{stub_server}/v1/chat/completions",
        model="stub",
        max_tokens=8,
    )
    workload = StaticWorkload(items=["hi"])
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=5, duration_s=0.6),
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["success"] > 0
    wm = result.summary["workload_metrics"]
    assert "ttft_s" in wm
    assert "tokens_out" in wm
    assert wm["tokens_out"]["mean"] == 5.0
    assert "itl_ms_mean" in wm
    assert "tokens_per_s" in wm


@pytest.mark.asyncio
async def test_http_dataset_dict_as_json_body(stub_server: str):
    """HttpWorkloadType should treat a plain dict item as the JSON body."""
    wt = HttpWorkloadType(url=f"{stub_server}/echo", method="POST")
    workload = StaticWorkload(items=[{"a": 1}, {"a": 2}, {"a": 3}])
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=20, duration_s=0.5),
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["success"] > 0
    assert result.summary["bytes_sent"] > 0  # JSON bodies were sent


@pytest.mark.asyncio
async def test_http_dataset_request_override(stub_server: str):
    """A dict with Request-like keys customizes the request."""
    wt = HttpWorkloadType(url=f"{stub_server}/echo", method="POST")
    workload = StaticWorkload(items=[
        {"json": {"x": 1}, "headers": {"X-A": "1"}},
        {"json": {"x": 2}, "headers": {"X-A": "2"}},
    ])
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=10, duration_s=0.4),
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["success"] > 0


@pytest.mark.asyncio
async def test_jsonl_workload(stub_server: str, tmp_path):
    path = tmp_path / "prompts.jsonl"
    with open(path, "w") as f:
        for i in range(3):
            f.write(json.dumps({"prompt": f"p{i}"}) + "\n")

    wt = HttpWorkloadType(url=f"{stub_server}/echo", method="POST")
    workload = JsonlWorkload(path=str(path), loop=True)
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=15, duration_s=0.4),
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["success"] > 0


@pytest.mark.asyncio
async def test_workload_exhaustion_halts_run(stub_server: str):
    wt = HttpWorkloadType(url=f"{stub_server}/hello")
    workload = StaticWorkload(items=[None], max_items=3)
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=100, duration_s=5.0),  # would fire 500 without exhaustion
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["total_requests"] == 3


def test_parse_rate_spec_variants():
    assert isinstance(parse_rate_spec("100"), _C)
    assert isinstance(parse_rate_spec(100), _C)
    assert isinstance(parse_rate_spec("100rps"), _C)
    from benchmaker.core.load import PoissonRPS as _P, ClosedLoop as _CL, Ramp as _R, Sweep as _S
    assert isinstance(parse_rate_spec("poisson:50"), _P)
    assert isinstance(parse_rate_spec("closed:32"), _CL)
    assert isinstance(parse_rate_spec("concurrency:8"), _CL)
    assert isinstance(parse_rate_spec("ramp:10..200:5s"), _R)
    assert isinstance(parse_rate_spec("ramp-poisson:1..50:10s"), _R)
    assert isinstance(parse_rate_spec("sweep:10,50,100@2s"), _S)


@pytest.mark.asyncio
async def test_function_monitor_ticks(stub_server: str):
    """A FunctionMonitor should tick periodically and aggregate values."""
    counter = {"n": 0}

    def tick():
        counter["n"] += 1
        return {"my_gauge": float(counter["n"]) * 10.0}

    wt = HttpWorkloadType(url=f"{stub_server}/hello")
    monitor = FunctionMonitor(fn=tick, name="my-mon", interval_s=0.1)
    runner = BenchRunner(BenchConfig(
        workload_type=wt, load=ConstantRPS(rps=20, duration_s=0.5),
        monitors=[monitor], progress_every_s=0,
    ))
    result = await runner.run()
    assert "monitors" in result.summary
    assert "my-mon" in result.summary["monitors"]
    mon = result.summary["monitors"]["my-mon"]
    assert mon["tick_count"] >= 3
    assert "my_gauge" in mon["metrics"]
    # last value > first value (monotonic counter)
    assert mon["metrics"]["my_gauge"]["last"] > mon["metrics"]["my_gauge"]["first"]


@pytest.mark.asyncio
async def test_prometheus_monitor_scrapes_stub(stub_server: str):
    wt = HttpWorkloadType(url=f"{stub_server}/hello")
    mon = PrometheusMonitor(
        url=f"{stub_server}/metrics",
        metric_names={"stub_requests_total", "stub_kv_cache_usage", "stub_gpu_util"},
        interval_s=0.1,
        name="vllm",
    )
    runner = BenchRunner(BenchConfig(
        workload_type=wt, load=ConstantRPS(rps=20, duration_s=0.5),
        monitors=[mon], progress_every_s=0,
    ))
    result = await runner.run()
    assert "monitors" in result.summary
    vllm = result.summary["monitors"]["vllm"]
    assert vllm["tick_count"] >= 3
    # Counter should monotonically increase across ticks.
    counter_stats = vllm["metrics"]["stub_requests_total"]
    assert counter_stats["last"] > counter_stats["first"]
    # Both GPU labels should be present.
    assert any("stub_gpu_util" in k and 'gpu="0"' in k for k in vllm["metrics"])
    assert any("stub_gpu_util" in k and 'gpu="1"' in k for k in vllm["metrics"])
    # Bare metric without labels works.
    assert "stub_kv_cache_usage" in vllm["metrics"]


def test_parse_prometheus_labelled_vs_summed():
    text = (
        "# HELP foo bar\n"
        "# TYPE foo gauge\n"
        'foo{a="1"} 10\n'
        'foo{a="2"} 5\n'
        "bare_metric 7.5\n"
    )
    labelled = parse_prometheus(text)
    assert labelled.get('foo{a="1"}') == 10.0
    assert labelled.get('foo{a="2"}') == 5.0
    assert labelled.get("bare_metric") == 7.5
    summed = parse_prometheus(text, labelled_keys=False)
    assert summed["foo"] == 15.0
    assert summed["bare_metric"] == 7.5


@pytest.mark.asyncio
async def test_monitor_failure_does_not_kill_bench(stub_server: str):
    """A monitor that raises should not crash the benchmark."""
    def bad_tick():
        raise RuntimeError("intentional")

    wt = HttpWorkloadType(url=f"{stub_server}/hello")
    monitor = FunctionMonitor(fn=bad_tick, name="bad", interval_s=0.1)
    runner = BenchRunner(BenchConfig(
        workload_type=wt, load=ConstantRPS(rps=10, duration_s=0.3),
        monitors=[monitor], progress_every_s=0,
    ))
    result = await runner.run()
    # Benchmark should have completed normally.
    assert result.summary["success"] > 0
    # Monitor should have recorded zero ticks (every one raised).
    mon = result.summary.get("monitors", {}).get("bad")
    assert mon is None or mon["tick_count"] == 0


def test_dotenv_loading_and_interpolation(tmp_path, monkeypatch):
    from benchmaker.env import interpolate, load_dotenv
    env_path = tmp_path / ".env"
    env_path.write_text(
        'OPENAI_API_KEY=sk-abc123\n'
        'OPENAI_API_BASE_URL="https://example.com/v1/"\n'
        '# a comment\n'
        'OPENAI_COMPATIBLE_MODEL=meta-llama/Llama-3.1-8B-Instruct\n'
        'export QUOTED_WITH_HASH="value # not a comment"\n'
    )
    # Ensure these aren't already set:
    for k in ("OPENAI_API_KEY", "OPENAI_API_BASE_URL", "OPENAI_COMPATIBLE_MODEL", "QUOTED_WITH_HASH"):
        monkeypatch.delenv(k, raising=False)
    loaded = load_dotenv(str(env_path))
    assert loaded["OPENAI_API_KEY"] == "sk-abc123"
    assert loaded["OPENAI_API_BASE_URL"] == "https://example.com/v1/"
    assert loaded["OPENAI_COMPATIBLE_MODEL"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert loaded["QUOTED_WITH_HASH"] == "value # not a comment"

    cfg = {
        "url": "${OPENAI_API_BASE_URL}chat/completions",
        "model": "${OPENAI_COMPATIBLE_MODEL}",
        "api_key": "${OPENAI_API_KEY}",
        "fallback": "${NOT_SET:-default-value}",
        "nested": {"list": ["${OPENAI_COMPATIBLE_MODEL}", "static"]},
    }
    out = interpolate(cfg)
    assert out["url"] == "https://example.com/v1/chat/completions"
    assert out["fallback"] == "default-value"
    assert out["nested"]["list"] == ["meta-llama/Llama-3.1-8B-Instruct", "static"]


def test_dotenv_does_not_override_existing(monkeypatch, tmp_path):
    from benchmaker.env import load_dotenv
    monkeypatch.setenv("EXISTING", "already-set")
    env_path = tmp_path / ".env"
    env_path.write_text("EXISTING=from-dotenv\n")
    load_dotenv(str(env_path))
    import os
    assert os.environ["EXISTING"] == "already-set"


def test_interpolate_missing_var_raises(monkeypatch):
    from benchmaker.env import interpolate
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    with pytest.raises(KeyError):
        interpolate("${DEFINITELY_NOT_SET}")


def test_openai_from_env_constructs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE_URL", "https://example.com/v1/")
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    wt = OpenAIChatWorkloadType.from_env(dotenv_path=None)
    assert wt._url == "https://example.com/v1/chat/completions"
    assert wt._model == "test-model"
    # Trailing slash handling — should produce exactly one slash:
    assert wt._url.count("/v1/chat") == 1
    # Authorization header present:
    assert any("Bearer sk-test" in v for v in wt._headers.values())


def test_build_config_yaml_shape():
    from benchmaker.config import build_config
    cfg = build_config({
        "workload_type": {"type": "http", "url": "http://x"},
        "workload": {"type": "static", "items": [{"a": 1}]},
        "load": "10",
        "duration": "1s",
    })
    assert cfg.workload_type.name == "http"
    assert cfg.workload.name == "static"


@pytest.mark.asyncio
async def test_sandbox_exec_lazy_creates_and_cleans_up(stub_server: str, sandbox_state):
    wt = SandboxWorkloadType(base_url=stub_server, default_command="echo hi")
    runner = BenchRunner(BenchConfig(
        workload_type=wt,
        load=ConstantRPS(rps=20, duration_s=0.5),
        progress_every_s=0,
    ))
    result = await runner.run()

    # Exactly one sandbox was created (lazy + locked), and it was deleted on aclose.
    assert len(sandbox_state["created"]) == 1
    sid, body = sandbox_state["created"][0]
    assert body["image"] == "alpine:3.20"
    assert sandbox_state["deleted"] == [sid]

    # All requests hit /exec on that one sandbox.
    assert result.summary["success"] == result.summary["total_requests"]
    assert all(s == sid and ep == "exec" for s, _b, ep in sandbox_state["exec_calls"])

    wm = result.summary["workload_metrics"]
    assert wm["exit_code"]["mean"] == 0.0
    assert wm["server_duration_s"]["mean"] > 0
    assert wm["stdout_bytes"]["mean"] > 0


@pytest.mark.asyncio
async def test_sandbox_exec_item_forms(stub_server: str, sandbox_state):
    wt = SandboxWorkloadType(base_url=stub_server)
    workload = StaticWorkload(items=[
        "echo string-form",                          # str
        ["echo", "argv-form"],                       # list[str]
        {"command": "echo dict-form", "env": {"X": "1"}},  # dict
    ])
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload,
        load=ConstantRPS(rps=10, duration_s=0.5),
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["success"] > 0
    # Verify the wire shape of each command form.
    cmds = [body["command"] for _sid, body, _ep in sandbox_state["exec_calls"]]
    assert ["sh", "-c", "echo string-form"] in cmds
    assert ["echo", "argv-form"] in cmds
    assert ["sh", "-c", "echo dict-form"] in cmds


@pytest.mark.asyncio
async def test_sandbox_exec_nonzero_marks_failed(stub_server: str, sandbox_state):
    wt = SandboxWorkloadType(base_url=stub_server, default_command="fail please")
    runner = BenchRunner(BenchConfig(
        workload_type=wt,
        load=ConstantRPS(rps=10, duration_s=0.3),
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["failed"] == result.summary["total_requests"]
    # workload_metrics only aggregates from successful samples; check raw samples instead.
    assert all(s.extra.get("exit_code") == 1.0 for s in result.samples)
    assert all("exit_code=1" in (s.error or "") for s in result.samples)


@pytest.mark.asyncio
async def test_sandbox_persistent_uses_pshell(stub_server: str, sandbox_state):
    wt = SandboxWorkloadType(
        base_url=stub_server, persistent=True, default_command="pwd"
    )
    runner = BenchRunner(BenchConfig(
        workload_type=wt,
        load=ConstantRPS(rps=10, duration_s=0.3),
        progress_every_s=0,
    ))
    await runner.run()
    assert all(ep == "pshell" for _s, _b, ep in sandbox_state["exec_calls"])


@pytest.mark.asyncio
async def test_sandbox_uses_existing_id(stub_server: str, sandbox_state):
    wt = SandboxWorkloadType(
        base_url=stub_server, sandbox_id="sb-existing", default_command="echo hi"
    )
    runner = BenchRunner(BenchConfig(
        workload_type=wt,
        load=ConstantRPS(rps=5, duration_s=0.3),
        progress_every_s=0,
    ))
    await runner.run()
    # Should never call create / delete for a user-provided sandbox.
    assert sandbox_state["created"] == []
    assert sandbox_state["deleted"] == []
    assert all(s == "sb-existing" for s, _b, _e in sandbox_state["exec_calls"])


@pytest.mark.asyncio
async def test_sandbox_create_mode(stub_server: str, sandbox_state):
    wt = SandboxWorkloadType(
        base_url=stub_server,
        operation="create",
        spec={"type": "kubernetes", "image": "alpine:3.20", "command": ["sh", "-c", "sleep 60"]},
        ttl_seconds=120,
    )
    runner = BenchRunner(BenchConfig(
        workload_type=wt,
        load=ConstantRPS(rps=10, duration_s=0.3),
        progress_every_s=0,
    ))
    result = await runner.run()
    n = result.summary["total_requests"]
    assert n > 0
    assert len(sandbox_state["created"]) == n
    # ttl_seconds was propagated into every create body.
    assert all(body.get("ttl_seconds") == 120 for _sid, body in sandbox_state["created"])
    assert result.summary["workload_metrics"]["server_created"]["mean"] == 1.0


@pytest.mark.asyncio
async def test_sandbox_node_prefix(stub_server: str, sandbox_state):
    wt = SandboxWorkloadType(
        base_url=stub_server,
        endpoint_prefix="/native/sandboxes",
        default_command="echo hi",
    )
    runner = BenchRunner(BenchConfig(
        workload_type=wt,
        load=ConstantRPS(rps=5, duration_s=0.3),
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.summary["success"] > 0
    assert len(sandbox_state["created"]) == 1


def test_sandbox_yaml_build():
    from benchmaker.config import build_config
    cfg = build_config({
        "workload_type": {
            "type": "sandbox",
            "base_url": "http://localhost:8080",
            "default_command": "echo hi",
        },
        "load": "10",
        "duration": "1s",
    })
    assert cfg.workload_type.name == "sandbox"
    assert isinstance(cfg.workload_type, SandboxWorkloadType)


def test_sandbox_lifecycle_rejects_sandbox_id():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        SandboxWorkloadType(
            base_url="http://x", operation="lifecycle", sandbox_id="sb-1",
        )


@pytest.mark.asyncio
async def test_sandbox_lifecycle_fires_create_exec_delete(stub_server: str, sandbox_state):
    wt = SandboxWorkloadType(
        base_url=stub_server,
        operation="lifecycle",
        spec={"type": "kubernetes", "image": "alpine:3.20",
              "command": ["sh", "-c", "sleep 60"]},
        ttl_seconds=60,
    )
    runner = BenchRunner(BenchConfig(
        workload_type=wt,
        workload=StaticWorkload(items=["echo hi", "uname -a"]),
        load=ConstantRPS(rps=10, duration_s=0.4),
        progress_every_s=0,
    ))
    result = await runner.run()

    n = result.summary["total_requests"]
    assert n > 0
    # one create + one delete per ticket; exec recorded for every ticket
    assert len(sandbox_state["created"]) == n
    assert len(sandbox_state["deleted"]) == n
    assert len(sandbox_state["exec_calls"]) == n
    # ttl propagated into each create body
    assert all(body.get("ttl_seconds") == 60 for _sid, body in sandbox_state["created"])
    # Every created sandbox is also deleted
    created_ids = {sid for sid, _ in sandbox_state["created"]}
    assert created_ids == set(sandbox_state["deleted"])
    # Sample fields
    for s in result.samples:
        assert s.ok
        assert s.extra["create_s"] >= 0.0
        assert s.extra["exec_s"] >= 0.0
        assert s.extra["delete_s"] >= 0.0
        assert s.extra["lifecycle_s"] >= s.extra["create_s"]  # total >= any leg
        assert s.extra["exit_code"] == 0.0
        assert s.meta.get("sandbox_id", "").startswith("sb-test-")


@pytest.mark.asyncio
async def test_sandbox_lifecycle_nonzero_exit_marks_failed(stub_server: str, sandbox_state):
    wt = SandboxWorkloadType(
        base_url=stub_server,
        operation="lifecycle",
    )
    runner = BenchRunner(BenchConfig(
        workload_type=wt,
        workload=StaticWorkload(items=["fail-me"]),
        load=ConstantRPS(rps=5, duration_s=0.3),
        progress_every_s=0,
    ))
    result = await runner.run()

    n = result.summary["total_requests"]
    assert n > 0
    # exec returned exit_code=1 → sample.ok=False, BUT delete still ran (no leak)
    assert all(not s.ok for s in result.samples)
    assert all(s.extra.get("exit_code") == 1.0 for s in result.samples)
    assert all("exit_code=1" in (s.error or "") for s in result.samples)
    assert len(sandbox_state["deleted"]) == n


@pytest.mark.asyncio
async def test_sandbox_lifecycle_runs_post_hooks(stub_server: str, sandbox_state):
    seen: list[str] = []

    def tag(req, resp, sample):
        # exec request is what gets passed in
        seen.append(req.meta.get("sandbox_operation") or "")
        sample.meta["tagged"] = True
        return sample

    wt = SandboxWorkloadType(base_url=stub_server, operation="lifecycle")
    runner = BenchRunner(BenchConfig(
        workload_type=wt,
        workload=StaticWorkload(items=["echo hi"]),
        load=ConstantRPS(rps=5, duration_s=0.2),
        post_hooks=[tag],
        progress_every_s=0,
    ))
    result = await runner.run()
    assert result.samples
    assert all(s.meta.get("tagged") is True for s in result.samples)
    assert all(stage == "exec" for stage in seen)

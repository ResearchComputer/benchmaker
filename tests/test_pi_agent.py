"""Unit tests for the pi (pi-coding-agent) harbor adapters.

Covers the pure config/command helpers and a live round-trip of the Mode-2
localhost exec bridge against a fake environment. The actual pi process + a real
harbor environment are not exercised here.
"""

import json
from dataclasses import dataclass

import pytest

from benchmaker.swebench import pi_agent as P


# --------------------------- model resolution ------------------------------ #

def test_resolve_model_strips_prefix_and_slash():
    base, model, key = P.resolve_model(
        {"OPENAI_API_BASE_URL": "https://api/v1/", "OPENAI_API_KEY": "sk-1",
         "OPENAI_COMPATIBLE_MODEL": "openai/zai-org/GLM"}, None)
    assert base == "https://api/v1"           # trailing slash dropped
    assert model == "zai-org/GLM"             # litellm "openai/" prefix stripped
    assert key == "sk-1"


def test_resolve_model_prefers_explicit_model_name():
    base, model, _ = P.resolve_model(
        {"OPENAI_API_BASE_URL": "http://h/v1"}, "my-model")
    assert model == "my-model" and base == "http://h/v1"


def test_resolve_model_expands_harbor_env_templates(monkeypatch):
    # harbor passes sensitive AgentConfig.env values as "${OPENAI_API_KEY}"
    # templates; resolve_model must expand them from os.environ (else pi sends
    # the literal template as its bearer and the endpoint 401s).
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    base, model, key = P.resolve_model(
        {"OPENAI_API_BASE_URL": "http://h/v1", "OPENAI_API_KEY": "${OPENAI_API_KEY}",
         "OPENAI_COMPATIBLE_MODEL": "m"}, None)
    assert key == "sk-real" and base == "http://h/v1" and model == "m"


def test_resolve_model_requires_base_and_model(monkeypatch):
    # Clear ambient env (the repo .env sets OPENAI_*) so the fallback finds nothing.
    for k in ("OPENAI_API_BASE_URL", "OPENAI_API_BASE", "OPENAI_BASE_URL",
              "OPENAI_COMPATIBLE_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError):
        P.resolve_model({"OPENAI_API_KEY": "x"}, None)   # no base url / model


# --------------------------- models.json ----------------------------------- #

def test_models_json_schema():
    cfg = json.loads(P.models_json("http://h/v1", "m", context_window=4096, max_tokens=512))
    prov = cfg["providers"]["bench"]
    assert prov["api"] == "openai-completions"
    assert prov["baseUrl"] == "http://h/v1"
    assert prov["apiKey"] == "$OPENAI_API_KEY"     # secret stays out of the file
    assert prov["models"][0] == {"id": "m", "contextWindow": 4096, "maxTokens": 512}


# --------------------------- command building ------------------------------ #

def test_pi_command_passes_prompt_by_file_and_extra_args():
    cmd = P.pi_command("m", "/tmp/t.txt", extra_args=["--yolo"])
    assert cmd == 'pi --mode json --provider bench --model m --yolo "$(cat /tmp/t.txt)"'


def test_build_prompt_includes_problem():
    p = P.build_prompt({"instance_id": "x__y-1", "repo": "x/y",
                        "problem_statement": "boom"})
    assert "boom" in p and "x__y-1" in p and "/testbed" in p


def test_b64_write_roundtrips_offline():
    import base64
    cmd = P._b64_write("/a/b.txt", "héllo\nworld")
    # the embedded base64 decodes back to the original content
    blob = cmd.split("echo ", 1)[1].split(" |", 1)[0]
    assert base64.b64decode(blob).decode("utf-8") == "héllo\nworld"
    assert "mkdir -p" in cmd and "/a/b.txt" in cmd


# --------------------------- exec bridge (Mode 2 crux) --------------------- #

@dataclass
class _Res:
    return_code: int
    stdout: str
    stderr: str


class _FakeEnv:
    def __init__(self):
        self.calls = []

    async def exec(self, command, cwd=None, timeout_sec=None):
        self.calls.append({"command": command, "cwd": cwd, "timeout": timeout_sec})
        return _Res(0, "ok-out", "")


async def test_exec_bridge_forwards_to_environment():
    import aiohttp

    env = _FakeEnv()
    bridge = P._ExecBridge(env, cwd="/testbed", exec_timeout_s=42.0)
    await bridge.start()
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"{bridge.url}/exec",
                                 json={"command": "ls -la", "timeout": 7}) as r:
                body = await r.json()
    finally:
        await bridge.stop()

    assert body == {"return_code": 0, "stdout": "ok-out", "stderr": ""}
    assert bridge.count == 1
    # command is anchored at cwd and the per-call timeout is honoured.
    assert env.calls[0]["command"] == "cd /testbed && ls -la"
    assert env.calls[0]["timeout"] == 7


async def test_exec_bridge_surfaces_environment_error():
    import aiohttp

    class _Boom(_FakeEnv):
        async def exec(self, command, cwd=None, timeout_sec=None):
            raise RuntimeError("pod gone")

    bridge = P._ExecBridge(_Boom(), cwd="/testbed", exec_timeout_s=10.0)
    await bridge.start()
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"{bridge.url}/exec", json={"command": "x"}) as r:
                body = await r.json()
    finally:
        await bridge.stop()
    assert body["return_code"] == -1 and "pod gone" in body["stderr"]

"""Unit tests for the pi (pi-coding-agent) harbor adapters.

Covers the pure config/command helpers and a live round-trip of the Mode-2
localhost exec bridge against a fake environment. The actual pi process + a real
harbor environment are not exercised here.
"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

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


def test_pi_extension_args_caps_only_when_positive():
    assert P.pi_extension_args("/x/max_turns.js", 5) == ["--extension", "/x/max_turns.js"]
    assert P.pi_extension_args("/x/max_turns.js", 0) == []
    assert P.pi_extension_args("/x/max_turns.js", -1) == []
    assert P.pi_extension_args(None, 5) == []


def test_pi_agent_coerces_and_stores_max_turns(tmp_path):
    capped = P.PiContainerAgent(logs_dir=tmp_path, model_name="m", pi_max_turns="7")
    assert capped._pi_max_turns == 7
    default = P.PiContainerAgent(logs_dir=tmp_path, model_name="m")
    assert default._pi_max_turns == 0
    junk = P.PiContainerAgent(logs_dir=tmp_path, model_name="m", pi_max_turns="oops")
    assert junk._pi_max_turns == 0


def test_max_turns_extension_file_present_and_shaped():
    assert P.MAX_TURNS_EXT.exists()
    txt = P.MAX_TURNS_EXT.read_text()
    assert "PI_MAX_TURNS" in txt and "turn_end" in txt and "export default" in txt


# --------------------------- approve-flag NOTE (#1) ------------------------ #

def test_pi_command_note_does_not_recommend_an_approve_flag():
    # Issue #6 (1): current pi --mode json runs tools with no approve flag, and
    # -a/--yolo make it exit with "Unknown option". The NOTE must say so and must
    # no longer recommend passing one.
    doc = (P.pi_command.__doc__ or "").lower()
    assert "without any approve flag" in doc
    assert "unknown option" in doc
    assert "pass the appropriate" not in doc  # the old recommendation is gone


# --------------------------- pi-host tool routing (#3) --------------------- #

def test_pi_host_route_tools_names_and_coercion(tmp_path):
    mk = lambda **kw: P.PiHostAgent(logs_dir=tmp_path, model_name="m", **kw)
    assert mk()._host_tool_names() == ["bash"]                       # default
    assert mk(route_tools="all")._host_tool_names() == ["bash", "read", "write", "edit"]
    assert mk(route_tools=" ALL ")._host_tool_names() == ["bash", "read", "write", "edit"]
    assert mk(route_tools="")._host_tool_names() == ["bash"]         # empty -> default
    assert mk(route_tools="bash")._host_tool_names() == ["bash"]


def test_pi_host_stage_default_routes_bash_only(tmp_path):
    agent = P.PiHostAgent(logs_dir=tmp_path, model_name="m")
    home = tmp_path / "h"
    agent._stage_host_config(home, "http://h/v1", "m")
    ad = home / ".pi" / "agent"
    settings = json.loads((ad / "settings.json").read_text())
    assert settings["tools"] == ["bash"]
    assert settings["defaultProvider"] == "bench" and settings["defaultModel"] == "m"
    exts = {p.name for p in (ad / "extensions").iterdir()}
    assert "remote_exec.js" in exts and "remote_exec_all.js" not in exts
    assert "register_provider.js" in exts          # provider registration (#2)
    assert (ad / "models.json").exists()           # fallback still staged


def test_pi_host_stage_all_routes_four_tools(tmp_path):
    agent = P.PiHostAgent(logs_dir=tmp_path, model_name="m", route_tools="all")
    home = tmp_path / "h"
    agent._stage_host_config(home, "http://h/v1", "m")
    ad = home / ".pi" / "agent"
    settings = json.loads((ad / "settings.json").read_text())
    assert settings["tools"] == ["bash", "read", "write", "edit"]
    exts = {p.name for p in (ad / "extensions").iterdir()}
    # exactly one bash-registering extension (loading both double-registers bash)
    assert "remote_exec_all.js" in exts and "remote_exec.js" not in exts
    assert "register_provider.js" in exts


def test_pi_host_env_inlines_api_key(tmp_path):
    # pi does NOT expand the "$OPENAI_API_KEY" ref in provider config (same as
    # container mode), so the resolved key must be inlined for register_provider.js
    # via PI_BENCH_API_KEY_REF — otherwise pi sends an empty bearer -> 401.
    agent = P.PiHostAgent(logs_dir=tmp_path, model_name="m")
    env = agent._pi_env(home=tmp_path / "h", base_url="http://h/v1", model="m",
                        api_key="sk-secret-xyz", bridge_url="http://127.0.0.1:9/")
    assert env["PI_BENCH_API_KEY_REF"] == "sk-secret-xyz"
    assert env["OPENAI_API_KEY"] == "sk-secret-xyz"
    assert env["PI_BENCH_BASE_URL"] == "http://h/v1"
    assert env["PI_BENCH_MODEL"] == "m"


def test_host_extensions_present_and_shaped():
    # provider-registration extension (#2)
    assert P.REGISTER_PROVIDER_EXT.exists()
    rp = P.REGISTER_PROVIDER_EXT.read_text()
    assert "registerProvider" in rp and "openai-completions" in rp
    assert "$OPENAI_API_KEY" in rp and "PI_BENCH_PROVIDER" in rp
    # all-tools routing extension (#3)
    assert P.REMOTE_EXEC_ALL_EXT.exists()
    ea = P.REMOTE_EXEC_ALL_EXT.read_text()
    for name in ('name: "bash"', 'name: "read"', 'name: "write"', 'name: "edit"'):
        assert name in ea
    # canonical pi tool-result shape (content/details), not legacy {output,exitCode}
    assert "content:" in ea and "details:" in ea


def test_remote_exec_uses_canonical_result_shape():
    txt = P.REMOTE_EXEC_EXT.read_text()
    assert "content:" in txt and "details:" in txt


def test_pi_ext_node_smoke():
    """Functionally exercise the JS extensions through a real shell round-trip."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    harness = Path(__file__).resolve().parent / "pi_ext_smoke.mjs"
    ext_dir = Path(P.__file__).resolve().parent / "pi_ext"
    res = subprocess.run([node, str(harness), str(ext_dir)],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, (res.stdout or "") + (res.stderr or "")


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
    import httpx2

    env = _FakeEnv()
    bridge = P._ExecBridge(env, cwd="/testbed", exec_timeout_s=42.0)
    await bridge.start()
    try:
        async with httpx2.AsyncClient() as sess:
            r = await sess.post(f"{bridge.url}/exec",
                                 json={"command": "ls -la", "timeout": 7})
            body = r.json()
    finally:
        await bridge.stop()

    assert body == {"return_code": 0, "stdout": "ok-out", "stderr": ""}
    assert bridge.count == 1
    # Command is anchored at cwd and wrapped in a LOGIN shell (bash -lc), so the
    # per-instance SWE-bench env setup in /etc/profile + ~/.bashrc runs first
    # (e.g. `conda activate testbed`). A bare `sh -c` would skip those and run
    # against base Python -> ModuleNotFoundError (issue #12). The per-call
    # timeout is honoured.
    assert env.calls[0]["command"] == "bash -lc 'cd /testbed && ls -la'"
    assert env.calls[0]["timeout"] == 7


async def test_exec_bridge_uses_login_shell_for_env_activation():
    """Regression for issue #12: routed commands must run via a login shell.

    SWE-bench images activate the per-instance ``testbed`` conda env (and set
    PATH/venv) from the login files (``/etc/profile`` / ``~/.bashrc``), which a
    non-login ``sh -c`` never sources. The bridge must therefore wrap every
    routed command in ``bash -lc`` (mirroring PiContainerAgent), with the inner
    ``cd <cwd> && <command>`` quoted as a single argument.
    """
    import httpx2
    import shlex

    env = _FakeEnv()
    bridge = P._ExecBridge(env, cwd="/testbed", exec_timeout_s=42.0)
    await bridge.start()
    try:
        async with httpx2.AsyncClient() as sess:
            # A command with shell metacharacters must survive the wrapping intact.
            r = await sess.post(f"{bridge.url}/exec",
                                 json={"command": "python -m pytest && echo 'done'"})
            r.json()
    finally:
        await bridge.stop()

    sent = env.calls[0]["command"]
    assert sent.startswith("bash -lc ")
    # The argument to `bash -lc` is exactly the cd+command, single-quoted safely.
    inner = sent[len("bash -lc "):]
    assert shlex.split(inner)[0] == "cd /testbed && python -m pytest && echo 'done'"


async def test_exec_bridge_login_shell_without_cwd():
    """With no cwd, still use a login shell (just the bare command, no `cd`)."""
    import httpx2

    env = _FakeEnv()
    bridge = P._ExecBridge(env, cwd="", exec_timeout_s=42.0)
    await bridge.start()
    try:
        async with httpx2.AsyncClient() as sess:
            r = await sess.post(f"{bridge.url}/exec", json={"command": "ls -la"})
            r.json()
    finally:
        await bridge.stop()

    assert env.calls[0]["command"] == "bash -lc 'ls -la'"


async def test_exec_bridge_surfaces_environment_error():
    import httpx2

    class _Boom(_FakeEnv):
        async def exec(self, command, cwd=None, timeout_sec=None):
            raise RuntimeError("pod gone")

    bridge = P._ExecBridge(_Boom(), cwd="/testbed", exec_timeout_s=10.0)
    await bridge.start()
    try:
        async with httpx2.AsyncClient() as sess:
            r = await sess.post(f"{bridge.url}/exec", json={"command": "x"})
            body = r.json()
    finally:
        await bridge.stop()
    assert body["return_code"] == -1 and "pod gone" in body["stderr"]

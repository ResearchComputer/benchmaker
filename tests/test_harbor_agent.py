"""Unit tests for BenchmakerHostAgent's environment exec bridge.

The host-loop agent runs the model loop on the host and routes each shell
``action`` into the per-instance environment via ``environment.exec``. Those
routed commands must run through a LOGIN shell so the SWE-bench image's
per-instance setup (e.g. ``conda activate testbed``, PATH/venv) — sourced from
``/etc/profile`` + ``~/.bashrc`` — runs before the command. See issue #12.
"""
from __future__ import annotations

import shlex

from benchmaker.swebench.harbor_agent import BenchmakerHostAgent


class _RecordingEnv:
    def __init__(self):
        self.calls = []

    async def exec(self, command, cwd=None, timeout_sec=None):
        self.calls.append({"command": command, "cwd": cwd, "timeout": timeout_sec})

        class _Res:
            return_code = 0
            stdout = "ok"
            stderr = ""

        return _Res()


async def test_executor_wraps_action_in_login_shell(tmp_path):
    """Regression for issue #12: routed actions must run via ``bash -lc``.

    A bare ``sh -c 'cd /testbed && ...'`` never sources the login files, so the
    agent runs against base Python with none of the repo's deps and the test
    suite fails with ModuleNotFoundError. The executor must anchor at cwd inside
    a login shell, mirroring PiContainerAgent / the pi-host bridge.
    """
    agent = BenchmakerHostAgent(logs_dir=tmp_path, model="m", api_base="http://x",
                                api_key="k")
    env = _RecordingEnv()
    executor = agent._make_executor(env)

    # An action with shell metacharacters must survive the wrapping intact.
    rc, out = await executor("python -m pytest && echo 'done'", 120.0)

    assert rc == 0 and out == "ok"
    sent = env.calls[0]["command"]
    assert sent.startswith("bash -lc ")
    inner = sent[len("bash -lc "):]
    assert shlex.split(inner)[0] == "cd /testbed && python -m pytest && echo 'done'"


async def test_executor_login_shell_without_cwd(tmp_path):
    """With no cwd, still run via a login shell (bare command, no ``cd``)."""
    agent = BenchmakerHostAgent(logs_dir=tmp_path, model="m", api_base="http://x",
                                api_key="k", cwd="")
    env = _RecordingEnv()
    executor = agent._make_executor(env)

    await executor("ls -la", 30.0)

    assert env.calls[0]["command"] == "bash -lc 'ls -la'"


async def test_executor_preserves_heredoc(tmp_path):
    """A trailing heredoc must survive the login-shell wrapping unbroken.

    The original bare-``cd`` form avoided a ``( ... )`` subshell precisely
    because parens break trailing heredocs; ``bash -lc <quoted>`` passes the
    whole command as one argument, so the heredoc parses normally.
    """
    agent = BenchmakerHostAgent(logs_dir=tmp_path, model="m", api_base="http://x",
                                api_key="k")
    env = _RecordingEnv()
    executor = agent._make_executor(env)

    action = "cat <<'EOF' > f.py\nprint('hi')\nEOF"
    await executor(action, 30.0)

    inner = shlex.split(env.calls[0]["command"][len("bash -lc "):])[0]
    assert inner == f"cd /testbed && {action}"

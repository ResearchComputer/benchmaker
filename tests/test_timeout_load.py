# tests/test_timeout_load.py
"""Unit tests for the command-timeout-under-load helpers."""
from __future__ import annotations

import math

from benchmaker.swebench.timeout_load import (
    effective_tau, task_survives, would_time_out,
)


def test_effective_tau():
    assert effective_tau(600, 1) == 600
    assert effective_tau(600, 20) == 30


def test_effective_tau_rejects_nonpositive_load():
    import pytest
    with pytest.raises(ValueError):
        effective_tau(600, 0)


def test_would_time_out_boundary():
    # strictly greater than the budget times out; equal does not
    assert would_time_out(5.0001, 5.0) is True
    assert would_time_out(5.0, 5.0) is False


def test_task_survives_uses_max_within_budget():
    assert task_survives(5.0, 5.0) is True
    assert task_survives(5.1, 5.0) is False
    assert task_survives(50.0, math.inf) is True


# append to tests/test_timeout_load.py
import json

from benchmaker.swebench.timeout_load import CommandTiming, recover_command_timings


def _log(tmp_path, rows):
    p = tmp_path / "pi-container.log"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_recover_command_timings(tmp_path):
    rows = [
        {"type": "message_start", "message": {"role": "user", "content": []}},
        {"type": "message_end",
         "message": {"role": "assistant", "timestamp": 1000,
                     "content": [{"type": "toolCall", "name": "bash"}]}},
        {"type": "message_end",
         "message": {"role": "toolResult", "timestamp": 3500, "content": []}},
        {"type": "message_end",
         "message": {"role": "assistant", "timestamp": 4000,
                     "content": [{"type": "toolCall", "name": "read"}]}},
        {"type": "message_end",
         "message": {"role": "toolResult", "timestamp": 4010, "content": []}},
    ]
    timings = recover_command_timings(_log(tmp_path, rows))
    assert timings == [CommandTiming("bash", 2.5), CommandTiming("read", 0.01)]


def test_recover_ignores_assistant_without_toolcall(tmp_path):
    rows = [
        {"type": "message_end",
         "message": {"role": "assistant", "timestamp": 1000,
                     "content": [{"type": "text", "text": "thinking"}]}},
        {"type": "message_end",
         "message": {"role": "toolResult", "timestamp": 9000, "content": []}},
    ]
    assert recover_command_timings(_log(tmp_path, rows)) == []

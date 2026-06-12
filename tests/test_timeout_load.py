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

# tests/test_pi_agent_timeout_injection.py
"""Unit tests for the load-factor timeout injection in _ExecBridge."""
from __future__ import annotations

from benchmaker.swebench import pi_agent as P


def _bridge(load_factor, inject_timeout_s):
    return P._ExecBridge(None, cwd="/testbed", exec_timeout_s=600.0,
                         load_factor=load_factor, inject_timeout_s=inject_timeout_s)


def test_no_injection_at_load_factor_1():
    b = _bridge(load_factor=1.0, inject_timeout_s=600.0)
    assert b._should_inject_timeout(10_000.0) is False


def test_injection_gate_at_load_factor_20():
    b = _bridge(load_factor=20.0, inject_timeout_s=600.0)  # tau = 30s
    assert b._should_inject_timeout(29.0) is False  # 29*20 = 580 < 600
    assert b._should_inject_timeout(31.0) is True   # 31*20 = 620 > 600


def test_injection_uses_inject_timeout_override():
    b = _bridge(load_factor=2.0, inject_timeout_s=10.0)  # tau = 5s
    assert b._should_inject_timeout(4.9) is False
    assert b._should_inject_timeout(5.1) is True


def _bridge_from_env():
    return P._ExecBridge(None, cwd="/testbed", exec_timeout_s=600.0)


def test_env_var_path_sets_load_factor(monkeypatch):
    monkeypatch.setenv("BENCH_LOAD_FACTOR", "10")
    monkeypatch.setenv("BENCH_INJECT_TIMEOUT_S", "100")
    b = _bridge_from_env()  # construct with no explicit load_factor/inject_timeout_s
    assert b._load_factor == 10.0
    assert b._inject_timeout_s == 100.0
    assert b._should_inject_timeout(11.0) is True   # 11 > 100/10 = 10
    assert b._should_inject_timeout(9.0) is False


def test_env_var_zero_load_factor_is_noop(monkeypatch):
    monkeypatch.setenv("BENCH_LOAD_FACTOR", "0")
    b = _bridge_from_env()
    assert b._should_inject_timeout(10_000.0) is False  # load_factor <= 1 is a strict no-op

# tests/test_swebench_replay_recipe.py
"""Tests for the swebench-replay recipe: registration, options, URL helper.

The full harbor run needs flash-sandbox and is not exercised here; we test the
pure pieces and that the command wires up."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from benchmaker.recipes import all_recipes, get
from benchmaker.recipes._factory import make_command
import benchmaker.recipes.swebench_replay as SR


def test_recipe_is_registered():
    assert "swebench-replay" in {r.name for r in all_recipes()}


def test_replay_url_localhost_and_reachable():
    assert SR._replay_url("127.0.0.1", 9100, None) == "http://127.0.0.1:9100/v1"
    # 0.0.0.0 bind is not dialable; fall back to loopback when no reachable host
    assert SR._replay_url("0.0.0.0", 9100, None) == "http://127.0.0.1:9100/v1"
    # explicit reachable host (container mode) wins
    assert SR._replay_url("0.0.0.0", 9100, "10.0.0.5") == "http://10.0.0.5:9100/v1"


def test_parse_sweep():
    assert SR._parse_concurrencies(None, 4) == [4]
    assert SR._parse_concurrencies("1,5,25", 4) == [1, 5, 25]
    assert SR._parse_concurrencies(" 2 , 2 ,3 ", 4) == [2, 2, 3]


def test_command_help_lists_key_options():
    cmd = make_command(get("swebench-replay"))
    out = CliRunner().invoke(cmd, ["--help"]).output
    for flag in ("--job", "--trajectories", "--concurrency", "--sweep",
                 "--mode", "--reachable-host"):
        assert flag in out


def test_requires_exactly_one_source(tmp_path):
    cmd = make_command(get("swebench-replay"))
    res = CliRunner().invoke(cmd, [])  # neither --job nor --trajectories
    assert res.exit_code != 0
    assert "exactly one of --job or --trajectories" in res.output

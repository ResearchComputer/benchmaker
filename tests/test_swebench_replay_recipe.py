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
    for flag in ("--job", "--trajectories", "--concurrency", "--concurrency-sweep",
                 "--mode", "--reachable-host"):
        assert flag in out


def test_requires_exactly_one_source(tmp_path):
    cmd = make_command(get("swebench-replay"))
    res = CliRunner().invoke(cmd, [])  # neither --job nor --trajectories
    assert res.exit_code != 0
    assert "exactly one of --job or --trajectories" in res.output


def test_rejects_both_sources(tmp_path):
    cmd = make_command(get("swebench-replay"))
    traj = tmp_path / "t.jsonl"
    traj.write_text("")
    res = CliRunner().invoke(cmd, ["--job", str(tmp_path),
                                   "--trajectories", str(traj)])
    assert res.exit_code != 0
    assert "exactly one of --job or --trajectories" in res.output


def test_resolve_task_filter_defaults_to_store_instance_ids():
    from benchmaker.swebench.trajectory import Trajectory
    store = {
        "k1": Trajectory(key="k1", instance_id="a__a-1", model="m", turns=[]),
        "k2": Trajectory(key="k2", instance_id="b__b-2", model="m", turns=[]),
        "k3": Trajectory(key="k3", instance_id=None, model="m", turns=[]),
    }
    ids, missing = SR._resolve_task_filter((), (), store)
    assert ids == ["a__a-1", "b__b-2"]  # only the recorded tasks, sorted
    assert missing == 1                 # the null-instance_id one is uncovered


def test_resolve_task_filter_honors_explicit_task():
    # An explicit --task wins over the store-derived default.
    ids, missing = SR._resolve_task_filter(("x__x-9",), (), {})
    assert ids == ["x__x-9"] and missing == 0


def test_resolve_task_filter_excludes_tasks():
    from benchmaker.swebench.trajectory import Trajectory
    store = {
        "k1": Trajectory(key="k1", instance_id="a__a-1", model="m", turns=[]),
        "k2": Trajectory(key="k2", instance_id="b__b-2", model="m", turns=[]),
    }
    # --exclude-task drops the named id from the store-derived default...
    ids, missing = SR._resolve_task_filter((), ("a__a-1",), store)
    assert ids == ["b__b-2"] and missing == 0
    # ...and from an explicit --task set too.
    ids, _ = SR._resolve_task_filter(("a__a-1", "b__b-2"), ("a__a-1",), {})
    assert ids == ["b__b-2"]


def test_resolve_url_host_loopback_and_reachable():
    assert SR._resolve_url_host("127.0.0.1", None) == "127.0.0.1"
    assert SR._resolve_url_host("0.0.0.0", None) == "127.0.0.1"      # bind-all -> loopback URL
    assert SR._resolve_url_host("0.0.0.0", "10.0.0.5") == "10.0.0.5"  # reachable host wins
    assert SR._resolve_url_host("localhost", None) == "localhost"


def test_validate_observations_flag_in_help():
    cmd = make_command(get("swebench-replay"))
    out = CliRunner().invoke(cmd, ["--help"]).output
    assert "--validate-observations" in out


def test_pi_container_loopback_fails_fast():
    # pi-container + a loopback URL (no --reachable-host) is unreachable from the
    # sandbox; the recipe must reject it up front with the fix.
    cmd = make_command(get("swebench-replay"))
    res = CliRunner().invoke(cmd, ["--mode", "pi-container",
                                   "--trajectories", "nonexistent.jsonl"])
    assert res.exit_code != 0
    assert "pi-container" in res.output and "reachable-host" in res.output

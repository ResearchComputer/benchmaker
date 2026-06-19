# tests/test_backfill_trajectory_status.py
"""Unit tests for scripts/backfill_trajectory_status.py.

The backfill adds the completion-status block (``exit_status``/``termination``/
``completed``/``status_source``) to an *already-collected* trajectory JSONL:
authoritative from the trial's ``result.json`` when its job dir still exists,
otherwise inferred from the record's own stored last turn.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
from pathlib import Path


def _load(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("backfill_trajectory_status", "scripts/backfill_trajectory_status.py")


def _rec(trial="aa__aa-1__x", finish="stop", tool_calls=None, **extra):
    rec = {"trial": trial, "instance_id": trial.rsplit("__", 1)[0],
           "turns": [{"index": 0, "finish_reason": finish,
                      "tool_calls": tool_calls or []}]}
    rec.update(extra)
    return rec


def _result_dir(jobs_root: Path, trial: str, exit_status: str) -> Path:
    d = jobs_root / "some-job" / trial
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(
        {"agent_result": {"metadata": {"exit_status": exit_status}}}))
    return d


# --------------------------- classification -------------------------------- #

def test_authoritative_when_result_json_exists(tmp_path):
    _result_dir(tmp_path, "aa__aa-1__x", "time_limit")
    fields = B.status_for_record(_rec("aa__aa-1__x"), tmp_path)
    assert fields["exit_status"] == "time_limit"
    assert fields["termination"] == "timeout"
    assert fields["completed"] is False
    assert fields["status_source"] == "result_json"


def test_inferred_completed_when_dir_missing_and_stop(tmp_path):
    fields = B.status_for_record(_rec("gone__gone-9__z", finish="stop"), tmp_path)
    assert fields["exit_status"] == "unknown"
    assert fields["termination"] == "completed"
    assert fields["status_source"] == "inferred"


def test_inferred_incomplete_when_dir_missing_and_tool_call(tmp_path):
    fields = B.status_for_record(
        _rec("gone__gone-9__z", finish="tool_calls",
             tool_calls=[{"name": "bash"}]), tmp_path)
    assert fields["exit_status"] == "unknown"
    assert fields["termination"] == "incomplete"
    assert fields["completed"] is False
    assert fields["status_source"] == "inferred"


# --------------------------- file rewrite ---------------------------------- #

def test_backfill_file_adds_fields_and_is_idempotent(tmp_path):
    _result_dir(tmp_path, "aa__aa-1__x", "ok")
    src = tmp_path / "traj.jsonl"
    src.write_text(
        json.dumps(_rec("aa__aa-1__x", finish="stop")) + "\n"
        + json.dumps(_rec("gone__gone-9__z", finish="tool_calls",
                          tool_calls=[{"name": "bash"}])) + "\n")

    n = B.backfill_file(src, jobs_root=tmp_path)
    assert n == 2
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    assert rows[0]["termination"] == "completed"
    assert rows[0]["status_source"] == "result_json"
    assert rows[1]["termination"] == "incomplete"
    assert rows[1]["status_source"] == "inferred"
    # existing fields are preserved
    assert rows[0]["instance_id"] == "aa__aa-1"

    # running again is a no-op in content (idempotent)
    B.backfill_file(src, jobs_root=tmp_path)
    rows2 = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    assert rows2 == rows

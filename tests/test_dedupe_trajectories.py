# tests/test_dedupe_trajectories.py
"""Unit tests for scripts/dedupe_trajectories.py.

Covers the pure dedup core (duplicate detection + best-graded keep policy) and
the CLI surface (inspect exit code, --out, --in-place).
"""
from __future__ import annotations

import importlib.util as _ilu
import json
from pathlib import Path


def _load_cli():
    spec = _ilu.spec_from_file_location(
        "dedupe_trajectories", "scripts/dedupe_trajectories.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load_cli()


def _rec(instance_id="aa__aa-1", *, trial=None, reward=1.0, passed=None,
         termination="completed", **extra):
    rec = {"instance_id": instance_id,
           "trial": trial or f"{instance_id}__x",
           "reward": reward,
           "termination": termination}
    rec["passed"] = (reward is not None and reward >= 1.0) if passed is None else passed
    rec.update(extra)
    return rec


# --------------------------- record_key ------------------------------------ #

def test_record_key_uses_instance_id():
    assert D.record_key(_rec("django__django-1"), "instance_id") == "django__django-1"


def test_record_key_falls_back_to_trial():
    rec = {"instance_id": None, "trial": "cc__cc-3__xy"}
    assert D.record_key(rec, "instance_id") == "cc__cc-3__xy"


def test_record_key_none_when_no_key():
    assert D.record_key({}, "instance_id") is None


# --------------------------- find_duplicates ------------------------------- #

def test_find_duplicates_groups_only_repeats():
    recs = [_rec("a"), _rec("b"), _rec("a")]
    groups = D.find_duplicates(recs, "instance_id")
    assert set(groups) == {"a"}
    assert len(groups["a"]) == 2


def test_find_duplicates_empty_when_unique():
    recs = [_rec("a"), _rec("b")]
    assert D.find_duplicates(recs, "instance_id") == {}


# --------------------------- dedupe keep-policy ---------------------------- #

def test_dedupe_keeps_graded_over_ungraded():
    ungraded = _rec("a", reward=None, passed=False, trial="a__ungraded")
    graded_fail = _rec("a", reward=0.0, passed=False, trial="a__graded")
    kept, dropped = D.dedupe([ungraded, graded_fail], "instance_id")
    assert [r["trial"] for r in kept] == ["a__graded"]
    assert [r["trial"] for r in dropped] == ["a__ungraded"]


def test_dedupe_keeps_pass_over_fail():
    fail = _rec("a", reward=0.0, passed=False, trial="a__fail")
    win = _rec("a", reward=1.0, passed=True, trial="a__pass")
    kept, _ = D.dedupe([fail, win], "instance_id")
    assert [r["trial"] for r in kept] == ["a__pass"]


def test_dedupe_keeps_real_termination_over_incomplete():
    incomplete = _rec("a", reward=0.0, passed=False, termination="incomplete",
                      trial="a__inc")
    timeout = _rec("a", reward=0.0, passed=False, termination="timeout",
                   trial="a__to")
    kept, _ = D.dedupe([incomplete, timeout], "instance_id")
    assert [r["trial"] for r in kept] == ["a__to"]


def test_dedupe_true_tie_keeps_last_seen():
    first = _rec("a", reward=1.0, passed=True, trial="a__first")
    last = _rec("a", reward=1.0, passed=True, trial="a__last")
    kept, _ = D.dedupe([first, last], "instance_id")
    assert [r["trial"] for r in kept] == ["a__last"]


def test_dedupe_preserves_first_appearance_order():
    recs = [_rec("b", trial="b__1"), _rec("a", trial="a__1"),
            _rec("a", trial="a__2"), _rec("c", trial="c__1")]
    kept, _ = D.dedupe(recs, "instance_id")
    # order keyed by where each instance first appeared: b, a, c
    assert [r["instance_id"] for r in kept] == ["b", "a", "c"]


def test_dedupe_clean_input_is_noop():
    recs = [_rec("a"), _rec("b")]
    kept, dropped = D.dedupe(recs, "instance_id")
    assert kept == recs and dropped == []


# --------------------------- CLI: inspect ---------------------------------- #

def _write(tmp_path: Path, *recs) -> Path:
    p = tmp_path / "in.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return p


def test_cli_inspect_exits_1_on_duplicates_and_writes_nothing(tmp_path, capsys):
    inp = _write(tmp_path, _rec("a", trial="a__1"), _rec("a", trial="a__2"))
    before = inp.read_text()
    rc = D.main([str(inp)])
    assert rc == 1
    assert inp.read_text() == before                 # read-only
    out = capsys.readouterr().out
    assert "a" in out and "KEEP" in out and "DROP" in out


def test_cli_inspect_exits_0_when_clean(tmp_path):
    inp = _write(tmp_path, _rec("a"), _rec("b"))
    assert D.main([str(inp)]) == 0


# --------------------------- CLI: write ------------------------------------ #

def test_cli_out_writes_deduped_store(tmp_path):
    inp = _write(tmp_path,
                 _rec("a", reward=0.0, passed=False, trial="a__fail"),
                 _rec("a", reward=1.0, passed=True, trial="a__pass"),
                 _rec("b", trial="b__1"))
    out = tmp_path / "out.jsonl"
    rc = D.main([str(inp), "--out", str(out)])
    assert rc == 0
    recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["instance_id"] for r in recs] == ["a", "b"]
    assert next(r for r in recs if r["instance_id"] == "a")["trial"] == "a__pass"


def test_cli_in_place_rewrites_input(tmp_path):
    inp = _write(tmp_path, _rec("a", trial="a__1"), _rec("a", trial="a__2"),
                 _rec("b", trial="b__1"))
    rc = D.main([str(inp), "--in-place"])
    assert rc == 0
    recs = [json.loads(l) for l in inp.read_text().splitlines() if l.strip()]
    assert [r["instance_id"] for r in recs] == ["a", "b"]


def test_cli_in_place_and_out_mutually_exclusive(tmp_path):
    inp = _write(tmp_path, _rec("a"))
    import pytest
    with pytest.raises(SystemExit):
        D.parse_args([str(inp), "--in-place", "--out", "x.jsonl"])


def test_cli_missing_input_errors(tmp_path):
    assert D.main([str(tmp_path / "nope.jsonl")]) != 0

import json, os
from benchmaker.swebench import cleanjobs as cj
from benchmaker.swebench.trajectory import parse_pi_conversation
from tests._cleanjobs_fixtures import (
    make_pi_host_trial, make_pi_container_trial,
    make_pi_container_killed_trial, make_failure_trial)


def _read_lines(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def test_clean_task_pi_host_collapses_and_deletes(tmp_path):
    td = make_pi_host_trial(str(tmp_path))
    status, ob, nb, n = cj.clean_task(td)
    assert status == "CLEANED"
    out = os.path.join(str(tmp_path), "django__django-11999__K75PXvM.jsonl")
    assert os.path.exists(out)
    assert not os.path.exists(td)            # dir removed
    recs = _read_lines(out)
    assert recs[0]["type"] == "benchmaker_meta"
    assert recs[1]["type"] == "session"
    assert n == 1


def test_clean_task_pi_container_keeps_turns_and_timings(tmp_path):
    from benchmaker.swebench.timeout_load import (
        recover_command_timings, recover_command_timings_from_records)
    td = make_pi_container_trial(str(tmp_path))
    orig_log = os.path.join(td, "agent", "pi-container.log")
    orig_timings = recover_command_timings(orig_log)
    orig_turns = parse_pi_conversation(open(orig_log).read()).turns
    status, ob, nb, n = cj.clean_task(td)
    assert status == "CLEANED"
    assert nb < ob
    out = os.path.join(str(tmp_path), "astropy__astropy-7606__B8sCm3m.jsonl")
    recs = _read_lines(out)
    assert recs[0]["trajectory_format"] == "pi_log"
    traj_text = "\n".join(json.dumps(r) for r in recs[1:])
    assert len(parse_pi_conversation(traj_text).turns) == len(orig_turns) > 0
    assert recover_command_timings_from_records(recs[1:]) == orig_timings


def test_clean_task_failure_meta_only(tmp_path):
    td = make_failure_trial(str(tmp_path))
    status, ob, nb, n = cj.clean_task(td)
    assert status == "CLEANED"
    out = os.path.join(str(tmp_path), "django__django-16100__PpdSN7s.jsonl")
    recs = _read_lines(out)
    assert len(recs) == 1 and recs[0]["trajectory_format"] == "none"
    assert not os.path.exists(td)


def test_clean_task_dry_run_changes_nothing(tmp_path):
    td = make_pi_host_trial(str(tmp_path))
    status, ob, nb, n = cj.clean_task(td, dry_run=True)
    assert status == "DRY"
    assert os.path.exists(td)
    assert not os.path.exists(os.path.join(str(tmp_path),
                                           "django__django-11999__K75PXvM.jsonl"))


def test_clean_task_unsafe_pi_log_zero_turns_keeps_dir(tmp_path, monkeypatch):
    td = make_pi_container_trial(str(tmp_path))
    # simulate a fold that drops turn_end -> parse_pi_conversation yields 0 turns
    monkeypatch.setattr(cj, "_PI_LOG_DROP", {"message_update", "agent_end", "turn_end"})
    status, *_ = cj.clean_task(td)
    assert status == "SKIPPED_UNSAFE"
    assert os.path.exists(td)
    assert not [f for f in os.listdir(str(tmp_path)) if f.startswith(".clean_")]


def test_clean_tree_idempotent(tmp_path):
    make_pi_host_trial(str(tmp_path))
    s1 = cj.clean_tree(str(tmp_path))
    assert s1["cleaned"] == 1
    s2 = cj.clean_tree(str(tmp_path))
    assert s2["cleaned"] == 0


def test_reward_txt_fallback(tmp_path):
    # build_meta should read verifier/reward.txt when result has no reward
    td = make_pi_host_trial(str(tmp_path))
    import json as _j
    rp = os.path.join(td, "result.json")
    r = _j.load(open(rp))
    r["verifier_result"] = {"rewards": {}}     # drop the inline reward
    _j.dump(r, open(rp, "w"))
    # reward.txt already says "1\n" from the fixture
    meta = cj.build_meta(td, trajectory_format="session")
    assert meta["reward"] == 1.0

def test_build_meta_pi_host(tmp_path):
    td = make_pi_host_trial(str(tmp_path))
    meta = cj.build_meta(td, trajectory_format="session")
    assert meta["type"] == "benchmaker_meta"
    assert meta["schema_version"] == 1
    assert meta["trial_name"].endswith("__K75PXvM")
    assert meta["task_name"] == "django__django-11999"
    assert meta["reward"] == 1.0
    assert meta["resolved"] is True
    assert meta["trajectory_format"] == "session"
    assert meta["result"]["agent_result"]["metadata"]["exec_count"] == 81
    assert meta["report"]["resolved"] is True
    assert meta["timeline_spans"][0]["name"] == "sandbox_exec"
    assert meta["exception_text"] is None

def test_find_trajectory_session(tmp_path):
    td = make_pi_host_trial(str(tmp_path))
    fmt, lines = cj.find_trajectory(td)
    assert fmt == "session"
    assert json.loads(lines[0])["type"] == "session"

def test_find_trajectory_pi_log_folds_and_keeps_turn_end(tmp_path):
    td = make_pi_container_trial(str(tmp_path))
    fmt, lines = cj.find_trajectory(td)
    assert fmt == "pi_log"
    types = [json.loads(b)["type"] for b in lines]
    assert "turn_end" in types and "message_end" in types
    assert "message_update" not in types

def test_find_trajectory_pi_log_unsafe_fold_keeps_raw(tmp_path):
    # Hard-killed stream: the final message_update partial has unique content
    # that never closed with a message_end, so folding it away would lose it.
    # The safety fallback must return the RAW log verbatim (message_update kept).
    td = make_pi_container_killed_trial(str(tmp_path))
    fmt, lines = cj.find_trajectory(td)
    assert fmt == "pi_log"
    types = [json.loads(b)["type"] for b in lines]
    assert "message_update" in types  # retained because folding was unsafe

def test_find_trajectory_none(tmp_path):
    td = make_failure_trial(str(tmp_path))
    fmt, lines = cj.find_trajectory(td)
    assert fmt == "none"
    assert lines == []

def test_build_meta_failure_has_exception_text(tmp_path):
    td = make_failure_trial(str(tmp_path))
    meta = cj.build_meta(td, trajectory_format="none")
    assert meta["reward"] is None
    assert "RuntimeError" in meta["exception_text"]


def test_iter_task_dirs_ignores_job_level_result(tmp_path):
    # a replay/job dir has its OWN result.json (run summary) + per-trial subdirs
    job = tmp_path / "replay_x"
    make_pi_host_trial(str(job))         # job/django__django-11999__K75PXvM/ (has agent/)
    make_pi_container_trial(str(job))    # job/astropy__astropy-7606__B8sCm3m/ (has agent/)
    (job / "result.json").write_text(json.dumps({"job": "summary"}))  # job-level, no agent/
    got = sorted(os.path.basename(d) for d in cj.iter_task_dirs(str(tmp_path)))
    assert got == ["astropy__astropy-7606__B8sCm3m", "django__django-11999__K75PXvM"]
    # clean_tree cleans the 2 trials and LEAVES the job-level result.json intact
    s = cj.clean_tree(str(tmp_path))
    assert s["cleaned"] == 2
    assert (job / "result.json").exists()                                  # job summary preserved
    assert (job / "django__django-11999__K75PXvM.jsonl").exists()
    assert not (job / "django__django-11999__K75PXvM").is_dir()            # trial dir collapsed


def test_clean_task_pi_host_log_only_preserves_trajectory(tmp_path):
    from tests._cleanjobs_fixtures import make_pi_host_log_only_trial
    from benchmaker.swebench.trajectory import parse_pi_conversation
    td = make_pi_host_log_only_trial(str(tmp_path))
    fmt, _ = cj.find_trajectory(td)
    assert fmt == "pi_log"                      # NOT "none"
    status, ob, nb, n = cj.clean_task(td)
    assert status == "CLEANED"
    out = os.path.join(str(tmp_path), "matplotlib__matplotlib-25122__QrBK5jY.jsonl")
    recs = [__import__("json").loads(x) for x in open(out) if x.strip()]
    assert recs[0]["trajectory_format"] == "pi_log"
    text = "\n".join(__import__("json").dumps(r) for r in recs[1:])
    assert len(parse_pi_conversation(text).turns) > 0   # trajectory preserved


import subprocess, sys

def test_cli_dry_run(tmp_path):
    make_pi_host_trial(str(tmp_path))
    out = subprocess.run([sys.executable, "scripts/clis/cleanjobs.py",
                          str(tmp_path), "--dry-run"],
                         capture_output=True, text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    assert "cleaned" in out.stdout.lower() or "saved" in out.stdout.lower()
    assert os.path.isdir(os.path.join(str(tmp_path),
                                      "django__django-11999__K75PXvM"))  # untouched

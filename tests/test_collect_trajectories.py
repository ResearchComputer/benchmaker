# tests/test_collect_trajectories.py
"""Unit tests for scripts/collect_trajectories.py.

Covers the two pure pieces — the harbor_eval argv builder and the on-disk
trajectory+grade fusion — plus the write/round-trip. No live harbor or Flash
Sandbox is exercised; the run path is a thin subprocess wrapper around the argv
builder (asserted here) and is not invoked.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
from pathlib import Path

from benchmaker.swebench import trajectory as T


def _load_cli():
    spec = _ilu.spec_from_file_location(
        "collect_trajectories", "scripts/collect_trajectories.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_cli()


# --------------------------- argv builder ---------------------------------- #

def _args(**over):
    argv = ["--mode", over.pop("mode", "pi-container")]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    args, passthrough = C.parse_args(argv)
    return args, passthrough


def test_parse_args_defaults():
    args, passthrough = C.parse_args(["--mode", "pi-host"])
    assert args.mode == "pi-host"
    assert args.route_tools == "all"          # tool-parity default
    assert args.dataset == "swebench-verified"
    assert args.job_name.startswith("pi-pi-host-") or args.job_name.startswith("pi-host-")
    assert passthrough == []


def test_build_argv_pi_host_routes_all_tools():
    args, passthrough = _args(mode="pi-host", job_name="j1")
    argv = C.build_harbor_argv(args, passthrough)
    assert argv[:4] == ["-m", "benchmaker.swebench.harbor_eval", "--agent", "pi-host"]
    # host mode routes all four tools by default for parity with container mode
    assert "--agent-kwarg" in argv
    assert "route_tools=all" in argv
    assert argv[argv.index("--job-name") + 1] == "j1"


def test_build_argv_pi_host_bash_only():
    args, passthrough = _args(mode="pi-host", route_tools="bash", job_name="j2")
    argv = C.build_harbor_argv(args, passthrough)
    assert "route_tools=bash" in argv
    assert "route_tools=all" not in argv


def test_build_argv_pi_container_has_no_route_kwarg():
    args, passthrough = _args(mode="pi-container", job_name="j3")
    argv = C.build_harbor_argv(args, passthrough)
    assert argv[:4] == ["-m", "benchmaker.swebench.harbor_eval", "--agent", "pi-container"]
    assert "--agent-kwarg" not in argv
    assert not any(a.startswith("route_tools=") for a in argv)


def test_build_argv_forwards_known_and_passthrough():
    args, passthrough = C.parse_args(
        ["--mode", "pi-container", "--n-tasks", "5", "--concurrency", "2",
         "--job-name", "j4", "--force-build"])
    argv = C.build_harbor_argv(args, passthrough)
    assert argv[argv.index("--n-tasks") + 1] == "5"
    assert argv[argv.index("--concurrency") + 1] == "2"
    # an unknown harbor_eval flag passes through verbatim
    assert "--force-build" in argv


# --------------------------- fusion ---------------------------------------- #

def _pi_log(*events) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _make_trial(job_dir: Path, trial_name: str, *, log_name="pi-container.log",
                reward="1", instance_id=None, with_turn=True, with_report=True,
                with_reward=True):
    iid = instance_id or trial_name.rsplit("__", 1)[0]
    tdir = job_dir / trial_name
    (tdir / "agent").mkdir(parents=True)
    events = [{"type": "message_start", "message": {"role": "user",
               "content": f"Fix the bug.\n# Task: ?\nRepository: {iid}"}}]
    if with_turn:
        events.append({"type": "turn_end", "message": {
            "role": "assistant",
            "content": [{"type": "toolCall", "id": "c1", "name": "bash",
                         "arguments": {"command": "ls"}}],
            "stopReason": "toolUse", "model": "zai-org/GLM-4.7-Flash",
            "usage": {"input": 10, "output": 2, "totalTokens": 12}}})
    (tdir / "agent" / log_name).write_text(_pi_log(*events))
    (tdir / "config.json").write_text(json.dumps(
        {"task": {"path": f"datasets/swebench-verified/{iid}"}, "trial_name": trial_name}))
    if with_reward:
        (tdir / "verifier").mkdir(exist_ok=True)
        (tdir / "verifier" / "reward.txt").write_text(f"{reward}\n")
    if with_report:
        (tdir / "verifier").mkdir(exist_ok=True)
        (tdir / "verifier" / "report.json").write_text(json.dumps({
            iid: {"resolved": str(reward) == "1", "tests_status": {
                "FAIL_TO_PASS": {"success": ["t1"], "failure": []},
                "PASS_TO_PASS": {"success": ["p1", "p2"], "failure": []}}}}))
    return tdir


def test_collect_fuses_trajectory_and_grade(tmp_path):
    _make_trial(tmp_path, "django__django-11095__hoUo2Vr", reward="1")
    recs = C.collect_from_job_dir(tmp_path)
    assert len(recs) == 1
    r = recs[0]
    assert r["instance_id"] == "django__django-11095"     # from config.json, not prompt
    assert r["trial"] == "django__django-11095__hoUo2Vr"
    assert r["mode"] == "pi-container"
    assert r["reward"] == 1.0 and r["passed"] is True
    assert r["resolved"] is True
    assert r["fail_to_pass"] == 1 and r["pass_to_pass"] == 2
    assert r["n_turns"] == 1
    assert r["turns"][0]["tool_calls"][0]["name"] == "bash"


def test_collect_marks_failed_grade(tmp_path):
    _make_trial(tmp_path, "aa__aa-1__x", reward="0")
    r = C.collect_from_job_dir(tmp_path)[0]
    assert r["reward"] == 0.0 and r["passed"] is False
    assert r["resolved"] is False


def test_collect_labels_pi_host_mode(tmp_path):
    _make_trial(tmp_path, "bb__bb-2__y", log_name="pi-host.log")
    r = C.collect_from_job_dir(tmp_path)[0]
    assert r["mode"] == "pi-host"


def test_collect_skips_zero_turn_trials(tmp_path):
    _make_trial(tmp_path, "cc__cc-3__z", with_turn=False)
    assert C.collect_from_job_dir(tmp_path) == []


def test_collect_tolerates_missing_reward(tmp_path):
    _make_trial(tmp_path, "dd__dd-4__w", with_reward=False, with_report=False)
    r = C.collect_from_job_dir(tmp_path)[0]
    assert r["reward"] is None and r["passed"] is False
    assert "resolved" not in r          # report absent -> field omitted
    assert r["n_turns"] == 1            # trajectory still collected


def test_collect_recovers_instance_id_without_config(tmp_path):
    tdir = _make_trial(tmp_path, "ee__ee-5__q")
    (tdir / "config.json").unlink()
    r = C.collect_from_job_dir(tmp_path)[0]
    assert r["instance_id"] == "ee__ee-5"   # recovered from the trial dir name


def test_collect_ignores_non_trial_entries(tmp_path):
    _make_trial(tmp_path, "ff__ff-6__a")
    (tmp_path / "result.json").write_text("{}")
    (tmp_path / "trajectories.jsonl").write_text("")   # harbor's own file
    recs = C.collect_from_job_dir(tmp_path)
    assert len(recs) == 1


# --------------------------- write / round-trip ---------------------------- #

def test_write_is_valid_replay_store(tmp_path):
    _make_trial(tmp_path, "django__django-11095__hoUo2Vr", reward="1")
    recs = C.collect_from_job_dir(tmp_path)
    out = tmp_path / "pi-trajectories.jsonl"
    n = C.write_trajectories(recs, out)
    assert n == 1
    # grading keys are additive -> the file is still a loadable replay store
    store = T.load_store(out)
    assert set(store) == {recs[0]["key"]}
    # and the grade survives a raw JSON round-trip
    line = json.loads(out.read_text().strip())
    assert line["passed"] is True and line["mode"] == "pi-container"

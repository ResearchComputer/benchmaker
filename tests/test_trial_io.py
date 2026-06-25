import os, json
from benchmaker.swebench import cleanjobs as cj
from benchmaker.swebench import trial_io as tio
from tests._cleanjobs_fixtures import make_pi_host_trial, make_pi_container_trial

def _fields(t):
    return (t.trial_name, t.task_name, t.reward, t.resolved, t.trajectory_format,
            t.timeline_spans, t.exception_text)

def test_legacy_vs_cleaned_equivalent_pi_host(tmp_path):
    leg = make_pi_host_trial(str(tmp_path / "leg"))
    cl = make_pi_host_trial(str(tmp_path / "cl"))
    t_leg = tio.load_trial(leg)
    cj.clean_task(cl)
    t_cl = tio.load_trial(os.path.join(str(tmp_path / "cl"),
                                       "django__django-11999__K75PXvM.jsonl"))
    assert _fields(t_leg) == _fields(t_cl)
    assert list(t_leg.iter_trajectory()) and \
        len(list(t_cl.iter_trajectory())) == len(list(t_leg.iter_trajectory()))

def test_legacy_vs_cleaned_equivalent_pi_container_nonempty(tmp_path):
    leg = make_pi_container_trial(str(tmp_path / "leg"))
    cl = make_pi_container_trial(str(tmp_path / "cl"))
    t_leg = tio.load_trial(leg)
    cj.clean_task(cl)
    t_cl = tio.load_trial(os.path.join(str(tmp_path / "cl"),
                                       "astropy__astropy-7606__B8sCm3m.jsonl"))
    from benchmaker.swebench.trajectory import parse_pi_conversation
    leg_turns = len(parse_pi_conversation(open(t_leg.legacy_agent_log()).read()).turns)
    cl_turns = len(parse_pi_conversation(
        "\n".join(json.dumps(r) for r in t_cl.iter_trajectory())).turns)
    assert cl_turns == leg_turns > 0
    assert tio.recover_command_timings_from_trial(t_cl) == \
           tio.recover_command_timings_from_trial(t_leg)

def test_iter_trials_finds_both_layouts(tmp_path):
    make_pi_host_trial(str(tmp_path / "a"))
    clb = make_pi_container_trial(str(tmp_path / "b"))
    cj.clean_task(clb)
    names = sorted(t.trial_name for t in tio.iter_trials(str(tmp_path)))
    assert names == ["astropy__astropy-7606__B8sCm3m",
                     "django__django-11999__K75PXvM"]


def test_trajectory_format_pi_host_log_only(tmp_path):
    from tests._cleanjobs_fixtures import make_pi_host_log_only_trial
    td = make_pi_host_log_only_trial(str(tmp_path))
    t = tio.load_trial(td)
    assert t.trajectory_format == "pi_log"
    assert len(list(t.iter_trajectory())) > 0


def test_iter_trials_ignores_job_level_result(tmp_path):
    import json
    from tests._cleanjobs_fixtures import make_pi_host_trial, make_pi_container_trial
    job = tmp_path / "replay_x"
    make_pi_host_trial(str(job)); make_pi_container_trial(str(job))
    (job / "result.json").write_text(json.dumps({"job": "summary"}))
    names = sorted(t.trial_name for t in tio.iter_trials(str(tmp_path)))
    assert names == ["astropy__astropy-7606__B8sCm3m", "django__django-11999__K75PXvM"]
    from benchmaker.swebench import cleanjobs as _cj
    _cj.clean_tree(str(tmp_path))
    names2 = sorted(t.trial_name for t in tio.iter_trials(str(tmp_path)))
    assert names2 == ["astropy__astropy-7606__B8sCm3m", "django__django-11999__K75PXvM"]

import importlib.util, os, pathlib

_CSD = pathlib.Path(__file__).parent.parent / "scripts" / "clis" / "collect_sweep_data.py"
spec = importlib.util.spec_from_file_location("csd", _CSD)
csd = importlib.util.module_from_spec(spec); spec.loader.exec_module(csd)

_trial_rows = csd._trial_rows


def test_verifier_timeout_true_on_real_shape():
    d = {"exception_info": {"exception_type": "VerifierTimeoutError",
                            "exception_message": "Verifier execution timed out after 600.0 seconds",
                            "exception_traceback": "..."}}
    assert csd._verifier_timed_out(d) == 1


def test_verifier_timeout_false_when_null_or_other():
    assert csd._verifier_timed_out({"exception_info": None}) == 0
    assert csd._verifier_timed_out({"exception_info": {"exception_type": "AgentTimeoutError"}}) == 0
    assert csd._verifier_timed_out({}) == 0


def test_verifier_timeout_false_when_only_in_message():
    d = {"exception_info": {"exception_type": "AgentTimeoutError",
                            "exception_message": "wrapped VerifierTimeoutError cause"}}
    assert csd._verifier_timed_out(d) == 0


from benchmaker.swebench import cleanjobs as cj
from tests._cleanjobs_fixtures import make_pi_host_trial


def _build_cell(root):
    cell = os.path.join(root, "timeout_T20_c16", "replay_2026-06-19__17-53-21_c16_80b8")
    make_pi_host_trial(cell)            # writes <cell>/django__django-11999__K75PXvM/
    return root


def test_rows_legacy_and_cleaned_match_expected(tmp_path):
    leg = _build_cell(str(tmp_path / "leg"))
    cl = _build_cell(str(tmp_path / "cl"))
    legacy_rows = list(_trial_rows("run", leg))
    cj.clean_tree(cl)
    cleaned_rows = list(_trial_rows("run", cl))
    # exactly one trial in the cell, in both layouts
    assert len(legacy_rows) == len(cleaned_rows) == 1
    lr, cr = legacy_rows[0], cleaned_rows[0]
    # stable columns: concrete expected values
    for row in (lr, cr):
        assert row["run"] == "run"
        assert row["T"] == "20"
        assert row["c"] == 16
        assert row["task"] == "django__django-11999"
        assert row["reward"] == 1.0
        assert row["solved"] == 1
        assert row["graded"] == 1
        assert row["verifier_timeout"] == 0
    # phase durations identical between layouts and plausibly non-None
    for col in ("env_setup_s", "agent_setup_s", "agent_exec_s", "verifier_s", "total_s"):
        assert lr[col] == cr[col] is not None, col

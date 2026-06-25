import importlib.util, os, pathlib

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts" / "clis" / "summarize_replay_timeout_sweep.py"
spec = importlib.util.spec_from_file_location("srts", _SCRIPT)
srts = importlib.util.module_from_spec(spec); spec.loader.exec_module(srts)

_solved = srts._solved
_spans = srts._spans

from benchmaker.swebench import cleanjobs as cj
from tests._cleanjobs_fixtures import make_pi_host_trial


def test_solved_and_spans_legacy_and_cleaned(tmp_path):
    leg = os.path.join(str(tmp_path), "leg", "timeout_T20_c16")
    cl = os.path.join(str(tmp_path), "cl", "timeout_T20_c16")
    make_pi_host_trial(leg)
    make_pi_host_trial(cl)
    # legacy layout
    assert _solved(leg) == (1, 1)
    assert _spans(leg, 20.0) == (1, 0, 1.5, 1.5, 1.5)
    # cleaned layout (same numbers)
    cj.clean_tree(cl)
    assert _solved(cl) == (1, 1)
    assert _spans(cl, 20.0) == (1, 0, 1.5, 1.5, 1.5)

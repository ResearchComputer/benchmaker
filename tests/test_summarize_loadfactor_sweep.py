import importlib.util, os, pathlib

from benchmaker.swebench import cleanjobs as cj
from tests._cleanjobs_fixtures import make_pi_container_trial

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts" / "clis" / "summarize_loadfactor_sweep.py"
spec = importlib.util.spec_from_file_location("summarize_loadfactor_sweep", _SCRIPT)
_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(_mod)


def test_c1_tasks_and_actual_solved_legacy_and_cleaned(tmp_path):
    leg, cl = str(tmp_path / "leg"), str(tmp_path / "cl")
    make_pi_container_trial(leg)
    make_pi_container_trial(cl)
    assert _mod._c1_tasks(leg) == [(1.0, 1.5)]
    assert _mod._actual_solved(leg) == (1, 1)
    cj.clean_tree(cl)
    assert _mod._c1_tasks(cl) == [(1.0, 1.5)]
    assert _mod._actual_solved(cl) == (1, 1)

import importlib.util, os, pathlib

from benchmaker.swebench import cleanjobs as cj
from tests._cleanjobs_fixtures import make_pi_container_trial

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts" / "analyze_timeout_curve.py"
spec = importlib.util.spec_from_file_location("analyze_timeout_curve", _SCRIPT)
_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(_mod)


def test_collect_tasks_legacy_and_cleaned_pi_container(tmp_path):
    leg, cl = str(tmp_path / "leg"), str(tmp_path / "cl")
    make_pi_container_trial(leg)
    make_pi_container_trial(cl)
    legacy = _mod.collect_tasks(leg)
    cj.clean_tree(cl)
    cleaned = _mod.collect_tasks(cl)
    assert legacy == cleaned == [(1.0, 1.5, "astropy__astropy-7606")]

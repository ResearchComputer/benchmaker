import importlib.util, pathlib

_CSD = pathlib.Path(__file__).parent.parent / "scripts" / "clis" / "collect_sweep_data.py"
spec = importlib.util.spec_from_file_location("csd", _CSD)
csd = importlib.util.module_from_spec(spec); spec.loader.exec_module(csd)


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

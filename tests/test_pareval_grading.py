import pytest

from benchmaker.pareval.grading import parse_run_output, RunParse


def test_parse_all_three_lines():
    out = "Time: 0.0125\nBestSequential: 0.0500\nValidation: PASS\n"
    r = parse_run_output(out)
    assert r.valid is True and r.time_s == 0.0125 and r.best_sequential_s == 0.05


def test_parse_fail_validation():
    r = parse_run_output("Validation: FAIL\n")
    assert r.valid is False and r.time_s is None


def test_parse_missing_lines_are_none():
    r = parse_run_output("garbage\n")
    assert r.valid is None and r.time_s is None and r.best_sequential_s is None


def test_parse_ignores_surrounding_noise():
    out = "starting...\nTime: 1.5\nsome log\nBestSequential: 3.0\nValidation: PASS\ndone\n"
    r = parse_run_output(out)
    assert r.valid is True and r.time_s == 1.5 and r.best_sequential_s == 3.0


from benchmaker.pareval.grading import sample_speedup


def test_speedup_picks_fastest_and_baseline():
    pc = [
        {"config": {"num_threads": 1}, "valid": True, "time_s": 0.10, "best_sequential_s": 0.10, "reps": 1},
        {"config": {"num_threads": 4}, "valid": True, "time_s": 0.025, "best_sequential_s": 0.10, "reps": 1},
    ]
    sp, nres = sample_speedup(pc, "omp")
    assert sp == pytest.approx(4.0)   # 0.10 / 0.025
    assert nres == 4                  # fastest config used 4 threads


def test_speedup_mpi_uses_procs():
    pc = [{"config": {"num_procs": 8}, "valid": True, "time_s": 0.5, "best_sequential_s": 2.0, "reps": 1}]
    sp, nres = sample_speedup(pc, "mpi")
    assert sp == pytest.approx(4.0) and nres == 8


def test_speedup_none_when_no_valid():
    pc = [{"config": {"num_threads": 4}, "valid": False, "time_s": None, "best_sequential_s": None, "reps": 1}]
    assert sample_speedup(pc, "omp") == (None, None)


def test_speedup_serial_one_resource():
    pc = [{"config": {}, "valid": True, "time_s": 1.0, "best_sequential_s": 1.0, "reps": 1}]
    sp, nres = sample_speedup(pc, "serial")
    assert sp == pytest.approx(1.0) and nres == 1

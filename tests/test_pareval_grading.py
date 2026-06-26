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


from benchmaker.pareval.grading import pass_at_k, expected_max_at_k, aggregate, SampleResult


def test_pass_at_k_basic():
    assert pass_at_k(5, 2, 1) == pytest.approx(0.4)


def test_pass_at_k_zero_correct():
    assert pass_at_k(5, 0, 1) == 0.0


def test_pass_at_k_all_correct():
    assert pass_at_k(5, 5, 1) == 1.0
    assert pass_at_k(5, 5, 3) == 1.0


def test_pass_at_k_k_exceeds_n():
    # k > n: guaranteed if any correct
    assert pass_at_k(3, 1, 5) == 1.0


def test_expected_max_k1_is_mean():
    assert expected_max_at_k([0.0, 0.0, 3.0], 1) == pytest.approx(1.0)


def test_expected_max_k2():
    assert expected_max_at_k([0.0, 0.0, 3.0], 2) == pytest.approx(2.0)


def test_expected_max_k_ge_n_is_max():
    assert expected_max_at_k([0.0, 0.0, 3.0], 3) == pytest.approx(3.0)
    assert expected_max_at_k([1.0, 2.0], 5) == pytest.approx(2.0)


def test_aggregate_two_problems_two_samples():
    samples = [
        # Problem A: omp / geometry. one correct (speedup 4.0 @ 4 threads), one wrong.
        SampleResult("A", "omp", "geometry", 0, built=True, correct=True,
                     per_config=[], speedup=4.0, best_n_resources=4),
        SampleResult("A", "omp", "geometry", 1, built=True, correct=False,
                     per_config=[], speedup=None, best_n_resources=None),
        # Problem B: mpi / graph. one correct (speedup 2.0 @ 8 procs), one didn't build.
        SampleResult("B", "mpi", "graph", 0, built=True, correct=True,
                     per_config=[], speedup=2.0, best_n_resources=8),
        SampleResult("B", "mpi", "graph", 1, built=False, correct=False,
                     per_config=[], speedup=None, best_n_resources=None),
    ]
    agg = aggregate(samples, ks=[1])

    # slice keys exist
    assert set(agg.keys()) == {"overall", "by_model", "by_problem_type"}
    assert set(agg["by_model"].keys()) == {"omp", "mpi"}
    assert set(agg["by_problem_type"].keys()) == {"geometry", "graph"}

    ov = agg["overall"]
    assert ov["n_problems"] == 2
    assert ov["n_samples"] == 4
    assert ov["build_rate"] == pytest.approx(0.75)   # 3 of 4 built
    assert ov["correct_rate"] == pytest.approx(0.5)  # 2 of 4 correct

    # pass@1 = avg over problems of pass_at_k(2,1,1)=0.5 -> 0.5
    assert ov["pass@k"][1] == pytest.approx(0.5)

    # speedup@1: A=mean([4.0,0.0])=2.0, B=mean([2.0,0.0])=1.0 -> avg 1.5
    # (the incorrect / not-built samples contribute 0.0)
    assert ov["speedup@k"][1] == pytest.approx(1.5)

    # efficiency@1: A=mean([4/4, 0])=0.5, B=mean([2/8, 0])=0.125 -> avg 0.3125
    assert ov["efficiency@k"][1] == pytest.approx(0.3125)

    # per-model slices
    omp = agg["by_model"]["omp"]
    assert omp["n_problems"] == 1 and omp["n_samples"] == 2
    assert omp["build_rate"] == pytest.approx(1.0)
    assert omp["correct_rate"] == pytest.approx(0.5)
    assert omp["pass@k"][1] == pytest.approx(0.5)
    assert omp["speedup@k"][1] == pytest.approx(2.0)
    assert omp["efficiency@k"][1] == pytest.approx(0.5)

    mpi = agg["by_model"]["mpi"]
    assert mpi["build_rate"] == pytest.approx(0.5)
    assert mpi["speedup@k"][1] == pytest.approx(1.0)
    assert mpi["efficiency@k"][1] == pytest.approx(0.125)

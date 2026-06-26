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

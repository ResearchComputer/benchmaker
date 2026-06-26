"""Grading and metrics for ParEval benchmark runs.

Pure functions, no I/O. Mirrors the ParEval C++ driver stdout contract and the
metric definitions from the ParEval paper (pass@k via Chen et al. unbiased
estimator; speedup@k / efficiency@k via expected-max over k samples).
"""

from dataclasses import dataclass


@dataclass
class RunParse:
    """Parsed result of a single ParEval driver run's stdout."""

    valid: bool | None
    time_s: float | None
    best_sequential_s: float | None


def parse_run_output(stdout: str) -> RunParse:
    """Parse the labeled stdout lines printed by the ParEval C++ driver.

    The driver prints (in any order, possibly amid other noise)::

        Time: <float>
        BestSequential: <float>
        Validation: PASS|FAIL

    Missing lines stay None. Mirrors upstream
    ``drivers/driver_wrapper.py::RunOutput._parse_output``.
    """
    valid: bool | None = None
    time_s: float | None = None
    best_sequential_s: float | None = None
    for line in stdout.split("\n"):
        if line.startswith("Time:"):
            time_s = float(line.split(":", 1)[1].strip())
        elif line.startswith("BestSequential:"):
            best_sequential_s = float(line.split(":", 1)[1].strip())
        elif line.startswith("Validation:"):
            valid = line.split(":", 1)[1].strip() == "PASS"
    return RunParse(valid=valid, time_s=time_s, best_sequential_s=best_sequential_s)

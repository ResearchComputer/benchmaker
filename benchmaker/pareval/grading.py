"""Grading and metrics for ParEval benchmark runs.

Pure functions, no I/O. Mirrors the ParEval C++ driver stdout contract and the
metric definitions from the ParEval paper (pass@k via Chen et al. unbiased
estimator; speedup@k / efficiency@k via expected-max over k samples).
"""

from dataclasses import dataclass, field


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


# Resource-count key by parallelism model (which run_config field counts as
# the "amount of parallelism" used). Missing key defaults to 1.
_RESOURCE_KEY = {
    "omp": "num_threads",
    "kokkos": "num_threads",
    "mpi": "num_procs",
}


@dataclass
class SampleResult:
    """Graded result for one generated sample of one problem."""

    name: str
    parallelism_model: str
    problem_type: str
    sample_idx: int
    built: bool
    correct: bool
    per_config: list[dict] = field(default_factory=list)
    speedup: float | None = None
    best_n_resources: int | None = None


def _config_resources(config: dict, parallelism_model: str) -> int:
    """Resource count used by a run config under the given parallelism model."""
    key = _RESOURCE_KEY.get(parallelism_model)
    if key is None:
        return 1
    return int(config.get(key, 1))


def sample_speedup(
    per_config: list[dict], parallelism_model: str
) -> tuple[float | None, int | None]:
    """Best speedup and resource count for one sample across its run configs.

    Considers only configs with ``valid is True`` and a positive ``time_s``.
    Speedup = baseline / best_parallel_time, where best_parallel_time is the
    minimum time over valid configs and baseline is the minimum available
    ``best_sequential_s``. Returns (None, None) if no valid config or no
    baseline is available.
    """
    valid = [
        c
        for c in per_config
        if c.get("valid") is True
        and c.get("time_s") is not None
        and c["time_s"] > 0
    ]
    if not valid:
        return (None, None)

    best = min(valid, key=lambda c: c["time_s"])
    best_parallel_time = best["time_s"]

    baselines = [
        c["best_sequential_s"]
        for c in valid
        if c.get("best_sequential_s") is not None
    ]
    if not baselines:
        return (None, None)
    baseline = min(baselines)

    speedup = baseline / best_parallel_time
    n_resources = _config_resources(best.get("config", {}), parallelism_model)
    return (speedup, n_resources)

"""Grading and metrics for ParEval benchmark runs.

Pure functions, no I/O. Mirrors the ParEval C++ driver stdout contract and the
metric definitions from the ParEval paper (pass@k via Chen et al. unbiased
estimator; speedup@k / efficiency@k via expected-max over k samples).
"""

import math
from dataclasses import dataclass, field
from typing import Optional


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
    build_err: Optional[str] = None


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


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Chen et al. (HumanEval) estimator: 1 - C(n-c, k)/C(n, k)."""
    if k <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if n - c < k:           # more correct than failures-left allow -> guaranteed hit
        return 1.0
    # 1 - prod_{i=n-c+1..n} (1 - k/i)  == 1 - C(n-c,k)/C(n,k)
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def expected_max_at_k(values: list[float], k: int) -> float:
    """Expected maximum of a uniformly random size-k subset drawn WITHOUT
    replacement from `values`. Incorrect samples must already be encoded as 0.0."""
    n = len(values)
    if n == 0 or k <= 0:
        return 0.0
    if k >= n:
        return max(values)
    vs = sorted(values)
    denom = math.comb(n, k)
    total = 0.0
    for i, v in enumerate(vs):       # i = count of elements strictly below position i
        # v is the max of a k-subset iff the other k-1 are drawn from the i below it
        total += v * math.comb(i, k - 1)
    return total / denom


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _sample_efficiency(s: "SampleResult") -> float:
    """Efficiency value encoding for a sample (0.0 if incorrect / unavailable)."""
    if not s.correct or s.speedup is None or not s.best_n_resources:
        return 0.0
    return s.speedup / s.best_n_resources


def _sample_speedup_value(s: "SampleResult") -> float:
    """Speedup value encoding for a sample (0.0 if incorrect / unavailable)."""
    if not s.correct or s.speedup is None:
        return 0.0
    return s.speedup


def _slice_metrics(slc: list["SampleResult"], ks: list[int]) -> dict:
    """Compute the metric block for one slice (list of SampleResults)."""
    by_name: dict[str, list[SampleResult]] = {}
    for s in slc:
        by_name.setdefault(s.name, []).append(s)

    n_samples = len(slc)
    n_problems = len(by_name)
    build_rate = _mean([1.0 if s.built else 0.0 for s in slc])
    correct_rate = _mean([1.0 if s.correct else 0.0 for s in slc])

    pass_block: dict[int, float] = {}
    speedup_block: dict[int, float] = {}
    efficiency_block: dict[int, float] = {}
    for k in ks:
        pass_vals: list[float] = []
        speedup_vals: list[float] = []
        efficiency_vals: list[float] = []
        for group in by_name.values():
            n = len(group)
            c = sum(1 for s in group if s.correct)
            pass_vals.append(pass_at_k(n, c, k))
            sp_list = [_sample_speedup_value(s) for s in group]
            eff_list = [_sample_efficiency(s) for s in group]
            speedup_vals.append(expected_max_at_k(sp_list, k))
            efficiency_vals.append(expected_max_at_k(eff_list, k))
        pass_block[k] = _mean(pass_vals)
        speedup_block[k] = _mean(speedup_vals)
        efficiency_block[k] = _mean(efficiency_vals)

    return {
        "n_problems": n_problems,
        "n_samples": n_samples,
        "build_rate": build_rate,
        "correct_rate": correct_rate,
        "pass@k": pass_block,
        "speedup@k": speedup_block,
        "efficiency@k": efficiency_block,
    }


def aggregate(samples: list["SampleResult"], ks: list[int]) -> dict:
    """Aggregate per-sample results into overall / per-model / per-type slices.

    Pure: does not add a ``trusted`` flag (the caller injects that later).
    """
    by_model: dict[str, list[SampleResult]] = {}
    by_type: dict[str, list[SampleResult]] = {}
    for s in samples:
        by_model.setdefault(s.parallelism_model, []).append(s)
        by_type.setdefault(s.problem_type, []).append(s)

    return {
        "overall": _slice_metrics(samples, ks),
        "by_model": {m: _slice_metrics(g, ks) for m, g in by_model.items()},
        "by_problem_type": {t: _slice_metrics(g, ks) for t, g in by_type.items()},
    }

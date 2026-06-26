import pytest
from benchmaker.pareval.sandbox_runner import CompletionGrader


def _grader(**kw):
    params = dict(max_threads=2, max_procs=2, run_reps=2,
                  build_timeout=30, run_timeout=60, cpuset="0-1")
    params.update(kw)
    return CompletionGrader(**params)


def test_compile_command_has_macro_and_sources():
    g = _grader()
    cmd = g.build_compile_command("03_dense_la_axpy", "omp",
                                  tmpdir="/t", problem_size="(1<<10)")[0]
    assert "-DUSE_OMP" in cmd
    assert "-I/t" in cmd
    assert "omp-driver.cc" in cmd and "cpu.cc" in cmd
    assert "generated-code.hpp" not in cmd          # written, not passed on cmdline
    assert '-DDRIVER_PROBLEM_SIZE="(1<<10)"' in cmd


def test_kokkos_compile_uses_cmake():
    g = _grader()
    cmds = g.build_compile_command("03_dense_la_axpy", "kokkos",
                                   tmpdir="/t", problem_size="(1<<9)")
    joined = " ".join(cmds)
    assert "cmake -B/t -S/t" in joined and "make -C/t" in joined
    assert "-DKokkos_ROOT=" in joined and "-DDRIVERS_CPP=" in joined
    assert "kokkos.cc" in joined
    assert ".o" not in joined
    assert '-DDRIVER_PROBLEM_SIZE="(1<<9)"' in joined


async def test_generation_error_is_unbuilt():
    g = _grader()
    comp = {"name": "03_dense_la_axpy", "parallelism_model": "omp",
            "problem_type": "t", "sample_idx": 0,
            "generated_code": None, "error": "no_code"}
    async def exec_fn(cmd, t): raise AssertionError("must not exec")
    async def write_file(p, b): raise AssertionError("must not write")
    res = await g.grade(comp, exec_fn, write_file)
    assert res.built is False and res.correct is False


async def test_compile_failure_records_build_err():
    g = _grader()
    comp = {"name": "03_dense_la_axpy", "parallelism_model": "omp",
            "problem_type": "t", "sample_idx": 0,
            "generated_code": "int f(){}", "error": None}
    calls = []
    async def exec_fn(cmd, t):
        calls.append(cmd)
        return (1, "error: expected ';'")
    async def write_file(p, b): pass
    res = await g.grade(comp, exec_fn, write_file)
    assert res.built is False and res.correct is False
    assert res.build_err and "expected" in res.build_err
    assert any("-DUSE_OMP" in c for c in calls)


async def test_passing_sample_computes_speedup():
    g = _grader()
    comp = {"name": "03_dense_la_axpy", "parallelism_model": "omp",
            "problem_type": "t", "sample_idx": 0,
            "generated_code": "int f(){}", "error": None}
    async def exec_fn(cmd, t):
        if "taskset" in cmd or "OMP_NUM_THREADS" in cmd:   # a run
            return (0, "Time: 0.01\nBestSequential: 0.04\nValidation: PASS\n")
        return (0, "")                                      # compile ok
    async def write_file(p, b): pass
    res = await g.grade(comp, exec_fn, write_file)
    assert res.built and res.correct
    assert res.speedup == pytest.approx(4.0)


async def test_kokkos_grade_writes_cmakelists():
    g = _grader()
    written = []
    comp = {"name": "03_dense_la_axpy", "parallelism_model": "kokkos",
            "problem_type": "t", "sample_idx": 0,
            "generated_code": "int f(){}", "error": None}
    async def exec_fn(cmd, t):
        return (0, "Time: 0.01\nBestSequential: 0.02\nValidation: PASS\n")
    async def write_file(p, b): written.append(str(p))
    await g.grade(comp, exec_fn, write_file)
    assert any(p.endswith("CMakeLists.txt") for p in written)
    assert any(p.endswith("generated-code.hpp") for p in written)


async def test_invalid_run_makes_incorrect():
    g = _grader()
    comp = {"name": "03_dense_la_axpy", "parallelism_model": "omp",
            "problem_type": "t", "sample_idx": 0,
            "generated_code": "int f(){}", "error": None}
    async def exec_fn(cmd, t):
        if "taskset" in cmd or "OMP_NUM_THREADS" in cmd:
            return (0, "Time: 0.01\nBestSequential: 0.04\nValidation: FAIL\n")
        return (0, "")
    async def write_file(p, b): pass
    res = await g.grade(comp, exec_fn, write_file)
    assert res.built is True and res.correct is False and res.speedup is None

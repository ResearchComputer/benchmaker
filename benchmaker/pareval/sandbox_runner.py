"""Per-completion grader: assemble -> compile -> run -> parse, over an injected
exec seam so it's testable without a real sandbox."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Awaitable, Callable, Optional

from benchmaker.pareval.grading import (
    SampleResult, parse_run_output, sample_speedup,
)
from benchmaker.pareval.launch_local import build_run_command, iter_run_configs

ExecFn = Callable[[str, float], Awaitable[tuple[int, str]]]      # (cmd, timeout)->(rc, stdout+stderr)
WriteFile = Callable[[str, bytes], Awaitable[None]]              # (path, content)->None

_HOST_DRIVERS = Path(__file__).parent / "drivers"               # packaged vendored copy


class CompletionGrader:
    def __init__(self, *, sandbox_drivers_cpp: str = "/opt/pareval/drivers/cpp",
                 kokkos_root: str = "/opt/kokkos", work_dir: str = "/tmp/pareval",
                 max_threads: int = 8, max_procs: int = 8, run_reps: int = 3,
                 build_timeout: float = 30.0, run_timeout: float = 120.0,
                 cpuset: str = "0-7"):
        self.sb = sandbox_drivers_cpp.rstrip("/")
        self.kokkos_root = kokkos_root
        self.work_dir = work_dir.rstrip("/")
        self.max_threads = max_threads
        self.max_procs = max_procs
        self.run_reps = run_reps
        self.build_timeout = build_timeout
        self.run_timeout = run_timeout
        self.cpuset = cpuset
        self._build_configs = json.loads((_HOST_DRIVERS / "build-configs.json").read_text())
        self._problem_sizes = json.loads((_HOST_DRIVERS / "problem-sizes.json").read_text())

    def _rel_problem_dir(self, name: str) -> str:
        """Resolve 'benchmarks/<category>/<name>' by globbing the host copy."""
        matches = list((_HOST_DRIVERS / "cpp" / "benchmarks").glob(f"*/{name}"))
        if not matches:
            raise FileNotFoundError(f"no benchmark dir for problem {name!r}")
        cat = matches[0].parent.name
        return f"benchmarks/{cat}/{name}"

    def _problem_size(self, name: str, model: str) -> str:
        return self._problem_sizes[name][model]

    def _kokkos_cmake_text(self) -> str:
        return (_HOST_DRIVERS / "cpp" / "ParEvalKokkos.cmake").read_text()

    def build_compile_command(self, problem: str, parallelism_model: str,
                              tmpdir: str, problem_size: str) -> list[str]:
        rel = self._rel_problem_dir(problem)
        sb = self.sb
        if parallelism_model == "kokkos":
            return [
                (f"cmake -B{tmpdir} -S{tmpdir} -DKokkos_ROOT={self.kokkos_root} "
                 f"-DDRIVERS_CPP={sb} -DPROBLEM_SRC={sb}/{rel}/kokkos.cc "
                 f'-DDRIVER_PROBLEM_SIZE="{problem_size}"'),
                f"make -C{tmpdir}",
            ]
        cfg = self._build_configs[parallelism_model]
        cxx, cxxflags = cfg["CXX"], cfg["CXXFLAGS"]
        macro = f"-DUSE_{parallelism_model.upper()}"
        cmd = (f'{cxx} {cxxflags} -I{sb} -I{sb}/models -I{tmpdir} {macro} '
               f'-DDRIVER_PROBLEM_SIZE="{problem_size}" '
               f"{sb}/models/{parallelism_model}-driver.cc {sb}/{rel}/cpu.cc "
               f"-o {tmpdir}/a.out")
        return [cmd]

    async def grade(self, completion: dict, exec_fn: ExecFn,
                    write_file: WriteFile) -> SampleResult:
        name = completion["name"]
        model = completion["parallelism_model"]
        ptype = completion.get("problem_type", "")
        idx = completion["sample_idx"]
        code = completion.get("generated_code")

        def _result(**kw):
            base = dict(name=name, parallelism_model=model, problem_type=ptype,
                        sample_idx=idx, built=False, correct=False, per_config=[],
                        speedup=None, best_n_resources=None, build_err=None)
            base.update(kw)
            return SampleResult(**base)

        if not code:
            return _result(build_err=completion.get("error") or "no_code")

        tmpdir = f"{self.work_dir}/{name}-{model}-{idx}"
        await write_file(f"{tmpdir}/generated-code.hpp", code.encode())
        if model == "kokkos":
            await write_file(f"{tmpdir}/CMakeLists.txt", self._kokkos_cmake_text().encode())

        size = self._problem_size(name, model)
        for cmd in self.build_compile_command(name, model, tmpdir, size):
            rc, out = await exec_fn(cmd, self.build_timeout)
            if rc != 0:
                return _result(built=False, build_err=out)

        exe = f"{tmpdir}/a.out"
        per_config = []
        for cfg in iter_run_configs(model, max_threads=self.max_threads,
                                    max_procs=self.max_procs):
            run_cmd = build_run_command(model, exe, cfg, self.cpuset)
            times, seqs, valids = [], [], []
            for _ in range(self.run_reps):
                rc, out = await exec_fn(run_cmd, self.run_timeout)
                p = parse_run_output(out)
                valids.append(p.valid is True)
                if p.valid and p.time_s is not None:
                    times.append(p.time_s)
                if p.best_sequential_s is not None:
                    seqs.append(p.best_sequential_s)
            per_config.append({
                "config": cfg,
                "valid": bool(valids) and all(valids),
                "time_s": min(times) if times else None,
                "best_sequential_s": min(seqs) if seqs else None,
                "reps": self.run_reps,
            })

        correct = bool(per_config) and all(pc["valid"] for pc in per_config)
        if correct:
            speedup, nres = sample_speedup(per_config, model)
        else:
            speedup, nres = None, None
        return _result(built=True, correct=correct, per_config=per_config,
                       speedup=speedup, best_n_resources=nres)

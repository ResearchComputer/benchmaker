"""End-to-end ParEval grading smoke test against a REAL local Docker container.

This compiles + runs hand-written, known-good (and deliberately wrong) C++
solutions for a simple ParEval problem through the full ``CompletionGrader``
pipeline, using a long-lived ``pareval-toolchain`` container as the exec
backend instead of Flash Sandbox. It proves the assemble -> compile -> run ->
parse pipeline works against real C++.

Gated: skipped unless ``docker`` and the ``pareval-toolchain`` image are both
present locally, so CI without them simply skips.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest

IMAGE = "pareval-toolchain"


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="docker or pareval-toolchain image unavailable",
)


class DockerBackend:
    """One long-lived container providing the exec + write_file seams.

    Files must persist across exec calls (compile writes ``a.out``, the run
    reads it), so a single container is reused for the whole grade.
    """

    def __init__(self) -> None:
        self.cid = subprocess.check_output(
            ["docker", "run", "-d", IMAGE, "sleep", "3600"]
        ).decode().strip()

    async def exec(self, command: str, timeout_s: float) -> tuple[int, str]:
        p = await asyncio.create_subprocess_exec(
            "docker", "exec", self.cid, "sh", "-c", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(p.communicate(), timeout=timeout_s + 30)
        return p.returncode, out.decode(errors="replace")

    async def write_file(self, path: str, content: bytes) -> None:
        # Create the parent dir, then stream the bytes in via stdin.
        p = await asyncio.create_subprocess_exec(
            "docker", "exec", "-i", self.cid, "sh", "-c",
            f'mkdir -p "$(dirname \'{path}\')" && cat > \'{path}\'',
            stdin=asyncio.subprocess.PIPE,
        )
        await p.communicate(content)
        if p.returncode != 0:
            raise RuntimeError(f"write_file failed for {path}")

    def close(self) -> None:
        subprocess.run(
            ["docker", "rm", "-f", self.cid],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


PROBLEM = "03_dense_la_axpy"

# Known-good axpy: z = alpha*x + y, elementwise. Matches the driver's
# correctAxpy reference exactly.
GOOD_SERIAL = """\
void axpy(double alpha, std::vector<double> const& x, std::vector<double> const& y, std::vector<double> &z) {
   for (size_t i = 0; i < x.size(); i += 1) {
      z[i] = alpha * x[i] + y[i];
   }
}"""

# Same kernel, parallelised with OpenMP.
GOOD_OMP = """\
void axpy(double alpha, std::vector<double> const& x, std::vector<double> const& y, std::vector<double> &z) {
   #pragma omp parallel for
   for (size_t i = 0; i < x.size(); i += 1) {
      z[i] = alpha * x[i] + y[i];
   }
}"""

# Compiles fine but produces the wrong answer (all zeros) -> validation FAIL.
WRONG_SERIAL = """\
void axpy(double alpha, std::vector<double> const& x, std::vector<double> const& y, std::vector<double> &z) {
   for (size_t i = 0; i < x.size(); i += 1) {
      z[i] = 0.0;
   }
}"""


def _completion(model: str, code: str) -> dict:
    return {
        "name": PROBLEM,
        "parallelism_model": model,
        "problem_type": "dense_la",
        "sample_idx": 0,
        "generated_code": code,
        "error": None,
    }


def _grader():
    from benchmaker.pareval.sandbox_runner import CompletionGrader

    return CompletionGrader(
        run_reps=1, max_threads=2, max_procs=2, cpuset="0-1",
        build_timeout=60, run_timeout=120,
    )


def _assemble(model: str, body: str) -> str:
    from benchmaker.pareval.dataset import load_prompts
    from benchmaker.pareval.generate import assemble_generated_code

    stub = [
        p
        for p in load_prompts(parallelism_models=[model], names=[PROBLEM])
    ][0].prompt
    return assemble_generated_code(stub, body)


@pytest.mark.asyncio
async def test_known_good_serial_axpy_passes():
    backend = DockerBackend()
    try:
        gen = _assemble("serial", GOOD_SERIAL)
        res = await _grader().grade(
            _completion("serial", gen), backend.exec, backend.write_file
        )
        assert res.built is True, res.build_err
        assert res.correct is True, res.per_config
        # Serial speedup is measured against its own best-sequential time.
        assert res.speedup is not None and res.speedup > 0, res.per_config
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_known_good_omp_axpy_passes():
    backend = DockerBackend()
    try:
        gen = _assemble("omp", GOOD_OMP)
        res = await _grader().grade(
            _completion("omp", gen), backend.exec, backend.write_file
        )
        assert res.built is True, res.build_err
        assert res.correct is True, res.per_config
        assert res.speedup is not None and res.speedup > 0, res.per_config
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_wrong_answer_fails_validation():
    backend = DockerBackend()
    try:
        gen = _assemble("serial", WRONG_SERIAL)
        res = await _grader().grade(
            _completion("serial", gen), backend.exec, backend.write_file
        )
        # Compiles fine, but the run's Validation must be FAIL.
        assert res.built is True, res.build_err
        assert res.correct is False
        assert res.speedup is None
    finally:
        backend.close()

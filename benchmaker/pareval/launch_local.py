"""Rewrite ParEval's SLURM ``srun`` launch-configs for single-node execution.

ParEval upstream drives benchmarks across a multi-node SLURM allocation,
emitting ``srun`` invocations that spread MPI ranks / OpenMP threads over
many nodes. Inside the benchmaker sandbox we run each generated executable
on a *single* node and pin it to an explicitly allocated cpuset. This module
holds the pure functions that perform that rewrite:

- :func:`build_run_command` turns a parallelism model + executable path +
  per-run config into a single-node shell command using ``taskset`` for CPU
  pinning, ``OMP_NUM_THREADS`` for OpenMP/Kokkos thread counts, and
  ``mpirun --bind-to core`` for MPI rank binding (no ``srun``).
- :func:`iter_run_configs` enumerates the thread/proc sweep as powers of two
  capped to the number of cores allocated to the sandbox.

These are pure functions: no I/O and no imports beyond the standard library.
"""

__all__ = ["build_run_command", "iter_run_configs"]


def _powers_of_two_up_to(n: int) -> list[int]:
    """Return ``[1, 2, 4, ...]`` up to and including the largest power of two <= n.

    Examples: n=8 -> [1, 2, 4, 8]; n=6 -> [1, 2, 4]; n=1 -> [1].
    For n < 1 the sweep is empty.
    """
    out: list[int] = []
    p = 1
    while p <= n:
        out.append(p)
        p *= 2
    return out


def iter_run_configs(
    parallelism_model: str, *, max_threads: int, max_procs: int
) -> list[dict]:
    """Enumerate the per-run sweep for a parallelism model, capped to cores.

    - ``serial`` -> ``[{}]`` (single run, no parallel knob)
    - ``omp`` / ``kokkos`` -> one config per power-of-two thread count up to
      ``max_threads``
    - ``mpi`` -> one config per power-of-two process count up to ``max_procs``

    Raises :class:`ValueError` for an unknown parallelism model.
    """
    if parallelism_model == "serial":
        return [{}]
    if parallelism_model in ("omp", "kokkos"):
        return [{"num_threads": n} for n in _powers_of_two_up_to(max_threads)]
    if parallelism_model == "mpi":
        return [{"num_procs": n} for n in _powers_of_two_up_to(max_procs)]
    raise ValueError(f"unknown parallelism_model: {parallelism_model!r}")


def build_run_command(
    parallelism_model: str, exec_path: str, run_config: dict, cpuset: str
) -> str:
    """Build a single-node, CPU-pinned shell command for one run.

    ``cpuset`` is a ``taskset -c`` spec such as ``"0-7"``. ``run_config``
    supplies ``num_threads`` (omp/kokkos) or ``num_procs`` (mpi).

    Raises :class:`ValueError` for an unknown parallelism model.
    """
    if parallelism_model == "serial":
        return f"taskset -c {cpuset} {exec_path}"
    if parallelism_model == "omp":
        nt = run_config["num_threads"]
        return f"OMP_NUM_THREADS={nt} taskset -c {cpuset} {exec_path} {nt}"
    if parallelism_model == "kokkos":
        nt = run_config["num_threads"]
        return (
            f"OMP_NUM_THREADS={nt} taskset -c {cpuset} {exec_path} "
            f"--kokkos-num-threads={nt}"
        )
    if parallelism_model == "mpi":
        np_ = run_config["num_procs"]
        return f"mpirun --bind-to core -n {np_} {exec_path}"
    raise ValueError(f"unknown parallelism_model: {parallelism_model!r}")

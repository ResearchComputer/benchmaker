import pytest
from benchmaker.pareval.launch_local import build_run_command, iter_run_configs


def test_serial_command():
    assert build_run_command("serial", "/x/a.out", {}, "0-3") == \
        "taskset -c 0-3 /x/a.out"

def test_omp_command_sets_threads_and_pins():
    cmd = build_run_command("omp", "/x/a.out", {"num_threads": 4}, "0-7")
    assert cmd == "OMP_NUM_THREADS=4 taskset -c 0-7 /x/a.out 4"

def test_kokkos_command():
    cmd = build_run_command("kokkos", "/x/a.out", {"num_threads": 2}, "0-1")
    assert cmd == "OMP_NUM_THREADS=2 taskset -c 0-1 /x/a.out --kokkos-num-threads=2"

def test_mpi_command_binds_ranks():
    cmd = build_run_command("mpi", "/x/a.out", {"num_procs": 8}, "0-7")
    assert cmd == "mpirun --bind-to core -n 8 /x/a.out"

def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_run_command("cuda", "/x/a.out", {}, "0-1")

def test_iter_configs_caps_threads_to_power_of_two():
    cfgs = iter_run_configs("omp", max_threads=8, max_procs=8)
    assert cfgs == [{"num_threads": n} for n in (1, 2, 4, 8)]

def test_iter_configs_serial_is_single_empty():
    assert iter_run_configs("serial", max_threads=8, max_procs=8) == [{}]

def test_iter_configs_caps_mpi_procs():
    cfgs = iter_run_configs("mpi", max_procs=4, max_threads=64)
    assert cfgs == [{"num_procs": n} for n in (1, 2, 4)]

def test_iter_configs_kokkos_uses_threads():
    cfgs = iter_run_configs("kokkos", max_threads=2, max_procs=64)
    assert cfgs == [{"num_threads": 1}, {"num_threads": 2}]

def test_powers_of_two_non_power_cap():
    # max_threads=6 -> largest power of two <= 6 is 4 -> [1,2,4]
    cfgs = iter_run_configs("omp", max_threads=6, max_procs=6)
    assert cfgs == [{"num_threads": n} for n in (1, 2, 4)]

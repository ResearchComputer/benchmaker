# ParEval CPU toolchain image

Docker image the Flash Sandbox runs to **compile + execute ParEval CPU code**.
It ships a ready-to-run C++ toolchain, a prebuilt Kokkos install, and the
vendored ParEval drivers so that **no per-sample dependency builds** happen at
grading time.

## What's inside

- **C++17 toolchain with OpenMP**: `g++` (Ubuntu 13.x)
- **MPI**: OpenMPI (`mpicxx` + `mpirun`)
- **Build tooling**: `cmake`, `make`, `git`
- **`taskset`** (from `util-linux`) — the grader uses it to pin cores
- **Kokkos `4.3.00`** (Threads backend) prebuilt and installed to `/opt/kokkos`
- **Vendored ParEval drivers** baked at `/opt/pareval/drivers/cpp`

### Paths the grader depends on

`benchmaker/pareval/sandbox_runner.py` builds compile commands that reference:

- driver sources at **`/opt/pareval/drivers/cpp`** (e.g. `models/`,
  `benchmarks/`, `*.hpp`, `ParEvalKokkos.cmake`, and the JSON configs)
- Kokkos at **`/opt/kokkos`**, resolved via `-DKokkos_ROOT=/opt/kokkos` passed to
  `find_package(Kokkos REQUIRED)`

If either path moves, the grader breaks.

## Build

The build **context is `benchmaker/pareval/`** so the Dockerfile can
`COPY drivers /opt/pareval/drivers`.

From the repo root:

```bash
docker build -f benchmaker/pareval/toolchain/Dockerfile -t pareval-toolchain benchmaker/pareval
```

Or from `benchmaker/pareval/`:

```bash
docker build -f toolchain/Dockerfile -t pareval-toolchain .
```

The Kokkos compile/install takes a few minutes. The Kokkos release tag is
pinned via the `KOKKOS_VERSION` build arg (default `4.3.00`, which is the
version verified to build here); override with
`--build-arg KOKKOS_VERSION=4.4.00` if needed.

## Shipping

The image **must be prebuilt and pushed** to wherever Flash Sandbox pulls
images from. `image_prepare` dominates provisioning cost, so the image has to
arrive ready-to-run — Kokkos and the drivers are baked in precisely so that no
dependency builds happen per sample.

## Verify

```bash
# Toolchain present
docker run --rm pareval-toolchain bash -lc \
  'g++ --version && mpicxx --version | head -1 && cmake --version | head -1 && which taskset mpirun'

# Vendored drivers present
docker run --rm pareval-toolchain bash -lc \
  'ls /opt/pareval/drivers/cpp/models && ls /opt/pareval/drivers/cpp/benchmarks | head && test -f /opt/pareval/drivers/cpp/ParEvalKokkos.cmake && echo DRIVERS_OK'

# OpenMP compile works
docker run --rm pareval-toolchain bash -lc \
  'echo "int main(){return 0;}" > /tmp/t.cc && g++ -std=c++17 -O3 -fopenmp /tmp/t.cc -o /tmp/t && /tmp/t && echo OMP_OK'

# Kokkos resolves via CMake
docker run --rm pareval-toolchain bash -lc '
  mkdir /tmp/kk && cd /tmp/kk &&
  printf "cmake_minimum_required(VERSION 3.16)\nproject(t CXX)\nfind_package(Kokkos REQUIRED)\nadd_executable(t t.cc)\ntarget_link_libraries(t Kokkos::kokkos)\n" > CMakeLists.txt &&
  printf "#include <Kokkos_Core.hpp>\nint main(int c,char**v){Kokkos::initialize(c,v);Kokkos::finalize();return 0;}\n" > t.cc &&
  cmake -B build -S . -DKokkos_ROOT=/opt/kokkos >/dev/null 2>&1 && cmake --build build >/dev/null 2>&1 && ./build/t && echo KOKKOS_OK'
```

Expect `DRIVERS_OK`, `OMP_OK`, and `KOKKOS_OK`.

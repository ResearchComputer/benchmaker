FROM ubuntu:24.04

# C++17 toolchain (g++/OpenMP), MPI (mpicxx + mpirun), build tooling, git,
# and taskset (from util-linux) used by the grader to pin cores.
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ make cmake git libopenmpi-dev openmpi-bin util-linux ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Prebuild Kokkos (Threads backend) to a fixed prefix so there is no per-task
# build. The grader passes -DKokkos_ROOT=/opt/kokkos to find_package(Kokkos).
ARG KOKKOS_VERSION=4.3.00
RUN git clone --depth 1 -b ${KOKKOS_VERSION} https://github.com/kokkos/kokkos /opt/kokkos-src \
 && cmake -S /opt/kokkos-src -B /opt/kokkos-build \
      -DCMAKE_INSTALL_PREFIX=/opt/kokkos -DKokkos_ENABLE_THREADS=ON -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /opt/kokkos-build --target install -j \
 && rm -rf /opt/kokkos-src /opt/kokkos-build
ENV KOKKOS_ROOT=/opt/kokkos

# Bake the vendored ParEval drivers at the sandbox-side path the grader uses.
# Build context is benchmaker/pareval/, so "drivers" is benchmaker/pareval/drivers.
COPY drivers /opt/pareval/drivers

WORKDIR /opt/pareval

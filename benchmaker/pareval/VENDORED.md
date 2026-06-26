# Vendored ParEval assets

Source: https://github.com/parallelcodefoundry/ParEval
Commit: 9e2a9afafa2c9686fdd3310defde0f9a8c3731c1
License: MIT (see drivers/LICENSE — UMD Parallel Software and Systems Group)

Vendored subset (CPU-only path):
- drivers/cpp/models/{serial,omp,mpi,kokkos}-driver.cc
- drivers/cpp/benchmarks/<problem>/ (baseline.hpp + per-problem driver sources)
- drivers/cpp/utilities.hpp (defines NO_INLINE) + shared headers
- drivers/{build,launch,problem-sizes}-configs.json
- data/generation-prompts.json (60 prompts x 7 models = 420; we use the 4 CPU models)
- drivers/cpp/ParEvalKokkos.cmake — OUR OWN CMakeLists for the kokkos build path
  (upstream's KokkosCMakeLists.txt is deliberately NOT vendored)

NOT vendored: cuda/hip drivers, generate/ scripts, analysis/ scripts, tpl/,
upstream KokkosCMakeLists.txt.

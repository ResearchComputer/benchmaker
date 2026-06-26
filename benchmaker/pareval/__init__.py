"""ParEval parallel-code benchmark integration.

A self-driving ``benchmaker pareval`` recipe: generate code completions for
ParEval prompts, compile + run them in a Flash Sandbox against the vendored
ParEval C++ drivers, and report pass@k / speedup@k / efficiency@k.

CPU-only v1: serial, omp, mpi, kokkos. See
docs/superpowers/specs/2026-06-26-pareval-integration-design.md.
"""

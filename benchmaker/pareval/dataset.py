"""Load + filter the vendored ParEval generation prompts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

CPU_MODELS = ("serial", "omp", "mpi", "kokkos")
_GPU_MODELS = ("cuda", "hip")
_DEFERRED = ("mpi+omp",)

_DATA = Path(__file__).parent / "data" / "generation-prompts.json"


@dataclass(frozen=True)
class ParEvalPrompt:
    name: str
    problem_type: str
    language: str
    parallelism_model: str
    prompt: str


def load_prompts(
    path=None,
    *,
    parallelism_models: Sequence[str] = CPU_MODELS,
    problem_types: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
) -> list[ParEvalPrompt]:
    models = tuple(parallelism_models)
    for m in models:
        if m in _GPU_MODELS:
            raise ValueError(f"GPU parallelism model {m!r} is out of scope (CPU-only v1)")
        if m in _DEFERRED:
            raise ValueError(f"parallelism model {m!r} is deferred (not in v1)")
        if m not in CPU_MODELS:
            raise ValueError(f"unknown parallelism model {m!r}; expected one of {CPU_MODELS}")

    raw = json.loads(Path(path or _DATA).read_text())
    want_models = set(models)
    want_types = set(problem_types) if problem_types else None
    want_names = set(names) if names else None

    out: list[ParEvalPrompt] = []
    for r in raw:
        if r["parallelism_model"] not in want_models:
            continue
        if want_types is not None and r["problem_type"] not in want_types:
            continue
        if want_names is not None and r["name"] not in want_names:
            continue
        out.append(ParEvalPrompt(
            name=r["name"], problem_type=r["problem_type"], language=r["language"],
            parallelism_model=r["parallelism_model"], prompt=r["prompt"],
        ))
    return out

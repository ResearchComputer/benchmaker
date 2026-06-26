"""ParEval 3-stage orchestrator: generate (skippable) -> sandbox grade
(resumable) -> aggregate. Writes completions.jsonl / runs.jsonl / metrics.json."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from benchmaker.pareval.dataset import load_prompts
from benchmaker.pareval.generate import generate_one, make_send_fn          # module-level seam (monkeypatchable)
from benchmaker.pareval.flash import FlashSandbox                            # module-level seam (monkeypatchable)
from benchmaker.pareval.grading import SampleResult, aggregate
from benchmaker.pareval.sandbox_runner import CompletionGrader


@dataclass
class ParEvalConfig:
    out_dir: Path
    parallelism_models: tuple[str, ...] = ("serial", "omp", "mpi", "kokkos")
    problem_types: Optional[tuple[str, ...]] = None
    names: Optional[tuple[str, ...]] = None
    num_samples: int = 1
    k: tuple[int, ...] = (1,)
    # generation source (exactly one path): live model OR a precomputed completions file
    completions_path: Optional[Path] = None
    model: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.2
    regenerate: bool = False
    # grading
    sandbox_url: str = ""
    image: str = "pareval-toolchain"
    endpoint_prefix: str = "/sandboxes"
    sandbox_drivers_cpp: str = "/opt/pareval/drivers/cpp"
    kokkos_root: str = "/opt/kokkos"
    max_threads: int = 8
    max_procs: int = 8
    run_reps: int = 3
    build_timeout: float = 30.0
    run_timeout: float = 120.0
    concurrency: int = 4
    exclusive_cpus: bool = False
    cpuset: Optional[str] = None


def _completion_key(rec: dict) -> tuple:
    return (rec["name"], rec["parallelism_model"], rec["sample_idx"])


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


async def _stage_completions(cfg: ParEvalConfig, completions_file: Path) -> None:
    """Ensure ``completions_file`` holds the completion records for Stage 2."""
    # Source 1: a precomputed completions file -> normalize into completions_file.
    if cfg.completions_path is not None:
        src = Path(cfg.completions_path)
        if src.resolve() == completions_file.resolve():
            return  # already the target file
        recs = _read_jsonl(src)
        completions_file.write_text(
            "".join(json.dumps(r) + "\n" for r in recs)
        )
        return

    # Source 2: a cached completions file already present -> reuse it.
    if completions_file.exists() and not cfg.regenerate:
        return

    # Source 3: live generation against the model.
    send_fn = make_send_fn(
        api_base=cfg.api_base, model=cfg.model,
        api_key=cfg.api_key, temperature=cfg.temperature,
    )
    try:
        prompts = load_prompts(
            parallelism_models=cfg.parallelism_models,
            problem_types=cfg.problem_types,
            names=cfg.names,
        )
        records: list[dict] = []
        for prompt in prompts:
            for sample_idx in range(cfg.num_samples):
                rec = await generate_one(send_fn, prompt, sample_idx=sample_idx)
                records.append(rec)
        completions_file.write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
    finally:
        await send_fn.aclose()


async def _stage_grade(cfg: ParEvalConfig, completions_file: Path, runs_file: Path) -> None:
    """Grade every completion not already present in ``runs_file`` (resumable)."""
    done: set[tuple] = set()
    if runs_file.exists():
        for rec in _read_jsonl(runs_file):
            done.add((rec["name"], rec["parallelism_model"], rec["sample_idx"]))

    completions = _read_jsonl(completions_file)
    todo = [c for c in completions if _completion_key(c) not in done]

    cpuset = cfg.cpuset or f"0-{max(cfg.max_threads, cfg.max_procs) - 1}"
    grader = CompletionGrader(
        sandbox_drivers_cpp=cfg.sandbox_drivers_cpp,
        kokkos_root=cfg.kokkos_root,
        max_threads=cfg.max_threads,
        max_procs=cfg.max_procs,
        run_reps=cfg.run_reps,
        build_timeout=cfg.build_timeout,
        run_timeout=cfg.run_timeout,
        cpuset=cpuset,
    )

    sem = asyncio.Semaphore(cfg.concurrency)
    write_lock = asyncio.Lock()

    async def _grade_one(rec: dict) -> SampleResult:
        async with FlashSandbox(
            cfg.sandbox_url, image=cfg.image, endpoint_prefix=cfg.endpoint_prefix
        ) as sb:
            return await grader.grade(rec, sb.exec, sb.write_file)

    async def _run_and_record(rec: dict) -> None:
        async with sem:
            try:
                result = await _grade_one(rec)
            except Exception as e:  # noqa: BLE001 - infra error must not kill the run
                print(
                    f"[pareval] grade failed for {_completion_key(rec)}: {e}",
                    file=sys.stderr,
                )
                result = SampleResult(
                    name=rec.get("name", ""),
                    parallelism_model=rec.get("parallelism_model", ""),
                    problem_type=rec.get("problem_type", ""),
                    sample_idx=rec.get("sample_idx", 0),
                    built=False,
                    correct=False,
                    build_err=f"infra: {e}",
                )
        async with write_lock:
            with runs_file.open("a") as f:
                f.write(json.dumps(dataclasses.asdict(result)) + "\n")

    if todo:
        await asyncio.gather(*(_run_and_record(rec) for rec in todo))


def _stage_aggregate(cfg: ParEvalConfig, runs_file: Path, cpuset: str) -> dict:
    samples = [SampleResult(**d) for d in _read_jsonl(runs_file)] if runs_file.exists() else []
    metrics = aggregate(samples, list(cfg.k))
    metrics["trusted"] = bool(cfg.exclusive_cpus)
    metrics["config"] = {
        "parallelism_models": list(cfg.parallelism_models),
        "num_samples": cfg.num_samples,
        "k": list(cfg.k),
        "run_reps": cfg.run_reps,
        "cpuset": cpuset,
        "exclusive_cpus": bool(cfg.exclusive_cpus),
    }
    (cfg.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


async def run_pareval(cfg: ParEvalConfig) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    completions_file = cfg.out_dir / "completions.jsonl"
    runs_file = cfg.out_dir / "runs.jsonl"

    await _stage_completions(cfg, completions_file)
    await _stage_grade(cfg, completions_file, runs_file)

    cpuset = cfg.cpuset or f"0-{max(cfg.max_threads, cfg.max_procs) - 1}"
    return _stage_aggregate(cfg, runs_file, cpuset)

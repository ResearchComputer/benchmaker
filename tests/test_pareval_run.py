import json
import dataclasses
import pytest
from pathlib import Path
from benchmaker.pareval.run import ParEvalConfig, run_pareval


class FakeFlash:
    def __init__(self, base_url, *, image, endpoint_prefix="/sandboxes", headers=None, create_timeout_s=120.0):
        pass
    async def __aenter__(self): return self
    async def __aexit__(self, *exc): return None
    async def exec(self, cmd, timeout_s):
        if "taskset" in cmd or "OMP_NUM_THREADS" in cmd or "mpirun" in cmd:
            return (0, "Time: 0.01\nBestSequential: 0.04\nValidation: PASS\n")
        return (0, "")   # compile
    async def write_file(self, path, content): return None


def _write_completions(p: Path, recs):
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))


def _completion(name="03_dense_la_axpy", model="omp", idx=0, code="int f(){}"):
    return {"name": name, "parallelism_model": model, "problem_type": "dense_la",
            "sample_idx": idx, "raw_reply": "", "generated_code": code,
            "error": None, "usage": {}}


async def test_grade_from_completions_file_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmaker.pareval.run.FlashSandbox", FakeFlash)
    comp = tmp_path / "in.jsonl"
    _write_completions(comp, [_completion(idx=0), _completion(idx=1)])
    cfg = ParEvalConfig(out_dir=tmp_path / "out", completions_path=comp,
                        parallelism_models=("omp",), num_samples=2, k=(1,),
                        sandbox_url="http://x", exclusive_cpus=True)
    metrics = await run_pareval(cfg)
    assert (cfg.out_dir / "runs.jsonl").exists()
    assert (cfg.out_dir / "metrics.json").exists()
    assert metrics["trusted"] is True
    assert metrics["overall"]["pass@k"][1] == pytest.approx(1.0)  # both samples pass


async def test_completions_path_bypasses_model(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmaker.pareval.run.FlashSandbox", FakeFlash)
    def boom(**kw): raise AssertionError("model must not be called")
    monkeypatch.setattr("benchmaker.pareval.run.make_send_fn", boom)
    comp = tmp_path / "in.jsonl"
    _write_completions(comp, [_completion()])
    cfg = ParEvalConfig(out_dir=tmp_path / "out", completions_path=comp,
                        parallelism_models=("omp",), num_samples=1, sandbox_url="http://x")
    await run_pareval(cfg)  # must not raise


async def test_cached_completions_skip_regeneration(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmaker.pareval.run.FlashSandbox", FakeFlash)
    def boom(**kw): raise AssertionError("should not regenerate")
    monkeypatch.setattr("benchmaker.pareval.run.make_send_fn", boom)
    out = tmp_path / "out"; out.mkdir()
    _write_completions(out / "completions.jsonl", [_completion()])
    cfg = ParEvalConfig(out_dir=out, parallelism_models=("omp",), num_samples=1,
                        sandbox_url="http://x", regenerate=False)
    await run_pareval(cfg)  # uses cached completions, no model call


async def test_live_generation_when_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmaker.pareval.run.FlashSandbox", FakeFlash)
    captured = {}
    def fake_make_send_fn(**kw):
        async def send_fn(messages): return ("```cpp\nint f(){ return 0; }\n```", None)
        send_fn.aclose = _noop_aclose
        captured["called"] = True
        return send_fn
    monkeypatch.setattr("benchmaker.pareval.run.make_send_fn", fake_make_send_fn)
    cfg = ParEvalConfig(out_dir=tmp_path / "out", parallelism_models=("omp",),
                        problem_types=("geometry",), num_samples=1,
                        model="m", api_base="http://api", sandbox_url="http://x")
    metrics = await run_pareval(cfg)
    assert captured.get("called") is True
    assert (cfg.out_dir / "completions.jsonl").exists()
    assert metrics["overall"]["n_samples"] >= 1


async def _noop_aclose(): return None


async def test_resume_skips_already_graded(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmaker.pareval.run.FlashSandbox", FakeFlash)
    out = tmp_path / "out"; out.mkdir()
    _write_completions(out / "completions.jsonl", [_completion(idx=0), _completion(idx=1)])
    # pre-seed runs.jsonl with sample 0 already graded -> only sample 1 should be graded
    pre = {"name": "03_dense_la_axpy", "parallelism_model": "omp", "problem_type": "dense_la",
           "sample_idx": 0, "built": True, "correct": True, "per_config": [],
           "speedup": None, "best_n_resources": None, "build_err": None}
    (out / "runs.jsonl").write_text(json.dumps(pre) + "\n")
    graded_idxs = []
    class TrackFlash(FakeFlash):
        async def write_file(self, path, content):
            # record which sample dir is being graded
            graded_idxs.append(path)
            return None
    monkeypatch.setattr("benchmaker.pareval.run.FlashSandbox", TrackFlash)
    cfg = ParEvalConfig(out_dir=out, parallelism_models=("omp",), num_samples=2,
                        completions_path=out / "completions.jsonl", sandbox_url="http://x")
    await run_pareval(cfg)
    # sample 1 graded (its tmpdir path mentions -1), sample 0 not re-graded
    assert any("-omp-1" in p for p in graded_idxs)
    assert not any("-omp-0" in p for p in graded_idxs)

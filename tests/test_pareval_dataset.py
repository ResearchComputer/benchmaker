import pytest
from benchmaker.pareval.dataset import load_prompts, CPU_MODELS, ParEvalPrompt


def test_load_defaults_to_cpu_models_only():
    prompts = load_prompts()
    assert prompts, "expected packaged prompts to load"
    models = {p.parallelism_model for p in prompts}
    assert models <= set(CPU_MODELS)
    # 60 problems per CPU model
    assert len(prompts) == 60 * len(CPU_MODELS)
    assert all(isinstance(p, ParEvalPrompt) for p in prompts)


def test_filter_by_model_and_type():
    prompts = load_prompts(parallelism_models=["omp"], problem_types=["geometry"])
    assert prompts
    assert all(p.parallelism_model == "omp" and p.problem_type == "geometry"
               for p in prompts)


def test_filter_by_name():
    prompts = load_prompts(parallelism_models=["omp"])
    one = prompts[0].name
    only = load_prompts(parallelism_models=["omp"], names=[one])
    assert {p.name for p in only} == {one}


def test_rejects_gpu_model():
    with pytest.raises(ValueError, match="cuda"):
        load_prompts(parallelism_models=["cuda"])

import pytest


def _base_params(**overrides):
    """A complete params dict for ``_build_config`` (defaults mirror the CLI)."""
    p = dict(
        model=None,
        api_base=None,
        api_key=None,
        completions=None,
        sandbox_url="http://x",
        parallelism_models="serial,omp,mpi,kokkos",
        problem_type=(),
        problem=(),
        num_samples=1,
        k="1",
        temperature=0.2,
        max_threads=8,
        max_procs=8,
        run_reps=3,
        build_timeout=30.0,
        run_timeout=120.0,
        concurrency=4,
        exclusive_cpus=False,
        cpuset=None,
        image="pareval-toolchain",
        sandbox_drivers_cpp="/opt/pareval/drivers/cpp",
        kokkos_root="/opt/kokkos",
        endpoint_prefix="/sandboxes",
        regenerate=False,
    )
    p.update(overrides)
    return p


def test_build_config_completions_with_model_env_no_conflict(tmp_path):
    # --completions provided AND OPENAI_MODEL in env -> completions wins, no UsageError.
    from benchmaker.recipes.pareval import _build_config
    comp = tmp_path / "c.jsonl"
    comp.write_text("{}\n")
    cfg = _build_config(
        _base_params(completions=str(comp)),
        {"OPENAI_MODEL": "m"},
        tmp_path / "out",
    )
    assert cfg.completions_path is not None
    assert cfg.model is None


def test_build_config_no_source_raises(tmp_path):
    import click
    from benchmaker.recipes.pareval import _build_config
    with pytest.raises(click.UsageError):
        _build_config(_base_params(), {}, tmp_path / "out")


def test_build_config_both_explicit_model_and_completions_raises(tmp_path):
    import click
    from benchmaker.recipes.pareval import _build_config
    comp = tmp_path / "c.jsonl"
    comp.write_text("{}\n")
    with pytest.raises(click.UsageError):
        _build_config(
            _base_params(model="m", completions=str(comp)),
            {},
            tmp_path / "out",
        )


def test_build_config_model_only_builds(tmp_path):
    from benchmaker.recipes.pareval import _build_config
    cfg = _build_config(_base_params(model="m"), {}, tmp_path / "out")
    assert cfg.model == "m"
    assert cfg.completions_path is None


def test_build_config_missing_sandbox_raises(tmp_path):
    import click
    from benchmaker.recipes.pareval import _build_config
    with pytest.raises(click.UsageError):
        _build_config(
            _base_params(model="m", sandbox_url=None), {}, tmp_path / "out"
        )


def test_pareval_recipe_registered():
    from benchmaker.recipes import get
    r = get("pareval")
    assert r.name == "pareval"
    assert r.wants_load_options is False


def test_pareval_help_lists_key_flags():
    from click.testing import CliRunner
    from benchmaker.recipes import get
    from benchmaker.recipes._factory import make_command
    cmd = make_command(get("pareval"))
    res = CliRunner().invoke(cmd, ["--help"])
    assert res.exit_code == 0
    for flag in ["--completions", "--num-samples", "--parallelism-models",
                 "--sandbox-url", "--k", "--model", "--out-dir"]:
        assert flag in res.output


def test_pareval_requires_generation_source(monkeypatch):
    # neither --model nor --completions, but a sandbox url set -> UsageError
    from click.testing import CliRunner
    from benchmaker.recipes import get
    from benchmaker.recipes._factory import make_command
    monkeypatch.delenv("OPENAI_COMPATIBLE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    cmd = make_command(get("pareval"))
    res = CliRunner().invoke(cmd, ["--sandbox-url", "http://x", "--dotenv", ""])
    assert res.exit_code != 0
    assert "completions" in res.output.lower() or "model" in res.output.lower()

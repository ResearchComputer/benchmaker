import pytest


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

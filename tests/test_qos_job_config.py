"""QoS plumbing: recipe flags -> EnvironmentConfig.kwargs + JobConfig.verifier_timeout_multiplier.

Task B2 — verify that ``_build_job_config`` propagates the QoS knobs into the
harbor ``JobConfig`` only when QoS is enabled. The ``base`` dict below mirrors
every attribute ``_build_job_config`` reads off the namespace; the QoS keys are
the ones under test.
"""
import argparse

from benchmaker.swebench.harbor_eval import _build_job_config


def _ns(**over):
    base = dict(
        dataset="swebench-verified",
        agent="pi-host",
        model="m",
        api_key="replay",
        api_base="http://x",
        agent_kwarg=[],
        agent_config_file=None,
        n_tasks=1,
        # Mirror production: the recipe's base_ns sets task=<list> (from
        # _resolve_task_filter) and exclude_task=None.
        task=[],
        exclude_task=None,
        n_attempts=1,
        timeout_multiplier=1.0,
        force_build=False,
        backend_type="docker",
        request_timeout_sec=120.0,
        agent_ready_timeout_sec=300.0,
        jobs_dir=None,
        concurrency=1,
        job_name="j",
        qos_enabled=False,
        on_demand_cpu_weight=10000,
        best_effort_cpu_weight=10,
        qos_verifier_timeout_multiplier=2.0,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_qos_off_no_env_kwargs():
    jc = _build_job_config(_ns(qos_enabled=False))
    assert "qos_enabled" not in jc.environment.kwargs
    assert jc.verifier_timeout_multiplier is None  # untouched


def test_qos_on_sets_env_kwargs_and_multiplier():
    jc = _build_job_config(_ns(qos_enabled=True))
    assert jc.environment.kwargs["qos_enabled"] is True
    assert jc.environment.kwargs["on_demand_cpu_weight"] == 10000
    assert jc.environment.kwargs["best_effort_cpu_weight"] == 10
    assert jc.verifier_timeout_multiplier == 2.0


def test_no_exclude_task_attr_does_not_raise():
    """The swebench recipe (and harbor_eval's own CLI) build a namespace with
    no ``exclude_task`` attribute at all — _build_job_config must read it
    defensively rather than AttributeError. Regression for the sweep crash."""
    ns = _ns()
    del ns.exclude_task
    jc = _build_job_config(ns)
    assert jc.datasets[0].exclude_task_names is None

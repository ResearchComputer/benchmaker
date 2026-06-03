"""SWE-bench coding-agent + grading primitives, native to benchmaker.

This subpackage holds the reusable machinery that used to live under
``examples/coding_agent`` and was forked into the agent-warmup tooling:

* :mod:`benchmaker.swebench.agent` — a compact mini-swe-agent control loop
  (:class:`CodingAgent`) whose shell actions run through an injected executor;
* :mod:`benchmaker.swebench.grading` — image resolution + authoritative
  grading via the official ``swebench`` package (the single source of truth);
* :mod:`benchmaker.swebench.native_eval` — a native SWE-bench rollout+grade
  workload (:class:`SWEBenchAgent`) that boots a prebuilt eval image per
  instance on a Flash Sandbox.

The harbor adapters (:mod:`benchmaker.swebench.harbor_agent`,
:mod:`benchmaker.swebench.harbor_eval`) are intentionally *not* re-exported
here: they import the optional ``harbor`` package, so import them directly only
where harbor is installed.
"""

from __future__ import annotations

from benchmaker.swebench.agent import SUBMIT_TOKEN, CodingAgent, Executor
from benchmaker.swebench.grading import (
    DEFAULT_IMAGE_ORG,
    DEFAULT_IMAGE_REGISTRY,
    as_list,
    grade,
    instance_image_key,
    make_test_spec,
    normalise_instance_id,
)
from benchmaker.swebench.native_eval import SWEBenchAgent

__all__ = [
    "CodingAgent",
    "Executor",
    "SUBMIT_TOKEN",
    "SWEBenchAgent",
    "grade",
    "make_test_spec",
    "instance_image_key",
    "normalise_instance_id",
    "as_list",
    "DEFAULT_IMAGE_ORG",
    "DEFAULT_IMAGE_REGISTRY",
]

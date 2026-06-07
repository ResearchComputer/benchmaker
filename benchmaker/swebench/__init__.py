"""SWE-bench coding-agent + grading primitives, native to benchmaker.

This subpackage holds the reusable machinery shared by the agent-warmup tooling
and the harbor SWE-bench path:

* :mod:`benchmaker.swebench.agent` — a compact mini-swe-agent control loop
  (:class:`CodingAgent`) whose shell actions run through an injected executor;
* :mod:`benchmaker.swebench.grading` — image resolution + authoritative
  grading via the official ``swebench`` package (the single source of truth,
  used by ``tools/agent_warmup``).

SWE-bench evaluation runs through **harbor** — see
:mod:`benchmaker.swebench.harbor_eval` (the engine behind ``benchmaker
swebench``) and :mod:`benchmaker.swebench.harbor_agent` /
:mod:`benchmaker.swebench.pi_agent` (the agents). Those import the ``harbor``
package at module top, so they're not re-exported here; import them directly.
"""

from __future__ import annotations

from benchmaker.swebench.agent import SUBMIT_TOKEN, CodingAgent, Executor
from benchmaker.swebench.grading import (
    DEFAULT_IMAGE_MIRROR,
    IMAGE_MIRRORS,
    as_list,
    grade,
    instance_image_key,
    make_test_spec,
    normalise_instance_id,
)

__all__ = [
    "CodingAgent",
    "Executor",
    "SUBMIT_TOKEN",
    "grade",
    "make_test_spec",
    "instance_image_key",
    "normalise_instance_id",
    "as_list",
    "DEFAULT_IMAGE_MIRROR",
    "IMAGE_MIRRORS",
]

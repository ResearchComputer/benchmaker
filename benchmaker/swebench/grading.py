"""SWE-bench image resolution + authoritative grading via the ``swebench`` package.

These are the harbor-equivalent *verification primitives*, native to benchmaker.
Harbor leans on a registered dataset + an in-sandbox verifier that writes
``reward.txt``; here we do the same job with the official ``swebench`` package:

- resolve each instance to its prebuilt per-instance eval image (the repo is
  already checked out at ``base_commit`` under ``/testbed`` with deps installed),
  mirrored to the public ghcr ``swe-images`` registry so we dodge Docker Hub
  rate limits — see ``tools/swe_images``;
- build the official ``swebench`` ``TestSpec`` (its ``eval_script`` resets the
  test files, applies the hidden ``test_patch``, and runs the tests, plus the
  per-repo log parser);
- classify a test log into FAIL_TO_PASS / PASS_TO_PASS resolution.

There is no async / no sandbox here on purpose: a caller runs ``eval_script``
wherever it likes (a Flash Sandbox pod) and feeds the captured log back to
:func:`grade`. That keeps these functions trivially unit-testable.
"""

from __future__ import annotations

import json
from typing import Any

# ghcr mirror of the SWE-bench per-instance images. The publish tool maps a
# source ``swebench/sweb.eval.<arch>.<id>`` to ``ghcr.io/<org>/sweb.eval.<arch>.<id>``
# (registry + first namespace dropped), so we construct the mirror ref directly
# from the instance id rather than rewriting swebench's Docker Hub key.
DEFAULT_IMAGE_ORG = "swe-images"
DEFAULT_IMAGE_REGISTRY = "ghcr.io"

# swebench's namespace for *Docker Hub* image keys. We only pass it to
# ``make_test_spec`` so the spec resolves cleanly; we don't use the resulting
# ``instance_image_key`` (we build the ghcr ref ourselves).
SWEBENCH_NAMESPACE = "swebench"


def normalise_instance_id(instance_id: str) -> str:
    """swebench's image-key normalisation: lower-case, ``__`` -> ``_1776_``."""
    return instance_id.lower().replace("__", "_1776_")


def instance_image_key(
    instance_id: str,
    *,
    org: str = DEFAULT_IMAGE_ORG,
    registry: str = DEFAULT_IMAGE_REGISTRY,
    arch: str = "x86_64",
) -> str:
    """Return the ghcr mirror ref for an instance's prebuilt eval image."""
    norm = normalise_instance_id(instance_id)
    return f"{registry}/{org}/sweb.eval.{arch}.{norm}:latest"


def make_test_spec(instance: dict[str, Any]) -> Any:
    """Build the swebench ``TestSpec`` (handles the cross-version import path).

    Accepts a raw SWE-bench row — including the HF form where ``FAIL_TO_PASS`` /
    ``PASS_TO_PASS`` are JSON strings; ``make_test_spec`` parses those itself.
    """
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec as _mk
    except ImportError:  # older swebench layout
        from swebench.harness.test_spec import make_test_spec as _mk  # type: ignore
    return _mk(instance, namespace=SWEBENCH_NAMESPACE)


def grade(spec: Any, log_text: str) -> dict[str, Any]:
    """Classify an eval log with swebench's authoritative grading.

    Returns ``{resolved, fail_to_pass, pass_to_pass, f2p_pass, f2p_total,
    p2p_pass, p2p_total}``. ``resolved`` is True only when every FAIL_TO_PASS
    test flipped to passing AND every PASS_TO_PASS test still passes (swebench's
    ``ResolvedStatus.FULL``).
    """
    from swebench.harness.constants import ResolvedStatus
    from swebench.harness.grading import get_eval_tests_report, get_resolution_status
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

    parser = MAP_REPO_TO_PARSER[spec.repo]
    status_map = parser(log_text, spec)  # {test_name: PASSED/FAILED/...}
    eval_ref = {
        "FAIL_TO_PASS": spec.FAIL_TO_PASS,
        "PASS_TO_PASS": spec.PASS_TO_PASS,
    }
    report = get_eval_tests_report(status_map, eval_ref)
    resolved = get_resolution_status(report) == ResolvedStatus.FULL.value
    ftp = report.get("FAIL_TO_PASS", {})
    ptp = report.get("PASS_TO_PASS", {})
    return {
        "resolved": resolved,
        "fail_to_pass": ftp,
        "pass_to_pass": ptp,
        "f2p_pass": len(ftp.get("success", [])),
        "f2p_total": len(ftp.get("success", [])) + len(ftp.get("failure", [])),
        "p2p_pass": len(ptp.get("success", [])),
        "p2p_total": len(ptp.get("success", [])) + len(ptp.get("failure", [])),
    }


def as_list(value: Any) -> list[Any]:
    """Coerce a SWE-bench test-name field to a list.

    HF rows store ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` as JSON strings; the raw
    dataset uses real lists. Tolerate both (and ``None``).
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []

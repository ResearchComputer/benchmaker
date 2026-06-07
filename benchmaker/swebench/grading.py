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

# Sources of prebuilt per-instance eval images on ghcr. Each is a "mirror"
# preset describing how to construct the image ref from a raw SWE-bench
# instance id. We build the ref directly rather than rewriting swebench's
# Docker Hub key, because the two public mirrors disagree on both the image
# name and whether the instance id is normalised:
#
#   swe-images      Our own mirror of swebench's *official* Docker Hub images,
#                   preserving swebench's normalised key (lower-case,
#                   ``__`` -> ``_1776_``): ``swebench/sweb.eval.<arch>.<id>`` ->
#                   ``ghcr.io/swe-images/sweb.eval.<arch>.<id>``. Default,
#                   because these are the exact images the dataset was generated
#                   from — their pytest/test-id rendering matches the dataset's
#                   FAIL_TO_PASS / PASS_TO_PASS strings, so grading is faithful.
#   epoch-research  Epoch AI's public, MIT-licensed eval images. Named with the
#                   RAW instance id (e.g. ``swe-bench.eval.x86_64.astropy__
#                   astropy-13236``); no ``_1776_`` normalisation. Covers
#                   2290/2294 of the *full* set on x86_64 (swe-images only
#                   mirrors what swebench publishes), but they are *rebuilt*, so
#                   package versions can drift from the official images and
#                   shift parametrised test-node-ids (e.g. ``[]`` -> ``[unit0]``),
#                   silently failing PASS_TO_PASS matches. Prefer only when you
#                   need full-set coverage and accept that risk.
IMAGE_MIRRORS: dict[str, dict[str, Any]] = {
    "swe-images": {
        "registry": "ghcr.io",
        "org": "swe-images",
        "name": "sweb.eval",
        "normalize": True,
    },
    "epoch-research": {
        "registry": "ghcr.io",
        "org": "epoch-research",
        "name": "swe-bench.eval",
        "normalize": False,
    },
}

DEFAULT_IMAGE_MIRROR = "swe-images"

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
    mirror: str = DEFAULT_IMAGE_MIRROR,
    arch: str = "x86_64",
    org: str | None = None,
    registry: str | None = None,
    name: str | None = None,
    normalize: bool | None = None,
) -> str:
    """Return the ghcr ref for an instance's prebuilt eval image.

    ``mirror`` selects a preset from :data:`IMAGE_MIRRORS` (``epoch-research``
    by default, or ``swe-images``). The explicit ``org``/``registry``/``name``/
    ``normalize`` kwargs override individual preset fields when ``not None``, so
    a custom registry can reuse a preset's naming convention.
    """
    preset = IMAGE_MIRRORS.get(mirror)
    if preset is None:
        raise ValueError(
            f"unknown image mirror {mirror!r}; "
            f"choose one of {sorted(IMAGE_MIRRORS)}"
        )
    registry = preset["registry"] if registry is None else registry
    org = preset["org"] if org is None else org
    name = preset["name"] if name is None else name
    normalize = preset["normalize"] if normalize is None else normalize
    iid = normalise_instance_id(instance_id) if normalize else instance_id
    return f"{registry}/{org}/{name}.{arch}.{iid}:latest"


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

import json

import pytest

from benchmaker.core.types import Request, Response
from benchmaker.workloads.sandbox import SandboxWorkloadType


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "duration_payload, expected_s",
    [
        ({"duration": 0.25}, 0.25),
        ({"Duration": 250_000_000}, 0.25),
        ({"Duration": "250000000"}, 0.25),
    ],
)
async def test_sandbox_make_sample_accepts_duration_formats(duration_payload, expected_s):
    wt = SandboxWorkloadType(base_url="http://example.invalid")
    req = Request(
        method="POST",
        url="http://example.invalid/sandboxes/sb/exec",
        meta={"sandbox_operation": "exec"},
    )
    body = {
        "stdout": "ok\n",
        "stderr": "",
        "exit_code": 0,
        **duration_payload,
    }
    resp = Response(
        status=200,
        headers={},
        body=json.dumps(body).encode(),
        elapsed_s=0.01,
        ok=True,
    )

    sample = await wt.make_sample(None, req, resp, start_ts=0.0)
    assert sample.extra["server_duration_s"] == pytest.approx(expected_s)

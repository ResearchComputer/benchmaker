"""Example: custom pre- and post-processing hooks + a dataset workload.

Pre-hook signs each request (adds HMAC headers).
Post-hook parses a JSON field out of the response and stores it on the sample.
The dataset is a StaticWorkload of dicts; the HttpWorkloadType treats each
dict as the JSON body.
"""

import asyncio
import hashlib
import hmac
import json
import os
import sys
import time

from benchmaker import (
    BenchConfig,
    BenchRunner,
    ConstantRPS,
    HttpWorkloadType,
    Request,
    Response,
    Sample,
    StaticWorkload,
)

SECRET = b"super-secret"


def sign_request(req: Request) -> Request:
    ts = str(int(time.time() * 1000))
    nonce = os.urandom(8).hex()
    payload = req.body or (
        json.dumps(req.json).encode("utf-8") if req.json is not None else b""
    )
    mac = hmac.new(SECRET, ts.encode() + nonce.encode() + payload, hashlib.sha256).hexdigest()
    req.headers["X-Ts"] = ts
    req.headers["X-Nonce"] = nonce
    req.headers["X-Sig"] = mac
    return req


def extract_result(req: Request, resp: Response, sample: Sample) -> Sample:
    try:
        obj = json.loads(resp.body)
        sample.meta["echoed_ts"] = obj.get("headers", {}).get("X-Ts")
        if "json" in obj and isinstance(obj["json"], dict):
            value = obj["json"].get("value")
            if isinstance(value, (int, float)):
                sample.extra["echoed_value"] = float(value)
    except (json.JSONDecodeError, ValueError):
        pass
    return sample


async def main():
    workload_type = HttpWorkloadType(url="https://httpbin.org/post", method="POST")
    workload = StaticWorkload(items=[{"value": i} for i in range(10)])
    load = ConstantRPS(rps=5, duration_s=4)
    runner = BenchRunner(BenchConfig(
        workload_type=workload_type,
        workload=workload,
        load=load,
        pre_hooks=[sign_request],
        post_hooks=[extract_result],
    ))
    await runner.run()
    runner.metrics.render(sys.stdout)


if __name__ == "__main__":
    asyncio.run(main())

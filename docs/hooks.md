# Hooks: pre and post-processing

Hooks are sync or async callables that run on every request. Two flavors:

- **Pre-request hooks** mutate (or replace) the `Request` before it goes out.
  Use for: authentication, signing, request tagging, header injection.
- **Post-response hooks** observe the `(Request, Response)` pair and update
  the `Sample` (typically attaching metrics).

Both run in order; later hooks see the effects of earlier ones.

## Signatures

```python
# Pre: gets a Request, returns a Request (may be the same instance, mutated).
def pre_request(req: Request) -> Request: ...

# Post: gets the original request, the response, and the workload-type's sample.
# Return the sample (possibly with .extra or .meta added).
def post_response(req: Request, resp: Response, sample: Sample) -> Sample: ...
```

Either may be `async`. The runner detects coroutines and awaits them.

## Examples

### Sign every request

```python
import hashlib, hmac, os, time

SECRET = os.environ["MY_SECRET"].encode()

def sign(req):
    ts = str(int(time.time() * 1000))
    nonce = os.urandom(8).hex()
    payload = req.body or b""
    mac = hmac.new(SECRET, ts.encode() + nonce.encode() + payload,
                   hashlib.sha256).hexdigest()
    req.headers["X-Ts"] = ts
    req.headers["X-Nonce"] = nonce
    req.headers["X-Sig"] = mac
    return req
```

### Extract a server-reported metric

```python
import json

def server_timing(req, resp, sample):
    try:
        obj = json.loads(resp.body)
        if "timing_ms" in obj:
            sample.extra["server_latency_ms"] = float(obj["timing_ms"])
    except (json.JSONDecodeError, ValueError):
        pass
    return sample
```

Any numeric value placed in `sample.extra` is automatically aggregated with
mean and percentiles in the final report.

### Mark certain responses as failures

```python
def reject_empty_results(req, resp, sample):
    try:
        obj = json.loads(resp.body)
        if not obj.get("results"):
            sample.ok = False
            sample.error = "empty results"
    except Exception:
        pass
    return sample
```

This affects the `success` / `failed` counts and the `goodput_rps` figure
without changing the underlying HTTP status histogram.

## Wiring hooks

```python
BenchConfig(
    workload_type=...,
    workload=...,
    load=...,
    pre_hooks=[sign],
    post_hooks=[server_timing, reject_empty_results],
)
```

In YAML, refer to hooks as `module:function`:

```yaml
pre_hooks:
  - my_pkg.hooks:sign
post_hooks:
  - my_pkg.hooks:server_timing
  - my_pkg.hooks:reject_empty_results
```

## Hooks vs workload-type `make_sample`

There's overlap: anything a post-hook can do, you can also do inside a custom
workload-type's `make_sample`. Rule of thumb:

- **Bake it into the workload-type** when the metric is intrinsic to the
  protocol (e.g., TTFT for OpenAI chat — every user of that workload-type
  wants it).
- **Use a hook** when the logic is specific to this particular experiment
  (e.g., parsing a custom field that only your service returns, or applying
  signing that only your environment requires).

## Hooks must not raise

If a hook raises, the runner records a failure sample for the request and
continues. The benchmark won't die, but you'll see those entries under
`errors` in the summary. Catch internally if you want to keep going silently.

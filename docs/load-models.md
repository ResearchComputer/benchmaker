# Load models

The load model decides *when* each request fires. Picking the right one
matters more than people usually think — closed-loop and open-loop answer
different questions.

## Open-loop vs closed-loop

**Open-loop** schedules arrivals on wall-clock time. If your target server is
slow, requests pile up. This is what real production traffic looks like, and
it's what you want when answering "how does latency degrade as load
increases?" or "what's the maximum throughput before the system saturates?"

**Closed-loop** keeps exactly N requests in flight at all times. Each worker
fires the next request only after the previous one completes. This answers
"what's the latency at concurrency level N?" but it *cannot* reveal queueing
because slower servers reduce the issued rate — there's no backpressure.

**Default to open-loop for capacity work; reach for closed-loop only when you
explicitly want fixed-concurrency latency.**

## ConstantRPS

Fire `rps` requests per second, evenly spaced.

```python
ConstantRPS(rps=100, duration_s=30)
ConstantRPS(rps=100, max_requests=10_000)   # stop after 10k
```

Spec strings: `"100"`, `"100rps"`.

## PoissonRPS

Open-loop, exponentially-distributed inter-arrival times with mean `1/rps`.
More realistic than constant RPS for serving workloads where arrivals are
independent.

```python
PoissonRPS(rps=100, duration_s=30, seed=42)
```

Spec strings: `"poisson:100"`.

## ClosedLoop

Exactly `concurrency` requests in flight. Each "worker" fires the next
request as soon as the previous one returns.

```python
ClosedLoop(concurrency=32, duration_s=30)
```

Spec strings: `"closed:32"`, `"concurrency:32"`.

## Ramp

Linearly ramp from `start_rps` to `end_rps` over `duration_s`. With
`poisson=True`, the *mean* arrival rate follows the ramp but inter-arrival
times are exponentially distributed.

```python
Ramp(start_rps=10, end_rps=500, duration_s=30)
Ramp(start_rps=10, end_rps=500, duration_s=30, poisson=True)
```

Spec strings: `"ramp:10..500:30s"`, `"ramp-poisson:10..500:30s"`.

## Sweep

Run multiple sub-load-models in sequence. Useful for finding the saturation
point — start at low RPS and step up.

```python
from benchmaker import Sweep, ConstantRPS

Sweep([
    ConstantRPS(rps=10,  duration_s=30),
    ConstantRPS(rps=50,  duration_s=30),
    ConstantRPS(rps=100, duration_s=30),
    ConstantRPS(rps=500, duration_s=30),
], labels=["10rps", "50rps", "100rps", "500rps"])
```

Spec strings: `"sweep:10,50,100,500@30s"` (constant stages, equal duration).

## Choosing a model

| Question                                                          | Use                                    |
| ----------------------------------------------------------------- | -------------------------------------- |
| What's the latency at a fixed offered load X?                     | `ConstantRPS(X)`                       |
| What's the latency under realistic Poisson arrivals?              | `PoissonRPS(X)`                        |
| What's the latency at concurrency N (e.g., for an LLM client)?    | `ClosedLoop(N)`                        |
| When does the system saturate?                                    | `Sweep` or `Ramp`                      |
| Does the system handle bursts cleanly?                            | `Ramp` followed by a steady stage      |

## Dict form (YAML)

In a YAML config the `load` key can be a dict instead of a rate-spec
string. This is useful when you need options that the short spec syntax
doesn't cover (e.g. `seed`, `sweep` labels).

```yaml
load:
  type: constant | poisson | closed | ramp | sweep
  rps: 100
  duration: 60s
  seed: 42
```

### Accepted keys per type

| Type | Keys |
| ---- | ---- |
| `constant` | `rps`, `duration`, `max_requests` |
| `poisson`  | `rps`, `duration`, `max_requests`, `seed` |
| `closed` / `closed-loop` / `concurrency` | `concurrency`, `duration`, `max_requests` |
| `ramp`     | `start_rps`, `end_rps`, `duration`, `poisson` (bool), `seed` |
| `sweep`    | `stages` (list of sub-spec dicts, each with its own `type` / keys); optional `label` per stage |

**`sweep` example:**

```yaml
load:
  type: sweep
  stages:
    - type: constant
      rps: 10
      duration: 30s
      label: warmup
    - type: constant
      rps: 50
      duration: 30s
    - type: constant
      rps: 200
      duration: 30s
      label: peak
```

## Duration parsing

The `duration` field (in both the Python API and the YAML dict form)
accepts a human-friendly string **or** a bare number (interpreted as
seconds):

| Value   | Meaning |
| ------- | ------- |
| `30s`   | 30 seconds |
| `500ms` | 0.5 seconds |
| `2m`    | 120 seconds |
| `1h`    | 3600 seconds |
| `60`    | 60 seconds (bare number) |

In Python code you can also use `duration_s` (a float) directly.

## Duration vs max-requests

Every load model accepts both:

```python
ConstantRPS(rps=100, duration_s=60)        # 60 seconds wall-clock
ConstantRPS(rps=100, max_requests=10_000)  # 10k requests, then stop
ConstantRPS(rps=100, duration_s=60, max_requests=10_000)  # whichever first
```

When neither is set, the model runs until the workload is exhausted
(`StopAsyncIteration` from `next_item`).

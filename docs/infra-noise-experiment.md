# Quantifying infrastructure noise in the SWE-bench harness

**Question.** How much of the measured SWE-bench resolve-rate is the *model's
capability* versus *infrastructure noise* injected by the flash-sandbox cluster,
the grader, and concurrency?

We separate the measured rate `R̂` from the true capability `R*`:

- **Bias** `B(C) = R* − R̂(C)` — systematic distortion. Crashes → false
  negatives; partial-state / flaky grading → either direction.
- **Variance** — run-to-run spread of `R̂` at a *fixed* config, split into
  *model* nondeterminism (temp-0 server batching still varies) vs *infra*
  nondeterminism.

## Three noise channels

| # | Channel | Stage | Loud? | Detector |
|---|---------|-------|-------|----------|
| 1 | **Sandbox crash** (HTTP 502 "all workers unreachable") | agent env | loud | `exit_status != "ok"`; gold-patch canary fails |
| 2 | **Grading noise** (image drift, flaky tests, partial-state, missing `uv`) | verifier | semi-loud | gold/empty-patch canaries; re-grade fixed patch offline |
| 3 | **Silent observation corruption** (congestion → truncated/empty/wrong output, or contention → spurious test failure) | agent env, mid-trajectory | **silent** | deterministic-command canary; shadow-replay; contention probe |

Channel 3 is the dangerous one: the model receives a plausible-but-false
observation, trusts it, and walks down a wrong path. A reward-0 from channel 3 is
indistinguishable from a genuine reasoning failure unless instrumented.

Measured (2026-06-05): channel 1 is ~0% at concurrency 5 and **~96%** at
concurrency 50; channel 2 produced the epoch-vs-swe-images `[]`→`[unit0]`
test-id drift and the missing-`uv` reward-0. Channel 3 is unmeasured — and the
high-concurrency trajectories that would show it were destroyed by channel 1
(crash → agent log not collected). Passive trajectory mining cannot find it,
because the same command legitimately returns different output when the model
mutates state. **Active, model-free probes are required.**

## Key idea: model-free probes with known answers

Send commands whose correct output is *known and fixed regardless of model
state* through the exact same infrastructure. Any deviation is 100% infra.

- **Canaries with ground truth.** Gold-patch (must resolve), empty-patch (must
  not resolve), and a **deterministic-command battery** (`echo <nonce>`,
  `seq|wc -l`, checksums, big integers, `sleep;echo DONE` under a short server
  timeout). Co-schedule them *inside every concurrency batch* so they feel the
  same load; their failure rate is the infra noise floor at that load.
- **Shadow-replay.** Re-run each *read-only* command back-to-back and diff.
  A mismatch under load = a non-reproducible observation the model may trust.
- **Decoupled grading.** Capture the agent's `git diff` once, then re-grade that
  *fixed* patch (a) in-band via harbor and (b) offline single-threaded via
  `native_eval` with swe-images (the oracle grader). Variation across re-grades
  = channel-2 variance; in-band-vs-offline disagreement = channel-2 bias.

## Isolating crashes from corruption: the 2×2

Channels 1 and 3 are entangled — to observe corruption you need runs that
*survive*. Decouple **control-plane saturation** (too many sandboxes → crashes)
from **data-plane contention** (starved resources → corrupt output) with two
independent axes:

```
                       data-plane contention (--stress)
                         off            cpu / mem
control-     low (C=1)   clean baseline  corruption, no crashes  <- isolates ch.3
plane    -----------------------------------------------------
load     high (C=50)    crashes          crashes + corruption
(--concurrency)
```

Inducing in-sandbox contention at **low** sandbox count reproduces channel-3
spurious failures *without* the control plane crashing — so OCR and test-flip
rates are measurable on complete runs.

## Metrics

- **OCR(C, stress)** — Observation Corruption Rate = fraction of
  deterministic-command runs whose output ≠ ground truth *while exit looks OK*
  (truncated / empty / wrong / crosstalk / silent-truncation).
- **Crash rate** — control-plane (create failures + exec 502s).
- **Replay instability** — read-only command giving ≠ output across back-to-back
  runs.
- **Test-flip rate** — a known-passing test flipping to fail under contention.
- `R̂_raw` vs `R_clean` (`exit_status==ok` only) vs `R*` (C=1 majority vote).
- **Variance decomposition** — σ²_total(C) − σ²_model(C=1) = σ²_infra.
- **SNR** — capability spread (e.g. between two models) ÷ infra noise std. If
  SNR ≲ 2–3 at the operating C, the measurement can't distinguish model from
  noise.

## Pre-registered acceptance thresholds

Pick the **largest** concurrency satisfying: crash rate < 2%, gold-canary
failure < 1%, empty-canary false-positive = 0, OCR < 1%, `R_clean − R̂_raw < 1`
pt, σ_infra < σ_model. That is the safe operating concurrency; report it.

## Figures (decide first)

1. Noise floor vs concurrency — crash rate + gold-canary failure + OCR.
2. Capability attenuation — `R̂_raw` vs `R_clean` vs `R*` across C.
3. Variance decomposition stacked bar (model / flaky-grading / crash / corruption).
4. Channel table — per-channel noise contribution and capability damage.

## Reproducibility

Fix `--seed` (nonces derive from it, so crosstalk detection is deterministic),
log full config + git commit + host per run, ≥3 repeats per cell, randomize cell
order to avoid time-of-day cluster-load confounds. Crashed runs are **excluded
from `R_clean` and counted separately**, never folded into the denominator.

## Findings (2026-06-05, first campaign)

- **Transport corruption (OCR) = 0** across 4000+ probe executions spanning
  concurrency 1→50 and cpu/mem contention. The exec path does **not** silently
  hand the model truncated/empty/wrong/cross-talked output. Congestion shows up
  as *latency* (p99 31ms→1670ms across C=1→50), not as corrupted observations.
- **The real channel-3 mechanism is spurious in-container failures**, not
  transport corruption: under cpu/mem contention, a command that passes clean
  (a 900MB alloc, a CPU-bound loop with a deadline) **fails** — OOM-kill or
  deadline-exceeded — i.e. a real-looking failure the model would trust. The
  `mem_alloc` / `cpu_deadline` probes go 0% → ~50% spurious-fail from off→stress
  while OCR stays 0%.
- **The crash channel is workload-weight dependent, not raw count.** Lightweight
  canaries (`python:3.11-slim`, short commands) at C=50 did **not** crash (0%),
  whereas the real pi-container SWE-bench workload (multi-GB image, 4GB RAM,
  ~40-min holds) crashed ~96% at C=50. To map the real crash knee the canary
  must match the heavy workload's footprint (`--image <swe-bench> --memory-mb
  4096` + a longer hold).
- **CPU contention is the demonstrated spurious-failure channel**: the
  `cpu_deadline` probe failed 100% under in-sandbox CPU burners (off→stress:
  0%→100%), independent of concurrency (identical at C=1 and C=8 — per-sandbox
  contention, not sandbox count). A real test would "time out" the same way.
- **`memory_mb` is a soft cap (overcommit)**: a 2048MB sandbox allocated 3000MB
  fine (exit 0) and only OOM-killed at 6000MB (exit 137). So OOM-spurious-
  failures need genuine *node-level* memory pressure (many heavy co-tenants),
  not the configured per-sandbox limit — a per-sandbox balloon at the limit does
  not reproduce it. A test the dataset assumes fits in 4GB won't be OOM-killed
  at 4GB here in isolation; only under real multi-tenant contention.
- Cluster facts: `timeout_ms` is not a prompt kill (command runs to completion,
  then returns exit=-1 + empty if it overran); `pgrep`/`pkill` are absent in
  slim images (verify the stressor via PID + shell-builtin `kill -0`).

## Tooling

- `tools/scripts/infra_noise_sweep.py` — the model-free driver: deterministic
  canary battery + shadow-replay + stress-sidecar contention mode, sweeping the
  `--concurrency` × `--stress` matrix. Writes `config.json` / `metrics.csv` /
  `summary.json` per cell.
- System under test = harbor path (`benchmaker.swebench.harbor_eval`); oracle
  grader = `benchmaker.swebench.native_eval` (fresh single-thread pod, swe-images).
- Gold/empty canaries come free from the dataset's `patch` field / empty diff.

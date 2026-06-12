# Command timeouts under load

A deterministic, (mostly) zero-inference toolkit for measuring how **per-command
timeouts** degrade SWE-bench accuracy when the machine is under load. It answers:
*"if every shell command ran `L`× slower, how many tasks would a command timeout
break, and how much accuracy would we lose?"*

The key reparameterization: a command of natural (uncontended) duration `d` times
out under load factor `L` and per-command timeout `T` iff `L·d > T`, i.e. `d > τ`
where **`τ = T / L`** is the *effective per-command time budget*. Raising load and
lowering the timeout are the **same axis** (both shrink `τ`). Under the strict rule
a task survives iff *all* its commands finish within `τ` (`max(d) ≤ τ`); its first
over-budget command times out, the replayed trajectory diverges, and the task is
counted as a failure.

---

## Components

| Piece | Path | Role |
| ----- | ---- | ---- |
| Pure helpers | `benchmaker/swebench/timeout_load.py` | `τ = T/L` predicates + duration recovery + curve builder |
| Tier‑1 CLI | `scripts/analyze_timeout_curve.py` | offline `accuracy(τ)` curve from existing replay logs — **no harness, no inference** |
| Tier‑2 hook | `benchmaker/swebench/pi_agent.py` (`_ExecBridge`) | `BENCH_LOAD_FACTOR` injection; strict no‑op by default |
| Tier‑2 sweep | `scripts/sweep_swebench_loadfactor.sh` | runs the replay over an `L` grid, prints predicted‑vs‑actual |

There are two ways to use it:

- **Tier 1 (offline, free):** predict the whole `accuracy(τ)` curve from an
  already-recorded **uncontended (c1)** jobs directory. No containers, no model.
- **Tier 2 (live, validates Tier 1):** actually replay the trajectories with
  injected slow-downs and confirm the prediction. Needs the replay endpoint.

---

## Tier 1 — offline `accuracy(τ)` curve

Point it at an uncontended (concurrency‑1) jobs directory. It recovers each
command's wall‑time from the agent logs (`toolResult.timestamp − toolCall.timestamp`)
and computes the strict curve.

```bash
python scripts/analyze_timeout_curve.py jobs/replay_2026-06-11__17-21-48_c1_0d73 --timeout 600
```

Sample output:

```
tasks=100  T=600s  baseline_solved=49
  tau(s)  L=T/tau  survive  solved   acc%  broken
     inf      1.0      100      49    49%       0
      60     10.0       93      45    45%       7
      30     20.0       88      42    42%      12
      10     60.0       78      35    35%      22
       5    120.0       68      30    30%      32
```

Reading it: at the production 600s timeout you need roughly a **20× slowdown**
(`τ=30s`) before accuracy moves off the 49 baseline; it only collapses past ~60×.
The same table doubles as a **timeout-tuning** curve — dropping the per-command
timeout from 600s→30s costs ~7 tasks, →10s costs ~14.

### CLI reference (`analyze_timeout_curve.py`)

| Flag | Meaning |
| ---- | ------- |
| `jobs_dir` (positional) | Uncontended (c1) jobs directory to read. |
| `--timeout T` | Per-command timeout `T` in seconds (default `600`). Sets the `L = T/τ` column. |
| `--taus …` | Explicit `τ` grid in seconds (e.g. `--taus 60 30 10 5`). Default is a log-ish grid. |
| `--out PREFIX` | Write `PREFIX.csv` and `PREFIX.json` (valid JSON; `τ=inf` is emitted as `"inf"`). |
| `--plot` | Also write `PREFIX.pdf` (lazy `matplotlib` import; only needed with this flag). |

Malformed/partial `result.json` files and tasks missing a log are skipped with a
warning, not fatal.

---

## Tier 2 — live load-factor sweep

Replays the **same** trajectories at a **fixed low real concurrency** (so durations
stay uncontended) while sweeping the synthetic load factor `L`. Each `L` writes to
`jobs/loadfactor_L<L>/`, then the script prints a strict‑prediction‑vs‑actual table.

```bash
# full default grid (L = 1 2 5 10 20 30 60  ->  tau = inf 300 120 60 30 20 10)
./scripts/sweep_swebench_loadfactor.sh

# smaller probe
LOAD_FACTORS="1 20 60" ./scripts/sweep_swebench_loadfactor.sh
```

Summary it prints at the end:

```
=== predicted (strict Tier-1) vs actual (Tier-2) solved ===
     L  tau(s)  predicted     actual
     1     inf         49      49/100
    20      30         42      45/100
    60      10         35      38/100
```

### Configuration (override via env)

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `FLASH_SANDBOX_URL` | `http://100.71.204.79:8080` | Flash sandbox endpoint. |
| `REACHABLE_HOST` | `100.101.144.78` | Host the in-container agent uses to reach the replay server. |
| `TRAJECTORIES` | `.local/replay-trajectories.jsonl` | Recorded trajectories to replay. |
| `N_TASKS` | `100` | Number of tasks. |
| `CONCURRENCY` | `1` | Real concurrency — keep low so measured durations are uncontended. |
| `BENCH_INJECT_TIMEOUT_S` | `600` | Per-command timeout `T` (seconds). |
| `LOAD_FACTORS` | `1 2 5 10 20 30 60` | Space-separated `L` grid. |
| `C1_DIR` | `jobs/replay_2026-06-11__17-21-48_c1_0d73` | Baseline used for the Tier‑1 prediction column. |
| `OUT_ROOT` | `jobs` | Parent directory for the per‑`L` output dirs. |

### How to read the two columns

- **`L=1` is a no-op control.** It must reproduce the ~49 baseline; if it doesn't,
  the injection plumbing or the endpoint is misconfigured — fix that before trusting
  any other row.
- **Expect `actual ≥ predicted`.** Tier‑1 strict counts *any* timeout as a failure,
  but a real run still passes when the fix already landed *before* the timed-out
  command. The gap between the columns measures how often the agent had already
  succeeded — i.e. how recoverable the timeouts are.

---

## The injection knob

The hook lives in the pi-container exec bridge (`_ExecBridge`). It runs the real
command, measures the elapsed time `d`, and if `L·d > T` returns a synthetic
timeout payload instead of the real output. Two environment variables control it,
read by the in-harness bridge process:

| Variable | Default | Effect |
| -------- | ------- | ------ |
| `BENCH_LOAD_FACTOR` | `1` | Load factor `L`. **`≤ 1` is a strict no‑op** (default behavior is unchanged). |
| `BENCH_INJECT_TIMEOUT_S` | `_exec_timeout_s` (600) | Modeled per-command timeout `T`. Named to avoid colliding with `bench_kimi.py`'s `BENCH_TIMEOUT_S`. |

So `BENCH_LOAD_FACTOR=20 BENCH_INJECT_TIMEOUT_S=600` models a 20× slowdown against a
600s timeout (`τ=30s`): any command whose real duration exceeds 30s is reported as
`command timed out after 30.0s (injected load_factor=20, T=600s)`.

Because the harness replays the *model* by hashing its input, a changed observation
(a timeout where the recording had real output) produces a **replay miss** → the run
terminates (a deterministic divergence signal; see `replay_server.get_misses`).

---

## Programmatic use

```python
from benchmaker.swebench.timeout_load import (
    recover_command_timings, accuracy_curve, effective_tau, would_time_out,
)

timings = recover_command_timings("jobs/.../agent/pi-container.log")
max_d = max((t.duration_s for t in timings), default=0.0)

# (reward, max_command_duration) per task -> strict accuracy(tau) curve
curve = accuracy_curve([(1.0, max_d)], taus=[float("inf"), 30, 10])
print(curve[0].n_solved, curve[0].n_broken)

effective_tau(600, 20)        # -> 30.0
would_time_out(31.0, 30.0)    # -> True
```

---

## Validity: is the data from real sandboxes?

Tier‑1's durations are only meaningful if the recorded commands really executed in
containers. The replay server is an LLM endpoint **only** — it has no layer for tool
results, so commands necessarily hit `environment.exec` (flash-sandbox / docker).
Observational evidence in the recorded `jobs/` data:

- The longest commands are real test suites (`pytest`, Django `runtests.py`) taking
  450–1180s — a mock returns instantly.
- Stateful FS: an `edit` followed by a `read` of the same file shows the agent's own
  change — impossible to fabricate from a model-keyed replay.
- Real hidden-test verifier stdout in every `verifier/test-stdout.txt`.
- The same workload's wall-time inflates with concurrency (median exec 18s → 154s
  from c1 → c100) and hits real `exit_137` (OOM) under load — mocked durations would
  be constant.

To prove it *actively* on a live run: inject `sleep 5` and confirm measured `d ≈ 5s`
(calibrates the exact quantity Tier‑1 uses); run `hostname; date +%s%N; echo $RANDOM`
and confirm a container hostname, current time, and run-to-run variation.

---

## Scope and caveats

- **Conservative lower bound.** Tier‑1 (and the strict sweep accounting) assume a
  timeout is unrecoverable; true under-load accuracy is `≥` this. It measures
  *exposure / potential harm*, not the agent's recovery behavior.
- **Only `bash`-class commands are timeout-prone.** `read`/`edit`/`write` run in
  the container in <0.1s and never trip the budget.
- **`d` includes a small bridge RPC overhead.** Negligible for the long commands
  that dominate timeout exposure; the `sleep` calibration quantifies it.
- **Out of scope (future work):** real-contention validation (the concurrency
  ladder), live recovery measurement (branch at the first timeout with live
  inference), and mitigations (adaptive timeout, retry-on-timeout, concurrency caps).

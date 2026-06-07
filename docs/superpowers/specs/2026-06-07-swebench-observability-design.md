# SWE-bench observability: timeline, machine utilization, tokens & trajectories

Date: 2026-06-07
Status: Approved design — ready for implementation plan

## Problem

The `benchmaker swebench` recipe drives a harbor `Job` (per-instance Flash
Sandbox env + agent + verifier) and only surfaces final accuracy. For analysis
of a run we want, without changing what the agents do:

1. **Timeline** of when each event starts/ends: environment setup, agent setup,
   agent execution, verifier — and, where we control the code, per-LLM-call and
   per-sandbox-exec spans.
2. **Machine utilization** over the run, polled from `FLASH_SANDBOX_URL/status`.
3. **Per-LLM-request input/output tokens** (and cache/cost where available).
4. **Trajectories** persisted and indexed for later analysis.

## Key facts established from harbor / flash-sandbox / pi

- Harbor runs trials as **in-process asyncio tasks** (single process, one event
  loop). Hooks fire in-process; our agent loop shares the loop. No IPC needed.
- Harbor already times the four phases: `TrialResult.{environment_setup,
  agent_setup, agent_execution, verifier}` are `TimingInfo(started_at,
  finished_at)` (UTC). `TrialResult.{started_at,finished_at}` bound the trial.
- `Job` exposes public lifecycle hooks: `on_trial_started`,
  `on_environment_started`, `on_agent_started`, `on_verification_started`,
  `on_trial_ended` (END carries the full `TrialResult`), `on_trial_cancelled`.
  Each callback receives a `TrialHookEvent(event, trial_id, task_name, config,
  timestamp, result)`.
- Harbor persists each trial's `result.json` (full `TrialResult`, incl.
  `agent_result: AgentContext`) to the trial dir.
- `AgentContext` carries `n_input_tokens / n_cache_tokens / n_output_tokens /
  cost_usd`; `TrialResult.compute_token_cost_totals()` aggregates them. Our
  `BenchmakerHostAgent` currently populates **none** of these (so harbor records
  no token data for our loop today).
- Flash Sandbox: `GET /status` → cluster summary; flash-sandbox's
  `AsyncHTTPClient(address=URL).cluster_status()` returns `ClusterStatus`
  (`node_count`, `available_node_count`, `unavailable_node_count`,
  `sandbox_count`, `nodes:[ClusterNode(node_id, status, available,
  running_count, …)]`). `?sandboxes=true` adds per-sandbox detail (not used).
- `CodingAgent._send` receives an OpenAI-compatible response whose `usage` block
  (`prompt_tokens`, `completion_tokens`, …) is currently discarded.
- pi `--mode json` emits JSONL to stdout (already captured to `pi-*.log`): a
  header line, then one `JSON.stringify(event)` per session event. Events have a
  `type` (`message`, `message_start`, `message_end`, `agent_start`/`agent_end`,
  …); assistant message events carry a `timestamp` and a `usage` object shaped
  `{input, output, cacheRead, cacheWrite, cost:{total}}`. The repo targets the
  `@earendil-works/pi-coding-agent` fork, so the schema is treated as **likely
  but not guaranteed** → the parser is defensive.

## Approach (selected)

A single observability module wires harbor's public hooks (coarse, all agents),
runs a background `/status` poller, and merges fine-grained spans that our own
loop emits. Rejected alternatives: hooks-only (drops requested per-call spans);
global monkeypatching of `FlashSandboxEnvironment.exec` + model HTTP (brittle,
and still blind inside pi's Node subprocess).

## Components

### 1. New module `benchmaker/swebench/observability.py`

Pure, unit-testable helpers (no I/O, no harbor/network deps in signatures):

- `phase_spans_from_result(result) -> list[Span]` — map a `TrialResult`'s four
  `TimingInfo`s (+ trial/task names, reward, exit_status) to `kind="phase"`
  span dicts. Skips phases whose `TimingInfo` is `None`/unfinished.
- `util_row_from_status(status, t, wall) -> dict` — `ClusterStatus` →
  utilization row.
- `summarize(spans, util_rows) -> dict` — per-phase count/mean/p90; agent-internal
  totals (n llm_calls, mean call s, total in/out tokens; n execs, mean exec s)
  when present; utilization peak/mean `sandbox_count`, `node_count`, mean
  available nodes.
- `merge_span_files(job_dir) -> list[Span]` — recursively glob the job dir for
  `timeline-spans.jsonl` (agents write under their `logs_dir`, nested in the
  trial dir), stamp `trial` from the first path component under `job_dir`, return
  spans.
- `parse_pi_token_spans(pi_log_text) -> list[Span]` — defensive pi JSONL parser
  (see §5).
- `trajectory_manifest_rows(results, job_dir) -> list[dict]` — one row per trial
  for `trajectories.jsonl` (see §4). Derives reward/passed from each result's
  `verifier_result` (same logic as harbor_eval's `_summarise`), so the helper is
  self-contained.
- `format_summary(summary) -> str` — render the printed table.

Orchestration (the only parts that touch harbor / network / fs):

- `class JobObserver`:
  - `attach(job)` — registers `on_trial_started/…/on_trial_ended/on_trial_cancelled`.
    The START-ish hooks append a lightweight progress log line
    (`{event, trial_id, task_name, timestamp}`) to an in-memory list; the **END**
    hook is the authoritative source — it stores the `TrialResult`. All hook
    bodies are wrapped best-effort (never raise into harbor).
  - `utilization()` — async context manager. On enter, spawns a background task
    polling `cluster_status()` every `interval` s, appending a util row each
    time; a failed poll logs once at debug and is skipped (the loop continues);
    if the first N polls all fail, log once at warning and keep trying at the
    same cadence (never give up silently, never crash). On exit, cancels the task
    and closes the client. Uses a fresh
    `flash_sandbox.AsyncHTTPClient(address=flash_url)` (benefits from the existing
    `_flash_hardening` monkeypatch).
  - `write(job_dir, rows, accuracy)` — from collected `TrialResult`s + merged
    fine-grained spans + pi token spans, write `timeline.jsonl`,
    `utilization.jsonl`, `trajectories.jsonl`; compute the summary; return it.
    Best-effort: a file-write failure logs and still returns the in-memory summary.

- `async def run_job_with_observability(job_config, *, flash_url, util_interval,
  enabled) -> (job, job_result, summary_text)` — shared entrypoint:
  ```
  job = await Job.create(job_config)
  obs = JobObserver(flash_url, util_interval) if enabled else None
  if obs: obs.attach(job)
  async with (obs.utilization() if obs else nullcontext()):
      job_result = await job.run()
  summary_text = obs.write(job.job_dir, *summarise(job_result)) if obs else None
  return job, job_result, summary_text
  ```
  Both `recipes/swebench.py` and `harbor_eval.py` call this (de-dups their
  near-identical job-run blocks).

### 2. Fine-grained tracer seam in `benchmaker/swebench/agent.py`

- `CodingAgent.run_loop` gains `tracer: Callable[[dict], None] | None = None`.
  - Around each `self._send(...)`: capture UTC start/end → if tracer,
    `tracer({"kind":"llm_call","seq":n_calls,"start":…,"end":…,
    "duration_s":…,"n_input_tokens":…,"n_output_tokens":…,"n_cache_tokens":…})`.
  - Around each `executor(...)`: `tracer({"kind":"sandbox_exec","seq":n_actions,
    "rc":rc,"start":…,"end":…,"duration_s":…})`.
  - `tracer is None` → zero added work. `CodingAgent.run` and `send_fn` tests are
    unaffected.
- Token surfacing: add `_send_with_usage(messages) -> (content, usage|None)` that
  parses the response `usage` (`prompt_tokens`→input, `completion_tokens`→output,
  cache fields if present). `_send` stays `-> str` for the `send_fn` path (usage
  `None`). `run_loop` uses the usage-aware variant, feeds tokens into the
  `llm_call` span, and accumulates per-trial totals. `LoopResult` gains
  `n_input_tokens / n_output_tokens / n_cache_tokens / cost_usd` (Optional).

### 3. Wiring in `benchmaker/swebench/harbor_agent.py`

- `BenchmakerHostAgent.run` builds a tracer that appends each span as one JSON
  line to `<logs_dir>/timeline-spans.jsonl` (idiomatic — it already writes
  `trajectory.json`/`*.log` there; append-as-you-go is crash-survivable), and
  passes it to `run_loop(tracer=…)`.
- Populate harbor's token data: in `run` (or via `populate_context_post_run`),
  set `context.n_input_tokens / n_output_tokens / n_cache_tokens / cost_usd`
  from the loop totals, so harbor's own `result.json` and any DB upload finally
  carry our token/cost numbers.

### 4. Trajectories — indexed, not re-copied

- Each agent already writes a per-trial trajectory to its `logs_dir`
  (`benchmaker-host.trajectory.json` for our loop; `pi-*.log` for pi). Keep that.
- `JobObserver.write` emits `<job_dir>/trajectories.jsonl` — one row per trial:
  `{trial, task, reward, passed, exit_status, n_calls, n_actions,
  n_input_tokens, n_output_tokens, n_cache_tokens, cost_usd,
  trajectory_paths:[<paths relative to job_dir>]}`. Token/cost come from the
  `TrialResult` (`compute_token_cost_totals`) with our loop's values folded in.
  Analysis loads one JSONL and joins to `timeline.jsonl` on `trial`. (We index
  rather than copy the large message blobs, which already live in the trial dirs.)

### 5. pi per-request tokens (defensive JSONL parse)

- `parse_pi_token_spans(text)` walks `pi-*.log` lines, `json.loads` each
  (ignoring non-JSON / header lines). For any object that contains a `usage`
  object with a numeric `input` or `output`, emit one `llm_call` span:
  `n_input_tokens=usage.input`, `n_output_tokens=usage.output`,
  `n_cache_tokens=(cacheRead or 0)+(cacheWrite or 0)`,
  `cost_usd=usage.cost.total` (each guarded), `seq` = running index. If the
  object carries a `timestamp`, use it for `start`/`end`; otherwise leave times
  `None` (tokens-only). Unknown/renamed fields → that field is `None`, never an
  error.
- The observer locates each pi trial's log via the trial's `logs_dir` (from the
  per-trial dir) and tags the resulting spans with `trial`/`task`. pi token spans
  flow into the same `timeline.jsonl` and feed `summarize`.
- pi-host `sandbox_exec` spans (the §3 bonus) are written by `_ExecBridge` to
  `PiHostAgent`'s own `<logs_dir>/timeline-spans.jsonl`, picked up by
  `merge_span_files` like any other agent's span file.
- This is best-effort and schema-tolerant; if a future pi build changes the
  shape, the run still succeeds with whatever it could parse.

## Artifacts (written into the harbor job dir)

`timeline.jsonl` — one span per line, single UTC wall-clock axis:
```json
{"trial":"…","task":"…","kind":"phase|llm_call|sandbox_exec",
 "name":"agent_execution","start":"<iso8601|null>","end":"<iso8601|null>",
 "duration_s":1.2,"seq":3,"rc":0,
 "n_input_tokens":1840,"n_output_tokens":210,"n_cache_tokens":0,
 "cost_usd":0.0031,"extra":{}}
```
`utilization.jsonl` — one poll per line:
```json
{"t":132.5,"wall":"<iso8601>","node_count":12,"available_node_count":11,
 "unavailable_node_count":1,"sandbox_count":8,
 "nodes":[{"id":"n1","available":true,"running_count":3}, …]}
```
`trajectories.jsonl` — one trial per line (manifest; see §4).

Printed end-of-run summary (after harbor's accuracy table): per-phase
count/mean/p90; agent-internal totals when present; utilization peak/mean
sandbox_count, node_count, mean available nodes.

## CLI flags (added to both `recipes/swebench.py` and `harbor_eval.py`)

- `--timeline / --no-timeline` (default **on**) — master switch for the whole
  feature (timeline + utilization + manifest + token capture).
- `--utilization-interval-sec` (float, default **5.0**).

Fine-grained spans are automatic for loop agents; no extra flag.

## Error handling

Every observer path is best-effort and wrapped: it may degrade (skip a poll,
drop fine-grained or pi spans, write a partial artifact) but must never fail a
trial or the job. If `/status` is unreachable, polling logs once and keeps
retrying at cadence. Artifact-write failures log and still print the in-memory
summary.

## Testing (TDD, alongside `tests/test_flash_hardening.py`)

Pure unit tests (no network/harbor):
- `phase_spans_from_result` over a hand-built `TrialResult`-shaped object
  (incl. missing/unfinished phases).
- `util_row_from_status` over a `ClusterStatus`.
- `summarize` (phase means/p90; token + exec totals; utilization peak/mean).
- `merge_span_files` over a temp job dir with two `timeline-spans.jsonl`.
- `parse_pi_token_spans` over sample pi JSONL: nominal `{input,output,cacheRead,
  cacheWrite,cost:{total},timestamp}`; missing fields; non-JSON/header lines;
  a renamed-field line (asserts graceful degradation, tokens-only or skip).
- `CodingAgent.run_loop` with a `send_fn` and fake executor + a recording
  tracer: asserts the ordered `llm_call`/`sandbox_exec` span sequence and that
  `usage` (when send path returns it) lands on the span. Drive via the existing
  no-LLM harness.

Integration (hooks firing, live poller, real pi stream) is **not** unit-tested.

## Out of scope

- Per-sandbox `?sandboxes=true` polling (per-node `running_count` is enough).
- Copying full trajectory blobs into a central folder (manifest indexes them;
  can be added later if a self-contained bundle is wanted).
- Real-time/streaming dashboards; artifacts are written at job end (utilization
  streams to file during the run; timeline/manifest are written at the end).

## Files touched

- **new** `benchmaker/swebench/observability.py`
- `benchmaker/swebench/agent.py` — tracer param + usage surfacing on `run_loop`/
  `_send`; `LoopResult` token fields.
- `benchmaker/swebench/harbor_agent.py` — wire tracer → `timeline-spans.jsonl`;
  populate `AgentContext` tokens.
- `benchmaker/swebench/pi_agent.py` — `_ExecBridge` writes `sandbox_exec` spans
  to `PiHostAgent`'s own `<logs_dir>/timeline-spans.jsonl` (pi-host bonus); ensure
  the pi log path is discoverable for token parsing.
- `benchmaker/swebench/harbor_eval.py` — call `run_job_with_observability`; add
  flags to argparse.
- `benchmaker/recipes/swebench.py` — call `run_job_with_observability`; add
  click flags.
- **new** `tests/test_observability.py` (+ pi-parse fixtures).
</content>
</invoke>

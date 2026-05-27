# Coding agent example

A compact SWE-style coding agent for bench-maker. It's a faithful but minimal
port of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)'s control
loop: the model emits one fenced `bash` block per turn, the agent runs it, the
stdout/stderr come back as the next user message, and the loop ends when the
model emits `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (or hits a step/wall-clock
limit).

## Files

- `coding_agent.py` — the `CodingAgent` class. Pluggable model side: pass an
  OpenAI-compatible `url` + `model` + `api_key`, or a `send_fn` callable for
  tests / custom clients.
- `config.yaml` — bench-maker YAML wiring. Static 2-item task list by default;
  swap for a `type: hf` workload to run a real eval.

## Run

```bash
bench-maker run examples/coding_agent/config.yaml \
    --out-dir ./runs --label agent=coding-tiny
```

Required env vars (loaded from `.env` automatically):

- `OPENAI_COMPATIBLE_MODEL` — e.g. `swiss-ai/Apertus-8B-Instruct-2509`
- `OPENAI_API_BASE_URL` — e.g. `https://api.swissai.svc.cscs.ch/v1/`
- `OPENAI_API_KEY` — bearer token (may be unused for local servers)

## Outputs

In `./runs/<run-id>/`:

- `summary.json` — `workload_metrics.steps.mean` is avg model-call count;
  `correct.mean` is task accuracy; the wrong/failed split distinguishes
  "agent submitted a bad answer" from "agent crashed or hit step_limit".
- `samples.jsonl` — one record per task: prediction, exit_status, cwd, steps.

## Knobs

Set under `workload_type.agent_kwargs` in `config.yaml`:

- `step_limit` — max model calls per task.
- `timeout_per_step_s` — per-command shell timeout.
- `total_wall_s` — hard wall-clock cap on the whole trajectory.
- `cwd_template` — pin each task to its own checkout (e.g.
  `/tmp/swe-{task_id}`). Combine with `keep_cwd: true` to inspect the
  workspace after the run.
- `max_obs_chars` — head+tail truncation cap on each observation fed back to
  the model.

### Sandbox execution

The default config runs each task inside a fresh
[Flash Sandbox](https://sandbox.swissai.cscs.ch) pod (one pod per task,
deleted on teardown). Drop `sandbox_url` to fall back to local subprocess
execution under a tmp dir.

- `sandbox_url` — Flash Sandbox base URL. Default in `config.yaml`:
  `https://sandbox.swissai.cscs.ch`.
- `sandbox_spec` — pod spec merged over the agent's default
  (`alpine:3.20`, keep-alive `sleep 3600`, `cpu_cores: 0.1`). Override
  `image`, `cpu_cores`, `memory_mb` here.
- `sandbox_ttl_seconds` — server-side safety net so leaked pods get reaped
  even if the agent's `DELETE` is skipped.
- `sandbox_persistent` — when `true` (default), exec via `/pshell` so
  `cd` / `export` / `source` persist across action steps; set `false` to
  use `/exec` (stateless, mirrors the local subprocess behavior).
- `sandbox_headers` — extra HTTP headers (e.g. `Authorization: "Bearer ..."`).
- `sandbox_endpoint_prefix` — `/sandboxes` (cluster coordinator, default) or
  `/native/sandboxes` (single-node).

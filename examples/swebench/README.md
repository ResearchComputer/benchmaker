# SWE-bench coding-agent examples

Runnable wiring for benchmaker's SWE-bench coding agent. The agent loop,
grading, and harbor adapters now live in the **`benchmaker.swebench`** package
(`benchmaker/swebench/`); this directory only holds the YAML configs and a
standalone slice runner that drive it.

The agent is a faithful but minimal port of
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)'s control loop:
the model emits one fenced `bash` block per turn, the agent runs it, the
stdout/stderr come back as the next user message, and the loop ends when the
model emits `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (or hits a step/wall-clock
limit).

## Where the code lives (`benchmaker.swebench`)

- `benchmaker/swebench/agent.py` — the `CodingAgent` class. Pluggable model
  side: pass an OpenAI-compatible `url` + `model` + `api_key`, or a `send_fn`
  callable for tests / custom clients.
- `benchmaker/swebench/grading.py` — pure helpers: ghcr image-key resolution,
  swebench `TestSpec` construction, and authoritative log grading
  (unit-testable, no sandbox). **Single source of truth** — the agent-warmup
  generator (`tools/agent_warmup`) imports the same functions.
- `benchmaker/swebench/native_eval.py` — `SWEBenchAgent`, a `CodingAgent`
  subclass that evaluates on SWE-bench Verified natively (no harbor).
- `benchmaker/swebench/harbor_agent.py` / `harbor_eval.py` — the harbor
  adapters (optional `harbor` dependency).

## Files here

- `config.yaml` — bench-maker YAML wiring for the plain `CodingAgent`. Static
  2-item task list by default; swap for a `type: hf` workload to run a real eval.
- `config_swebench.yaml` — wires `SWEBenchAgent` through `AgentWorkloadType`
  against the `SWE-bench/SWE-bench_Verified` HF dataset. `AgentResult.ok` (the
  summary's pass rate) is the swebench `resolved` verdict.
- `run_slice.py` — drive `SWEBenchAgent` directly over a JSON manifest of
  instances (a smoke harness, outside the benchmaker runner).

### SWE-bench Verified (harbor-equivalent, native)

`SWEBenchAgent` evaluates on SWE-bench Verified the same way
[`flash-sandbox/examples/harbor`](../../../flash-sandbox/examples/harbor) does,
but without harbor or mini-swe-agent as a dependency:

- boots the **prebuilt per-instance eval image** (repo already at `base_commit`
  under `/testbed`, deps installed) from the public ghcr `swe-images` mirror —
  no GitHub clone, no `pip install` on a bare image;
- runs the agent loop, then collects the diff straight from `git`;
- **grades in a fresh pod** using the `swebench` package as the source of truth
  (FAIL_TO_PASS / PASS_TO_PASS → `RESOLVED_FULL`). The agent never sees the
  hidden `test_patch`.

Run a small slice end-to-end (raise `workload.max_items` once it passes):

```bash
benchmaker run examples/swebench/config_swebench.yaml \
    --out-dir ./runs --label agent=swebench
```

Extra env var beyond the model ones below: `FLASH_SANDBOX_URL` (e.g.
`https://sandbox.swissai.cscs.ch`). The CSCS cluster is a **kubernetes**
backend with no auth. The ghcr mirror is produced by `tools/swe_images`; point
`image_org` / `image_registry` at a different registry if you mirror elsewhere.

### SWE-bench via harbor (harbor *is* the eval engine)

The other way to run SWE-bench: let [harbor](https://github.com/) own the
evaluation (per-instance environment, agent execution, and verifier), and use
benchmaker only as the launcher with a **pluggable agent registry**. This is the
same machinery as
[`flash-sandbox/examples/harbor`](../../../flash-sandbox/examples/harbor), driven
from `benchmaker.swebench.harbor_eval`.

- `benchmaker.swebench.harbor_eval` — driver: builds a harbor `Job`
  (flash-sandbox env + registered dataset like `swebench-verified`), runs it,
  prints accuracy. The `--agent` flag accepts a **registry key**
  (`mini-swe-agent`, `coding-agent`, `claude-code`), a bare harbor built-in name
  (`openhands`, `swe-agent`, …), or a custom `module.path:ClassName` (a harbor
  `BaseAgent` subclass) — harbor's `BaseAgent` is the general plug-in interface.
- `benchmaker.swebench.harbor_agent` — `BenchmakerHostAgent`, a harbor
  `BaseAgent` that wraps **our** loop: it runs `CodingAgent.run_loop` on the host
  and pushes each shell action into the sandbox via harbor's `environment.exec`.
  This is the `coding-agent` registry entry.

The loop seam is `CodingAgent.run_loop(task, executor, ...)`: the model/observe
loop is decoupled from *where* commands run via an injected `executor`
(`async (action, timeout) -> (returncode, output)`). The same loop runs under
harbor (`harbor_agent`) and under benchmaker's own Flash Sandbox client
(`config_swebench.yaml`) — only the executor differs.

Setup (one-time): install harbor + the flash-sandbox SDK into the venv:

```bash
uv pip install --python .venv/bin/python -e /pub/scratch/xiayao/research/harbor
uv pip install --python .venv/bin/python -e /pub/scratch/xiayao/research/flash-sandbox/libs/python
```

Run (boot a flash-sandbox cluster first — see the harbor example's README):

```bash
python -m benchmaker.swebench.harbor_eval --list-agents

# harbor's mini-swe-agent on a 5-task slice:
FLASH_SANDBOX_URL=http://localhost:8080 \
    python -m benchmaker.swebench.harbor_eval \
        --agent mini-swe-agent --model "$OPENAI_COMPATIBLE_MODEL" --n-tasks 5

# our wrapped CodingAgent loop, evaluated by harbor:
FLASH_SANDBOX_URL=http://localhost:8080 \
    python -m benchmaker.swebench.harbor_eval --agent coding-agent --n-tasks 5
```

Notes: harbor resolves a dataset *name* (`swebench-verified`) through its task
registry, so the first run downloads/caches the tasks. Harbor's flash-sandbox
env defaults to a **docker** backend (`--backend-type`); SWE-bench cold-start
needs `--timeout-multiplier 4`–`6`. To wrap a *different* benchmaker loop, point
`coding-agent`'s `loop_agent` kwarg at another `module:Class`.

### Which SWE-bench path?

- `config_swebench.yaml` (native) — runs under **benchmaker's** runner, so you
  get load models, metrics, and `samples.jsonl`; grading is the `swebench`
  package. No harbor dependency.
- `benchmaker.swebench.harbor_eval` (harbor) — runs under **harbor's** runner;
  reuses harbor's registered datasets, built-in agents, and verifier. Pick this
  when you want parity with the harbor example or its agent zoo.

## Run the plain coding agent

```bash
bench-maker run examples/swebench/config.yaml \
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

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
  (unit-testable, no sandbox). Used by the agent-warmup generator
  (`tools/agent_warmup`).
- `benchmaker/swebench/harbor_agent.py` / `harbor_eval.py` / `pi_agent.py` —
  the harbor adapters + agents (require the `harbor` package). `harbor_eval` is
  the engine behind the `benchmaker swebench` recipe.

## Files here

- `config.yaml` — benchmaker YAML wiring for the plain `CodingAgent`. Static
  2-item task list by default; swap for a `type: hf` workload to run a real eval.

## Run SWE-bench: the `swebench` recipe (harbor)

SWE-bench evaluation runs through **harbor** — harbor owns the per-instance
Flash Sandbox environment, the agent run, and the verifier; benchmaker is the
launcher with a pluggable agent registry. The recipe is *self-driving*: it
prints harbor's accuracy summary + job dir (no benchmaker run-bundle).

Model URL/model/key fall back to the `OPENAI_*` env vars; the sandbox to
`$FLASH_SANDBOX_URL`:

```bash
# Default pi agent on a 5-task slice:
benchmaker swebench --n-tasks 5 --concurrency 4

# Our own CodingAgent loop, evaluated by harbor:
benchmaker swebench --agent coding-agent --n-tasks 5

benchmaker swebench --list-agents
```

`scripts/run_swebench.sh` is a thin wrapper (a 5-task smoke run by default); any
flags pass straight through to the recipe:

```bash
scripts/run_swebench.sh --n-tasks 50 --concurrency 16
scripts/run_swebench.sh --agent coding-agent
```

Note: the CSCS cluster is a **kubernetes** backend with no auth (`--backend-type
kubernetes`); SWE-bench cold-start needs `--timeout-multiplier 4`–`6`. Harbor
resolves the per-instance images from its registered dataset (`--dataset`).

### Under the hood: `benchmaker.swebench.harbor_eval`

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
harbor (`harbor_agent`) and under benchmaker's own Flash Sandbox executor
(`config.yaml`) — only the executor differs.

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

#### pi (`@earendil-works/pi-coding-agent`)

pi is a **Node CLI** that runs shell commands locally in its working directory,
so it can't take an injected executor like our `CodingAgent`. Two harbor agents
(`benchmaker/swebench/pi_agent.py`) bridge that gap:

- **`--agent pi`** (`PiContainerAgent`) — installs Node + pi *inside* the
  per-instance environment (`setup()`), writes a `models.json` pointing pi at
  the OpenAI-compatible endpoint, and runs `pi --mode json` at `/testbed`. pi's
  local shell is the container shell; edits land in `/testbed` and harbor grades.
- **`--agent pi-host`** (`PiHostAgent`) — runs pi on the **host** and routes
  every shell action into the environment. A localhost HTTP bridge forwards to
  `environment.exec`, and the `pi_ext/remote_exec.js` extension replaces pi's
  built-in `bash` tool to POST to it. Model reasoning runs on the host; all file
  edits land in the environment.

```bash
# pi installed in-container:
FLASH_SANDBOX_URL=http://localhost:8080 \
    python -m benchmaker.swebench.harbor_eval --agent pi \
        --model "$OPENAI_COMPATIBLE_MODEL" --n-tasks 5

# pi on the host, shell routed into the sandbox:
python -m benchmaker.swebench.harbor_eval --agent pi-host --n-tasks 5
```

Model config is a `models.json` custom provider (`api: "openai-completions"`)
built from `OPENAI_API_BASE_URL` / `OPENAI_API_KEY` — pi's built-in `openai`
provider can't be pointed at a custom base URL. Useful `--agent-kwarg`s:
`pi_extra_args` (extra pi CLI flags), `total_wall_s`, `context_window`,
`max_tokens`, and (container mode) `install_script`.

**Verify against your pi build** (the docs left these unspecified, so they're
runtime knobs rather than hard-coded): the non-interactive/auto-approve flag for
`--mode json` (pass via `pi_extra_args`, e.g. `--yolo`); that the extension
auto-loads from `~/.pi/agent/extensions/`; and that environment pods have
network egress to npm + the model endpoint (container mode installs Node at
runtime — bake a Node+pi image layer via `tools/swe_images` if you want to skip
the per-instance install).

### Which entry point?

- **`benchmaker swebench`** (recipe) → `harbor_eval` — the way to run a real
  SWE-bench evaluation: harbor's registered datasets, built-in/registry agents,
  and verifier. Self-driving (harbor's accuracy summary + job dir, no benchmaker
  run-bundle).
- **`config.yaml`** (below) — runs the plain `CodingAgent` under **benchmaker's**
  runner over a tiny static task list, so you get load models, metrics, and
  `samples.jsonl`. A demo of the agent loop, not a full SWE-bench eval.

## Run the plain coding agent

```bash
benchmaker run examples/swebench/config.yaml \
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

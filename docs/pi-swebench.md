# pi on SWE-bench

Run **pi** (`@earendil-works/pi-coding-agent`) as a coding-agent harness on
SWE-bench Verified, driven through **harbor** from benchmaker. harbor owns the
per-instance environment (the prebuilt eval image on Flash Sandbox) and the
verifier; benchmaker contributes the entrypoint, model wiring, and the two pi
agents.

This is one of benchmaker's coding-agent paths — see
[`examples/swebench/`](../examples/swebench/README.md) for the others
(`mini-swe-agent`, the native `SWEBenchAgent`, the host-loop `coding-agent`).

## Why two modes

pi is a **Node CLI that runs shell commands locally in its working directory** —
unlike benchmaker's `CodingAgent`, it has no Python-injectable executor. So
there are exactly two ways to make pi edit a task's `/testbed` (which is what
harbor's verifier grades), and `benchmaker/swebench/pi_agent.py` ships one
harbor `BaseAgent` for each:

```text
  --agent pi  (PiContainerAgent)            --agent pi-host  (PiHostAgent)
  ───────────────────────────────          ──────────────────────────────────
       host (harbor trial)                       host (harbor trial)
       │ setup(): install node+pi                │  ┌─────────────────────┐
       │ exec: pi --mode json                    │  │ pi --mode json      │  ← model
       ▼                                         │  │  (runs on host)     │    reasoning
  ┌─────────────────────┐                        │  └─────────┬───────────┘    on host
  │ ENVIRONMENT /testbed│                        │    bash tool (overridden)
  │  pi runs here       │                        │            │ POST $PI_EXEC_BRIDGE
  │  local shell = pod  │                        │            ▼
  │  edits land here    │                        │   localhost _ExecBridge
  └─────────────────────┘                        │            │ environment.exec
       │                                         ▼            ▼
       ▼                                  ┌─────────────────────┐
  harbor verifier grades                  │ ENVIRONMENT /testbed│  ← edits land here
                                          └─────────────────────┘
                                                 │
                                                 ▼  harbor verifier grades
```

| Mode | Registry key | Where pi runs | How shell reaches `/testbed` |
| ---- | ------------ | ------------- | ---------------------------- |
| In-container | `pi` / `pi-container` | inside the environment | pi's own local shell *is* the pod shell |
| Host + remote bash | `pi-host` | on the host | bash tool overridden → localhost bridge → `environment.exec` |

**In-container** (`PiContainerAgent`): `setup()` installs Node + pi into the
per-instance environment (a static Node tarball + `npm i -g`, overridable via
the `install_script` kwarg); `run()` writes a `models.json` and the task prompt
into the environment, then execs `pi --mode json` at `/testbed`.

**Host + remote bash** (`PiHostAgent`): pi runs on the host; a localhost aiohttp
bridge (`_ExecBridge`, `POST /exec` → `environment.exec`) routes shell actions
into the environment, and the `pi_ext/remote_exec.js` extension replaces pi's
built-in `bash` tool to POST to it. `settings.json` restricts pi to the `bash`
tool so file tools never touch the host filesystem. This mirrors the
`coding-agent` host-loop pattern (`harbor_agent.py`) but for pi.

## Model wiring

pi's built-in `openai` provider can't be pointed at a custom base URL, so both
modes write a **`models.json`** custom provider and launch with
`--provider`/`--model`:

```json
{
  "providers": {
    "bench": {
      "baseUrl": "https://api.example/v1",
      "api": "openai-completions",
      "apiKey": "$OPENAI_API_KEY",
      "models": [{ "id": "<model>", "contextWindow": 128000, "maxTokens": 8192 }]
    }
  }
}
```

The endpoint + key come from harbor's `AgentConfig.env` (`OPENAI_API_BASE_URL` /
`OPENAI_API_KEY`) or the host environment — the same `.env` the other agents
read. The secret stays out of the file (`"$OPENAI_API_KEY"` is resolved from the
environment at runtime).

## Run it

The one-command wrapper (reads model / key / `FLASH_SANDBOX_URL` from `.env`):

```bash
tools/scripts/run_pi_experiment.sh                              # pi in-container, 5 tasks
AGENT=pi-host N_TASKS=20 tools/scripts/run_pi_experiment.sh     # pi on host, shell → sandbox
tools/scripts/run_pi_experiment.sh --task astropy__astropy-12907   # extra harbor_eval flags pass through
```

Env knobs (all overridable): `AGENT`, `DATASET`, `N_TASKS`, `CONCURRENCY`,
`BACKEND`, `TIMEOUT_MULT`, `JOB_NAME`, `PYTHON`. The script expands to:

```bash
.venv/bin/python -m benchmaker.swebench.harbor_eval \
    --agent pi --dataset swebench-verified --n-tasks 5 \
    --concurrency 1 --backend-type kubernetes --timeout-multiplier 5 \
    --job-name pi-<timestamp>
```

Or call the driver directly:

```bash
python -m benchmaker.swebench.harbor_eval --list-agents
python -m benchmaker.swebench.harbor_eval --agent pi-host --n-tasks 5
```

Useful `--agent-kwarg key=value` (repeatable):

| Kwarg | Applies to | Meaning |
| ----- | ---------- | ------- |
| `pi_extra_args` | both | extra pi CLI flags (e.g. a non-interactive/auto-approve flag) |
| `total_wall_s` | both | hard wall-clock cap on the pi run |
| `context_window` / `max_tokens` | both | the `models.json` model limits |
| `install_script` | `pi` | override the in-container Node + pi install |
| `exec_timeout_s` | `pi-host` | per-command timeout on the bridge |

## Setup

Install harbor + the Flash Sandbox SDK into benchmaker's venv (one-time):

```bash
uv pip install --python .venv/bin/python -e /pub/scratch/xiayao/research/harbor
uv pip install --python .venv/bin/python -e /pub/scratch/xiayao/research/flash-sandbox/libs/python
```

`.env` must define `OPENAI_COMPATIBLE_MODEL`, `OPENAI_API_BASE_URL`,
`OPENAI_API_KEY`, and `FLASH_SANDBOX_URL` (`harbor_eval` loads `.env`
automatically).

## Grading & output

harbor's verifier grades the post-run `/testbed` (applies the hidden
`test_patch`, runs FAIL_TO_PASS / PASS_TO_PASS). A trial passes when its reward
≥ 1. The driver prints a per-trial table + aggregate accuracy and writes a job
directory under `jobs/<job_name>/`.

## Runtime checklist

These were left unspecified by the pi docs, so they're exposed as knobs rather
than hard-coded — verify them against your installed pi build on the first run:

- **Tool auto-approval.** `--mode json` is headless; if your pi build still
  gates tool calls, pass the auto-approve flag via `pi_extra_args` (e.g.
  `--agent-kwarg pi_extra_args=--yolo`).
- **Secret templating.** harbor rewrites sensitive `AgentConfig.env` values
  (e.g. `OPENAI_API_KEY`) to `${OPENAI_API_KEY}` placeholders for safe
  persistence; harbor's `AgentFactory` and our `resolve_model` expand them via
  `resolve_env_vars` before handing the key to pi.
- **In-container key (container mode).** `PiContainerAgent` embeds the *resolved*
  key directly in the in-container `models.json` (`api_key_ref=<key>`) instead of
  the `$OPENAI_API_KEY` ref: pi's in-container env-var resolution does not pick up
  the inline-exported key, so the `$`-ref path sends an empty bearer →
  `401 status code (no body)` — even though the key (and the endpoint) are
  reachable from the pod (verified: in-pod `curl` with the key returns 200). The
  sandbox is deleted after the trial, so the key does not linger. Host mode keeps
  the `$OPENAI_API_KEY` ref (resolves fine from the host process env).
- **Config dir.** Both modes pin `PI_CODING_AGENT_DIR` to an explicit path
  (`/tmp/pi-agent` in-container; the per-run temp `HOME/.pi/agent` on the host)
  rather than letting pi derive `~/.pi/agent` from `$HOME` — bash's `$HOME` at
  staging time and Node's `os.homedir()` at runtime can disagree inside the
  environment, which otherwise hides `models.json` (→ `Unknown provider "bench"`).
- **Extension auto-load (host mode).** `remote_exec.js` is written to
  `$PI_CODING_AGENT_DIR/extensions/` (under a per-run temp `HOME`); confirm pi loads it.
- **JS tool return shape (host mode).** `remote_exec.js`'s `execute()` returns
  `{output, exitCode}`; older pi builds may expect a bare string.
- **Network egress.** In-container mode installs Node at runtime, so the pods
  need egress to npm + the model endpoint. To skip the per-instance install,
  bake a Node + pi image layer (see [`tools/swe_images`](../tools/swe_images/)).

## CSCS gotchas

On the CSCS cluster (see the harbor-eval notes):

- `--backend-type kubernetes` is required (docker → `503 no healthy nodes`).
  The wrapper defaults to `kubernetes`.
- Keep `--concurrency` low (1–2); the shared sandbox coordinator overloads at 4.
- `--job-name` must be unique — an empty name collides with the existing
  `jobs/` dir (`FileExistsError`). The wrapper stamps a timestamp.

"""SWE-bench Verified agent driver — harbor-equivalent, native to benchmaker.

This is the benchmaker port of how ``flash-sandbox/examples/harbor`` evaluates a
coding agent. Harbor runs its ``Job`` machinery against a *registered*
SWE-bench dataset whose trials boot the prebuilt per-instance eval images, then
grades with an in-sandbox verifier. We do the same thing natively:

  1. Resolve the instance's **prebuilt SWE-bench eval image** (repo already
     checked out at ``base_commit`` under ``/testbed``, deps installed) from the
     public ghcr ``swe-images`` mirror, and boot one Flash Sandbox pod from it.
     No GitHub clone, no ``pip install`` on a bare ``python:3.11`` — the env is
     exactly what the official harness uses.
  2. Run the model loop with a SWE-bench-shaped prompt. The agent edits source
     files in place under ``/testbed``; each command runs via stateless
     ``/exec`` (anchored at ``/testbed``).
  3. Collect the agent's diff straight from git (``git diff`` of the working
     tree vs a baseline commit) — robust against the model pasting a malformed
     patch.
  4. **Grade** in a *fresh* pod (mirroring harbor's separate verifier): apply
     the agent's patch, run swebench's ``eval_script`` (which resets the test
     files, applies the hidden ``test_patch``, and runs FAIL_TO_PASS /
     PASS_TO_PASS), and classify resolution with the ``swebench`` package as the
     source of truth. The agent never sees ``test_patch``.

An instance counts as a pass (``AgentResult.ok``) iff swebench grades it
``RESOLVED_FULL``. Plug this into ``AgentWorkloadType`` for the full benchmaker
machinery (metrics, load models, summary) — see ``config_swebench.yaml`` — or
drive it directly via ``run_swe_bench_slice.py``.

Expected item dict shape (a raw SWE-bench row, or HF form with JSON-string
test-name fields)::

    {
        "instance_id":     "sympy__sympy-20916",
        "repo":            "sympy/sympy",
        "base_commit":     "82298df...",
        "problem_statement": "...",
        "test_patch":      "diff --git a/...",
        "FAIL_TO_PASS":    ["test_foo", ...],   # or '["test_foo", ...]'
        "PASS_TO_PASS":    ["test_bar", ...],
        "version":         "1.8",
        "environment_setup_commit": "...",
    }
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, Optional

import aiohttp

from benchmaker import AgentContext, AgentResult
from examples.coding_agent.coding_agent import SUBMIT_TOKEN, CodingAgent, _excerpt
from examples.coding_agent.swe_bench_grading import (
    DEFAULT_IMAGE_ORG,
    DEFAULT_IMAGE_REGISTRY,
    as_list,
    grade,
    instance_image_key,
    make_test_spec,
)

WORKDIR = "/testbed"


SWEBENCH_SYSTEM_PROMPT = """You are a senior Python engineer fixing a bug in an
open-source repository. The repository is already cloned and fully set up at
`/testbed`, checked out at the buggy commit, with all dependencies installed in
the active environment — you do NOT need to clone, install, or build anything.

Each turn, emit **exactly one** shell command in a fenced bash block:

```bash
your-command-here
```

You will see the command's combined stdout+stderr in the next message. **Each
command runs in a fresh shell starting at `/testbed`** — the working directory
and environment variables do NOT persist across turns, but filesystem edits DO.
Use non-interactive flags; avoid editors that need a TTY (vi, nano). Use
`python -c "..."` or heredocs (`cat <<'EOF' > path ... EOF`) to edit files.

Investigate the codebase, identify the fix, and edit the **source** files in
place. Do NOT modify the tests — a hidden test suite grades your fix.

When you are confident the fix is complete, emit a fenced block whose **first
line** is exactly:

    COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

Nothing else is required after it — your in-tree edits under `/testbed` are
collected automatically as the patch (via `git diff`), so you do NOT need to
print or paste a diff yourself.

Guidelines:
- Prefer `grep -rn`, `sed -n`, `cat`, `python -c "..."` to explore.
- Make a minimal, correct change to the source. Then submit.
"""


SWEBENCH_INSTANCE_TEMPLATE = """# Task: fix {instance_id}

Repository: **{repo}** (already checked out at the buggy commit in `/testbed`).

## Problem statement

{problem_statement}

Investigate and fix the bug, then submit. Begin.
"""


class SWEBenchAgent(CodingAgent):
    """CodingAgent specialised for SWE-bench Verified instances.

    Beyond ``CodingAgent``'s model/sandbox knobs, the SWE-bench-specific kwargs:

        image_org / image_registry / arch: locate the prebuilt per-instance eval
            image. Default ``ghcr.io/swe-images/sweb.eval.x86_64.<id>:latest``.
        image_override: force a fixed image for every instance (tests/smoke).
        sandbox_type: Flash Sandbox backend (``kubernetes`` on the CSCS cluster).
        cpu_cores / memory_mb: per-pod resources (SWE-bench needs a real env).
        grade: run the swebench verifier in a fresh pod (default True). Set
            False to only produce the patch (generation, no score).
        eval_timeout_s / apply_patch_timeout_s / baseline_timeout_s: timeouts for
            the grading phase.
        create_retry_attempts / create_retry_delay_s: retry pod creation on
            transient 502/503 ("no healthy nodes available").
    """

    def __init__(
        self,
        *args: Any,
        image_org: str = DEFAULT_IMAGE_ORG,
        image_registry: str = DEFAULT_IMAGE_REGISTRY,
        arch: str = "x86_64",
        image_override: Optional[str] = None,
        sandbox_type: str = "kubernetes",
        cpu_cores: float = 2.0,
        memory_mb: int = 4096,
        grade: bool = True,
        eval_timeout_s: float = 1800.0,
        apply_patch_timeout_s: float = 180.0,
        baseline_timeout_s: float = 120.0,
        create_retry_attempts: int = 12,
        create_retry_delay_s: float = 8.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("system_prompt", SWEBENCH_SYSTEM_PROMPT)
        kwargs.setdefault("instance_template", SWEBENCH_INSTANCE_TEMPLATE)
        # The per-instance image is chosen per task (see _image_for), not from a
        # fixed sandbox_spec. Each exec is a stateless /exec call anchored at
        # /testbed (this cluster's /pshell doesn't persist state reliably).
        kwargs.setdefault("sandbox_persistent", False)
        kwargs.setdefault("sandbox_create_timeout_s", 300.0)
        super().__init__(*args, **kwargs)
        self._image_org = image_org
        self._image_registry = image_registry
        self._arch = arch
        self._image_override = image_override
        self._sb_type = sandbox_type
        self._cpu_cores = cpu_cores
        self._memory_mb = memory_mb
        self._grade = grade
        self._eval_timeout_s = eval_timeout_s
        self._apply_patch_timeout_s = apply_patch_timeout_s
        self._baseline_timeout_s = baseline_timeout_s
        self._create_retry_attempts = create_retry_attempts
        self._create_retry_delay_s = create_retry_delay_s
        # Set by run() so a driver can inspect the model trajectory afterward.
        self.last_messages: Optional[list[dict]] = None

    # ---- sandbox plumbing (per-task image) --------------------------- #

    def _image_for(self, item: dict) -> str:
        if self._image_override:
            return self._image_override
        return instance_image_key(
            item.get("instance_id", ""),
            org=self._image_org,
            registry=self._image_registry,
            arch=self._arch,
        )

    def _build_spec(self, image: str) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "type": self._sb_type,
            "image": image,
            "command": ["sh", "-c", "sleep infinity"],
        }
        if self._cpu_cores is not None:
            spec["cpu_cores"] = self._cpu_cores
        if self._memory_mb is not None:
            spec["memory_mb"] = self._memory_mb
        if self._sandbox_ttl_seconds is not None:
            spec["ttl_seconds"] = self._sandbox_ttl_seconds
        return spec

    async def _create(self, spec: dict[str, Any]) -> str:
        sess = await self._ensure_session()
        url = f"{self._sandbox_url}{self._sandbox_prefix}"
        timeout = aiohttp.ClientTimeout(total=self._sandbox_create_timeout_s)
        async with sess.post(url, headers=self._sandbox_headers,
                             json=spec, timeout=timeout) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"sandbox create HTTP {resp.status}: {_excerpt(text)}")
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"sandbox create non-JSON: {_excerpt(text)}") from e
        if not isinstance(data, dict) or "id" not in data:
            raise RuntimeError(f"sandbox create unexpected body: {data!r}")
        return str(data["id"])

    async def _create_with_retry(self, spec: dict[str, Any]) -> str:
        """Create a pod, retrying transient 502/503 ("no healthy nodes")."""
        last: Optional[RuntimeError] = None
        for attempt in range(1, self._create_retry_attempts + 1):
            try:
                return await self._create(spec)
            except RuntimeError as e:
                msg = str(e)
                transient = "HTTP 503" in msg or "HTTP 502" in msg
                if not transient or attempt >= self._create_retry_attempts:
                    raise
                last = e
                await asyncio.sleep(self._create_retry_delay_s)
        raise last or RuntimeError("sandbox create failed")

    async def _exec(self, sid: str, command: str, timeout_s: float, *,
                    raw: bool = False) -> tuple[int, str]:
        """Run ``bash -lc`` in the pod; return ``(exit_code, stdout+stderr)``.

        Commands are anchored at ``/testbed`` (the exec endpoint has no cwd and
        does not persist ``cd`` between calls); we anchor with ``cd … && …`` and
        deliberately avoid a ``(...)`` subshell — it breaks trailing heredocs.
        ``raw=True`` skips the anchor (for absolute-path setup like writing /tmp).
        """
        full = command if raw else f"cd {WORKDIR} && {command}"
        body: dict[str, Any] = {"command": ["bash", "-lc", full]}
        if timeout_s:
            body["timeout"] = int(timeout_s)
        url = f"{self._sandbox_url}{self._sandbox_prefix}/{sid}/exec"
        sess = await self._ensure_session()
        timeout = aiohttp.ClientTimeout(total=timeout_s + 30.0)
        try:
            async with sess.post(url, headers=self._sandbox_headers,
                                 json=body, timeout=timeout) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    return -1, f"<exec HTTP {resp.status}: {_excerpt(text)}>"
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    return -1, f"<exec non-JSON: {_excerpt(text)}>"
        except asyncio.TimeoutError:
            return -1, f"<exec timed out after {timeout_s}s>"
        rc = data.get("exit_code")
        try:
            rc_i = int(rc) if rc is not None else 0
        except (TypeError, ValueError):
            rc_i = 0
        stdout = data.get("stdout") or ""
        stderr = data.get("stderr") or ""
        combined = stdout if not stderr else stdout + ("\n" if stdout else "") + stderr
        return rc_i, combined

    async def _write_file(self, sid: str, path: str, content: str) -> tuple[int, str]:
        """Write text to a pod path (base64 to dodge shell-escaping issues)."""
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        return await self._exec(
            sid, f"echo {b64} | base64 -d > {path}", 120.0, raw=True,
        )

    async def _extract_patch(self, sid: str) -> str:
        """The agent's diff: working tree vs the baseline commit."""
        rc, out = await self._exec(
            sid, "git add -A && git diff --cached HEAD 2>/dev/null | base64 -w0", 120.0,
        )
        try:
            return base64.b64decode(out.strip()).decode("utf-8", "replace")
        except Exception:
            return ""

    # ---- grading (fresh pod, swebench source of truth) --------------- #

    async def _verify(self, item: dict, model_patch: str) -> dict[str, Any]:
        block: dict[str, Any] = {
            "method": "swebench:FAIL_TO_PASS+PASS_TO_PASS", "resolved": False,
        }
        try:
            spec = make_test_spec(item)
            eval_script = spec.eval_script
        except Exception as e:
            block["error"] = f"swebench spec unavailable: {type(e).__name__}: {e}"
            return block

        image = self._image_for(item)
        sid = await self._create_with_retry(self._build_spec(image))
        try:
            await self._write_file(sid, "/tmp/patch.diff", model_patch)
            apply_rc, apply_log = await self._exec(
                sid,
                "git apply -v /tmp/patch.diff "
                "|| git apply --3way /tmp/patch.diff "
                "|| patch -p1 < /tmp/patch.diff",
                self._apply_patch_timeout_s,
            )
            if apply_rc != 0:
                block["error"] = "model patch did not apply"
                block["apply_log_tail"] = _excerpt(apply_log, 800)
                return block
            await self._write_file(sid, "/tmp/eval.sh", eval_script)
            _, log = await self._exec(sid, "bash /tmp/eval.sh 2>&1", self._eval_timeout_s)
            block.update(grade(spec, log))
        except Exception as e:
            block["error"] = f"verification run failed: {type(e).__name__}: {e}"
        finally:
            await self._sandbox_delete(sid)
        return block

    # ---- main loop --------------------------------------------------- #

    async def run(self, ctx: AgentContext) -> AgentResult:
        item = ctx.item if isinstance(ctx.item, dict) else {}
        if not self._sandbox_url:
            raise RuntimeError("SWEBenchAgent requires sandbox_url to be set")

        instance_id = item.get("instance_id", "?")
        image = self._image_for(item)
        start = time.monotonic()
        n_calls = n_actions = 0
        submitted = False
        exit_status = "step_limit"
        model_patch = ""
        grading: dict[str, Any] = {}

        sid = await self._create_with_retry(self._build_spec(image))
        try:
            # Baseline commit so the later diff captures only the agent's edits.
            await self._exec(
                sid,
                "git config user.email a@b.c && git config user.name a "
                "&& git add -A && git commit -q -m baseline --allow-empty",
                self._baseline_timeout_s,
            )

            user_msg = SWEBENCH_INSTANCE_TEMPLATE.format(
                instance_id=instance_id,
                repo=item.get("repo", ""),
                problem_statement=item.get("problem_statement", ""),
            )
            messages: list[dict] = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_msg},
            ]
            self.last_messages = messages

            while True:
                if n_calls >= self._step_limit:
                    exit_status = "step_limit"
                    break
                if time.monotonic() - start > self._total_wall_s:
                    exit_status = "time_limit"
                    break

                reply = await self._send(messages)
                n_calls += 1
                messages.append({"role": "assistant", "content": reply})

                action = self._parse_action(reply)
                if action is None:
                    exit_status = "no_action"
                    break

                first_line = action.splitlines()[0].strip() if action.strip() else ""
                if first_line == SUBMIT_TOKEN:
                    submitted = True
                    exit_status = "submitted"
                    break

                rc, output = await self._exec(sid, action, self._timeout_per_step_s)
                n_actions += 1
                obs = self._truncate(output)
                messages.append({
                    "role": "user",
                    "content": f"$ {action}\nreturncode: {rc}\n```\n{obs}\n```",
                })

            model_patch = await self._extract_patch(sid)
        finally:
            await self._sandbox_delete(sid)

        # Grade in a fresh pod (the rollout pod is gone — the verifier must not
        # see the agent's shell history or any leftover state).
        if not self._grade:
            grading = {"method": "skipped", "resolved": False}
        elif not model_patch.strip():
            grading = {"method": "swebench:FAIL_TO_PASS+PASS_TO_PASS",
                       "resolved": False, "error": "empty patch"}
        else:
            grading = await self._verify(item, model_patch)

        resolved = bool(grading.get("resolved"))
        elapsed = time.monotonic() - start
        f2p_total = grading.get("f2p_total", len(as_list(item.get("FAIL_TO_PASS"))))
        p2p_total = grading.get("p2p_total", len(as_list(item.get("PASS_TO_PASS"))))

        if resolved:
            error = None
        elif grading.get("error"):
            error = grading["error"]
        elif model_patch.strip():
            error = "unresolved"
        else:
            error = exit_status

        return AgentResult(
            output=model_patch.strip(),
            ok=resolved,
            request_ok=True,
            error=error,
            metrics={
                "steps": float(n_calls),
                "actions": float(n_actions),
                "elapsed_s": float(elapsed),
                "patch_chars": float(len(model_patch)),
                "f2p_pass": float(grading.get("f2p_pass", 0)),
                "f2p_total": float(f2p_total),
                "p2p_pass": float(grading.get("p2p_pass", 0)),
                "p2p_total": float(p2p_total),
                "resolved": 1.0 if resolved else 0.0,
            },
            meta={
                "exit_status": exit_status,
                "instance_id": instance_id,
                "repo": item.get("repo"),
                "image": image,
                "submitted": submitted,
                "resolved": resolved,
                "grading_method": grading.get("method"),
                "grading_error": grading.get("error"),
                "apply_log_tail": grading.get("apply_log_tail"),
            },
        )

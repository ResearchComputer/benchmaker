"""SWE-bench Verified agent driver — a thin subclass of CodingAgent.

For each task item the agent:

  1. Allocates a fresh Flash Sandbox pod (``python:3.11`` by default).
  2. Bootstraps the repo: clones ``{repo}`` from GitHub, checks out
     ``base_commit``, and installs the package in editable mode.
  3. Runs the model loop with a SWE-bench-shaped prompt that asks for a
     single unified diff as the final submission.
  4. Before tearing the pod down, grades the submission inside the same
     sandbox: applies ``test_patch`` + the agent's diff, then runs the
     ``FAIL_TO_PASS`` and ``PASS_TO_PASS`` tests with pytest. An instance
     counts as a pass iff every F2P test went from failing-on-base to
     passing-with-patch AND every P2P test still passes.

The grader is intentionally lightweight — it does not pre-run the F2P tests
against the unpatched base to verify they were originally failing, and it
relies on pytest's ``-k`` matcher to resolve the bare test names that
SWE-bench provides. That's enough for a smoke check, not the official score.

Expected item dict shape::

    {
        "instance_id":     "sympy__sympy-20916",
        "repo":            "sympy/sympy",
        "base_commit":     "82298df...",
        "problem_statement": "...",
        "test_patch":      "diff --git a/...",
        "FAIL_TO_PASS":    ["test_foo", ...],
        "PASS_TO_PASS":    ["test_bar", ...],
        "version":         "1.8",
    }
"""

from __future__ import annotations

import asyncio
import re
import shlex
import time
from typing import Any, Optional

from benchmaker import AgentContext, AgentResult
from examples.coding_agent.coding_agent import (
    SUBMIT_TOKEN,
    CodingAgent,
    _BASH_FENCE,
    _excerpt,
)


SWEBENCH_SYSTEM_PROMPT = """You are a senior Python engineer fixing a bug in an
open-source repository. You work in a sandbox with the target repo cloned at
the buggy commit under `/repo`.

Each turn, emit **exactly one** shell command in a fenced bash block:

```bash
your-command-here
```

You will see the command's combined stdout+stderr in the next message. **Each
command runs in a fresh shell** — working directory and environment variables
do NOT persist across turns. Always prefix path-sensitive work with
`cd /repo && ...`. Filesystem edits DO persist between commands.

Investigate the codebase, identify the fix, and edit the source files in place
(use `python -c` or heredocs with `cat <<'EOF' > path/to/file.py ... EOF`).

When you are ready to submit, emit a fenced block whose **first line** is
exactly:

    COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

…followed by a single unified diff against `/repo` (`diff --git a/... b/...`
headers). The diff is what's evaluated — your in-tree edits ARE preserved
across commands, so a clean workflow is: edit files, then submit the output
of `git -C /repo diff` verbatim. Example:

```bash
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
diff --git a/sympy/path/to/file.py b/sympy/path/to/file.py
--- a/sympy/path/to/file.py
+++ b/sympy/path/to/file.py
@@
-old line
+new line
```

Guidelines:
- Prefer `cat`, `grep -rn`, `sed -n`, `python -c "..."` over heavy edits.
- After editing files, run `git -C /repo diff` to capture the unified diff,
  then submit it verbatim.
- Do not include explanation text outside the bash block.
"""


SWEBENCH_INSTANCE_TEMPLATE = """# Task: fix {instance_id}

Repository: **{repo}** (checked out at `{base_commit}` in `/repo`).

## Problem statement

{problem_statement}

## How you will be graded

A separate evaluator will apply your unified diff to `/repo` on top of the
project's hidden test patch, then run these tests with pytest:

- FAIL_TO_PASS (must go from failing to passing): {fail_to_pass}
- PASS_TO_PASS (must keep passing): {pass_to_pass_summary}

Begin.
"""


_DIFF_HEAD = re.compile(r"^(?:diff --git |--- |\+\+\+ |Index: |@@ )", re.MULTILINE)


class SWEBenchAgent(CodingAgent):
    """CodingAgent specialised for SWE-bench Verified instances."""

    def __init__(
        self,
        *args: Any,
        repo_url_template: str = "https://github.com/{repo}",
        bootstrap_timeout_s: float = 600.0,
        grade_timeout_s: float = 600.0,
        pytest_timeout_s: float = 300.0,
        extra_pip: tuple[str, ...] = ("mpmath",),
        create_retry_attempts: int = 6,
        create_retry_delay_s: float = 5.0,
        **kwargs: Any,
    ):
        kwargs.setdefault("system_prompt", SWEBENCH_SYSTEM_PROMPT)
        kwargs.setdefault("instance_template", SWEBENCH_INSTANCE_TEMPLATE)
        # SWE-bench needs a real Python + git environment, not alpine.
        spec = dict(kwargs.get("sandbox_spec") or {})
        spec.setdefault("image", "python:3.11")
        spec.setdefault("cpu_cores", 1.0)
        spec.setdefault("memory_mb", 2048)
        kwargs["sandbox_spec"] = spec
        # /exec on this cluster preserves the container's filesystem across
        # calls; /pshell does NOT (it runs in a busybox sidecar without the
        # image's userspace). So we use /exec for everything and instruct the
        # model that working-directory does not persist between commands.
        kwargs.setdefault("sandbox_persistent", False)
        super().__init__(*args, **kwargs)
        self._repo_url_template = repo_url_template
        self._bootstrap_timeout_s = bootstrap_timeout_s
        self._grade_timeout_s = grade_timeout_s
        self._pytest_timeout_s = pytest_timeout_s
        self._extra_pip = tuple(extra_pip)
        self._create_retry_attempts = create_retry_attempts
        self._create_retry_delay_s = create_retry_delay_s
        # Set by run() so a driver can inspect the model trajectory afterward.
        self.last_messages: Optional[list[dict]] = None

    async def _sandbox_create_with_retry(self) -> str:
        """Wrap _sandbox_create with bounded retry on transient 503s.

        The Flash Sandbox cluster occasionally responds with
        ``HTTP 503: no healthy nodes available`` when at capacity. Retry a
        handful of times so a single instance doesn't get nuked by a flake.
        """
        last: Optional[RuntimeError] = None
        for attempt in range(1, self._create_retry_attempts + 1):
            try:
                return await self._sandbox_create()
            except RuntimeError as e:
                msg = str(e)
                transient = "HTTP 503" in msg or "HTTP 502" in msg
                if not transient or attempt >= self._create_retry_attempts:
                    raise
                last = e
                await asyncio.sleep(self._create_retry_delay_s)
        # Unreachable — raise is inside the loop on the final attempt.
        raise last or RuntimeError("sandbox create failed")

    # ---- bootstrap & grade ------------------------------------------- #

    async def _bootstrap(self, sid: str, item: dict) -> tuple[bool, str]:
        """Clone {repo}@{base_commit} into /repo and install deps.

        Bootstrap is run as a single one-shot ``/exec`` (not ``/pshell``)
        because long-running multi-step pshell sessions on this cluster have
        been observed to terminate mid-command on this image.
        """
        repo = item["repo"]
        base = item["base_commit"]
        url = self._repo_url_template.format(repo=repo)
        pip_extra = " ".join(shlex.quote(p) for p in self._extra_pip) or ""
        # The `python:3.11` image on this cluster doesn't always ship git;
        # install it if missing. apt-get is silent on hit, ~10s on miss.
        script = (
            f"set -e; "
            f"command -v git >/dev/null 2>&1 || "
            f"  (apt-get update -qq && apt-get install -y -qq git >/dev/null); "
            f"git clone --quiet {shlex.quote(url)} /repo && "
            f"cd /repo && "
            f"git checkout --quiet {shlex.quote(base)} && "
            f"python -m pip install --quiet --no-input "
            f"  pytest {pip_extra} && "
            f"python -m pip install --quiet --no-input -e . && "
            f"echo BOOTSTRAP_OK"
        )
        rc, out = await self._sandbox_oneshot(
            sid, script, total_timeout_s=self._bootstrap_timeout_s,
        )
        return (rc == 0 and "BOOTSTRAP_OK" in out), out

    async def _sandbox_oneshot(
        self, sid: str, cmd: str, *, total_timeout_s: float,
    ) -> tuple[int, str]:
        """POST one command to ``/exec`` (stateless) and decode the result.

        Bypasses the agent's ``_sandbox_exec`` so we can pin the endpoint
        independently of ``sandbox_persistent``.
        """
        import json as _json
        url = f"{self._sandbox_url}{self._sandbox_prefix}/{sid}/exec"
        body = {"command": ["sh", "-c", cmd]}
        sess = await self._ensure_session()
        timeout = __import__("aiohttp").ClientTimeout(total=total_timeout_s + 30.0)
        try:
            async with sess.post(url, headers=self._sandbox_headers,
                                 json=body, timeout=timeout) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    return -1, f"<exec HTTP {resp.status}: {_excerpt(text)}>"
                try:
                    data = _json.loads(text)
                except _json.JSONDecodeError:
                    return -1, f"<exec non-JSON: {_excerpt(text)}>"
        except asyncio.TimeoutError:
            return -1, f"<exec timed out after {total_timeout_s}s>"
        rc = data.get("exit_code")
        try:
            rc_i = int(rc) if rc is not None else 0
        except (TypeError, ValueError):
            rc_i = 0
        stdout = data.get("stdout") or ""
        stderr = data.get("stderr") or ""
        combined = stdout if not stderr else (stdout + ("\n" if stdout else "") + stderr)
        return rc_i, combined

    @staticmethod
    def _looks_like_diff(text: str) -> bool:
        return bool(_DIFF_HEAD.search(text or ""))

    @staticmethod
    def _extract_test_files(test_patch: str) -> list[str]:
        files: list[str] = []
        for line in (test_patch or "").splitlines():
            if line.startswith("+++ b/"):
                p = line[6:].strip()
                if p and p != "/dev/null":
                    files.append(p)
        # Preserve order, dedupe.
        seen, ordered = set(), []
        for f in files:
            if f not in seen:
                seen.add(f); ordered.append(f)
        return ordered

    async def _apply_patch(
        self, sid: str, label: str, patch: str
    ) -> tuple[bool, str]:
        # Base64 the patch so multi-line content with `EOF` markers / backticks
        # doesn't fight our shell quoting. One-shot exec — no pshell state needed.
        import base64
        b64 = base64.b64encode(patch.encode("utf-8")).decode("ascii")
        cmd = (
            f"cd /repo && "
            f"printf %s {shlex.quote(b64)} | base64 -d > /tmp/{label}.patch && "
            f"git apply --whitespace=nowarn /tmp/{label}.patch && "
            f"echo APPLY_{label}_OK"
        )
        rc, out = await self._sandbox_oneshot(sid, cmd, total_timeout_s=120.0)
        return (rc == 0 and f"APPLY_{label}_OK" in out), out

    async def _run_pytest(
        self, sid: str, test_files: list[str], test_names: list[str]
    ) -> dict[str, str]:
        """Run pytest on (test_files) filtered by `-k <name>` for each test.

        Returns dict[test_name → 'passed' | 'failed' | 'error' | 'missing'].
        """
        if not test_names:
            return {}
        # `-k "a or b or c"` — names are bare function names from F2P/P2P.
        k_expr = " or ".join(test_names)
        files_arg = " ".join(shlex.quote(f) for f in test_files) if test_files else ""
        # `-v` so we get one PASSED/FAILED line per test we can grep.
        cmd = (
            f"cd /repo && "
            f"timeout {int(self._pytest_timeout_s)} "
            f"python -m pytest --tb=no -v -p no:cacheprovider "
            f"  -k {shlex.quote(k_expr)} {files_arg} 2>&1 | tail -200"
        )
        _, out = await self._sandbox_oneshot(
            sid, cmd, total_timeout_s=self._pytest_timeout_s,
        )
        return _parse_pytest_outcomes(out, test_names)

    async def _grade(
        self, sid: str, item: dict, submission: str
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "submitted_diff": False,
            "test_patch_applied": False,
            "pred_patch_applied": False,
            "f2p_outcomes": {},
            "p2p_outcomes": {},
            "f2p_pass": 0, "f2p_fail": 0,
            "p2p_pass": 0, "p2p_fail": 0,
            "grading_log": "",
        }
        if not submission or not self._looks_like_diff(submission):
            result["grading_log"] = "no diff in submission"
            return result
        result["submitted_diff"] = True

        # Apply test_patch (the hidden tests for this instance).
        ok_t, log_t = await self._apply_patch(sid, "test", item.get("test_patch") or "")
        result["test_patch_applied"] = ok_t
        if not ok_t:
            result["grading_log"] = f"test_patch failed to apply:\n{_excerpt(log_t)}"
            return result

        # Apply the agent's predicted patch on top.
        ok_p, log_p = await self._apply_patch(sid, "pred", submission)
        result["pred_patch_applied"] = ok_p
        if not ok_p:
            result["grading_log"] = f"predicted patch failed to apply:\n{_excerpt(log_p)}"
            return result

        # Resolve test files from the test_patch headers so pytest does
        # targeted discovery instead of crawling the whole repo.
        test_files = self._extract_test_files(item.get("test_patch") or "")

        f2p = list(item.get("FAIL_TO_PASS") or [])
        p2p = list(item.get("PASS_TO_PASS") or [])
        result["f2p_outcomes"] = await self._run_pytest(sid, test_files, f2p)
        result["p2p_outcomes"] = await self._run_pytest(sid, test_files, p2p)

        result["f2p_pass"] = sum(1 for v in result["f2p_outcomes"].values() if v == "passed")
        result["f2p_fail"] = len(f2p) - result["f2p_pass"]
        result["p2p_pass"] = sum(1 for v in result["p2p_outcomes"].values() if v == "passed")
        result["p2p_fail"] = len(p2p) - result["p2p_pass"]
        return result

    # ---- main loop --------------------------------------------------- #

    async def run(self, ctx: AgentContext) -> AgentResult:
        item = ctx.item if isinstance(ctx.item, dict) else {}
        if not self._sandbox_url:
            raise RuntimeError("SWEBenchAgent requires sandbox_url to be set")

        instance_id = item.get("instance_id", "?")
        sandbox_id = await self._sandbox_create_with_retry()
        bootstrap_log = ""
        bootstrap_ok = False
        grading: dict[str, Any] = {}
        submission: Optional[str] = None
        exit_status = "step_limit"
        n_calls = n_actions = 0
        start = time.monotonic()

        try:
            # Step 1: bootstrap the repo.
            bootstrap_ok, bootstrap_log = await self._bootstrap(sandbox_id, item)
            if not bootstrap_ok:
                exit_status = "bootstrap_failed"
                return AgentResult(
                    output="",
                    ok=False, error="bootstrap_failed",
                    metrics={"steps": 0.0, "actions": 0.0,
                             "elapsed_s": float(time.monotonic() - start),
                             "f2p_pass": 0.0,
                             "f2p_total": float(len(item.get("FAIL_TO_PASS") or [])),
                             "p2p_pass": 0.0,
                             "p2p_total": float(len(item.get("PASS_TO_PASS") or [])),
                             "instance_pass": 0.0},
                    meta={"exit_status": exit_status,
                          "sandbox_id": sandbox_id,
                          "instance_id": instance_id,
                          "bootstrap_ok": False,
                          "submitted_diff": False,
                          "pred_patch_applied": False,
                          "test_patch_applied": False,
                          "f2p_outcomes": {},
                          "p2p_outcomes": {},
                          "bootstrap_log_tail": _excerpt(bootstrap_log, 800),
                          "grading_log_tail": ""},
                )

            # Step 2: prompt the model. Customise the instance text with
            # task-specific fields.
            f2p_list = list(item.get("FAIL_TO_PASS") or [])
            p2p_list = list(item.get("PASS_TO_PASS") or [])
            p2p_summary = (
                ", ".join(p2p_list[:3])
                + (f" ... (+{len(p2p_list) - 3} more)" if len(p2p_list) > 3 else "")
                if p2p_list else "(none)"
            )
            user_msg = SWEBENCH_INSTANCE_TEMPLATE.format(
                instance_id=instance_id,
                repo=item.get("repo", ""),
                base_commit=item.get("base_commit", "")[:12],
                problem_statement=item.get("problem_statement", ""),
                fail_to_pass=", ".join(f2p_list) or "(none)",
                pass_to_pass_summary=p2p_summary,
            )
            messages: list[dict] = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_msg},
            ]
            self.last_messages = messages

            # Step 3: main loop.
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

                stripped = action.strip()
                first_line = stripped.splitlines()[0].strip() if stripped else ""
                if first_line == SUBMIT_TOKEN:
                    submission = "\n".join(action.splitlines()[1:]).strip()
                    exit_status = "submitted"
                    break

                rc, output = await self._sandbox_exec(sandbox_id, action)
                n_actions += 1
                obs = self._truncate(output)
                messages.append({
                    "role": "user",
                    "content": f"$ {action}\nreturncode: {rc}\n```\n{obs}\n```",
                })

            # Step 4: grade.
            if submission is not None:
                grading = await self._grade(sandbox_id, item, submission)
        finally:
            await self._sandbox_delete(sandbox_id)

        f2p_pass = int(grading.get("f2p_pass", 0))
        f2p_total = len(item.get("FAIL_TO_PASS") or [])
        p2p_pass = int(grading.get("p2p_pass", 0))
        p2p_total = len(item.get("PASS_TO_PASS") or [])
        instance_pass = bool(
            grading.get("submitted_diff")
            and grading.get("pred_patch_applied")
            and f2p_pass == f2p_total
            and p2p_pass == p2p_total
            and (f2p_total + p2p_total) > 0
        )

        elapsed = time.monotonic() - start
        return AgentResult(
            output=(submission or "").strip(),
            ok=instance_pass,
            error=None if instance_pass else (
                exit_status if not grading.get("submitted_diff") else "tests_failed"
            ),
            metrics={
                "steps": float(n_calls),
                "actions": float(n_actions),
                "elapsed_s": float(elapsed),
                "f2p_pass": float(f2p_pass),
                "f2p_total": float(f2p_total),
                "p2p_pass": float(p2p_pass),
                "p2p_total": float(p2p_total),
                "instance_pass": 1.0 if instance_pass else 0.0,
            },
            meta={
                "exit_status": exit_status,
                "sandbox_id": sandbox_id,
                "instance_id": instance_id,
                "bootstrap_ok": bootstrap_ok,
                "submitted_diff": bool(grading.get("submitted_diff")),
                "pred_patch_applied": bool(grading.get("pred_patch_applied")),
                "test_patch_applied": bool(grading.get("test_patch_applied")),
                "f2p_outcomes": grading.get("f2p_outcomes", {}),
                "p2p_outcomes": grading.get("p2p_outcomes", {}),
                "grading_log_tail": _excerpt(grading.get("grading_log", ""), 800),
            },
        )


# ---- pytest output parsing ------------------------------------------- #

# Match lines like:
#   sympy/.../test_foo.py::test_super_sub PASSED                       [ 50%]
#   sympy/.../test_foo.py::TestX::test_y FAILED                        [100%]
_PYTEST_LINE = re.compile(
    r"^(?P<file>[^\s:]+\.py)::(?:(?P<cls>[^\s:]+)::)?(?P<test>[^\s\[]+)"
    r"(?P<param>\[.*?\])?\s+(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)",
    re.MULTILINE,
)


def _parse_pytest_outcomes(output: str, wanted: list[str]) -> dict[str, str]:
    """Map each wanted bare test name → 'passed'/'failed'/'error'/'missing'.

    Pytest with `-k` matches by substring; the bare names in F2P/P2P should
    match the function-name component of one or more node ids. If any matched
    line for a test is non-PASSED, the test is marked failed/error/skipped
    accordingly. Parametrised variants of the same name all roll up under that
    name, so any failure of a parameterised variant counts as a failure.
    """
    outcomes: dict[str, list[str]] = {name: [] for name in wanted}
    wanted_set = set(wanted)
    for m in _PYTEST_LINE.finditer(output or ""):
        test_name = m.group("test")
        status = m.group("status")
        # The bare F2P/P2P name may match either the method name (no class)
        # or a class-qualified name. Try both.
        candidates = {test_name}
        cls = m.group("cls")
        if cls:
            candidates.add(f"{cls}::{test_name}")
        for cand in candidates:
            if cand in wanted_set:
                outcomes[cand].append(status)
    final: dict[str, str] = {}
    for name in wanted:
        statuses = outcomes[name]
        if not statuses:
            final[name] = "missing"
        elif all(s == "PASSED" for s in statuses):
            final[name] = "passed"
        elif any(s == "ERROR" for s in statuses):
            final[name] = "error"
        else:
            final[name] = "failed"
    return final

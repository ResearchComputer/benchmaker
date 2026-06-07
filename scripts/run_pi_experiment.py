#!/usr/bin/env python3
"""Run a pi (or any registered agent) SWE-bench experiment via harbor_eval.

Thin driver around ``benchmaker.swebench.harbor_eval``: builds the command
line, wires the Flash Sandbox URL into the environment, and optionally repeats
the same run several times (continuing past per-run failures and reporting a
summary at the end).

Any arguments after ``--`` (or any flags this script does not recognise) are
forwarded verbatim to ``harbor_eval``.

Examples::

    tools/scripts/run_pi_experiment.py --repeat 3 --sandbox-url http://localhost:8080
    tools/scripts/run_pi_experiment.py --agent pi-host --n-tasks 5 -- --force-build
"""

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path

# Repo root: tools/scripts/<this file> -> ../../
REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_job_name(agent: str) -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{agent}-{stamp}"


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable),
                        help="Python interpreter used to launch harbor_eval "
                             "(default: the active interpreter running this script).")
    parser.add_argument("--agent", default="pi")
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "swebench-verified"))
    parser.add_argument("--n-tasks", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--backend", default=os.environ.get("BACKEND", "docker"))
    parser.add_argument("--timeout-multiplier", type=int,
                        default=int(os.environ.get("TIMEOUT_MULT", "5000")))
    parser.add_argument("--job-name", default=os.environ.get("JOB_NAME"),
                        help="Base job name (default: <agent>-<UTC timestamp>).")
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("REPEAT", "5")),
                        help="Number of times to run the same command.")
    parser.add_argument("--sandbox-url",
                        default="https://sandbox.yao.sh",
                        help="Flash Sandbox URL (exported as FLASH_SANDBOX_URL).")
    # Everything else is forwarded to harbor_eval.
    args, passthrough = parser.parse_known_args(argv)
    # Drop a lone "--" separator if argparse left it in the remainder.
    passthrough = [a for a in passthrough if a != "--"]
    return args, passthrough


def main(argv: list[str]) -> int:
    args, passthrough = parse_args(argv)

    env = os.environ.copy()
    env.setdefault("SWE_IMAGE_MIRROR", "swe-images")
    if args.sandbox_url:
        env["FLASH_SANDBOX_URL"] = args.sandbox_url

    base_job_name = args.job_name or _default_job_name(args.agent)

    failures = 0
    for i in range(1, args.repeat + 1):
        if args.repeat > 1:
            run_job_name = f"{base_job_name}-run{i}"
            print(f"=== Run {i}/{args.repeat}: {run_job_name} ===", file=sys.stderr, flush=True)
        else:
            run_job_name = base_job_name

        cmd = [
            args.python, "-m", "benchmaker.swebench.harbor_eval",
            "--agent", args.agent,
            "--dataset", args.dataset,
            "--n-tasks", str(args.n_tasks),
            "--concurrency", str(args.concurrency),
            "--backend-type", args.backend,
            "--timeout-multiplier", str(args.timeout_multiplier),
            "--job-name", run_job_name,
            *passthrough,
        ]
        status = subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode
        if status != 0:
            failures += 1
            print(f"!!! Run {i}/{args.repeat} ({run_job_name}) failed with exit code {status}",
                  file=sys.stderr, flush=True)

    if failures:
        print(f"=== {failures}/{args.repeat} run(s) failed ===", file=sys.stderr, flush=True)
        return 1
    print(f"=== All {args.repeat} run(s) completed ===", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

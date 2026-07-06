#!/usr/bin/env python3
"""Collect **graded pi trajectories** on SWE-bench in two execution modes.

This runs **pi** (`@earendil-works/pi-coding-agent`) on SWE-bench via harbor and
then produces one consolidated JSONL where every recorded trajectory carries its
harbor verifier grade. Two modes (the two harbor agents in
``benchmaker.swebench.pi_agent``):

* ``pi-host``      — pi runs on the host; its ``bash``/``read``/``write``/``edit``
  tools are routed into the per-instance environment (``/testbed``) via the
  localhost bridge. Defaults to routing **all** four tools (``--route-tools all``)
  so a host run is tool-identical to a container run.
* ``pi-container`` — pi runs entirely inside the per-instance environment.

The script has two responsibilities, cleanly split:

1. **run** — shell out to ``benchmaker.swebench.harbor_eval`` with the chosen
   ``--agent`` (the ``run_pi_experiment.py`` convention). harbor owns environment
   provisioning, agent execution, and grading. Unrecognised flags pass through
   verbatim to ``harbor_eval``.
2. **collect** — :func:`collect_from_job_dir`, a pure function that walks the
   resulting job dir and fuses each trial's parsed trajectory
   (``trajectory.parse_pi_conversation``) with its grade
   (``verifier/reward.txt`` + ``verifier/report.json``).

The grading fields are *additive* keys on top of ``Trajectory.to_dict()``, so the
output stays a valid replay store (``trajectory.load_store`` ignores them).

**Incremental save + resume.** The ``--out`` JSONL is the checkpoint. Each trial
is appended the moment it finishes (its ``result.json`` lands in the job dir), so
a crash — e.g. the systemd memory cap OOM-killing the run — never loses a graded
trajectory. On the next run the already-saved instance ids are read back and
handed to ``harbor_eval`` as ``--exclude-task`` while ``--n-tasks`` is lowered by
how many are already done, so harbor only runs the tasks still missing and the
file tops up toward ``--n-tasks`` total. Pass ``--no-resume`` to ignore an
existing ``--out`` and re-run everything. (Resume keys off the output file, not
harbor's job dir, so it survives changing ``--concurrency`` between runs.)

Examples::

    # run pi on the host (all tools routed), 5 tasks, then collect graded trajectories
    FLASH_SANDBOX_URL=http://localhost:8080 \\
        scripts/collect_trajectories.py --mode pi-host --n-tasks 5

    # pi in-container; forward an extra harbor_eval flag verbatim
    scripts/collect_trajectories.py --mode pi-container --n-tasks 5 -- --force-build

    # re-fuse an already-collected job dir (no re-run)
    scripts/collect_trajectories.py --collect-only jobs/pi-host-20260614T101500Z
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Optional, TextIO

from benchmaker.env import load_dotenv
from benchmaker.swebench.trajectory import parse_pi_conversation, parse_pi_session

# Repo root: scripts/<this file> -> ../
REPO_ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("benchmaker.collect_trajectories")

# pi writes the same `--mode json` event stream under one of these per-trial
# agent logs; which one tells us the execution mode.
_LOG_NAMES = (("pi-container.log", "pi-container"), ("pi-host.log", "pi-host"))

# harbor writes this file when a trial is fully done (agent + verifier). We use
# its presence as the "trial is final, safe to collect" signal while streaming.
_TRIAL_DONE_MARKER = "result.json"

# How often the streamer polls the job dir for newly-finished trials.
_POLL_SEC = 5.0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _default_job_name(mode: str) -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{mode}-{stamp}"


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse our flags; unknown flags become harbor_eval passthrough."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["pi-host", "pi-container"],
                   help="pi execution mode (required unless --collect-only).")
    p.add_argument("--dataset", default="swebench-verified",
                   help="Harbor dataset slug (default: swebench-verified).")
    p.add_argument("--n-tasks", type=int, default=None, help="Cap dataset tasks.")
    p.add_argument("--concurrency", type=int, default=16,
                   help="Concurrent trials (default: harbor_eval's).")
    p.add_argument("--model", default=os.environ.get("OPENAI_COMPATIBLE_MODEL")
                   or os.environ.get("OPENAI_MODEL"),
                   help="Model id. Default: $OPENAI_COMPATIBLE_MODEL / $OPENAI_MODEL.")
    p.add_argument("--timeout-multiplier", type=float, default=4.0,
                   help="Multiplier on harbor's timeouts (SWE-bench cold-start "
                        "needs 4-6x).")
    p.add_argument("--backend-type", default="docker", help="harbor backend type.")
    p.add_argument("--route-tools", choices=["all", "bash"], default="all",
                   help="pi-host only: route all four tools (default) or bash only.")
    p.add_argument("--job-name", default=None,
                   help="Job name (default: <mode>-<UTC stamp>).")
    p.add_argument("--jobs-dir", default=None,
                   help="Parent dir for the job bundle (default: harbor's 'jobs').")
    p.add_argument("--out", default=None,
                   help="Output JSONL (default: <job_dir>/pi-trajectories.jsonl).")
    p.add_argument("--collect-only", default=None, metavar="JOB_DIR",
                   help="Skip the run; fuse an existing job dir only.")
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                   help="Ignore an existing --out and re-run every task (default: "
                        "resume — skip tasks already saved in --out).")
    p.add_argument("--exclude-task", action="append", default=[],
                   help="Instance id(s) for harbor_eval to skip (repeatable). "
                        "Normally auto-filled from the --out checkpoint on resume.")
    p.add_argument("--python", default=sys.executable,
                   help="Interpreter used to launch harbor_eval.")
    args, passthrough = p.parse_known_args(argv)
    passthrough = [a for a in passthrough if a != "--"]
    if args.job_name is None:
        args.job_name = _default_job_name(args.mode or "pi")
    return args, passthrough


def build_harbor_argv(args: argparse.Namespace,
                      passthrough: Optional[list[str]] = None) -> list[str]:
    """The ``python -m ...`` argv (without the interpreter) for harbor_eval."""
    argv = ["-m", "benchmaker.swebench.harbor_eval", "--agent", args.mode]
    if args.mode == "pi-host":
        argv += ["--agent-kwarg", f"route_tools={args.route_tools}"]
    argv += ["--dataset", args.dataset]
    if args.n_tasks is not None:
        argv += ["--n-tasks", str(args.n_tasks)]
    if args.concurrency is not None:
        argv += ["--concurrency", str(args.concurrency)]
    if args.model:
        argv += ["--model", args.model]
    argv += ["--timeout-multiplier", str(args.timeout_multiplier)]
    argv += ["--backend-type", args.backend_type]
    argv += ["--job-name", args.job_name]
    if args.jobs_dir:
        argv += ["--jobs-dir", str(args.jobs_dir)]
    for task in getattr(args, "exclude_task", None) or []:
        argv += ["--exclude-task", task]
    argv += list(passthrough or [])
    return argv


def _job_dir_for(args: argparse.Namespace) -> Path:
    base = Path(args.jobs_dir) if args.jobs_dir else (REPO_ROOT / "jobs")
    if not base.is_absolute():
        base = REPO_ROOT / base
    return base / args.job_name


# The run path (subprocess + streaming collection) lives in run_job_streaming,
# below the collect helpers it depends on.


# --------------------------------------------------------------------------- #
# Collect (pure fusion)
# --------------------------------------------------------------------------- #


def _trial_log(trial_dir: Path) -> tuple[Optional[Path], Optional[str]]:
    for name, mode in _LOG_NAMES:
        p = trial_dir / "agent" / name
        if p.exists():
            return p, mode
    return None, None


# pi persists the live conversation here too. When a run is killed at the
# wall-clock cap, the `--mode json` log (pi-host.log/pi-container.log) is never
# flushed but this session jsonl survives — so it is our fallback source for the
# (validly graded) cap-hit trials that would otherwise be dropped.
_SESSION_GLOB = "agent/pi-home/.pi/agent/sessions/*/*.jsonl"

# result.json's `agent_result.metadata.mode` ("host"/"container") -> our label.
_MODE_BY_META = {"host": "pi-host", "container": "pi-container"}


def _trial_session_log(trial_dir: Path) -> Optional[Path]:
    """The most recent pi session jsonl for a trial, or None if there is none."""
    try:
        cands = sorted(trial_dir.glob(_SESSION_GLOB))
    except OSError:
        return None
    return cands[-1] if cands else None


def _mode_from_result_data(data: Optional[dict]) -> Optional[str]:
    """pi execution mode from a parsed ``result.json`` dict (the session log
    can't tell host from container on its own)."""
    md = (data.get("agent_result") or {}).get("metadata") if isinstance(data, dict) else None
    meta_mode = md.get("mode") if isinstance(md, dict) else None
    return _MODE_BY_META.get(meta_mode) if isinstance(meta_mode, str) else None


def _mode_from_result(trial_dir: Path) -> Optional[str]:
    """pi execution mode from ``result.json`` metadata (legacy file path)."""
    try:
        data = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    return _mode_from_result_data(data)


def _instance_id_from_config(trial_dir: Path) -> Optional[str]:
    """Clean instance id from the trial config (the prompt only has `# Task: ?`)."""
    try:
        data = json.loads((trial_dir / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    task = data.get("task") if isinstance(data, dict) else None
    path = task.get("path") if isinstance(task, dict) else None
    if isinstance(path, str) and path:
        return PurePosixPath(path).name
    return None


def _read_reward(trial_dir: Path) -> Optional[float]:
    try:
        return float((trial_dir / "verifier" / "reward.txt").read_text().strip())
    except Exception:
        return None


def _report_fields(rep: Optional[dict]) -> dict[str, Any]:
    """``resolved`` + FAIL_TO_PASS/PASS_TO_PASS counts from ONE report inner dict."""
    out: dict[str, Any] = {}
    if not isinstance(rep, dict):
        return out
    if isinstance(rep.get("resolved"), bool):
        out["resolved"] = rep["resolved"]
    ts = rep.get("tests_status")
    if isinstance(ts, dict):
        for field, key in (("FAIL_TO_PASS", "fail_to_pass"),
                           ("PASS_TO_PASS", "pass_to_pass")):
            bucket = ts.get(field)
            if isinstance(bucket, dict) and isinstance(bucket.get("success"), list):
                out[key] = len(bucket["success"])
    return out


def _read_report(trial_dir: Path) -> dict[str, Any]:
    """Best-effort ``resolved`` + FAIL_TO_PASS/PASS_TO_PASS counts from report.json.

    Legacy file path: report.json is keyed by instance id; take the first inner
    dict and delegate to :func:`_report_fields`.
    """
    try:
        data = json.loads((trial_dir / "verifier" / "report.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    for rep in data.values():  # report is keyed by instance id; take the first
        if isinstance(rep, dict):
            return _report_fields(rep)
    return {}


# The agent's raw ``exit_status`` (see ``benchmaker.swebench.pi_agent``) mapped
# to a coarse termination class. Anything not listed is an error (an
# ``exit_<code>`` string or an exception class name from pi_agent's except path).
_TERMINATION_BY_EXIT = {"ok": "completed", "time_limit": "timeout"}


def _exit_status_from_result(data: Optional[dict]) -> Optional[str]:
    """Raw agent ``exit_status`` from a parsed ``result.json`` dict.

    harbor records the agent's own status under ``agent_result.metadata`` —
    ``"ok"`` (the agent stopped on its own), ``"time_limit"`` (killed at the
    wall-clock cap), or an ``exit_<code>`` / exception-class string (it errored).
    A top-level ``exception_info`` is a harbor-level error even when no agent
    status was recorded.
    """
    if not isinstance(data, dict):
        return None
    md = (data.get("agent_result") or {}).get("metadata")
    es = md.get("exit_status") if isinstance(md, dict) else None
    if isinstance(es, str) and es:
        return es
    if data.get("exception_info") is not None:
        return "error"
    return None


def _read_exit_status(trial_dir: Path) -> Optional[str]:
    """Raw agent ``exit_status`` from ``result.json`` (legacy file path)."""
    try:
        data = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    return _exit_status_from_result(data)


def _termination_from_finish(finish_reason: Optional[str],
                             has_tool_calls: bool) -> str:
    """Completed-vs-cut-off from a trajectory's *last* turn.

    A trailing turn that issues no tool call ends pi's agent loop — the agent
    stopped on its own (``completed``). A trailing tool-call turn is the
    fingerprint of a run cut off before the model's next reply (a timeout *or*
    an error — the trajectory alone cannot tell which), so ``incomplete``. No
    last turn at all -> ``unknown``.
    """
    if finish_reason is None:
        return "unknown"
    return ("completed" if finish_reason == "stop" and not has_tool_calls
            else "incomplete")


def _termination_from_last_turn(traj: Any) -> str:
    """Infer termination from a parsed ``Trajectory``'s last recorded turn."""
    if not traj.turns:
        return "unknown"
    last = traj.turns[-1]
    return _termination_from_finish(last.finish_reason, bool(last.tool_calls))


def _completion_fields_from_status(exit_status: Optional[str],
                                   traj: Any) -> dict[str, Any]:
    """The completion-status block, given an already-resolved ``exit_status``.

    Authoritative when ``exit_status`` is not None (``status_source`` =
    ``result_json``); otherwise inferred from the trajectory's last turn
    (``inferred``) — coarser, since inference cannot split timeout from error.
    """
    if exit_status is not None:
        termination = _TERMINATION_BY_EXIT.get(exit_status, "error")
        source = "result_json"
    else:
        exit_status = "unknown"
        termination = _termination_from_last_turn(traj)
        source = "inferred"
    return {
        "exit_status": exit_status,
        "termination": termination,
        "completed": termination == "completed",
        "status_source": source,
    }


def _completion_fields(trial_dir: Path, traj: Any) -> dict[str, Any]:
    """The completion-status block fused onto a record (legacy file path)."""
    return _completion_fields_from_status(_read_exit_status(trial_dir), traj)


def _phase_secs(section: Any) -> Optional[float]:
    """finished_at - started_at (seconds) for a {started_at, finished_at} dict."""
    if not isinstance(section, dict):
        return None
    s, f = section.get("started_at"), section.get("finished_at")
    if not s or not f:
        return None
    try:
        ds = _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        df = _dt.datetime.fromisoformat(str(f).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (df - ds).total_seconds()


def _runtime_fields_from(result_data: Any, spans: list[dict],
                         exec_timeout_s: Any) -> dict[str, Any]:
    """Per-trial runtime block the parsed trajectory itself can't carry.

    Closes the three gaps a pure trajectory record has: (1) wall-clock + phase
    durations (from ``result.json``), (2) per-command exec timing — ``rc`` and
    ``duration_s`` for every ``sandbox_exec`` span (from ``timeline-spans.jsonl``),
    and (3) the per-command exec-timeout cap ``exec_timeout_s`` (from
    ``config.json``). Storing the raw spans + cap lets a consumer recompute any
    duration statistic or apply the exact ``rc<0 & d>=0.9*T`` timeout gate the
    sweep summary uses. Pure (takes already-loaded data) so the legacy and
    cleaned collection paths share it. Defensive: missing data degrades a field,
    never raises.
    """
    out: dict[str, Any] = {"exec_timeout_s": exec_timeout_s}
    if isinstance(result_data, dict):
        out["wall_s"] = _phase_secs(result_data.get("agent_execution"))
        out["phases"] = {
            "environment_setup_s": _phase_secs(result_data.get("environment_setup")),
            "agent_setup_s": _phase_secs(result_data.get("agent_setup")),
            "agent_execution_s": _phase_secs(result_data.get("agent_execution")),
            "verifier_s": _phase_secs(result_data.get("verifier")),
        }
    compact = [{"seq": s.get("seq"), "rc": s.get("rc"), "duration_s": s.get("duration_s")}
               for s in spans if isinstance(s, dict) and s.get("name") == "sandbox_exec"]
    out["exec_spans"] = compact
    if compact:
        durs = sorted(float(s["duration_s"] or 0.0) for s in compact)
        n = len(durs)
        out["exec_stats"] = {
            "n_exec": n,
            "n_timeout": sum(1 for s in compact if int(s["rc"] or 0) < 0),  # rc<0 == killed at cap
            "dur_mean": sum(durs) / n,
            "dur_p95": durs[int(0.95 * (n - 1))],
            "dur_max": durs[-1],
        }
    return out


def _read_result_json(trial_dir: Path) -> Any:
    try:
        return json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_spans(trial_dir: Path) -> list[dict]:
    sp = trial_dir / "agent" / "timeline-spans.jsonl"
    if not sp.exists():
        return []
    spans: list[dict] = []
    try:
        for line in sp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if isinstance(o, dict):
                spans.append(o)
    except Exception:
        return []
    return spans


def _exec_timeout_from_config_data(cfg: Any) -> Any:
    agent = cfg.get("agent") if isinstance(cfg, dict) else None
    kwargs = agent.get("kwargs") if isinstance(agent, dict) else None
    return kwargs.get("exec_timeout_s") if isinstance(kwargs, dict) else None


def _runtime_fields(trial_dir: Path) -> dict[str, Any]:
    """Legacy-layout runtime block: read result.json, timeline-spans, config.json."""
    cfg: Any = None
    try:
        cfg = json.loads((trial_dir / "config.json").read_text(encoding="utf-8"))
    except Exception:
        cfg = None
    return _runtime_fields_from(_read_result_json(trial_dir), _read_spans(trial_dir),
                                _exec_timeout_from_config_data(cfg))


def _collect_trial(trial_dir: Path) -> Optional[dict[str, Any]]:
    """Fuse one trial's parsed trajectory with its grade, or None to skip."""
    log_path, mode = _trial_log(trial_dir)
    if log_path is not None:
        parse = parse_pi_conversation
    else:
        # No `--mode json` log (e.g. killed at the wall-clock cap before it
        # flushed) — recover the conversation from pi's surviving session log.
        log_path = _trial_session_log(trial_dir)
        if log_path is None:
            log.debug("no pi log or session log under %s; skipping", trial_dir)
            return None
        parse = parse_pi_session
        mode = _mode_from_result(trial_dir)
    try:
        text = log_path.read_text(encoding="utf-8")
    except Exception as e:  # pragma: no cover - fs failure
        log.warning("could not read %s: %s", log_path, e)
        return None
    traj = parse(text)
    if not traj.turns:
        log.warning("no turns parsed from %s; skipping", log_path)
        return None
    iid = (_instance_id_from_config(trial_dir) or traj.instance_id
           or (trial_dir.name.rsplit("__", 1)[0] or None))
    reward = _read_reward(trial_dir)
    rec = traj.to_dict()
    rec["trial"] = trial_dir.name
    rec["instance_id"] = iid
    rec["mode"] = mode
    rec["reward"] = reward
    rec["passed"] = reward is not None and reward >= 1.0
    rec.update(_read_report(trial_dir))
    rec.update(_completion_fields(trial_dir, traj))
    rec.update(_runtime_fields(trial_dir))
    return rec


def _collect_cleaned_trial(trial: Any) -> Optional[dict[str, Any]]:
    """Build the same graded-trajectory record from a cleaned ``Trial``.

    Mirrors :func:`_collect_trial` but sources every field from the cleaned
    single-file layout via ``trial_io`` rather than from on-disk trial files.
    """
    fmt = trial.trajectory_format
    if fmt == "none":
        return None
    text = "\n".join(json.dumps(r) for r in trial.iter_trajectory())
    parse = parse_pi_session if fmt == "session" else parse_pi_conversation
    traj = parse(text)
    if not traj.turns:
        return None
    task = trial.config.get("task") or {}
    path = task.get("path") if isinstance(task, dict) else None
    iid = ((PurePosixPath(path).name if isinstance(path, str) and path else None)
           or traj.instance_id or (trial.trial_name.rsplit("__", 1)[0] or None))
    rec = traj.to_dict()
    rec["trial"] = trial.trial_name
    rec["instance_id"] = iid
    rec["mode"] = _mode_from_result_data(trial.result)
    rec["reward"] = trial.reward
    rec["passed"] = trial.reward is not None and trial.reward >= 1.0
    rec.update(_report_fields(trial.report))
    rec.update(_completion_fields_from_status(
        _exit_status_from_result(trial.result), traj))
    spans = list(getattr(trial, "timeline_spans", None) or [])
    rec.update(_runtime_fields_from(
        trial.result, spans, _exec_timeout_from_config_data(trial.config)))
    return rec


def collect_from_job_dir(job_dir: Any) -> list[dict[str, Any]]:
    """Walk a harbor job dir and return one graded-trajectory record per trial.

    A legacy trial subdir is any directory containing an ``agent/`` subdir;
    cleaned trials are ``<trial>.jsonl`` files (the legacy subdirs are gone). A
    job dir is one layout or the other, so the two passes never double-count.
    Trials with no parsable pi turns are skipped (logged). Defensive throughout:
    a corrupt or missing file degrades a single field/trial, never raises.
    """
    job_dir = Path(job_dir)
    records: list[dict[str, Any]] = []
    for trial_dir in sorted(p for p in job_dir.iterdir()
                            if p.is_dir() and (p / "agent").is_dir()):
        rec = _collect_trial(trial_dir)
        if rec is not None:
            records.append(rec)
    # cleaned single-file trials (<trial>.jsonl); legacy trees have none
    from benchmaker.swebench import trial_io
    for trial in trial_io.iter_trials(job_dir):
        if trial.layout == "cleaned":
            rec = _collect_cleaned_trial(trial)
            if rec is not None:
                records.append(rec)
    return records


def write_trajectories(records: list[dict[str, Any]], out_path: Any) -> int:
    """Write one record per line; returns the number written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    return len(records)


# --------------------------------------------------------------------------- #
# Incremental save + resume
# --------------------------------------------------------------------------- #


def _record_id(rec: dict[str, Any]) -> Optional[str]:
    """The dedup/resume key for a record: its instance id, trial name as backup."""
    return rec.get("instance_id") or rec.get("trial")


def load_collected_ids(out_path: Any) -> set[str]:
    """Instance ids already saved in ``out_path`` — the resume checkpoint.

    Tolerant of a partially written file: blank or unparsable lines (e.g. a line
    truncated by an OOM-kill mid-append) are skipped, not fatal.
    """
    out_path = Path(out_path)
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rid = _record_id(rec) if isinstance(rec, dict) else None
            if rid:
                done.add(rid)
    return done


def _append_record(fh: TextIO, rec: dict[str, Any]) -> None:
    """Append one JSONL record and flush so it survives an abrupt kill.

    ``flush`` hands the bytes to the OS, which persists them even if this process
    is later SIGKILLed (the systemd memory cap) — only a machine crash could lose
    them, which ``fsync`` would not meaningfully change here.
    """
    fh.write(json.dumps(rec, default=str) + "\n")
    fh.flush()


class _TrialStreamer:
    """Append each trial to ``out`` the moment it finishes, in a poller thread.

    A trial counts as *finished* once harbor has written its
    ``result.json``; only then is its parsed trajectory + grade appended (and
    only if its id is not already in ``seen``). ``seen`` is shared with the
    caller — seeded from the resume checkpoint and grown as we write — so a trial
    is never written twice, within a run or across resumes.
    """

    def __init__(self, job_dir: Any, fh: TextIO, seen: set[str],
                 poll_sec: float = _POLL_SEC) -> None:
        self.job_dir = Path(job_dir)
        self._fh = fh
        self._seen = seen
        self._poll_sec = poll_sec
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="trial-streamer",
                                         daemon=True)
        self.n_new = 0

    def start(self) -> "_TrialStreamer":
        self._thread.start()
        return self

    def stop_and_join(self) -> int:
        """Stop polling, drain a final sweep, and return how many were written."""
        self._stop.set()
        self._thread.join()
        self.sweep()  # final pass on the main thread once harbor has exited
        return self.n_new

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sweep()
            self._stop.wait(self._poll_sec)

    def sweep(self) -> int:
        """Append every finished, not-yet-seen trial; return how many this pass."""
        written = 0
        try:
            trial_dirs = sorted(p for p in self.job_dir.iterdir()
                                if p.is_dir() and (p / "agent").is_dir())
        except FileNotFoundError:
            return 0  # job dir not created yet
        for trial_dir in trial_dirs:
            if not (trial_dir / _TRIAL_DONE_MARKER).exists():
                continue  # still running / not yet graded
            # Cheap pre-filter on the trial-dir name before parsing the log.
            if trial_dir.name.rsplit("__", 1)[0] in self._seen:
                continue
            # One bad trial must never stop the poller (and the incremental
            # saves it provides) for the rest of the run.
            try:
                rec = _collect_trial(trial_dir)
                if rec is None:
                    continue
                rid = _record_id(rec)
                if rid is None or rid in self._seen:
                    continue
                _append_record(self._fh, rec)
            except Exception as e:  # pragma: no cover - defensive
                log.warning("streamer: could not save %s: %s", trial_dir.name, e)
                continue
            self._seen.add(rid)
            self.n_new += 1
            written += 1
            log.info("saved %s (reward=%s)", rid, rec.get("reward"))
        return written


def run_job_streaming(args: argparse.Namespace, passthrough: list[str],
                      out_path: Any, seen: set[str]) -> tuple[Path, int]:
    """Run harbor_eval while streaming each finished trial into ``out_path``.

    Returns the job dir and the number of trajectories newly appended this run.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    job_dir = _job_dir_for(args)
    env = os.environ.copy()
    env.setdefault("SWE_IMAGE_MIRROR", "swe-images")
    argv = [args.python] + build_harbor_argv(args, passthrough)
    log.info("running: %s", " ".join(argv))
    with out_path.open("a", encoding="utf-8") as fh:
        streamer = _TrialStreamer(job_dir, fh, seen).start()
        try:
            status = subprocess.run(argv, cwd=str(REPO_ROOT), env=env).returncode
        finally:
            n_new = streamer.stop_and_join()
    if status != 0:
        log.warning("harbor_eval exited %s; kept the %d trials that finished",
                    status, n_new)
    return job_dir, n_new


def _summarise_out(out_path: Any) -> tuple[int, int]:
    """(total, passed) over the records currently in ``out_path``."""
    total = passed = 0
    for line in Path(out_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        total += 1
        if rec.get("passed"):
            passed += 1
    return total, passed


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(str(REPO_ROOT / ".env"))
    args, passthrough = parse_args(sys.argv[1:] if argv is None else argv)

    # --collect-only: one-shot (re-)fuse of an existing job dir into --out.
    if args.collect_only:
        job_dir = Path(args.collect_only)
        if not job_dir.exists():
            print(f"ERROR: job dir not found: {job_dir}", file=sys.stderr)
            return 1
        records = collect_from_job_dir(job_dir)
        out = Path(args.out) if args.out else (job_dir / "pi-trajectories.jsonl")
        n = write_trajectories(records, out)
        n_pass = sum(1 for r in records if r.get("passed"))
        pct = (n_pass / n * 100.0) if n else 0.0
        print(f"wrote {n} trajectories ({n_pass} passed, {pct:.1f}%) -> {out}")
        return 0

    if not args.mode:
        print("ERROR: --mode {pi-host,pi-container} required (or use --collect-only).",
              file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else (_job_dir_for(args) / "pi-trajectories.jsonl")

    # Resume off the --out checkpoint: skip tasks already saved and only top up
    # to --n-tasks total. (Each finished trial is streamed into --out during the
    # run, so a crashed run still shrinks the work the next run has to do.)
    done = load_collected_ids(out) if args.resume else set()
    if done:
        log.info("resume: %d trajectory(ies) already in %s; skipping them",
                 len(done), out)
    # Merge the resume set with any hand-supplied --exclude-task overrides.
    args.exclude_task = sorted(set(args.exclude_task) | done)
    if args.n_tasks is not None:
        remaining = max(0, args.n_tasks - len(done))
        if remaining == 0:
            total, passed = _summarise_out(out)
            pct = (passed / total * 100.0) if total else 0.0
            print(f"nothing to do: {len(done)} >= --n-tasks {args.n_tasks} already "
                  f"in {out} ({passed}/{total} passed, {pct:.1f}%)")
            return 0
        args.n_tasks = remaining

    job_dir, n_new = run_job_streaming(args, passthrough, out, set(done))

    if not job_dir.exists():
        print(f"ERROR: job dir not found: {job_dir}", file=sys.stderr)
        return 1

    total, passed = _summarise_out(out)
    pct = (passed / total * 100.0) if total else 0.0
    print(f"appended {n_new} this run; {total} trajectories total "
          f"({passed} passed, {pct:.1f}%) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

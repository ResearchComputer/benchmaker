"""Collapse a SWE-bench task directory into a single <trial>.jsonl (meta line +
agent trajectory) and delete the directory, behind a verify-before-delete gate.

The importable core; scripts/clis/cleanjobs.py is a thin CLI, and the
swebench-replay recipe calls clean_tree() after each run."""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import tempfile

from benchmaker.swebench.foldlogs import fold_lines
from benchmaker.swebench import trajectory as _traj

RESULT = "result.json"
_SESSION_GLOB = os.path.join("agent", "pi-home", ".pi", "agent", "sessions", "*", "*.jsonl")
_CONTAINER_LOG = os.path.join("agent", "pi-container.log")
_HOST_LOG = os.path.join("agent", "pi-host.log")
_PI_LOG_DROP = {"message_update", "agent_end"}   # keep turn_end + message_end
_UUID8 = re.compile(r"([0-9a-fA-F]{8})")


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _reward(result):
    rr = (result.get("verifier_result") or {}).get("rewards") or {}
    return rr.get("reward")


def build_meta(task_dir, *, trajectory_format, is_secondary_session=False):
    """Build the superset metadata record for a task directory.

    Returns a dict with type="benchmaker_meta" containing all fields needed
    for downstream analysis. `report` is the inner per-instance dict (not
    keyed by task name).
    """
    result = _read_json(os.path.join(task_dir, RESULT)) or {}
    task_name = result.get("task_name")
    report_full = _read_json(os.path.join(task_dir, "verifier", "report.json")) or {}
    report = report_full.get(task_name) if task_name else None
    reward = _reward(result)
    if reward is None:
        try:
            with open(os.path.join(task_dir, "verifier", "reward.txt")) as f:
                reward = float(f.read().strip())
        except Exception:
            reward = None
    spans = []
    sp = os.path.join(task_dir, "agent", "timeline-spans.jsonl")
    if os.path.exists(sp):
        with open(sp) as f:
            spans = [json.loads(x) for x in f if x.strip()]
    exc_text = None
    ep = os.path.join(task_dir, "exception.txt")
    if os.path.exists(ep):
        with open(ep) as f:
            exc_text = f.read()
    agent = (result.get("config") or {}).get("agent") or {}
    return {
        "type": "benchmaker_meta",
        "schema_version": 1,
        "trial_name": result.get("trial_name") or os.path.basename(task_dir.rstrip("/")),
        "task_name": task_name,
        "source": result.get("source"),
        "model_name": agent.get("model_name"),
        "reward": reward,
        "resolved": (report or {}).get("resolved") if report else None,
        "is_secondary_session": is_secondary_session,
        "trajectory_format": trajectory_format,
        "result": result,
        "report": report,
        "timeline_spans": spans,
        "exception_text": exc_text,
    }


def _session_files(task_dir):
    files = glob.glob(os.path.join(task_dir, _SESSION_GLOB))
    return sorted(files, key=lambda p: os.path.basename(p))


def find_trajectory(task_dir):
    """Return (format, raw_byte_lines). format in {session, pi_log, none}.

    For pi-host mode: returns the first session jsonl (sorted by filename).
    For pi-container mode: returns the folded pi-container.log, retaining
    turn_end and message_end while dropping message_update and agent_end.
    For failure/empty dirs: returns ("none", []).
    """
    sessions = _session_files(task_dir)
    if sessions:
        with open(sessions[0], "rb") as f:
            return "session", f.readlines()
    def _fold_log(path):
        with open(path, "rb") as f:
            raw = f.readlines()
        kept, missing = fold_lines(raw, drop=_PI_LOG_DROP)
        # if folding would be unsafe (lost content), keep the raw log verbatim
        return raw if missing else kept

    clog = os.path.join(task_dir, _CONTAINER_LOG)
    if os.path.exists(clog):
        return "pi_log", _fold_log(clog)
    hlog = os.path.join(task_dir, _HOST_LOG)
    if os.path.exists(hlog):
        return "pi_log", _fold_log(hlog)
    return "none", []


def _write_atomic(parent, name, byte_lines):
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".clean_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as out:
            for b in byte_lines:
                if not b.endswith(b"\n"):
                    b = b + b"\n"
                out.write(b)
        target = os.path.join(parent, name)
        os.replace(tmp, target)
        return target
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _verify(target, *, trajectory_format, n_traj):
    with open(target, "rb") as f:
        lines = f.readlines()
    objs = [json.loads(b) for b in lines]      # raises if unparseable
    if not objs or objs[0].get("type") != "benchmaker_meta" or "reward" not in objs[0]:
        raise ValueError("bad meta line")
    if objs[0].get("trajectory_format") != trajectory_format:
        raise ValueError("trajectory_format mismatch")
    if len(objs) != n_traj + 1:
        raise ValueError("line count mismatch")
    if trajectory_format == "session":
        if objs[1].get("type") != "session":
            raise ValueError("session record missing")
        if not _traj.parse_pi_session("\n".join(json.dumps(o) for o in objs[1:])).turns:
            raise ValueError("session parses to zero turns")
    elif trajectory_format == "pi_log":
        text = "\n".join(json.dumps(o) for o in objs[1:])
        if not _traj.parse_pi_conversation(text).turns:
            raise ValueError("pi_log parses to zero turns")
    elif trajectory_format == "none":
        if len(objs) != 1:
            raise ValueError("meta-only expected")


def _dir_bytes(d):
    total = 0
    for dp, _dirs, files in os.walk(d):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total


def clean_task(task_dir, *, dry_run=False):
    if not os.path.exists(os.path.join(task_dir, RESULT)):
        return ("NOT_A_TASK", 0, 0, 0)
    parent = os.path.dirname(task_dir.rstrip("/"))
    trial = os.path.basename(task_dir.rstrip("/"))
    orig_bytes = _dir_bytes(task_dir)

    fmt, traj = find_trajectory(task_dir)
    sessions = _session_files(task_dir)
    targets = []        # (name, trajectory_format, n_traj, byte_lines)

    meta = build_meta(task_dir, trajectory_format=fmt)
    meta_line = (json.dumps(meta) + "\n").encode()
    targets.append((f"{trial}.jsonl", fmt, len(traj), [meta_line] + traj))

    for extra in sessions[1:]:
        m = _UUID8.search(os.path.basename(extra))
        suffix = m.group(1) if m else os.path.basename(extra)[:8]
        with open(extra, "rb") as f:
            elines = f.readlines()
        emeta = build_meta(task_dir, trajectory_format="session",
                           is_secondary_session=True)
        targets.append((f"{trial}__{suffix}.jsonl", "session", len(elines),
                        [(json.dumps(emeta) + "\n").encode()] + elines))

    written = []
    try:
        for name, tfmt, n_traj, blines in targets:
            t = _write_atomic(parent, name, blines)
            written.append(t)
            _verify(t, trajectory_format=tfmt, n_traj=n_traj)
    except Exception:
        for w in written:
            if os.path.exists(w):
                os.unlink(w)
        return ("SKIPPED_UNSAFE", orig_bytes, orig_bytes, 0)

    new_bytes = sum(os.path.getsize(w) for w in written)
    if dry_run:
        for w in written:
            os.unlink(w)
        return ("DRY", orig_bytes, new_bytes, len(written))
    shutil.rmtree(task_dir)
    return ("CLEANED", orig_bytes, new_bytes, len(written))


def iter_task_dirs(root):
    if os.path.isfile(root):
        return
    for dp, _dirs, files in os.walk(root):
        # A genuine trial dir has BOTH result.json and an agent/ subdir; a
        # job/replay dir has a (job-level) result.json but no agent/ — descend
        # into it to reach its trial subdirs instead of treating it as a trial.
        if RESULT in files and "agent" in _dirs:
            yield dp
            _dirs[:] = []          # don't descend into a task dir's internals


def clean_tree(root, *, dry_run=False):
    s = {"cleaned": 0, "skipped": 0, "errors": 0, "orig_bytes": 0,
         "new_bytes": 0, "skipped_paths": []}
    for td in list(iter_task_dirs(root)):
        try:
            status, ob, nb, _n = clean_task(td, dry_run=dry_run)
        except Exception:
            s["errors"] += 1
            continue
        if status in ("CLEANED", "DRY"):
            s["cleaned"] += 1
            s["orig_bytes"] += ob
            s["new_bytes"] += nb
        elif status == "SKIPPED_UNSAFE":
            s["skipped"] += 1
            s["skipped_paths"].append(td)
    s["saved_bytes"] = s["orig_bytes"] - s["new_bytes"]
    return s

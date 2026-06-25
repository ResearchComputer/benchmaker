"""Read a SWE-bench trial regardless of layout: legacy nested task dir or cleaned
<trial>.jsonl (meta line + trajectory). The migration target for all collectors.

See docs/superpowers/specs/2026-06-25-cleanjobs-trajectory-collapse-design.md."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

_LOG_NAMES = ("pi-host.log", "pi-container.log")
_SESSION_GLOB = os.path.join("agent", "pi-home", ".pi", "agent", "sessions", "*", "*.jsonl")


def _read_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


@dataclass
class Trial:
    path: str
    layout: str                      # "legacy" | "cleaned"
    _meta: dict | None = None
    _result: dict | None = None
    _traj_lines: list | None = None

    @property
    def result(self):
        if self._result is None:
            if self.layout == "cleaned":
                self._result = self._meta.get("result") or {}
            else:
                self._result = _read_json(os.path.join(self.path, "result.json")) or {}
        return self._result

    @property
    def trial_name(self):
        return (self._meta or {}).get("trial_name") or self.result.get("trial_name") \
            or os.path.basename(self.path).removesuffix(".jsonl")

    @property
    def task_name(self):
        return (self._meta or {}).get("task_name") or self.result.get("task_name")

    @property
    def config(self):
        return self.result.get("config") or {}

    @property
    def report(self):
        if self.layout == "cleaned":
            return self._meta.get("report")
        rep = _read_json(os.path.join(self.path, "verifier", "report.json")) or {}
        return rep.get(self.task_name)

    @property
    def reward(self):
        if self.layout == "cleaned":
            return self._meta.get("reward")
        rr = (self.result.get("verifier_result") or {}).get("rewards") or {}
        r = rr.get("reward")
        if r is None:
            try:
                with open(os.path.join(self.path, "verifier", "reward.txt")) as f:
                    r = float(f.read().strip())
            except Exception:
                r = None
        return r

    @property
    def resolved(self):
        return (self.report or {}).get("resolved") if self.report else None

    @property
    def tests_status(self):
        return (self.report or {}).get("tests_status") if self.report else None

    @property
    def exception_info(self):
        return self.result.get("exception_info")

    @property
    def exception_text(self):
        if self.layout == "cleaned":
            return self._meta.get("exception_text")
        p = os.path.join(self.path, "exception.txt")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return f.read()

    @property
    def timeline_spans(self):
        if self.layout == "cleaned":
            return self._meta.get("timeline_spans") or []
        p = os.path.join(self.path, "agent", "timeline-spans.jsonl")
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return [json.loads(x) for x in f if x.strip()]

    @property
    def trajectory_format(self):
        if self.layout == "cleaned":
            return self._meta.get("trajectory_format")
        if glob.glob(os.path.join(self.path, _SESSION_GLOB)):
            return "session"
        if os.path.exists(os.path.join(self.path, "agent", "pi-container.log")):
            return "pi_log"
        if os.path.exists(os.path.join(self.path, "agent", "pi-host.log")):
            return "pi_log"
        return "none"

    def legacy_agent_log(self):
        if self.layout != "legacy":
            return None
        for n in _LOG_NAMES:
            p = os.path.join(self.path, "agent", n)
            if os.path.exists(p):
                return p
        return None

    def iter_trajectory(self):
        """Yield agent trajectory records (the meta line excluded)."""
        if self.layout == "cleaned":
            yield from (self._traj_lines or [])
            return
        sess = sorted(glob.glob(os.path.join(self.path, _SESSION_GLOB)))
        if sess:
            with open(sess[0]) as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
            return
        log = self.legacy_agent_log()
        if log:
            with open(log) as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)


def load_trial(path):
    if os.path.isdir(path):
        return Trial(path=path, layout="legacy")
    with open(path) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    if not lines or lines[0].get("type") != "benchmaker_meta":
        raise ValueError(f"not a cleaned trial file: {path}")
    return Trial(path=path, layout="cleaned", _meta=lines[0], _traj_lines=lines[1:])


def iter_trials(root):
    for dp, _dirs, files in os.walk(root):
        if "result.json" in files and "agent" in _dirs:   # genuine trial, not a job dir
            yield Trial(path=dp, layout="legacy")
            _dirs[:] = []
            continue
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            p = os.path.join(dp, fn)
            try:
                with open(p) as f:
                    first = f.readline()
                obj = json.loads(first) if first.strip() else {}
            except Exception:
                continue
            if obj.get("type") == "benchmaker_meta" and not obj.get("is_secondary_session"):
                yield load_trial(p)


def recover_command_timings_from_records(records):
    """Dispatch by record schema: pi-log message_end stream vs session message
    envelope. Returns list[CommandTiming]."""
    records = list(records)
    types = {r.get("type") for r in records if isinstance(r, dict)}
    if "message_end" in types:
        return _ct_from_records(records)
    if "message" in types or "session" in types:
        return _ct_from_session(records)
    return []


def recover_command_timings_from_trial(trial):
    return recover_command_timings_from_records(trial.iter_trajectory())


def _ct_from_records(records):
    from benchmaker.swebench.timeout_load import recover_command_timings_from_records as f
    return f(records)


def _ct_from_session(records):
    """Session envelope: {'type':'message','message':{role, timestamp(ms), content}}.
    Pair each assistant message that issues a toolCall with the next toolResult."""
    from benchmaker.swebench.timeout_load import CommandTiming
    timings, pending = [], None
    for obj in records:
        if not isinstance(obj, dict) or obj.get("type") != "message":
            continue
        msg = obj.get("message", {})
        role, ts = msg.get("role"), msg.get("timestamp")
        if role == "assistant":
            tools = [c.get("name") for c in msg.get("content", [])
                     if c.get("type") == "toolCall"]
            pending = (ts, tools[0]) if (ts is not None and tools) else None
        elif role == "toolResult" and pending is not None:
            a_ts, tool = pending
            if a_ts is not None and ts is not None:
                timings.append(CommandTiming(tool, (ts - a_ts) / 1000.0))
            pending = None
    return timings

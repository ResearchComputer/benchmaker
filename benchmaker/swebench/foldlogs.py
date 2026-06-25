"""Losslessly fold `pi-host.log` (the per-task agent event stream) by removing
REDUNDANT records — not by compression. The complete, authoritative copy of every
message (with its real timestamp) lives in `message_end`; tool results live in
`tool_execution_end`. The following are duplicates of that content:

  * `message_update`  — streaming deltas that re-emit the whole message every token
                        (`partial` + top-level `message`); carry NO timestamp.
  * `turn_end`        — byte-identical re-emission of a `message_end` message.
  * `agent_end`       — a full re-dump of the conversation already in `message_end`s.

We drop those three classes, with per-file SAFETY GUARANTEES so nothing unique is
lost (e.g. a run hard-killed mid-stream whose final partial message never reached a
`message_end`):

  - `turn_end`  kept unless its message is byte-identical to some `message_end`.
  - `agent_end` kept unless ALL its messages are covered by `message_end`s.
  - `message_update` dropped, EXCEPT the single most-complete partial of any stream
    that never closed with a `message_end` (the only surviving copy of that content).

Before replacing the original, we verify the folded file's message inventory is a
SUPERSET of the original's; otherwise the file is left untouched and reported.

This is the importable core. `scripts/clis/fold_pihost_log.py` is a thin CLI over
it, and the swebench-replay recipe calls `fold_tree()` after each run to reclaim
space automatically.
"""
from __future__ import annotations

import json
import os
import tempfile

REDUNDANT = {"message_update", "turn_end", "agent_end"}
LOG_NAME = "pi-host.log"


def _canon(m):
    try:
        return json.dumps(m, sort_keys=True, ensure_ascii=False)
    except Exception:
        return repr(m)


def _final_partial(ev):
    """The most-complete message snapshot carried by a message_update event."""
    if not isinstance(ev, dict):
        return None
    amev = ev.get("assistantMessageEvent") or {}
    # the streamed snapshot lives in top-level .message or assistantMessageEvent.partial
    return ev.get("message") or amev.get("partial")


def fold_lines(raw_lines, *, drop=REDUNDANT):
    """Pure core of the fold: given raw JSONL byte-lines, return (kept_lines,
    missing). `drop` is the set of record `type`s to remove (default REDUNDANT).
    `missing` is the set of message-content canonical strings that would be lost
    if `kept_lines` replaced the input — empty means the fold is content-lossless.
    Callers treat non-empty `missing` as "unsafe, keep original".
    """
    parsed = []          # (idx, type, obj_or_None, raw_bytes)
    end_msgs = set()
    orig_inventory = set()
    for i, b in enumerate(raw_lines):
        try:
            o = json.loads(b)
        except Exception:
            o = None
        t = o.get("type") if isinstance(o, dict) else None
        parsed.append((i, t, o, b))
        if t == "message_end":
            cm = _canon(o.get("message"))
            end_msgs.add(cm)
            orig_inventory.add(cm)

    cur_last_mu = None

    def flush_stream(streamed_partial):
        if streamed_partial is not None:
            orig_inventory.add(_canon(streamed_partial))

    for i, t, o, b in parsed:
        if t == "message_start":
            flush_stream(cur_last_mu)
            cur_last_mu = None
        elif t == "message_update":
            fp = _final_partial(o)
            if fp is not None:
                cur_last_mu = fp
        elif t == "turn_end":
            orig_inventory.add(_canon(o.get("message")))
        elif t == "agent_end":
            for m in (o.get("messages") or []):
                orig_inventory.add(_canon(m))
    flush_stream(cur_last_mu)

    stream_last_mu_idx = {}
    stream_has_end = {}
    boundary = -1
    for i, t, o, b in parsed:
        if t == "message_start":
            boundary = i
            stream_has_end.setdefault(boundary, False)
        elif t == "message_update":
            stream_last_mu_idx[boundary] = i
        elif t == "message_end":
            stream_has_end[boundary] = True
    keep_mu_idx = {idx for b, idx in stream_last_mu_idx.items()
                   if not stream_has_end.get(b, False)}

    keep = []
    folded_inventory = set()
    for i, t, o, b in parsed:
        if t not in drop:
            keep.append(b)
            if t == "message_end":
                folded_inventory.add(_canon(o.get("message")))
            elif t == "turn_end":
                folded_inventory.add(_canon(o.get("message")))
            elif t == "agent_end":
                folded_inventory.update(_canon(m) for m in (o.get("messages") or []))
            continue
        if t == "turn_end":
            if _canon(o.get("message")) in end_msgs:
                continue
            keep.append(b)
            folded_inventory.add(_canon(o.get("message")))
        elif t == "agent_end":
            msgs = [_canon(m) for m in (o.get("messages") or [])]
            if all(m in end_msgs for m in msgs):
                continue
            keep.append(b)
            folded_inventory.update(msgs)
        elif t == "message_update":
            if i in keep_mu_idx:
                keep.append(b)
                fp = _final_partial(o)
                if fp is not None:
                    folded_inventory.add(_canon(fp))

    missing = orig_inventory - folded_inventory
    return keep, missing


def fold_file(path, dry_run=False):
    """Fold one pi-host.log in place (atomic). Returns (status, orig_bytes,
    new_bytes, n_missing) where status is one of:
      FOLDED          — rewrote the file smaller (lossless).
      NOOP            — nothing to drop (already folded); file untouched.
      DRY             — --dry-run; reports sizes, changes nothing.
      SKIPPED_UNSAFE  — folding would lose unique content; file untouched.
    """
    with open(path, "rb") as fh:
        raw = fh.readlines()
    orig_bytes = sum(len(b) for b in raw)
    keep, missing = fold_lines(raw)          # default drop=REDUNDANT
    new_bytes = sum(len(b) for b in keep)
    if missing:
        return ("SKIPPED_UNSAFE", orig_bytes, orig_bytes, len(missing))
    if new_bytes == orig_bytes:
        # nothing was dropped (already folded) -> avoid a pointless rewrite/mtime churn.
        return ("DRY" if dry_run else "NOOP", orig_bytes, orig_bytes, 0)
    if dry_run:
        return ("DRY", orig_bytes, new_bytes, 0)

    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".fold_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as out:
            out.writelines(keep)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return ("FOLDED", orig_bytes, new_bytes, 0)


def iter_logs(root):
    """Yield every pi-host.log path under `root` (or `root` itself if it is one)."""
    if os.path.isfile(root):
        if os.path.basename(root) == LOG_NAME:
            yield root
        return
    for dirpath, _dirs, files in os.walk(root):
        if LOG_NAME in files:
            yield os.path.join(dirpath, LOG_NAME)


def fold_tree(root, *, dry_run=False):
    """Fold every pi-host.log under `root`. Best-effort: a file that errors is
    counted, not raised. Returns a summary dict."""
    s = {"folded": 0, "noop": 0, "skipped": 0, "errors": 0,
         "orig_bytes": 0, "new_bytes": 0, "skipped_paths": []}
    for p in iter_logs(root):
        try:
            status, ob, nb, miss = fold_file(p, dry_run=dry_run)
        except Exception:
            s["errors"] += 1
            continue
        s["orig_bytes"] += ob
        s["new_bytes"] += nb
        if status in ("FOLDED", "DRY"):
            s["folded"] += 1
        elif status == "NOOP":
            s["noop"] += 1
        elif status == "SKIPPED_UNSAFE":
            s["skipped"] += 1
            s["skipped_paths"].append(p)
    s["saved_bytes"] = s["orig_bytes"] - s["new_bytes"]
    return s

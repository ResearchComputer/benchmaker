#!/usr/bin/env python3
"""Densify pi-container.log: drop per-token deltas, keep the real conversation.

A pi-container.log is a streaming JSONL trace where the bulk of the bytes are
incremental events — ``thinking_delta``, ``text_delta``, ``toolcall_delta`` and
especially ``message_update`` (which re-emits the *entire* accumulated message on
every token). The fully-assembled content already lives in the ``message_end``
events, one per logical step:

    role=user        -> the task prompt (and any later user turns)
    role=assistant   -> {thinking, text, toolCall[]} for that turn
    role=toolResult  -> {toolName, toolCallId, isError, content} command output

So a faithful, much smaller trace is just: the ``session`` header, every
``message_end`` in order, plus the ``auto_retry_*`` / ``agent_end`` markers.
Everything else is redundant and dropped. Typical reduction is 50-500x.

Usage
-----
    # one file -> writes pi-container.dense.jsonl next to it
    python scripts/densify_pi_log.py jobs/.../agent/pi-container.log

    # a whole tree of trajectories (recurses for pi-container.log)
    python scripts/densify_pi_log.py jobs/2026-06-07__22-12-29_56a0e8

    # human-readable transcript instead of jsonl, capping huge tool outputs
    python scripts/densify_pi_log.py FILE --format md --max-output 2000

    # stream one file to stdout (e.g. to pipe into jq / less)
    python scripts/densify_pi_log.py FILE --stdout

Output schema (``--format jsonl``), one object per line:
    {"t":"session","id":..,"cwd":..,"ts":..}
    {"t":"user","text":..,"ts":..}
    {"t":"assistant","thinking":..,"text":..,"tools":[{"id","name","args"}],"ts":..}
    {"t":"tool_result","id":..,"name":..,"isError":bool,"output":..,"ts":..}
    {"t":"retry","attempt":..,"max":..,"error":..}
    {"t":"retry_end","success":bool,"attempt":..}
    {"t":"agent_end"}            # marker only; its huge messages[] dump is skipped
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Iterator, Optional

# Substrings that gate JSON parsing — we only parse complete, non-delta events.
# Cheap `in` checks on the raw line skip the ~99% of lines that are deltas.
_KEEP_MARKERS = (
    '"type":"message_end"',
    '"type":"session"',
    '"type":"auto_retry_start"',
    '"type":"auto_retry_end"',
    '"type":"agent_end"',
)


def _text_of(content: Any) -> str:
    """Concatenate the ``text`` fields of a content array."""
    if not isinstance(content, list):
        return ""
    return "".join(
        c["text"] for c in content if isinstance(c, dict) and isinstance(c.get("text"), str)
    )


def _coerce_args(arguments: Any) -> Any:
    """toolCall.arguments is usually a dict; some builds stringify it. Normalize."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except Exception:
            return arguments
    return arguments


def densify(path: str, max_output: int = 0) -> Iterator[dict]:
    """Yield dense records for one pi-container.log, in chronological order.

    ``max_output`` > 0 truncates each tool-result output to that many chars
    (a ``…(+N chars)`` marker is appended). Malformed/truncated lines — e.g. a
    trajectory killed mid-write — are skipped, so partial logs still convert.
    """
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if not any(m in line for m in _KEEP_MARKERS):
                continue
            try:
                d = json.loads(line, strict=False)
            except Exception:
                continue
            t = d.get("type")

            if t == "session":
                yield {"t": "session", "id": d.get("id"),
                       "cwd": d.get("cwd"), "ts": d.get("timestamp")}

            elif t == "auto_retry_start":
                yield {"t": "retry", "attempt": d.get("attempt"),
                       "max": d.get("maxAttempts"), "error": d.get("errorMessage")}

            elif t == "auto_retry_end":
                yield {"t": "retry_end", "success": d.get("success"),
                       "attempt": d.get("attempt")}

            elif t == "agent_end":
                # The payload duplicates the whole conversation; keep only a marker.
                yield {"t": "agent_end"}

            elif t == "message_end":
                m = d.get("message") or {}
                role = m.get("role")
                ts = m.get("timestamp")

                if role == "assistant":
                    thinking, text, tools = [], [], []
                    for c in m.get("content") or []:
                        if not isinstance(c, dict):
                            continue
                        ct = c.get("type")
                        if ct == "thinking" and c.get("thinking"):
                            thinking.append(c["thinking"])
                        elif ct == "text" and c.get("text"):
                            text.append(c["text"])
                        elif ct == "toolCall":
                            tools.append({"id": c.get("id"), "name": c.get("name"),
                                          "args": _coerce_args(c.get("arguments"))})
                    rec: dict = {"t": "assistant", "ts": ts}
                    if thinking:
                        rec["thinking"] = "".join(thinking)
                    if text:
                        rec["text"] = "".join(text)
                    if tools:
                        rec["tools"] = tools
                    yield rec

                elif role == "toolResult":
                    out = _text_of(m.get("content"))
                    if max_output and len(out) > max_output:
                        out = out[:max_output] + f"\n…(+{len(out) - max_output} chars truncated)"
                    yield {"t": "tool_result", "id": m.get("toolCallId"),
                           "name": m.get("toolName"), "isError": bool(m.get("isError")),
                           "output": out, "ts": ts}

                else:  # user (task prompt) or any other role -> keep its text
                    yield {"t": role or "message", "text": _text_of(m.get("content")), "ts": ts}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _to_md(records: Iterator[dict]) -> Iterator[str]:
    for r in records:
        t = r["t"]
        if t == "session":
            yield f"# pi trajectory `{r.get('id')}`\n\n- cwd: `{r.get('cwd')}`\n"
        elif t == "user":
            yield f"\n## 👤 user\n\n{r.get('text','')}\n"
        elif t == "assistant":
            yield "\n## 🤖 assistant\n"
            if r.get("thinking"):
                yield f"\n<details><summary>thinking</summary>\n\n{r['thinking']}\n\n</details>\n"
            if r.get("text"):
                yield f"\n{r['text']}\n"
            for tc in r.get("tools", []):
                args = tc.get("args")
                cmd = args.get("command") if isinstance(args, dict) and "command" in args else \
                    (args.get("path") if isinstance(args, dict) and "path" in args else json.dumps(args))
                yield f"\n**→ {tc.get('name')}**\n```bash\n{cmd}\n```\n"
        elif t == "tool_result":
            flag = " ❌" if r.get("isError") else ""
            yield f"\n<details><summary>↳ {r.get('name')} result{flag}</summary>\n\n```\n{r.get('output','')}\n```\n\n</details>\n"
        elif t == "retry":
            yield f"\n> 🔄 retry #{r.get('attempt')}/{r.get('max')}: {r.get('error')}\n"
        elif t == "retry_end":
            yield f"> 🔄 retry {'recovered' if r.get('success') else 'failed'}\n"
        elif t == "agent_end":
            yield "\n---\n*(agent_end)*\n"
        else:
            yield f"\n## {t}\n\n{r.get('text','')}\n"


def _find_logs(target: str) -> list[str]:
    if os.path.isfile(target):
        return [target]
    out = []
    for root, _dirs, files in os.walk(target):
        for fn in files:
            if fn == "pi-container.log":
                out.append(os.path.join(root, fn))
    return sorted(out)


def _convert_one(src: str, fmt: str, max_output: int,
                 to_stdout: bool, out_path: Optional[str]) -> tuple[int, int]:
    """Returns (src_bytes, dst_bytes)."""
    records = densify(src, max_output=max_output)
    if to_stdout:
        sink = sys.stdout
        if fmt == "md":
            for chunk in _to_md(records):
                sink.write(chunk)
        else:
            for r in records:
                sink.write(json.dumps(r, ensure_ascii=False) + "\n")
        return os.path.getsize(src), 0

    if out_path is None:
        base = src[:-4] if src.endswith(".log") else src
        out_path = base + (".dense.md" if fmt == "md" else ".dense.jsonl")
    written = 0
    with open(out_path, "w") as fh:
        if fmt == "md":
            for chunk in _to_md(records):
                written += fh.write(chunk)
        else:
            for r in records:
                written += fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return os.path.getsize(src), written


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+",
                    help="pi-container.log file(s) and/or directories to recurse.")
    ap.add_argument("--format", choices=["jsonl", "md"], default="jsonl",
                    help="dense jsonl (default) or markdown transcript.")
    ap.add_argument("--max-output", type=int, default=0, metavar="N",
                    help="truncate each tool-result output to N chars (0 = full).")
    ap.add_argument("--stdout", action="store_true",
                    help="write to stdout instead of a sidecar file (single target).")
    ap.add_argument("-o", "--out", default=None,
                    help="explicit output path (single input file only).")
    ap.add_argument("--quiet", action="store_true", help="suppress per-file stats.")
    args = ap.parse_args(argv)

    files: list[str] = []
    for tgt in args.targets:
        files.extend(_find_logs(tgt))
    if not files:
        print("no pi-container.log found", file=sys.stderr)
        return 1
    if args.stdout and len(files) != 1:
        print("--stdout requires exactly one input file", file=sys.stderr)
        return 2
    if args.out and len(files) != 1:
        print("--out requires exactly one input file", file=sys.stderr)
        return 2

    tot_src = tot_dst = 0
    for src in files:
        try:
            s, d = _convert_one(src, args.format, args.max_output, args.stdout, args.out)
        except BrokenPipeError:  # downstream closed (head/less) — let top level handle
            raise
        except Exception as e:  # keep going across a batch
            print(f"ERROR {src}: {e}", file=sys.stderr)
            continue
        tot_src += s
        tot_dst += d
        if not args.quiet and not args.stdout:
            ratio = f"{s / d:.0f}x" if d else "—"
            print(f"{src}  {s/1e6:.1f}MB -> {d/1e6:.3f}MB  ({ratio} smaller)")
    if not args.quiet and not args.stdout and len(files) > 1:
        ratio = f"{tot_src / tot_dst:.0f}x" if tot_dst else "—"
        print(f"--- {len(files)} files: {tot_src/1e9:.2f}GB -> {tot_dst/1e6:.1f}MB ({ratio}) ---")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # downstream (head/less) closed the pipe. Redirect stdout to devnull so
        # the interpreter's shutdown flush can't fail and print a traceback.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os._exit(0)

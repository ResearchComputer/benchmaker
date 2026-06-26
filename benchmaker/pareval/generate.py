"""Instruct-mode generation: prompt templating, code extraction, TU assembly."""
from __future__ import annotations

import re

_FENCE = re.compile(r"```[a-zA-Z0-9+]*\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str:
    m = _FENCE.search(reply or "")
    if m:
        return m.group(1).strip()
    return (reply or "").strip()


def split_preamble(prompt_stub: str) -> tuple[str, str]:
    lines = prompt_stub.splitlines()
    sig_idx = max(i for i, ln in enumerate(lines) if ln.strip())
    preamble = "\n".join(lines[:sig_idx])
    return preamble, lines[sig_idx]


def patch_no_inline(signature_line: str) -> str:
    parts = signature_line.split(" ")
    if len(parts) < 2:
        return signature_line
    return " ".join([parts[0], "NO_INLINE", *parts[1:]])


def assemble_generated_code(prompt_stub: str, completion: str) -> str:
    preamble, sig = split_preamble(prompt_stub)
    sig_key = sig.split("(")[0].strip()
    if sig_key and sig_key in completion:
        c_lines = completion.splitlines()
        start = next((i for i, ln in enumerate(c_lines)
                      if sig_key in ln and "(" in ln), 0)
        body_after_sig = "\n".join(c_lines[start + 1:])
        func = patch_no_inline(c_lines[start]) + "\n" + body_after_sig
    else:
        func = patch_no_inline(sig) + "\n" + completion
    return (preamble + "\n" if preamble else "") + func

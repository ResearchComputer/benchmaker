"""Instruct-mode generation: prompt templating, code extraction, TU assembly."""
from __future__ import annotations

import re
from typing import Awaitable, Callable, Optional

from benchmaker.pareval.dataset import ParEvalPrompt

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


INSTRUCT_TEMPLATE = """\
Complete the following {model} C++ function. Output ONLY the complete function \
(signature and body) in a single ```cpp code block. Do not include any prose, \
explanation, or additional declarations.

```cpp
{prompt}
```
"""


async def generate_one(
    send_fn: Callable[[list[dict]], Awaitable[tuple[str, Optional[dict]]]],
    prompt: ParEvalPrompt,
    *,
    sample_idx: int,
) -> dict:
    """Call the model once and assemble a completion record."""
    rec = {
        "name": prompt.name,
        "parallelism_model": prompt.parallelism_model,
        "problem_type": prompt.problem_type,
        "sample_idx": sample_idx,
        "raw_reply": None,
        "generated_code": None,
        "error": None,
        "usage": None,
    }
    content = INSTRUCT_TEMPLATE.format(
        prompt=prompt.prompt, model=prompt.parallelism_model
    )
    messages = [{"role": "user", "content": content}]
    try:
        text, usage = await send_fn(messages)
    except Exception as e:  # noqa: BLE001 - record, never raise
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    rec["raw_reply"] = text
    rec["usage"] = usage
    code = extract_code(text)
    if not code:
        rec["error"] = "no_code"
        return rec
    rec["generated_code"] = assemble_generated_code(prompt.prompt, code)
    return rec

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
    # Mirrors upstream ParEval patch_prompt's parts.insert(1, "NO_INLINE"):
    # NO_INLINE goes after the return type; multi-qualifier GPU signatures
    # (e.g. CUDA `__global__ void`) are a deferred concern.
    parts = signature_line.split(" ")
    if len(parts) < 2:
        return signature_line
    return " ".join([parts[0], "NO_INLINE", *parts[1:]])


def assemble_generated_code(prompt_stub: str, completion: str) -> str:
    preamble, sig = split_preamble(prompt_stub)
    sig_key = sig.split("(")[0].strip()
    c_lines = completion.splitlines()
    start = None
    if sig_key:
        # Match the line that *starts with* the signature key (a leading
        # comment that merely mentions the function name must not match).
        start = next((i for i, ln in enumerate(c_lines)
                      if ln.lstrip().startswith(sig_key) and "(" in ln), None)
    if start is not None:
        # Keep any pre-signature lines the model added (e.g. comments) but
        # drop lines that merely re-emit the stub preamble.
        preamble_set = set(preamble.splitlines())
        kept_pre = [ln for ln in c_lines[:start] if ln not in preamble_set]
        func_lines = kept_pre + [patch_no_inline(c_lines[start])] + c_lines[start + 1:]
        func = "\n".join(func_lines)
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


def _build_chat_request(
    api_base: str,
    model: str,
    api_key: Optional[str],
    temperature: float,
    messages: list[dict],
) -> tuple[str, dict, dict]:
    """Return (url, headers, json_body) for an OpenAI-compatible chat request."""
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    return url, headers, body


def make_send_fn(
    *,
    api_base: str,
    model: str,
    api_key: Optional[str],
    temperature: float,
) -> Callable[[list[dict]], Awaitable[tuple[str, Optional[dict]]]]:
    """Return an async (messages) -> (text, usage) that POSTs to chat/completions."""
    import httpx2

    from benchmaker.swebench.agent import parse_openai_usage

    session: Optional[httpx2.AsyncClient] = None

    def _get_session() -> httpx2.AsyncClient:
        nonlocal session
        if session is None or session.is_closed:
            session = httpx2.AsyncClient()
        return session

    async def send_fn(messages: list[dict]) -> tuple[str, Optional[dict]]:
        url, headers, body = _build_chat_request(
            api_base, model, api_key, temperature, messages
        )
        sess = _get_session()
        resp = await sess.post(url, headers=headers, json=body)
        text = resp.text
        if resp.status_code >= 400:
            excerpt = text if len(text) <= 500 else text[:500] + "...[truncated]"
            raise RuntimeError(f"model endpoint HTTP {resp.status_code}: {excerpt}")
        import json as _json

        data = _json.loads(text)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"model endpoint returned no choices: {data!r}")
        content = (choices[0].get("message") or {}).get("content") or ""
        return content, parse_openai_usage(data)

    async def _aclose() -> None:
        nonlocal session
        if session is not None and not session.is_closed:
            await session.aclose()
        session = None

    send_fn.aclose = _aclose
    return send_fn

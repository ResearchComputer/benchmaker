"""Instruct-mode generation: prompt templating, code extraction, TU assembly."""
from __future__ import annotations

import re

_FENCE = re.compile(r"```[a-zA-Z0-9+]*\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str:
    m = _FENCE.search(reply or "")
    if m:
        return m.group(1).strip()
    return (reply or "").strip()

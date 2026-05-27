"""Lightweight .env loading and ${VAR} interpolation for YAML configs."""

from __future__ import annotations

import os
import re
from typing import Any, Mapping, Optional


def load_dotenv(path: str = ".env", override: bool = False) -> dict[str, str]:
    """Parse a .env file and inject KEY=VALUE pairs into `os.environ`.

    Minimal parser — handles:
        KEY=value
        KEY="quoted value"
        KEY='single-quoted'
        # comment lines
        export KEY=value         (the `export` prefix is stripped)
        KEY=value with spaces    (unquoted; trailing whitespace stripped)

    Returns the dict of values loaded (also injected into `os.environ`).
    Silently returns `{}` if the file doesn't exist.

    By default, existing env vars are NOT overwritten (set `override=True` to
    force).
    """
    if not os.path.exists(path):
        return {}

    out: dict[str, str] = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes if matched.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            # Strip inline comments only for unquoted lines (rough heuristic).
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            out[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return out


_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z_0-9]*)(?::-([^}]*))?\}")


def interpolate(value: Any, env: Optional[Mapping[str, str]] = None) -> Any:
    """Recursively walk `value` and substitute ${VAR} / ${VAR:-default}.

    Lookups go to `env` if given, otherwise `os.environ`. Missing vars without
    a default raise `KeyError`.
    """
    src: Mapping[str, str] = env if env is not None else os.environ
    return _walk(value, src)


def _walk(value: Any, env: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _substitute(value, env)
    if isinstance(value, dict):
        return {k: _walk(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk(v, env) for v in value]
    return value


def _substitute(s: str, env: Mapping[str, str]) -> str:
    def repl(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        if name in env:
            return env[name]
        if default is not None:
            return default
        raise KeyError(f"environment variable {name!r} is not set (used in config)")

    return _VAR_RE.sub(repl, s)

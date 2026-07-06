"""TraceLab coding-agent trace workload.

`TraceLab <https://github.com/uw-syfi/TraceLab>`_ is a public, sanitized dataset
of real coding-agent (Claude Code / Codex) LLM rounds — 357K invocations from
43 developers. Each row is one LLM round with full **token accounting** and
**session structure** but **no prompt text** (it is sanitized for privacy):

.. code-block:: json

    {
      "provider": "claude", "model": "claude-opus-4-8",
      "session_id": "claude:...", "round_index": 0, "round_id": "msg_...",
      "input_tokens_total": 32272, "prefix_tokens": 27217,
      "newly_append_tokens": 5055, "output_tokens": 931,
      "claude_cache_read_input_tokens": 27217, "tools": [...], ...
    }

That token accounting — `input_tokens_total = prefix_tokens + newly_append_tokens`
plus `output_tokens` — is exactly what an LLM-serving benchmark needs to
reproduce a realistic coding-agent load shape (prefill bytes, decode length,
prefix-cache locality). Because the text is gone, this workload **synthesizes
token-faithful prompts**: each request's prompt is sized to the recorded
`input_tokens_total` and `max_tokens` is set to the recorded `output_tokens`.

Two emission modes:

* **flat** (default) — every request is independent; the prompt for round *i* is
  sized to its own `input_tokens_total`. Reproduces the marginal token
  distribution (prefill/decode shape) the way a throughput/latency sweep wants.
* **prefix-cache** (``prefix_cache=True``) — rounds are grouped by ``session_id``
  and replayed in session order. Within a session each round's prompt is a
  *byte-exact prefix* of the next round's, so the server's prefix cache is
  exercised the way it is on real coding agents (growing conversation history).
  Round *i* reuses the session's shared prefix (sized to ``prefix_tokens[i]``)
  and appends ``newly_append_tokens[i]`` of fresh text.

Token sizing has two backends:

* **char-based** (default, dependency-free) — ``target_chars =
  round(target_tokens * chars_per_token)``. Robust; guarantees byte-prefix
  nesting within a session. The server's reported ``prompt_tokens`` is always
  the authoritative realized count.
* **tokenizer-based** (``tokenizer=<hf-id>``, needs ``pip install -e .[tokenizer]``)
  — exact. Finds a single-token filler unit and repeats it, so the prompt has
  exactly the target token count and prefix nesting holds trivially.

Emitted items are shaped for
:class:`~benchmaker.workloads.llm.OpenAIChatWorkloadType` with
``passthrough_meta=True``: ``messages`` + ``max_tokens`` go to the body, while
``meta`` carries the recorded token accounting + provenance into each sample's
``meta`` (and ``prompt_tokens_hint`` / ``prefix_tokens_hint`` into ``extra``).
Use :mod:`tools.tracelab.prepare` to fetch the release asset first.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import random
import zipfile
from typing import Any, Iterable, Optional

from benchmaker.workloads.datasets import Workload


# Field name conventions in the published TraceLab JSONL.
F_INPUT_TOTAL = "input_tokens_total"
F_PREFIX = "prefix_tokens"
F_NEWLY_APPEND = "newly_append_tokens"
F_OUTPUT = "output_tokens"
F_PROVIDER = "provider"
F_MODEL = "model"
F_SESSION = "session_id"
F_ROUND_INDEX = "round_index"
F_ROUND_ID = "round_id"
F_TRACE_KEY = "trace_key"
F_TOOLS = "tools"

DEFAULT_CHARS_PER_TOKEN = 4.0

# A small pool of coding-ish words used to build filler text in char mode.
# Joined with spaces it looks vaguely like prose, tokenizes sanely, and — being
# a pure function of position within a session — keeps prefix nesting exact.
_FILLER_WORDS = (
    "def", "return", "self", "data", "value", "context", "model", "request",
    "response", "config", "cache", "input", "output", "token", "prompt",
    "message", "session", "round", "tool", "result", "agent", "stream",
    "client", "server", "param", "state", "event", "trace", "parse", "load",
)


class TraceLabWorkload(Workload):
    """Replay TraceLab coding-agent rounds as token-faithful chat requests.

    Args:
        path: path to a TraceLab ``.jsonl`` (or ``.jsonl.gz`` / ``.gz``) file.
            A ``.zip`` containing exactly one JSONL member is also accepted.
        prefix_cache: when True, group rounds by ``session_id`` and replay each
            session with byte-exact growing prefixes (exercises the server's
            prefix cache). When False (default), every request is independent.
        match_output_tokens: when True, also set ``min_tokens`` and
            ``ignore_eos`` per request so the server decodes exactly
            ``output_tokens`` tokens (reproduces the true decode-length load;
            best on vLLM/SGLang). When False (default), only ``max_tokens`` is
            set, so the model may stop early.
        max_tokens_cap: optional ceiling on the per-request ``max_tokens``
            (and ``min_tokens``) derived from ``output_tokens`` — guards against
            a handful of pathologically long rounds dominating a run.

        provider: keep only rows whose ``provider`` matches (e.g. ``"claude"``).
        model_filter: keep only rows whose ``model`` matches. ``None`` keeps all.
        min_input_tokens / max_input_tokens: inclusive bounds on
            ``input_tokens_total``.
        min_output_tokens / max_output_tokens: inclusive bounds on
            ``output_tokens`` (also the lower bound used when
            ``match_output_tokens`` would otherwise force a tiny decode).
        max_items: cap on the total number of rows kept (after filtering).
        max_sessions: cap on the number of sessions kept (prefix-cache mode).
        default_output_tokens: ``max_tokens`` used when a row has no usable
            ``output_tokens`` (default 128).

        chars_per_token: char-mode token-size approximation (default 4.0).
        tokenizer: optional HuggingFace tokenizer id; switches to exact
            token-count sizing (needs the ``tokenizer`` extra).

        shuffle: permute the emission order (flat mode: rows; prefix-cache
            mode: sessions — rounds within a session always stay ordered).
        seed: RNG seed for shuffle and for any tie-breaking.
        loop: restart the (filtered) plan when exhausted (default True).
        workload_name: override the recorded workload name.
    """

    name = "tracelab"

    def __init__(
        self,
        path: str,
        *,
        prefix_cache: bool = False,
        match_output_tokens: bool = False,
        max_tokens_cap: Optional[int] = None,

        provider: Optional[str] = None,
        model_filter: Optional[str] = None,
        min_input_tokens: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
        min_output_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        max_items: Optional[int] = None,
        max_sessions: Optional[int] = None,
        default_output_tokens: int = 128,

        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
        tokenizer: Optional[str] = None,

        shuffle: bool = True,
        seed: Optional[int] = 0,
        loop: bool = True,
        workload_name: Optional[str] = None,
    ):
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")
        if default_output_tokens <= 0:
            raise ValueError("default_output_tokens must be > 0")

        self._path = path
        self._prefix_cache = prefix_cache
        self._match_output = match_output_tokens
        self._max_tokens_cap = max_tokens_cap
        self._provider = provider
        self._model_filter = model_filter
        self._min_in = min_input_tokens
        self._max_in = max_input_tokens
        self._min_out = min_output_tokens
        self._max_out = max_output_tokens
        self._max_items = max_items
        self._max_sessions = max_sessions
        self._default_out = default_output_tokens
        self._chars_per_token = chars_per_token
        self._tokenizer_id = tokenizer
        self._shuffle = shuffle
        self._seed = seed
        self._loop = loop

        self.name = workload_name or ("tracelab:prefix" if prefix_cache else "tracelab")

        self._sizer = _build_sizer(tokenizer, chars_per_token)
        # Materialized emission plan: a flat list of per-request records. In
        # flat mode one record == one row; in prefix-cache mode one record ==
        # one round, carrying the session seed + cumulative token target so
        # round i's prompt is a byte-exact prefix of round i+1's.
        self._plan: list[dict[str, Any]] = []
        self._cursor = 0
        self._n = 0
        self._build_plan()

    # ------------------------------------------------------------------ plan

    def _build_plan(self) -> None:
        rows = list(self._iter_filtered_rows())
        if self._prefix_cache:
            self._plan = self._expand_sessions(self._group_sessions(rows))
        else:
            self._plan = [
                self._flat_record(row, index)
                for index, row in enumerate(rows)
            ]
        if self._shuffle:
            rng = random.Random(self._seed)
            if self._prefix_cache:
                # Shuffle the *sessions*, not the rounds within them — otherwise
                # a session's prefix-nesting would be broken across shuffles.
                self._plan = self._shuffle_preserving_sessions(self._plan, rng)
            else:
                rng.shuffle(self._plan)

    def _flat_record(self, row: dict[str, Any], index: int) -> dict[str, Any]:
        seed = row.get(F_TRACE_KEY)
        if seed is None:
            seed = f"{row.get(F_SESSION)}:{row.get(F_ROUND_ID)}:{index}"
        return {
            "row": row,
            "seed": seed,
            "target": row["input_tokens_total"],
            "prefix_hint": None,
            "session_rounds": None,
        }

    def _iter_filtered_rows(self) -> Iterable[dict[str, Any]]:
        kept = 0
        for row in self._read_jsonl(self._path):
            shaped = self._shape_row(row)
            if shaped is None:
                continue
            if self._max_items is not None and kept >= self._max_items:
                break
            yield shaped
            kept += 1

    def _shape_row(self, row: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Validate + normalize a raw row into a flat record, or None to skip."""
        if not isinstance(row, dict):
            return None
        if self._provider is not None and row.get(F_PROVIDER) != self._provider:
            return None
        if self._model_filter is not None and row.get(F_MODEL) != self._model_filter:
            return None

        input_total = _as_int(row.get(F_INPUT_TOTAL))
        prefix = _as_int(row.get(F_PREFIX))
        newly = _as_int(row.get(F_NEWLY_APPEND))
        if input_total is None and (prefix is not None or newly is not None):
            input_total = (prefix or 0) + (newly or 0)
        if input_total is None or input_total <= 0:
            return None
        if prefix is None or prefix < 0:
            prefix = 0
        if newly is None or newly < 0:
            newly = max(0, input_total - prefix)

        out = _as_int(row.get(F_OUTPUT))
        if out is None or out <= 0:
            out = self._default_out

        if self._min_in is not None and input_total < self._min_in:
            return None
        if self._max_in is not None and input_total > self._max_in:
            return None
        if self._min_out is not None and out < self._min_out:
            return None
        if self._max_out is not None and out > self._max_out:
            return None

        return {
            F_PROVIDER: row.get(F_PROVIDER),
            F_MODEL: row.get(F_MODEL),
            F_SESSION: row.get(F_SESSION),
            F_ROUND_INDEX: row.get(F_ROUND_INDEX),
            F_ROUND_ID: row.get(F_ROUND_ID),
            F_TRACE_KEY: row.get(F_TRACE_KEY),
            "input_tokens_total": input_total,
            "prefix_tokens": prefix,
            "newly_append_tokens": newly,
            "output_tokens": out,
            "tool_count": len(row[F_TOOLS]) if isinstance(row.get(F_TOOLS), list) else 0,
        }

    def _group_sessions(self, rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        sessions: dict[Any, list[dict[str, Any]]] = {}
        order: list[Any] = []
        for row in rows:
            sid = row.get(F_SESSION)
            key = sid if sid is not None else row.get(F_TRACE_KEY) or id(row)
            if key not in sessions:
                sessions[key] = []
                order.append(key)
            sessions[key].append(row)
        grouped = []
        for key in order:
            rounds = sessions[key]
            rounds.sort(key=_round_sort_key)
            grouped.append(rounds)
            if self._max_sessions is not None and len(grouped) >= self._max_sessions:
                break
        return grouped

    def _expand_sessions(
        self, sessions: list[list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Flatten sessions into one record per round, ordered round-robin.

        Emitting round 0 of every session, then round 1 of every session, …
        keeps prefix locality strong (a session's consecutive rounds land
        close together in the stream) while still mixing sessions — the load
        pattern an LLM server actually sees from many concurrent agents.
        """
        max_rounds = max((len(s) for s in sessions), default=0)
        plan: list[dict[str, Any]] = []
        for depth in range(max_rounds):
            for session in sessions:
                if depth >= len(session):
                    continue
                plan.append(self._prefix_record(session, depth))
        return plan

    def _prefix_record(
        self, rounds: list[dict[str, Any]], depth: int
    ) -> dict[str, Any]:
        seed = rounds[0].get(F_SESSION) or rounds[0].get(F_TRACE_KEY) or id(rounds)
        base_prefix = rounds[0]["prefix_tokens"]
        running = base_prefix
        for r in rounds[: depth + 1]:
            running += r["newly_append_tokens"]
        return {
            "row": rounds[depth],
            "seed": seed,
            "target": running,
            "prefix_hint": rounds[depth]["prefix_tokens"],
            "session_rounds": len(rounds),
        }

    @staticmethod
    def _shuffle_preserving_sessions(
        plan: list[dict[str, Any]], rng: random.Random
    ) -> list[dict[str, Any]]:
        # Group consecutive records by their session seed (round-robin emission
        # interleaves sessions, so regroup by seed), shuffle the session order,
        # and re-emit each session's rounds in their internal round order.
        order: list[Any] = []
        by_seed: dict[Any, list[dict[str, Any]]] = {}
        for rec in plan:
            s = rec["seed"]
            if s not in by_seed:
                by_seed[s] = []
                order.append(s)
            by_seed[s].append(rec)
        rng.shuffle(order)
        out: list[dict[str, Any]] = []
        for s in order:
            out.extend(by_seed[s])
        return out

    # --------------------------------------------------------------- emission

    async def next_item(self) -> dict[str, Any]:
        if self._cursor >= len(self._plan):
            if not self._loop or not self._plan:
                raise StopAsyncIteration
            self._cursor = 0

        rec = self._plan[self._cursor]
        self._cursor += 1
        self._n += 1
        prompt = self._sizer.fill(rec["seed"], rec["target"])
        return self._wrap(
            rec["row"],
            prompt,
            prefix_tokens_hint=rec["prefix_hint"],
            session_rounds=rec.get("session_rounds"),
        )

    def _wrap(
        self,
        row: dict[str, Any],
        prompt: str,
        *,
        prefix_tokens_hint: Optional[int],
        session_rounds: Optional[int] = None,
    ) -> dict[str, Any]:
        out = row["output_tokens"]
        cap = self._max_tokens_cap
        max_tokens = out if cap is None else min(out, cap)
        item: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if self._match_output:
            forced = max_tokens
            item["min_tokens"] = forced
            item["ignore_eos"] = True

        meta: dict[str, Any] = {
            "prompt_tokens_hint": row["input_tokens_total"],
            "tracelab_provider": row.get(F_PROVIDER),
            "tracelab_model": row.get(F_MODEL),
            "tracelab_session_id": row.get(F_SESSION),
            "tracelab_round_index": row.get(F_ROUND_INDEX),
            "tracelab_input_tokens": row["input_tokens_total"],
            "tracelab_prefix_tokens": row["prefix_tokens"],
            "tracelab_newly_append_tokens": row["newly_append_tokens"],
            "tracelab_output_tokens_target": row["output_tokens"],
            "tracelab_tool_count": row["tool_count"],
        }
        if prefix_tokens_hint is not None:
            meta["prefix_tokens_hint"] = prefix_tokens_hint
        if session_rounds is not None:
            meta["tracelab_session_rounds"] = session_rounds
        if row.get(F_TRACE_KEY) is not None:
            meta["tracelab_trace_key"] = row[F_TRACE_KEY]
        item["meta"] = meta
        return item

    # --------------------------------------------------------------- file io

    @staticmethod
    def _read_jsonl(path: str) -> Iterable[dict[str, Any]]:
        """Yield parsed JSON objects from a .jsonl / .jsonl.gz / .gz / .zip."""
        opener = _open_trace(path)
        with opener as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield obj

    async def aclose(self) -> None:
        closer = getattr(self._sizer, "aclose", None)
        if closer is not None:
            await closer()
        return None


# --------------------------------------------------------------------------- #
# Sizing backends
# --------------------------------------------------------------------------- #


def _build_sizer(tokenizer_id: Optional[str], chars_per_token: float) -> "_Sizer":
    if tokenizer_id:
        return _TokenizerSizer(tokenizer_id)
    return _CharSizer(chars_per_token)


class _CharSizer:
    """Char-based filler: deterministic word stream sliced to a char target."""

    def __init__(self, chars_per_token: float):
        self._cpt = chars_per_token

    def fill(self, seed: Any, target_tokens: int) -> str:
        target_tokens = max(1, int(target_tokens))
        target_chars = max(1, round(target_tokens * self._cpt))
        return _filler_text(seed, target_chars)


class _TokenizerSizer:
    """Exact filler: repeat a single-token unit ``target_tokens`` times."""

    def __init__(self, tokenizer_id: str):
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised via message
            raise ImportError(
                "TraceLabWorkload(tokenizer=...) needs the `transformers` "
                "package. Install with `pip install -e .[tokenizer]`."
            ) from e
        self._tok = AutoTokenizer.from_pretrained(tokenizer_id)
        self._unit = _single_token_unit(self._tok)

    def fill(self, seed: Any, target_tokens: int) -> str:
        n = max(1, int(target_tokens))
        return self._unit * n


def _single_token_unit(tokenizer: Any) -> str:
    """Find a short string that encodes to exactly one token.

    Repeating it then yields exactly N tokens, and any prefix of ``unit * N`` is
    byte-exact with ``unit * (N + k)`` — keeping session prefix nesting exact.
    """
    candidates = ("x", " x", "a", " a", "0", ".", "word")
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return cand
    # Fall back to whatever the first candidate encodes to; sizes will be close
    # but not exact. (Unlikely path for mainstream tokenizers.)
    return "x"


def _filler_text(seed: Any, char_count: int) -> str:
    """Deterministic filler of ~char_count chars, stable per seed+position."""
    rng = random.Random(_hash_seed(seed))
    words = _FILLER_WORDS
    nwords = len(words)
    out = io.StringIO()
    written = 0
    i = 0
    while written < char_count:
        w = words[(i + rng.randrange(0, nwords)) % nwords]
        if i:
            out.write(" ")
            written += 1
        out.write(w)
        written += len(w)
        i += 1
    text = out.getvalue()
    return text[:char_count]


def _hash_seed(seed: Any) -> int:
    """Reduce an arbitrary seed to a stable int for random.Random."""
    if isinstance(seed, int):
        return seed
    return abs(hash(str(seed))) % (2 ** 31)


def _round_sort_key(row: dict[str, Any]) -> tuple:
    idx = row.get(F_ROUND_INDEX)
    if isinstance(idx, int):
        return (0, idx)
    if isinstance(idx, str) and idx.lstrip("-").isdigit():
        return (0, int(idx))
    return (1, str(row.get(F_ROUND_ID) or ""))


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return None


def _open_trace(path: str):
    """Open a TraceLab JSONL source as a text file handle (handles gz/zip)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lower = path.lower()
    if lower.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if n.lower().endswith(".jsonl")]
        if not names:
            raise ValueError(f"zip {path!r} contains no .jsonl member")
        member = io.TextIOWrapper(zf.open(names[0]), encoding="utf-8")
        # Keep the zip open for the lifetime of the member.
        member._zip = zf  # type: ignore[attr-defined]
        return member
    if lower.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

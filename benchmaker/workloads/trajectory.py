"""Multi-turn trajectory replay workload.

Expands each agent trajectory into one chat request per assistant turn, sending
the sanitized message prefix up to (but excluding) that turn. Built for
prefix-cache / HiCache parity benchmarking against an OpenAI-compatible chat
endpoint — pair with ``OpenAIChatWorkloadType(passthrough_meta=True)``.

Source: a HuggingFace dataset id (needs the ``datasets`` package) or a local
JSONL file. The ``SWE-bench/SWE-smith-trajectories`` ``messages`` column is a
JSON-encoded *string*; ``parse_messages`` handles both string and list.

Recorded per request (into Request.meta -> samples.jsonl): conversation_id,
instance_id, turn_index, prefix_messages, model_label, and -- when a tokenizer
is configured -- expected_prefix_tokens (the theoretical upper bound of cacheable
prefix, i.e. the previous turn's prompt length; compare to the server's
cached_tokens).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any, Callable, Iterator, Optional

from benchmaker.core.load import parse_duration
from benchmaker.workloads.datasets import Workload


def parse_gap_spec(
    spec: Optional[str], seed: Optional[int] = None
) -> Callable[[], float]:
    """Build an inter-turn think-time sampler: ``() -> delay_seconds``.

    Models the per-session pause between a session's consecutive turns (agent
    tool-execution / user think time). Accepted forms::

        None / "" / "0" / "none"  -> always 0.0 (no gap)
        "2s" / "500ms"            -> constant (bare duration)
        "const:1500ms"            -> constant
        "exp:1.5"                 -> exponential, mean 1.5s
        "uniform:1s..3s"          -> uniform in [1s, 3s]

    ``seed`` makes the random forms reproducible. Constant forms ignore it.
    """
    if spec is None:
        return lambda: 0.0
    s = spec.strip().lower()
    if s in ("", "0", "none"):
        return lambda: 0.0

    rng = random.Random(seed)

    if s.startswith("exp:"):
        mean = parse_duration(s.split(":", 1)[1])
        if mean <= 0:
            return lambda: 0.0
        lam = 1.0 / mean
        return lambda: rng.expovariate(lam)

    if s.startswith("uniform:"):
        rest = s.split(":", 1)[1]
        if ".." not in rest:
            raise ValueError(f"uniform gap needs 'lo..hi', got {spec!r}")
        lo_s, hi_s = rest.split("..", 1)
        lo, hi = parse_duration(lo_s), parse_duration(hi_s)
        if hi < lo:
            raise ValueError(f"uniform gap hi<lo in {spec!r}")
        return lambda: rng.uniform(lo, hi)

    if s.startswith("const:"):
        val = parse_duration(s.split(":", 1)[1])
        return lambda: val

    if ":" in s:
        raise ValueError(f"unknown inter-turn gap distribution: {spec!r}")

    # Bare duration -> constant.
    val = parse_duration(s)
    return lambda: val


# Message keys accepted by OpenAI-compatible chat endpoints. All dataset-specific
# keys (agent, message_type, ...) are dropped.
_OPENAI_MSG_KEYS = ("role", "content", "name", "tool_calls", "tool_call_id")


def sanitize_message(msg: dict) -> dict:
    """Keep only OpenAI-valid keys; default missing content to ''."""
    out: dict[str, Any] = {}
    for k in _OPENAI_MSG_KEYS:
        if k in msg and msg[k] is not None:
            out[k] = msg[k]
    out.setdefault("role", "user")
    if "content" not in out and "tool_calls" not in out:
        out["content"] = ""
    return out


def parse_messages(raw: Any) -> list[dict]:
    """Parse a `messages` value (JSON string or list) into sanitized dicts."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise TypeError(f"messages must be a list or JSON-list string, got {type(raw).__name__}")
    return [sanitize_message(m) for m in raw if isinstance(m, dict)]


def expand_trajectory(
    messages: list[dict],
    *,
    meta_base: dict,
    max_tokens: int,
    max_turns: Optional[int],
    count_tokens: Optional[Callable[[list[dict]], int]],
) -> list[dict]:
    """One item per assistant turn; prompt = sanitized prefix before that turn.

    `expected_prefix_tokens` (only when `count_tokens` is given) is the token
    count of the *previous emitted turn's* prompt — the nested-prefix upper bound
    of what the server could serve from cache (0 for the first turn).
    """
    items: list[dict] = []
    turn = 0
    prev_tokens = 0
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        prefix = messages[:i]
        if not prefix:
            continue
        meta = dict(meta_base)
        meta["turn_index"] = turn
        meta["prefix_messages"] = len(prefix)
        if count_tokens is not None:
            meta["expected_prefix_tokens"] = prev_tokens
        items.append({"messages": prefix, "max_tokens": max_tokens, "meta": meta})
        if count_tokens is not None:
            prev_tokens = count_tokens(prefix)
        turn += 1
        if max_turns is not None and turn >= max_turns:
            break
    return items


class _Session:
    """One in-flight conversation in interleaved mode.

    A session holds its expanded turn items and a cursor; it is *eligible* to
    emit its next turn only when no turn is in flight and the inter-turn gap has
    elapsed (``ready_at`` is in the past).
    """

    __slots__ = ("cid", "turns", "idx", "in_flight", "ready_at")

    def __init__(self, cid: Any, turns: list[dict]):
        self.cid = cid
        self.turns = turns
        self.idx = 0
        self.in_flight = False
        self.ready_at = 0.0  # monotonic time the next turn becomes eligible

    @property
    def has_remaining(self) -> bool:
        return self.idx < len(self.turns)

    @property
    def done(self) -> bool:
        return not self.has_remaining and not self.in_flight


class TrajectoryReplayWorkload(Workload):
    name = "trajectory"

    def __init__(
        self,
        *,
        dataset: Optional[str] = None,
        split: str = "tool",
        path: Optional[str] = None,
        messages_field: str = "messages",
        id_field: str = "instance_id",
        model_field: str = "model",
        max_tokens: int = 1024,
        max_turns_per_trajectory: Optional[int] = None,
        max_trajectories: Optional[int] = None,
        tokenizer: Optional[str] = None,
        concurrent_sessions: Optional[int] = None,
        inter_turn_gap: Optional[str] = None,
        gap_seed: Optional[int] = 0,
        name: Optional[str] = None,
        **hf_kwargs: Any,
    ):
        if (dataset is None) == (path is None):
            raise ValueError("provide exactly one of `dataset` or `path`.")
        if concurrent_sessions is not None and concurrent_sessions < 1:
            raise ValueError("concurrent_sessions must be >= 1")
        self._dataset = dataset
        self._split = split
        self._path = path
        self._messages_field = messages_field
        self._id_field = id_field
        self._model_field = model_field
        self._max_tokens = max_tokens
        self._max_turns = max_turns_per_trajectory
        self._max_trajectories = max_trajectories
        self._hf_kwargs = hf_kwargs
        if name:
            self.name = name
        elif dataset:
            self.name = f"trajectory:{dataset}/{split}"
        else:
            self.name = f"trajectory:{path}"

        self._count_tokens = self._make_counter(tokenizer)
        self._gen: Optional[Iterator[dict]] = None
        self._lock = asyncio.Lock()

        # Interleaved / concurrent-session scheduling. When off (default), turns
        # replay contiguously per trajectory (prefix-cache locality). When on,
        # up to N sessions stay active and round-robin their turns, with each
        # session's turn k+1 gated on turn k's completion (+ inter-turn gap) —
        # producing realistic reuse-after-eviction working-set pressure.
        self._interleave = concurrent_sessions is not None
        self._concurrency = concurrent_sessions or 0
        self._gap = parse_gap_spec(inter_turn_gap, seed=gap_seed)
        self._sessions: list[_Session] = []
        self._by_cid: dict[Any, _Session] = {}
        self._rr = 0
        self._source: Optional[Iterator[tuple[Any, list[dict]]]] = None
        self._source_done = False
        self._wakeup: Optional[asyncio.Event] = None

    def _make_counter(self, tokenizer: Optional[str]):
        if not tokenizer:
            return None
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "trajectory replay with --tokenizer needs `transformers`. "
                "Install with `pip install transformers` or `pip install -e .[hf]`."
            ) from e
        tok = AutoTokenizer.from_pretrained(tokenizer)

        def _count(msgs: list[dict]) -> int:
            try:
                text = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = "\n".join(str(m.get("content") or "") for m in msgs)
            return len(tok(text)["input_ids"])

        return _count

    def _iter_rows(self) -> Iterator[dict]:
        if self._path:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        else:
            try:
                from datasets import load_dataset
            except ImportError as e:
                raise ImportError(
                    "trajectory replay from a HF dataset needs `datasets`. "
                    "Install with `pip install datasets` or `pip install -e .[hf]`."
                ) from e
            ds = load_dataset(self._dataset, split=self._split, streaming=True,
                              **self._hf_kwargs)
            for row in ds:
                yield dict(row)

    def _iter_sessions(self) -> Iterator[tuple[Any, list[dict]]]:
        """Yield ``(conversation_id, turn_items)`` per non-empty trajectory.

        Honors ``max_trajectories`` (degenerate trajectories with no assistant
        turns are skipped and don't count) and ``max_turns_per_trajectory``.
        """
        n_traj = 0
        for row in self._iter_rows():
            conv_id = row.get(self._id_field) or row.get("traj_id")
            meta_base = {
                "conversation_id": conv_id,
                "instance_id": row.get(self._id_field),
                "model_label": row.get(self._model_field),
            }
            messages = parse_messages(row[self._messages_field])
            items = expand_trajectory(
                messages, meta_base=meta_base, max_tokens=self._max_tokens,
                max_turns=self._max_turns, count_tokens=self._count_tokens,
            )
            if not items:
                continue  # degenerate trajectory (no assistant turns) — don't count it
            if self._max_trajectories is not None and n_traj >= self._max_trajectories:
                return
            n_traj += 1
            yield conv_id, items

    def _iter_items(self) -> Iterator[dict]:
        for _, items in self._iter_sessions():
            for item in items:
                yield item

    async def next_item(self) -> Any:
        if self._interleave:
            return await self._next_interleaved()
        async with self._lock:
            if self._gen is None:
                self._gen = self._iter_items()
            try:
                return next(self._gen)
            except StopIteration:
                raise StopAsyncIteration

    # ----- interleaved scheduling -----

    def _admit(self) -> None:
        """Fill the active pool up to ``concurrency`` from the trajectory source."""
        if self._source is None:
            self._source = self._iter_sessions()
        while not self._source_done and len(self._sessions) < self._concurrency:
            try:
                cid, items = next(self._source)
            except StopIteration:
                self._source_done = True
                break
            sess = _Session(cid, items)
            self._sessions.append(sess)
            # Last writer wins if a conversation_id is concurrently active twice
            # (not expected for unique instance_ids); completion still advances
            # exactly one in-flight turn at a time.
            self._by_cid[cid] = sess

    def _reap(self) -> None:
        """Drop finished sessions, keeping the round-robin cursor in range."""
        if not any(s.done for s in self._sessions):
            return
        kept = []
        for s in self._sessions:
            if s.done:
                if self._by_cid.get(s.cid) is s:
                    del self._by_cid[s.cid]
            else:
                kept.append(s)
        self._sessions = kept
        if self._sessions:
            self._rr %= len(self._sessions)
        else:
            self._rr = 0

    def _pick_ready(self, now: float) -> Optional[_Session]:
        """Round-robin the next eligible session (not in flight, gap elapsed)."""
        n = len(self._sessions)
        for offset in range(n):
            i = (self._rr + offset) % n
            s = self._sessions[i]
            if not s.in_flight and s.has_remaining and s.ready_at <= now:
                self._rr = i + 1
                return s
        return None

    def _soonest_ready_in(self, now: float) -> Optional[float]:
        """Seconds until the earliest gapping session becomes eligible (or None)."""
        waits = [
            s.ready_at - now
            for s in self._sessions
            if not s.in_flight and s.has_remaining and s.ready_at > now
        ]
        return min(waits) if waits else None

    async def _next_interleaved(self) -> Any:
        if self._wakeup is None:
            self._wakeup = asyncio.Event()
        while True:
            # Reap finished sessions first, then refill from the source so the
            # active pool stays topped up and end-of-source is detected promptly.
            self._reap()
            self._admit()
            now = time.monotonic()
            sess = self._pick_ready(now)
            if sess is not None:
                item = sess.turns[sess.idx]
                sess.idx += 1
                sess.in_flight = True
                return item
            # Nothing eligible right now.
            if not self._sessions and self._source_done:
                raise StopAsyncIteration
            # Wait for a completion (wakeup) or the next gap to elapse.
            timeout = self._soonest_ready_in(now)
            self._wakeup.clear()
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass  # a gap elapsed; re-evaluate eligibility

    def notify_turn_complete(self, conversation_id: Any) -> None:
        """Mark a session's in-flight turn done; schedule its next turn (+ gap).

        Called by the completion post-hook (see :meth:`note_complete_hook`).
        Idempotent for unknown / already-completed conversation ids.
        """
        sess = self._by_cid.get(conversation_id)
        if sess is None or not sess.in_flight:
            return
        sess.in_flight = False
        if sess.has_remaining:
            sess.ready_at = time.monotonic() + self._gap()
        if self._wakeup is not None:
            self._wakeup.set()

    def note_complete_hook(self, request: Any, response: Any, sample: Any) -> Any:
        """Post-hook: advance the session whose turn just completed.

        Wired into the run only in interleaved mode. Reads ``conversation_id``
        off ``Request.meta`` and returns ``sample`` unchanged.
        """
        cid = (getattr(request, "meta", None) or {}).get("conversation_id")
        if cid is not None:
            self.notify_turn_complete(cid)
        return sample

    def completion_hook(self) -> Optional[Callable]:
        """The interleaved scheduler needs a per-turn completion signal."""
        return self.note_complete_hook if self._interleave else None

    async def aclose(self) -> None:
        if self._gen is not None:
            self._gen.close()
            self._gen = None
        if self._source is not None:
            self._source.close()
            self._source = None

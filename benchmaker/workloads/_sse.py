"""Shared helper for parsing streamed Server-Sent-Events (SSE) responses.

The runner captures a streaming response as ``Response.stream_chunks`` — the raw
byte chunks yielded by httpx2's ``resp.aiter_raw()``, which land on
*arbitrary* byte boundaries. A single SSE event (a ``data: {...}`` line) may
therefore be split across two chunks, and — because it is the last and largest
line — the final ``usage`` / ``meta_info`` event is the most likely to be split.

Splitting each chunk independently with ``bytes.splitlines()`` turns such a split
event into two unparseable JSON fragments that get silently dropped, losing
per-request token accounting (``prompt_tokens`` / ``cached_tokens``). This helper
reassembles the chunk stream into complete lines before parsing, attributing each
completed line to the arrival time of the chunk that finished it.
"""

from __future__ import annotations

from typing import Optional


def reassemble_sse_lines(
    chunks: Optional[list[bytes]],
    chunk_times: Optional[list[float]],
) -> list[tuple[bytes, float]]:
    """Reassemble arbitrarily-split byte chunks into ``(line, arrival_time)``.

    A line is emitted only once its terminating newline has arrived, so an SSE
    event split across chunk boundaries is rejoined instead of dropped. The
    arrival time is that of the chunk which completed the line (i.e. when the
    full event first became available) — the correct instant for TTFT / ITL.

    Trailing bytes with no final newline (a well-behaved SSE stream ends every
    event with ``\\n\\n``, but be defensive) are emitted best-effort at the last
    chunk's time. A trailing ``\\r`` is left on the line for the caller's
    ``.strip()`` to remove, matching ``splitlines()`` handling of ``\\r\\n``.

    ``chunk_times`` may be shorter than ``chunks`` (or omitted entirely, for
    callers that only need the reassembled text): missing entries reuse the last
    known time, defaulting to ``0.0``.
    """
    chunks = chunks or []
    chunk_times = chunk_times or []
    lines: list[tuple[bytes, float]] = []
    buf = b""
    last_t = 0.0
    for i, raw in enumerate(chunks):
        t = chunk_times[i] if i < len(chunk_times) else last_t
        last_t = t
        buf += raw
        start = 0
        nl = buf.find(b"\n", start)
        while nl != -1:
            lines.append((buf[start:nl], t))
            start = nl + 1
            nl = buf.find(b"\n", start)
        buf = buf[start:]
    if buf:
        lines.append((buf, last_t))
    return lines

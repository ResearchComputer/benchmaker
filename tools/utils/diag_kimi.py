#!/usr/bin/env python3
"""Single-request diagnostic for the Kimi-K2.6 2-node serve.

Probes the OpenAI endpoint in stages to localize where the 8-rank serve falls
over under real load:
  1. SHORT single request   (tiny prompt) — baseline, should match warmup.
  2. LONG  single request   (longest ShareGPT row) — isolates long prefill.
  3. CONCURRENT small burst  (6 at once)  — isolates concurrency.

Stops at the FIRST stage that hangs/errors and writes that verdict to
$DIAG_MARKER (the sbatch then py-spies all ranks while the hang is live).
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import aiohttp

URL = os.environ.get("DIAG_URL", "http://127.0.0.1:8000/v1/chat/completions")
JSONL = os.environ["BENCH_JSONL"]
MARKER = os.environ["DIAG_MARKER"]
MAX_PROMPT_CHARS = int(os.environ.get("DIAG_MAX_PROMPT_CHARS", "8000"))  # ~2k tokens < max_model_len


def longest_messages(path: str):
    best, bestlen = None, -1
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            msgs = row.get("messages") or []
            if not msgs or msgs[-1].get("role") != "user":
                continue
            tot = sum(len(m.get("content", "")) for m in msgs)
            if bestlen < tot <= MAX_PROMPT_CHARS:
                bestlen, best = tot, msgs
    return best, bestlen


async def send(session, messages, max_tokens, timeout_s):
    body = {
        "model": "kimi",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        async with session.post(
            URL, json=body, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as r:
            data = await r.json()
            dt = time.monotonic() - t0
            txt = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return ("ok", dt, txt[:160], data.get("usage", {}))
    except asyncio.TimeoutError:
        return ("HANG", time.monotonic() - t0, "", {})
    except Exception as e:
        return ("ERR", time.monotonic() - t0, f"{type(e).__name__}: {str(e)[:160]}", {})


def _flag(stage, status):
    with open(MARKER, "w") as f:
        f.write(f"{status}:{stage}")


async def main():
    short = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    longmsgs, longlen = longest_messages(JSONL)
    print(f"[diag] longest ShareGPT row: {longlen} chars, {len(longmsgs or [])} turns", flush=True)

    async with aiohttp.ClientSession() as s:
        print("[diag] STAGE 1: SHORT single (max_tokens=16, timeout=90s)", flush=True)
        st, dt, txt, u = await send(s, short, 16, 90)
        print(f"[diag] SHORT: {st} dt={dt:.1f}s usage={u} txt={txt!r}", flush=True)
        if st != "ok":
            _flag("short", st)
            print("[diag] -> first failure at SHORT; triggering py-spy", flush=True)
            return

        print("[diag] STAGE 2: LONG single (max_tokens=128, timeout=240s)", flush=True)
        st, dt, txt, u = await send(s, longmsgs, 128, 240)
        print(f"[diag] LONG: {st} dt={dt:.1f}s usage={u} txt={txt!r}", flush=True)
        if st != "ok":
            _flag("long", st)
            print("[diag] -> first failure at LONG; triggering py-spy", flush=True)
            return

        print("[diag] STAGE 3: BURST 6 concurrent (max_tokens=64, timeout=300s)", flush=True)
        res = await asyncio.gather(*[send(s, short, 64, 300) for _ in range(6)])
        statuses = [r[0] for r in res]
        nbad = sum(1 for x in statuses if x != "ok")
        print(f"[diag] BURST: {len(res) - nbad}/{len(res)} ok statuses={statuses}", flush=True)
        if nbad:
            _flag("burst", f"{nbad}bad")
            print("[diag] -> failure at BURST; triggering py-spy", flush=True)
            return

    _flag("none", "ALLOK")
    print("[diag] ALL STAGES OK: serve survived short + long + concurrent", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

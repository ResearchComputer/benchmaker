#!/usr/bin/env python3
"""RPS-sweep driver for an OpenAI-compatible endpoint, via the benchmaker library.

Runs one open-loop Poisson load per RPS point against
``$BENCH_URL`` (default localhost:8000), feeding ShareGPT ``messages`` rows,
and prints a per-rate summary plus a final table. Uses only the benchmaker
library API (no recipe registry), so it does not pull the swebench/harbor deps.

Env:
  BENCH_URL        chat-completions URL (default http://127.0.0.1:8000/v1/chat/completions)
  BENCH_MODEL      served model name (default "kimi")
  BENCH_JSONL      ShareGPT JSONL (one {"messages": [...]} per line) [required]
  BENCH_RATES      comma list of Poisson req/s, e.g. "1,2,4,8,16"
  BENCH_DURATION_S seconds per rate point (default 45)
  BENCH_MAX_TOKENS max output tokens per request (default 128)
  BENCH_OUTDIR     directory for per-rate summary JSON (default ./runs)
"""
from __future__ import annotations

import asyncio
import json
import os
import time

from benchmaker import (
    BenchConfig,
    BenchRunner,
    JsonlWorkload,
    OpenAIChatWorkloadType,
    parse_rate_spec,
)

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("BENCH_MODEL", "kimi")
JSONL = os.environ["BENCH_JSONL"]
RATES = [float(x) for x in os.environ.get("BENCH_RATES", "1,2,4,8,16").split(",")]
DURATION_S = float(os.environ.get("BENCH_DURATION_S", "45"))
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "128"))
OUTDIR = os.environ.get("BENCH_OUTDIR", "./runs")
# "poisson" = open-loop req/s; "closed" = bounded in-flight (concurrency). For a
# capacity-bound server, closed-loop avoids an unbounded backlog from arrivals
# outrunning completions.
MODE = os.environ.get("BENCH_MODE", "poisson").strip().lower()
REQ_TIMEOUT_S = float(os.environ.get("BENCH_TIMEOUT_S", "600"))
# Optional marker file: written "HANG:<stage>" on the first stage with failures
# so the sbatch can py-spy the (likely hung) ranks.
MARKER = os.environ.get("BENCH_MARKER", "")


def _metric(summary: dict, name: str, stat: str):
    """Pull a percentile for a metric, whether it's a workload_metrics entry
    (ttft_s, tokens_per_s, tokens_out, ...) or a top-level one (latency_s)."""
    wm = summary.get("workload_metrics", {})
    if name in wm and isinstance(wm[name], dict):
        return wm[name].get(stat)
    if name in summary and isinstance(summary[name], dict):
        return summary[name].get(stat)
    return None


def _fmt(v, nd=1):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


async def main():
    os.makedirs(OUTDIR, exist_ok=True)
    spec_kind = "closed" if MODE == "closed" else "poisson"
    unit = "in-flight" if spec_kind == "closed" else "req/s"
    print(
        f"[bench] url={URL} model={MODEL} jsonl={JSONL}\n"
        f"[bench] mode={spec_kind} levels={RATES} ({unit}) duration_s={DURATION_S} "
        f"max_tokens={MAX_TOKENS} req_timeout_s={REQ_TIMEOUT_S}",
        flush=True,
    )
    table = []
    for rps in RATES:
        cfg = BenchConfig(
            workload_type=OpenAIChatWorkloadType(
                url=URL, model=MODEL, max_tokens=MAX_TOKENS, temperature=0.0
            ),
            # fresh workload iterator per level (reads from the top of the file)
            workload=JsonlWorkload(path=JSONL, field="messages"),
            load=parse_rate_spec(f"{spec_kind}:{rps:g}", duration_s=DURATION_S),
            timeout_s=REQ_TIMEOUT_S,
        )
        print(f"\n========== {spec_kind} {rps:g} {unit} for {DURATION_S:g}s ==========", flush=True)
        res = await BenchRunner(cfg).run()
        s = res.summary
        with open(os.path.join(OUTDIR, f"summary_rps{rps:g}.json"), "w") as f:
            json.dump(s, f, indent=2)
        # human-readable text summary if benchmaker provides one
        txt = getattr(res, "summary_text", None)
        print(txt if isinstance(txt, str) else json.dumps(s, indent=2), flush=True)
        ok = s.get("success") or 0
        total = s.get("total_requests") or (ok + (s.get("failed") or 0))
        table.append(
            {
                "level": rps,
                "achieved_rps": s.get("goodput_rps", s.get("throughput_rps")),
                "success": ok,
                "total": total,
                "ttft_p50": _metric(s, "ttft_s", "p50"),
                "ttft_p99": _metric(s, "ttft_s", "p99"),
                "lat_p50": _metric(s, "latency_s", "p50"),
                "lat_p99": _metric(s, "latency_s", "p99"),
                "tok_s_p50": _metric(s, "tokens_per_s", "p50"),
                "tokens_out_p50": _metric(s, "tokens_out", "p50"),
            }
        )
        # Fail-fast: a stage with failed requests means the server likely hung;
        # stop sweeping and flag for py-spy rather than push further load into it.
        if total and ok < total:
            print(
                f"[bench] {total - ok}/{total} requests FAILED at {spec_kind} {rps:g} "
                f"{unit} -> stopping sweep",
                flush=True,
            )
            if MARKER:
                with open(MARKER, "w") as f:
                    f.write(f"HANG:{spec_kind}{rps:g}")
            break

    print("\n\n==================== SWEEP SUMMARY ====================", flush=True)
    hdr = (
        f"{'level':>8} {'achieved':>9} {'ok/total':>10} "
        f"{'ttft_p50':>9} {'ttft_p99':>9} {'lat_p50':>8} {'lat_p99':>8} "
        f"{'tok/s_p50':>9} {'out_p50':>8}"
    )
    print(hdr, flush=True)
    for r in table:
        print(
            f"{r['level']:>8g} {_fmt(r['achieved_rps'],2):>9} "
            f"{str(r['success'])+'/'+str(r['total']):>10} "
            f"{_fmt(r['ttft_p50'],3):>9} {_fmt(r['ttft_p99'],3):>9} "
            f"{_fmt(r['lat_p50'],2):>8} {_fmt(r['lat_p99'],2):>8} "
            f"{_fmt(r['tok_s_p50'],1):>9} {_fmt(r['tokens_out_p50'],0):>8}",
            flush=True,
        )
    with open(os.path.join(OUTDIR, "sweep_table.json"), "w") as f:
        json.dump(table, f, indent=2)
    print(f"\n[bench] wrote per-rate summaries + sweep_table.json to {OUTDIR}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

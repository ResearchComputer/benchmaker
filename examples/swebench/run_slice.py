"""Run a small SWE-bench Verified slice through SWEBenchAgent + Flash Sandbox.

A lightweight harness *outside* the benchmaker runner: it picks instances from a
JSON manifest (a list of raw SWE-bench rows) and prints per-instance pass/fail +
aggregate pass rate. Each instance boots its prebuilt ghcr eval image and is
graded authoritatively by the `swebench` package (same flow as the
`benchmaker run examples/swebench/config_swebench.yaml` path, minus the
metrics/load/summary machinery).

Run:
    python examples/swebench/run_slice.py /tmp/swe_bench_slice.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from benchmaker import AgentContext
from benchmaker.swebench import SWEBenchAgent

def _load_env(path: str = ".env") -> None:
    """Tiny KEY=VALUE loader so we don't need python-dotenv."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

async def run_one(agent: SWEBenchAgent, instance: dict,
                  trajectories_dir: Path) -> dict:
    instance_id = instance["instance_id"]
    print(f"\n--- {instance_id} ---", flush=True)
    t0 = time.monotonic()
    # Reset so we don't carry messages from a prior instance.
    agent.last_messages = None
    try:
        result = await agent.run(AgentContext(item=instance))
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"  CRASH after {elapsed:.1f}s: {type(e).__name__}: {e}",
              flush=True)
        return {
            "instance_id": instance_id, "ok": False, "crashed": True,
            "error": f"{type(e).__name__}: {e}",
            "elapsed_s": elapsed,
        }

    elapsed = time.monotonic() - t0
    meta, mx = result.meta, result.metrics
    # Save the model trajectory + extracted patch for diagnosis.
    if getattr(agent, "last_messages", None):
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        out_path = trajectories_dir / f"{instance_id}.json"
        out_path.write_text(json.dumps({
            "instance_id": instance_id,
            "exit_status": meta.get("exit_status"),
            "model_patch": result.output,
            "messages": agent.last_messages,
        }, indent=2))
        print(f"  trajectory: {out_path}", flush=True)
    print(f"  exit_status={meta['exit_status']}  image={meta.get('image')}"
          f"  submitted={meta.get('submitted')}  resolved={meta.get('resolved')}",
          flush=True)
    print(f"  steps={int(mx['steps'])} actions={int(mx['actions'])}  "
          f"F2P {int(mx['f2p_pass'])}/{int(mx['f2p_total'])}  "
          f"P2P {int(mx['p2p_pass'])}/{int(mx['p2p_total'])}  "
          f"patch={int(mx['patch_chars'])}c  elapsed={elapsed:.1f}s", flush=True)
    if meta.get("grading_error"):
        print(f"  grading_error: {meta['grading_error']}", flush=True)
    if meta.get("apply_log_tail"):
        print(f"  apply_log_tail:\n{_indent(meta['apply_log_tail'])}", flush=True)
    return {
        "instance_id": instance_id, "ok": result.ok,
        "exit_status": meta["exit_status"],
        "image": meta.get("image"),
        "submitted": meta.get("submitted"),
        "resolved": meta.get("resolved"),
        "f2p_pass": int(mx["f2p_pass"]), "f2p_total": int(mx["f2p_total"]),
        "p2p_pass": int(mx["p2p_pass"]), "p2p_total": int(mx["p2p_total"]),
        "grading_error": meta.get("grading_error"),
        "elapsed_s": elapsed,
        "patch_head": (result.output or "")[:600],
    }


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in (text or "").splitlines())


async def main(argv: list[str]) -> int:
    _load_env()
    if len(argv) < 1:
        print("usage: run_slice.py <slice.json> [--limit N]",
              file=sys.stderr)
        return 2
    slice_path = argv[0]
    limit = None
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])

    instances = json.load(open(slice_path))
    if limit is not None:
        instances = instances[:limit]
    print(f"Running {len(instances)} instance(s)", flush=True)

    base_url = os.environ["OPENAI_API_BASE_URL"]
    model = os.environ["OPENAI_COMPATIBLE_MODEL"]
    api_key = os.environ["OPENAI_API_KEY"]
    sandbox_url = os.environ.get("FLASH_SANDBOX_URL",
                                 "https://sandbox.swissai.cscs.ch")

    agent = SWEBenchAgent(
        url=base_url.rstrip("/") + "/chat/completions",
        model=model,
        api_key=api_key,
        step_limit=40,
        timeout_per_step_s=120,
        total_wall_s=2400,
        temperature=0.0,
        max_tokens=4096,
        sandbox_url=sandbox_url,
        sandbox_type="kubernetes",
        cpu_cores=2.0,
        memory_mb=4096,
        sandbox_ttl_seconds=3600,
        # Per-instance eval images from the ghcr swe-images mirror.
        image_org="swe-images",
        image_registry="ghcr.io",
        grade=True,
        eval_timeout_s=1800,
        create_retry_attempts=20,
        create_retry_delay_s=8.0,
    )
    trajectories_dir = Path("/tmp/swe_bench_trajectories")
    rows: list[dict] = []
    try:
        for inst in instances:
            rows.append(await run_one(agent, inst, trajectories_dir))
    finally:
        await agent.aclose()

    n_pass = sum(1 for r in rows if r["ok"])
    print("\n========== Summary ==========")
    for r in rows:
        verdict = "PASS" if r["ok"] else ("CRASH" if r.get("crashed") else "FAIL")
        f2p = f"{r.get('f2p_pass',0)}/{r.get('f2p_total',0)}"
        p2p = f"{r.get('p2p_pass',0)}/{r.get('p2p_total',0)}"
        print(f"  {verdict:6s}  {r['instance_id']:30s}  "
              f"F2P {f2p}  P2P {p2p}  "
              f"({r.get('exit_status') or r.get('error', '')})  "
              f"{r['elapsed_s']:.1f}s")
    print(f"\nPass rate: {n_pass}/{len(rows)} = "
          f"{n_pass / max(len(rows), 1) * 100:.0f}%")

    Path("/tmp/swe_bench_results.json").write_text(json.dumps(rows, indent=2))
    print("Wrote /tmp/swe_bench_results.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))

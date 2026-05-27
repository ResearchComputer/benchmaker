"""Run a small SWE-bench Verified slice through SWEBenchAgent + Flash Sandbox.

Picks instances from a JSON manifest (use `prepare_slice.py` or the snippet in
the README to generate one) and prints per-instance pass/fail + aggregate pass
rate. This is a smoke check, NOT the official SWE-bench score — see the
docstring of `swe_bench_agent.py` for caveats.

Run:
    python examples/coding_agent/run_swe_bench_slice.py /tmp/swe_bench_slice.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from benchmaker import AgentContext
from examples.coding_agent.swe_bench_agent import SWEBenchAgent


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
    # Save the model trajectory for diagnosis.
    if getattr(agent, "last_messages", None):
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        out_path = trajectories_dir / f"{instance_id}.json"
        out_path.write_text(json.dumps({
            "instance_id": instance_id,
            "exit_status": meta.get("exit_status"),
            "submission": result.output,
            "messages": agent.last_messages,
        }, indent=2))
        print(f"  trajectory: {out_path}", flush=True)
    print(f"  exit_status={meta['exit_status']}  bootstrap_ok={meta['bootstrap_ok']}"
          f"  diff={meta['submitted_diff']}  applied={meta['pred_patch_applied']}",
          flush=True)
    print(f"  steps={int(mx['steps'])} actions={int(mx['actions'])}  "
          f"F2P {int(mx['f2p_pass'])}/{int(mx['f2p_total'])}  "
          f"P2P {int(mx['p2p_pass'])}/{int(mx['p2p_total'])}  "
          f"elapsed={elapsed:.1f}s", flush=True)
    if not meta["bootstrap_ok"]:
        tail = meta.get("bootstrap_log_tail") or ""
        print(f"  bootstrap_log_tail:\n{_indent(tail)}", flush=True)
    if meta.get("grading_log_tail"):
        print(f"  grading_log_tail:\n{_indent(meta['grading_log_tail'])}",
              flush=True)
    return {
        "instance_id": instance_id, "ok": result.ok,
        "exit_status": meta["exit_status"],
        "bootstrap_ok": meta["bootstrap_ok"],
        "submitted_diff": meta["submitted_diff"],
        "pred_patch_applied": meta["pred_patch_applied"],
        "f2p_pass": int(mx["f2p_pass"]), "f2p_total": int(mx["f2p_total"]),
        "p2p_pass": int(mx["p2p_pass"]), "p2p_total": int(mx["p2p_total"]),
        "f2p_outcomes": meta.get("f2p_outcomes") or {},
        "p2p_outcomes": meta.get("p2p_outcomes") or {},
        "elapsed_s": elapsed,
        "output_head": (result.output or "")[:600],
    }


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in (text or "").splitlines())


async def main(argv: list[str]) -> int:
    _load_env()
    if len(argv) < 1:
        print("usage: run_swe_bench_slice.py <slice.json> [--limit N]",
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
        step_limit=20,
        timeout_per_step_s=60,
        total_wall_s=900,
        temperature=0.1,
        max_tokens=2048,
        sandbox_url=sandbox_url,
        sandbox_spec={"cpu_cores": 1.0, "memory_mb": 2048},
        sandbox_ttl_seconds=1200,
        bootstrap_timeout_s=600,
        pytest_timeout_s=300,
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

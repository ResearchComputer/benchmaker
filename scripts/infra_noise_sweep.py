#!/usr/bin/env python3
"""infra_noise_sweep.py — quantify flash-sandbox infrastructure noise, model-free.

The SWE-bench harness mixes the model's capability with infrastructure noise:
sandbox crashes, grading flakiness, and — the subtle one — *silent observation
corruption*, where the model issues a reasonable command but congestion returns
truncated / empty / wrong output that the model then trusts. Passive trajectory
mining cannot isolate this (a command legitimately returns different output when
the model mutates state), so this driver sends commands whose *correct output is
known and fixed* through the same flash-sandbox infrastructure. Any deviation is
100% infrastructure.

Two independent axes isolate the two failure families (see
docs/infra-noise-experiment.md):

    --concurrency  number of simultaneous sandboxes (CONTROL-PLANE load ->
                   HTTP 502 crashes; the loud channel)
    --stress       in-sandbox CPU/memory contention (DATA-PLANE pressure ->
                   silent output corruption / spurious timeouts; the quiet
                   channel that misguides the model)

Probes
    1. Deterministic-command canary: a battery of commands with fixed, known
       outputs (per-sandbox nonce for cross-talk, line counts, checksums, a big
       integer, and `sleep; echo DONE` under a short server-side timeout to
       catch the "killed but looks successful" silent-truncation case).
    2. Shadow-replay: each read-only canary runs twice back-to-back; differing
       output => a non-reproducible observation under load.

Outputs, per (concurrency, stress) cell, a results dir with config.json,
metrics.csv (one row per probe execution) and summary.json, following the lab
experiment conventions.

Usage
    # smoke: 1 sandbox, no stress
    .venv/bin/python tools/scripts/infra_noise_sweep.py --concurrency 1 --rounds 2

    # the 2x2 isolation matrix
    .venv/bin/python tools/scripts/infra_noise_sweep.py \
        --concurrency 1,8,32 --stress off,cpu,mem --rounds 5

Reads FLASH_SANDBOX_URL from the environment or repo .env.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

from flash_sandbox import AsyncHTTPClient, AsyncSandbox, SandboxHTTPError

# --------------------------------------------------------------------------- #
# Verdicts: how a single probe execution turned out.
# --------------------------------------------------------------------------- #
OK = "ok"                          # output matched ground truth
TIMEOUT_CLEAN = "timeout_clean"    # command killed by server timeout, loud (good)
COMPLETED = "completed"            # sleep probe finished (server didn't enforce timeout)
EMPTY = "empty"                    # expected output, got nothing
TRUNCATED = "truncated"            # output is a strict prefix of ground truth
WRONG = "wrong"                    # output non-empty and not a prefix
CROSSTALK = "crosstalk"            # got another sandbox's nonce -> mis-routing
SILENT_TRUNCATION = "silent_trunc" # timed-out command returned empty + exit 0
SPURIOUS_FAIL = "spurious_fail"    # a clean-passing computation failed under contention
EXEC_ERROR = "exec_error"          # unexpected exception (not a transient 50x)
CRASH = "crash"                    # transient HTTP 50x / unreachable (control-plane)

# Transport corruption: the exec path silently handed the model a bad observation.
CORRUPTION = {EMPTY, TRUNCATED, WRONG, CROSSTALK, SILENT_TRUNCATION}
# In-container contention damage: a real command that passes clean but fails/times
# out under resource starvation (a real-looking failure the model would trust).
SPURIOUS = {SPURIOUS_FAIL}
# Verdicts where the infra behaved or failed *loudly* (model not misled).
CLEAN = {OK, TIMEOUT_CLEAN, COMPLETED}


# --------------------------------------------------------------------------- #
# Deterministic-command battery. Each probe knows its own ground truth.
# --------------------------------------------------------------------------- #
@dataclass
class Probe:
    name: str
    readonly: bool
    timeout_ms: Optional[int]
    build: Callable[[str], str]                 # nonce -> shell command
    classify: Callable[[Any, str], str]         # (ExecResult, nonce) -> verdict


def _exact(expected: str) -> Callable[[Any, str], str]:
    def f(res: Any, nonce: str) -> str:
        out = (res.stdout or "").strip()
        if out == expected:
            return OK
        if out == "":
            return EMPTY
        if expected.startswith(out):
            return TRUNCATED
        return WRONG
    return f


def _classify_nonce(res: Any, nonce: str) -> str:
    expected = f"CANARY-{nonce}"
    out = (res.stdout or "").strip()
    if out == expected:
        return OK
    if out == "":
        return EMPTY
    if out.startswith("CANARY-") and out != expected:
        return CROSSTALK          # a different sandbox's nonce came back
    if expected.startswith(out):
        return TRUNCATED
    return WRONG


def _classify_lines(n: int, last: str) -> Callable[[Any, str], str]:
    def f(res: Any, nonce: str) -> str:
        out = (res.stdout or "")
        lines = out.splitlines()
        if not lines:
            return EMPTY
        if len(lines) == n and lines[-1] == last:
            return OK
        if lines[0] == "LINE00000" and len(lines) < n:
            return TRUNCATED
        return WRONG
    return f


def _classify_spurious(ok_token: str) -> Callable[[Any, str], str]:
    """Probe that must succeed clean; a failure under contention is spurious.

    Clean baseline (no stress) MUST yield ``ok`` or the probe is mis-sized.
    Under contention the same command failing (OOM kill, deadline exceeded) is a
    real-looking failure the model would trust => SPURIOUS_FAIL.
    """
    def f(res: Any, nonce: str) -> str:
        out = (res.stdout or "")
        if ok_token in out and res.exit_code == 0:
            return OK
        return SPURIOUS_FAIL
    return f


def _classify_sleep(res: Any, nonce: str) -> str:
    out = (res.stdout or "")
    if "DONE" in out:
        return COMPLETED          # server did not enforce timeout_ms; ran full 30s
    if res.exit_code != 0:
        return TIMEOUT_CLEAN      # killed loudly -> model would see an error
    if out.strip() == "":
        return SILENT_TRUNCATION  # killed but exit 0 + empty -> looks successful (!)
    return WRONG


def build_battery() -> list[Probe]:
    big_int = str(2 ** 4000)
    seq_blob = ("\n".join(str(i) for i in range(1, 100001)) + "\n").encode()
    seq_sha = hashlib.sha256(seq_blob).hexdigest()
    return [
        Probe("nonce_echo", True, 10_000,
              lambda n: f"echo CANARY-{n}", _classify_nonce),
        Probe("line_count", True, 30_000,
              lambda n: "seq 1 5000 | wc -l", _exact("5000")),
        Probe("big_int", True, 30_000,
              lambda n: 'python3 -c "print(2**4000)"', _exact(big_int)),
        Probe("checksum", True, 60_000,
              lambda n: "seq 1 100000 | sha256sum | cut -c1-64", _exact(seq_sha)),
        Probe("multiline", True, 30_000,
              lambda n: "python3 -c \"[print('LINE%05d' % i) for i in range(2000)]\"",
              _classify_lines(2000, "LINE01999")),
        # Spurious-failure probes: pass clean, fail under contention -> a
        # real-looking failure the model would trust. mem_alloc OOMs when the
        # ~900MB alloc + the mem stressor balloon exceed the sandbox limit;
        # cpu_deadline "times out" (like a test) when CPU burners starve it.
        # Both MUST be `ok` in the stress=off baseline.
        Probe("mem_alloc", True, 30_000,
              lambda n: ('python3 -c "b=bytearray(900*1024*1024);'
                         ' [b.__setitem__(i,1) for i in range(0,len(b),4096)];'
                         ' print(\'ALLOC_OK\')"'),
              _classify_spurious("ALLOC_OK")),
        Probe("cpu_deadline", True, 30_000,
              lambda n: ('python3 -c "import time;t=time.time();x=0\n'
                         'while x<15000000: x+=1\n'
                         "print('DEADLINE_OK' if time.time()-t<3.0 "
                         "else 'SLOW:%.2f'%(time.time()-t))\""),
              _classify_spurious("DEADLINE_OK")),
        # Not read-only-replayed: probes the killed-but-looks-ok case. The
        # cluster does not kill promptly at timeout_ms (it returns exit=-1 +
        # empty stdout only once the command finishes), so keep the sleep short
        # to bound cost; we still detect a silent exit-0-empty result if it ever
        # occurs under contention.
        Probe("sleep_timeout", False, 3_000,
              lambda n: "sh -c 'sleep 8; echo DONE'", _classify_sleep),
    ]


# --------------------------------------------------------------------------- #
# In-sandbox stressor (data-plane contention). Detached so it survives the
# one-shot exec that launches it; the container's `sleep infinity` keeps it alive.
# --------------------------------------------------------------------------- #
_STRESSOR_PY = r'''
import multiprocessing as mp, sys, time
W = int(sys.argv[1]); MB = int(sys.argv[2]); D = float(sys.argv[3])
def burn():
    x = 0
    while True:
        x = (x + 1) % 1_000_000
procs = [mp.Process(target=burn, daemon=True) for _ in range(W)]
for p in procs:
    p.start()
held = []
if MB > 0:
    try:
        b = bytearray(MB * 1024 * 1024)
        for i in range(0, len(b), 4096):
            b[i] = 1            # touch pages so they are resident
        held.append(b)
    except MemoryError:
        pass
time.sleep(D)
'''
_STRESSOR_PATH = "/tmp/infra_stressor.py"
_STRESSOR_MARK = "infra_stressor.py"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    url: str
    concurrency_list: list[int]
    stress_list: list[str]
    rounds: int
    image: str
    backend: str
    memory_mb: int
    cpu_cores: float
    stress_cpu_workers: int
    stress_mem_mb: int
    stress_duration_s: int
    cmd_http_timeout_s: float
    seed: int
    run_id: str
    out_root: str
    keep_sandboxes: bool


# --------------------------------------------------------------------------- #
# One probe execution -> a metrics row.
# --------------------------------------------------------------------------- #
async def _exec(client: AsyncHTTPClient, sid: str, cmd: str,
                timeout_ms: Optional[int]) -> tuple[Optional[Any], Optional[str]]:
    """Return (ExecResult, None) or (None, crash_reason)."""
    try:
        res = await client.exec_command(sid, cmd, timeout_ms=timeout_ms)
        return res, None
    except SandboxHTTPError as e:
        return None, f"http_{getattr(e, 'status_code', '?')}"
    except asyncio.TimeoutError:
        return None, "client_timeout"
    except Exception as e:  # noqa: BLE001 — record, don't crash the sweep
        return None, f"exc_{type(e).__name__}"


async def _run_probe(client, sid, probe: Probe, nonce: str, worker: int,
                     rnd: int) -> dict:
    res, crash = await _exec(client, sid, probe.build(nonce), probe.timeout_ms)
    row = {
        "worker": worker, "round": rnd, "probe": probe.name,
        "verdict": CRASH if crash else probe.classify(res, nonce),
        "crash_reason": crash or "",
        "exit_code": "" if res is None else res.exit_code,
        "elapsed_ms": "" if res is None else round(res.elapsed_ms, 1),
        "out_len": 0 if res is None else len(res.stdout or ""),
        "replay_verdict": "", "replay_stable": "",
    }
    # Shadow-replay: read-only probe, re-run and diff.
    if probe.readonly and not crash:
        res2, crash2 = await _exec(client, sid, probe.build(nonce), probe.timeout_ms)
        if crash2:
            row["replay_verdict"] = CRASH
        else:
            row["replay_verdict"] = probe.classify(res2, nonce)
            row["replay_stable"] = ((res.stdout or "") == (res2.stdout or ""))
    return row


# --------------------------------------------------------------------------- #
# Worker: own sandbox, optional stressor, run the battery R rounds.
# --------------------------------------------------------------------------- #
async def _start_stressor(client, sb, cfg: Config, stress: str) -> bool:
    workers = cfg.stress_cpu_workers
    mem = cfg.stress_mem_mb if stress == "mem" else 0
    try:
        await sb.write_files(
            [{"path": _STRESSOR_PATH, "content": _STRESSOR_PY.encode()}],
            parents=True,
        )
        # Background + echo $! to capture the PID; a backgrounded process
        # reparents to PID 1 (`sleep infinity`) and survives the one-shot exec.
        # nohup guards against SIGHUP; pgrep is absent in slim images so we
        # verify liveness with the shell builtin `kill -0` against the PID.
        launch = (
            f"nohup python3 {_STRESSOR_PATH} "
            f"{workers} {mem} {cfg.stress_duration_s} >/dev/null 2>&1 & echo $!"
        )
        r, _c = await _exec(client, sb.id, launch, 10_000)
        pid = ""
        if r is not None:
            toks = [t for t in (r.stdout or "").split() if t.isdigit()]
            pid = toks[-1] if toks else ""
        if not pid:
            return False
        await asyncio.sleep(1.0)
        chk, _ = await _exec(
            client, sb.id, f"kill -0 {pid} 2>/dev/null && echo ALIVE || echo DEAD", 10_000)
        return chk is not None and "ALIVE" in (chk.stdout or "")
    except Exception:
        return False


async def _run_worker(client, cfg: Config, stress: str, worker: int,
                      battery: list[Probe], rows: list[dict],
                      meta: dict) -> None:
    nonce = f"{cfg.run_id}-w{worker:03d}"
    sb = None
    try:
        sb = await AsyncSandbox.create(
            client, type=cfg.backend, image=cfg.image,
            command=["sleep", "infinity"],
            memory_mb=cfg.memory_mb, cpu_cores=cfg.cpu_cores,
        )
    except Exception as e:  # noqa: BLE001 — control-plane crash on create
        meta["create_crashes"] += 1
        rows.append({"worker": worker, "round": -1, "probe": "<create>",
                     "verdict": CRASH, "crash_reason": f"create_{type(e).__name__}",
                     "exit_code": "", "elapsed_ms": "", "out_len": 0,
                     "replay_verdict": "", "replay_stable": ""})
        return

    try:
        # Readiness: container may need a beat before exec works.
        for _ in range(10):
            r, _c = await _exec(client, sb.id, "echo ready", 10_000)
            if r is not None and "ready" in (r.stdout or ""):
                break
            await asyncio.sleep(1.0)

        if stress != "off":
            meta["stress_started"] += int(await _start_stressor(client, sb, cfg, stress))

        for rnd in range(cfg.rounds):
            for probe in battery:
                rows.append(await _run_probe(client, sb.id, probe, nonce, worker, rnd))
    finally:
        try:
            if stress != "off":
                await _exec(client, sb.id, f"pkill -f {_STRESSOR_MARK} || true", 10_000)
        except Exception:
            pass
        if sb is not None and not cfg.keep_sandboxes:
            try:
                await client.stop_sandbox(sb.id, cleanup=True)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# One cell of the (concurrency, stress) matrix.
# --------------------------------------------------------------------------- #
async def _run_cell(cfg: Config, concurrency: int, stress: str) -> dict:
    battery = build_battery()
    rows: list[dict] = []
    meta = {"create_crashes": 0, "stress_started": 0}
    client = AsyncHTTPClient(address=cfg.url, timeout=cfg.cmd_http_timeout_s)
    try:
        await asyncio.gather(*[
            _run_worker(client, cfg, stress, w, battery, rows, meta)
            for w in range(concurrency)
        ])
    finally:
        close = getattr(client, "close", None) or getattr(client, "aclose", None)
        if close:
            try:
                maybe = close()
                if asyncio.iscoroutine(maybe):
                    await maybe
            except Exception:
                pass

    summary = _summarize(rows, meta, concurrency, stress, cfg)
    _write_cell(cfg, concurrency, stress, rows, summary)
    return summary


def _summarize(rows, meta, concurrency, stress, cfg: Config) -> dict:
    primary = [r for r in rows if r["round"] >= 0]           # exclude <create>
    n = len(primary)
    vc: dict[str, int] = {}
    for r in primary:
        vc[r["verdict"]] = vc.get(r["verdict"], 0) + 1
    corruption = sum(vc.get(v, 0) for v in CORRUPTION)
    crashes = vc.get(CRASH, 0)
    executed = n - crashes - vc.get(EXEC_ERROR, 0)
    # Spurious-failure probes (in-container contention channel), tracked apart
    # from transport corruption.
    spur_probes = {"mem_alloc", "cpu_deadline"}
    spur = [r for r in primary if r["probe"] in spur_probes and r["verdict"] != CRASH]
    spur_fail = sum(1 for r in spur if r["verdict"] == SPURIOUS_FAIL)
    # Latency over the fast read-only probes only; the sleep probe's elapsed is
    # an intentional fixed delay, not infra latency, so it would skew p99/max.
    lat = [r["elapsed_ms"] for r in primary
           if r["probe"] != "sleep_timeout"
           and isinstance(r["elapsed_ms"], (int, float))]
    replayed = [r for r in primary if r["replay_stable"] != ""]
    unstable = [r for r in replayed if r["replay_stable"] is False]

    def pct(a, b):
        return round(100.0 * a / b, 2) if b else 0.0

    return {
        "run_id": cfg.run_id,
        "concurrency": concurrency,
        "stress": stress,
        "workers": concurrency,
        "create_crashes": meta["create_crashes"],
        "stress_started": meta["stress_started"],
        "probe_runs": n,
        "verdicts": vc,
        "OCR_pct": pct(corruption, executed),          # observation corruption rate
        "spurious_fail_pct": pct(spur_fail, len(spur)),  # contention-induced failures
        "spurious_runs": len(spur),
        "crash_pct": pct(crashes + meta["create_crashes"], n + meta["create_crashes"]),
        "replay_instability_pct": pct(len(unstable), len(replayed)),
        "latency_ms": {
            "p50": round(statistics.median(lat), 1) if lat else None,
            "p99": round(sorted(lat)[int(0.99 * (len(lat) - 1))], 1) if lat else None,
            "max": round(max(lat), 1) if lat else None,
        },
    }


# --------------------------------------------------------------------------- #
# Output (lab conventions: config.json, metrics.csv, summary.json per cell).
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _cell_dir(cfg: Config, concurrency: int, stress: str) -> Path:
    d = Path(cfg.out_root) / f"{cfg.run_id}_c{concurrency}_{stress}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_cell(cfg, concurrency, stress, rows, summary) -> None:
    d = _cell_dir(cfg, concurrency, stress)
    fields = ["worker", "round", "probe", "verdict", "crash_reason", "exit_code",
              "elapsed_ms", "out_len", "replay_verdict", "replay_stable"]
    with open(d / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (d / "summary.json").write_text(json.dumps(summary, indent=2))


def _write_run_config(cfg: Config) -> None:
    root = Path(cfg.out_root)
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "git_commit": _git_commit(),
        "config": asdict(cfg),
    }
    (root / f"{cfg.run_id}_config.json").write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=None,
                   help="flash-sandbox URL (default: $FLASH_SANDBOX_URL).")
    p.add_argument("--concurrency", default="1", type=_parse_int_list,
                   help="comma list of simultaneous sandboxes, e.g. 1,8,32.")
    p.add_argument("--stress", default="off", type=_parse_str_list,
                   help="comma list of contention levels: off,cpu,mem.")
    p.add_argument("--rounds", type=int, default=3,
                   help="battery repeats per worker (per cell).")
    p.add_argument("--image", default="python:3.11-slim",
                   help="sandbox image (use a swe-bench image for representativeness).")
    p.add_argument("--backend", default=os.environ.get("FLASH_BACKEND", "docker"))
    p.add_argument("--memory-mb", type=int, default=2048)
    p.add_argument("--cpu-cores", type=float, default=1.0)
    p.add_argument("--stress-cpu-workers", type=int, default=4)
    p.add_argument("--stress-mem-mb", type=int, default=1536,
                   help="memory balloon for --stress mem (keep < memory-mb).")
    p.add_argument("--stress-duration-s", type=int, default=300)
    p.add_argument("--cmd-http-timeout-s", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default=None,
                   help="default: YYYYMMDD_infranoise_seed<seed>.")
    p.add_argument("--out-root", default="results")
    p.add_argument("--keep-sandboxes", action="store_true",
                   help="do not delete sandboxes after the run (debug).")
    a = p.parse_args(argv)

    if load_dotenv:
        load_dotenv()
    url = a.url or os.environ.get("FLASH_SANDBOX_URL")
    if not url:
        p.error("no flash-sandbox URL: pass --url or set FLASH_SANDBOX_URL.")
    run_id = a.run_id or (
        f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%d}_infranoise_seed{a.seed}")
    bad = [s for s in a.stress if s not in ("off", "cpu", "mem")]
    if bad:
        p.error(f"unknown --stress level(s): {bad}; choose off,cpu,mem.")
    return Config(
        url=url, concurrency_list=a.concurrency, stress_list=a.stress,
        rounds=a.rounds, image=a.image, backend=a.backend, memory_mb=a.memory_mb,
        cpu_cores=a.cpu_cores, stress_cpu_workers=a.stress_cpu_workers,
        stress_mem_mb=a.stress_mem_mb, stress_duration_s=a.stress_duration_s,
        cmd_http_timeout_s=a.cmd_http_timeout_s, seed=a.seed, run_id=run_id,
        out_root=a.out_root, keep_sandboxes=a.keep_sandboxes,
    )


def _print_table(summaries: list[dict]) -> None:
    print("\n" + "=" * 90)
    print(f"{'conc':>5} {'stress':>6} {'runs':>5} {'OCR%':>6} {'spurFail%':>9} "
          f"{'crash%':>7} {'replayUnst%':>11} {'p50ms':>7} {'p99ms':>8}")
    print("-" * 90)
    for s in summaries:
        lat = s["latency_ms"]
        print(f"{s['concurrency']:>5} {s['stress']:>6} {s['probe_runs']:>5} "
              f"{s['OCR_pct']:>6} {s['spurious_fail_pct']:>9} {s['crash_pct']:>7} "
              f"{s['replay_instability_pct']:>11} "
              f"{str(lat['p50']):>7} {str(lat['p99']):>8}")
    print("=" * 90)
    print("OCR       = transport corruption (truncated/empty/wrong/crosstalk/silent-trunc).")
    print("spurFail% = clean-passing mem/cpu probe that failed under contention.")
    print("crash%    = create failures + exec 50x (control-plane).")


async def _main_async(cfg: Config) -> None:
    _write_run_config(cfg)
    print(f"[infra_noise_sweep] run_id={cfg.run_id} url={cfg.url}")
    print(f"  matrix: concurrency={cfg.concurrency_list} x stress={cfg.stress_list}"
          f"  rounds={cfg.rounds} image={cfg.image}")
    summaries: list[dict] = []
    # Sequential over cells so each cell's concurrency is measured in isolation.
    for stress in cfg.stress_list:
        for conc in cfg.concurrency_list:
            print(f"\n>> cell concurrency={conc} stress={stress} ...")
            s = await _run_cell(cfg, conc, stress)
            print(f"   OCR={s['OCR_pct']}%  crash={s['crash_pct']}%  "
                  f"replay_unstable={s['replay_instability_pct']}%  "
                  f"verdicts={s['verdicts']}")
            summaries.append(s)
    (Path(cfg.out_root) / f"{cfg.run_id}_sweep_summary.json").write_text(
        json.dumps(summaries, indent=2))
    _print_table(summaries)
    print(f"\nresults under: {Path(cfg.out_root).resolve()}/{cfg.run_id}_*")


def main(argv=None) -> int:
    cfg = parse_args(argv)
    asyncio.run(_main_async(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())

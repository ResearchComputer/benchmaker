set -uo pipefail

# --- config (override via env) ---------------------------------------------
: "${FLASH_SANDBOX_URL:=http://100.71.204.79:8080}"
: "${REACHABLE_HOST:=100.71.204.79}"
: "${TRAJECTORIES:=.local/pi-host-traj-v1-500.jsonl}"
: "${N_TASKS:=100}"
: "${EXCLUDE_TASKS:=psf__requests-2317}"
: "${TIMEOUTS:=0.00001}"
: "${CONCURRENCIES:=100}"
OUT_ROOT="${OUT_ROOT:-jobs}"

export FLASH_SANDBOX_URL

exclude_args=()
for t in ${EXCLUDE_TASKS}; do exclude_args+=(--exclude-task "${t}"); done

echo "Real timeout x concurrency sweep (pi-host)  n_tasks=${N_TASKS}"
echo "  T grid:           ${TIMEOUTS}"
echo "  concurrency grid: ${CONCURRENCIES}"
echo "  excluding:        ${EXCLUDE_TASKS:-<none>}"
echo

# --- run one replay per (T, C) cell ----------------------------------------
for T in ${TIMEOUTS}; do
    # Normalize T the same way the summary does (Python f"{T:g}"), so the
    # directory created here matches the one the summary stats. Without this,
    # e.g. T=0.00001 creates timeout_T0.00001_c100 but the summary looks up
    # timeout_T1e-05_c100 -> "(not run)".
    Tg=$(python3 -c "print(f'{float(\"${T}\"):g}')")
    for C in ${CONCURRENCIES}; do
        jobs_dir="${OUT_ROOT}/timeout_T${Tg}_c${C}"
        echo "=== T=${T}s  c=${C}  -> ${jobs_dir} ==="
        benchmaker swebench-replay \
            --trajectories "${TRAJECTORIES}" \
            --mode pi-host \
            --route-tools all \
            --exec-timeout-sec "${T}" \
            --concurrency "${C}" \
            --reachable-host "${REACHABLE_HOST}" \
            --host 0.0.0.0 \
            --n-tasks "${N_TASKS}" \
            "${exclude_args[@]}" \
            --jobs-dir "${jobs_dir}" \
            || echo "WARNING: T=${T} c=${C} run exited non-zero; continuing"
        echo
    done
done

# --- summary: solved + timeouts + duration tail per cell -------------------
echo "=== solved / timeouts / duration tail per (T, c) cell ==="
OUT_ROOT="${OUT_ROOT}" python3 - "${TIMEOUTS}" "${CONCURRENCIES}" <<'PY'
import glob
import json
import os
import sys

timeouts = [float(x) for x in sys.argv[1].split()]
concurrencies = [int(x) for x in sys.argv[2].split()]
out_root = os.environ.get("OUT_ROOT", "jobs")


def _reward(result_json):
    try:
        with open(result_json) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return (data.get("verifier_result") or {}).get("rewards", {}).get("reward")


def _solved(jobs_dir):
    solved = total = 0
    for rj in glob.glob(os.path.join(jobs_dir, "**", "result.json"), recursive=True):
        r = _reward(rj)
        if r is None:
            continue
        total += 1
        solved += 1 if r == 1.0 else 0
    return solved, total


def _spans(jobs_dir, T):
    # A real exec timeout shows up as a bridge span with rc<0 whose duration
    # ran up to the wall; rc<0 with ~0 duration is an instant exec error, not a
    # timeout, so gate on duration to avoid conflating the two.
    durs, n_exec, n_timeout = [], 0, 0
    for sp in glob.glob(os.path.join(jobs_dir, "**", "timeline-spans.jsonl"),
                        recursive=True):
        with open(sp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("name") != "sandbox_exec":
                    continue
                d = float(obj.get("duration_s") or 0.0)
                n_exec += 1
                durs.append(d)
                if int(obj.get("rc", 0)) < 0 and d >= 0.9 * T:
                    n_timeout += 1
    durs.sort()
    p95 = durs[int(0.95 * (len(durs) - 1))] if durs else 0.0
    mx = durs[-1] if durs else 0.0
    return n_exec, n_timeout, p95, mx


hdr = f"{'T(s)':>6} {'c':>5} {'solved':>9} {'timeouts':>9} {'execs':>7} {'p95(s)':>8} {'max(s)':>8}"
print(hdr)
print("-" * len(hdr))
for T in timeouts:
    for C in concurrencies:
        jobs_dir = os.path.join(out_root, f"timeout_T{T:g}_c{C}")
        if not os.path.isdir(jobs_dir):
            print(f"{T:>6g} {C:>5} {'(not run)':>9}")
            continue
        s, n = _solved(jobs_dir)
        n_exec, n_to, p95, mx = _spans(jobs_dir, T)
        solved = f"{s}/{n}" if n else "(empty)"
        print(f"{T:>6g} {C:>5} {solved:>9} {n_to:>9} {n_exec:>7} {p95:>8.1f} {mx:>8.1f}")
PY

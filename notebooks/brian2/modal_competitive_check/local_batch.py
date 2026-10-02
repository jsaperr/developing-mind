"""Laptop side of the competitive fidelity check: runs the SAME function bodies as modal_competitive_check.py
(imported, so the code is identical), one subprocess per seed, 8 concurrent (the arc-06 sweet spot), writing
out/laptop_<arm>_seed<N>.json.gz. A failed seed writes status 'failed'. Resume-safe: completed files are skipped.

Usage (repo root):  python notebooks/brian2/modal_competitive_check/local_batch.py          (orchestrator)
                    python notebooks/brian2/modal_competitive_check/local_batch.py <arm> <seed>   (one job)
"""
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAP = 8


def one(arm, seed):
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
    import modal_competitive_check as M
    body = {"probe": lambda s: M._stg_body(s, 5.0, 1.0), "stg": lambda s: M._stg_body(s, 600.0, 1.0), "op": M._op_body}[arm]
    try:
        r = body(seed)
    except Exception as e:  # never leave a truncated file that looks complete
        r = dict(status="failed", seed=seed, error=repr(e))
    M.save("laptop", arm, r)
    print(f"{arm} {seed} {r['status']}", flush=True)


def done(arm, seed):
    f = HERE / "out" / f"laptop_{arm}_seed{seed}.json.gz"
    if not f.exists():
        return False
    try:
        with gzip.open(f, "rt") as fh:
            return json.load(fh).get("status") == "completed"
    except Exception:
        return False


def orchestrate():
    sys.path.insert(0, str(HERE))
    from modal_competitive_check import OP_SEEDS, PROBE_SEEDS, STG_SEEDS
    jobs = [("probe", s) for s in PROBE_SEEDS] + [("op", s) for s in OP_SEEDS] + [("stg", s) for s in STG_SEEDS]
    pending = [j for j in jobs if not done(*j)]
    print(f"{len(jobs) - len(pending)} done, {len(pending)} to run", flush=True)
    running, t0 = {}, time.time()
    while pending or running:
        while pending and len(running) < CAP:
            arm, s = pending.pop(0)
            running[(arm, s)] = subprocess.Popen([sys.executable, "-u", __file__, arm, str(s)],
                                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ROOT))
        for k, p in list(running.items()):
            if p.poll() is not None:
                print(f"finished {k} rc={p.returncode} at {time.time() - t0:.0f}s", flush=True)
                del running[k]
        time.sleep(3)
    print(f"LAPTOP BATCH DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        one(sys.argv[1], int(sys.argv[2]))
    else:
        orchestrate()

"""Boundary-mapping sweep: is the rich (genuine-reorganization) zone a single fragile point
(strong_tight_gate, 10mV/1.0) or a real, workable region? Per web's message: focused grid
between the two known operating points (10mV/1.0, where reorganization is real but
differentiation itself is only ~50% reliable; 13mV/1.5, where differentiation is 100% reliable at
both N=3 and N=7 but reorganization was never observed in any of 8 tested seeds).

Reuses run_competitive_seed.py directly (N_POST=3 fixed there, matching every prior batch at this
scale) -- only inhib_strength/gap_scale vary here, not n_post.

Two metrics tracked SEPARATELY per grid point, per web's explicit instruction -- not collapsed
into one richness score:
  - reliability: fraction of seeds classified 'differentiate' (classify_hierarchy).
  - reentry_rate: fraction of ALL seeds (not just differentiating ones) showing genuine tier
    reentry (detect_tier_reentry) -- an excluded neuron actually regaining top-tier membership
    after the initial settling window, not noise-trading among neurons that never left.

A real, stated methodological limitation, not discovered after the fact: the original
strong_tight_gate typology's one-time-reorganization events happened at t~1000-2600s (5003,
6007, 6001) -- well past this sweep's 600s duration. A 600s window can reliably catch the
"never settles at all" pattern (visible from the first few hundred seconds, per 6008/6009's
established timing) but will systematically UNDERCOUNT true late reorganization events, even at
10mV/1.0 itself. A 'reentry_rate' of 0 at any grid point therefore means "no reentry observed
within 600s," not "this setting never reorganizes at any timescale" -- stated here before
running, not added as a post-hoc excuse if the numbers look flat.

Falsification criteria, stated before running: the question is whether the reliability curve and
the reentry curve overlap anywhere in this grid -- a region with both meaningfully high
reliability (>=60%) and non-zero reentry rate -- or whether they trade off cleanly with no
overlap across the whole tested range. Either is a real, reportable answer.
"""
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONDA_PYTHON = r"C:\Users\urawi\miniconda3\envs\developing-mind\python.exe"
SEED_SCRIPT = REPO_ROOT / "notebooks" / "brian2" / "competitive_population_data" / "run_competitive_seed.py"
OUT_DIR = Path(__file__).resolve().parent
PROGRESS_PATH = OUT_DIR / "boundary_progress.json"

APRE_VAL = 0.005
DURATION_S = 600.0
INHIB_VALUES = [10.0, 11.0, 12.0, 13.0]        # mV -- spans the two known endpoints
GAP_SCALE_VALUES = [1.0, 1.17, 1.33, 1.5]      # spans the two known endpoints
N_SEEDS_PER_POINT = 8
SEED_BASE = 24000  # fresh block, no overlap with any prior batch (21000s/22000s/23000s used)
MAX_CONCURRENT = 6


def build_jobs():
    jobs = []
    seed_val = SEED_BASE
    for inhib, gap_scale in product(INHIB_VALUES, GAP_SCALE_VALUES):
        combo_name = f"inhib{inhib:g}_gap{gap_scale:.2f}"
        for _ in range(N_SEEDS_PER_POINT):
            out_path = OUT_DIR / f"boundary_{combo_name}_seed{seed_val}.json"
            jobs.append({
                "seed": seed_val, "inhib": inhib, "gap_scale": gap_scale,
                "combo_name": combo_name, "out_path": str(out_path),
            })
            seed_val += 1
    return jobs


def launch(job):
    log_path = OUT_DIR / f"boundary_{job['combo_name']}_seed{job['seed']}.log"
    args = [CONDA_PYTHON, "-u", str(SEED_SCRIPT), str(job["seed"]), str(APRE_VAL),
            str(job["inhib"]), str(job["gap_scale"]), str(DURATION_S), job["combo_name"], job["out_path"]]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(args, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    return proc, log_f


def main(max_jobs=None):
    jobs = build_jobs()
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
    total = len(jobs)

    progress = {
        "status": "running", "total_jobs": total, "completed": 0, "failed": 0,
        "started_at": time.time(),
        "grid": {"inhib_values": INHIB_VALUES, "gap_scale_values": GAP_SCALE_VALUES},
        "n_seeds_per_point": N_SEEDS_PER_POINT, "duration_s": DURATION_S,
    }
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"launching {total} jobs, max_concurrent={MAX_CONCURRENT}", flush=True)

    pending = list(jobs)
    running = []
    completed_jobs = []

    while pending or running:
        while pending and len(running) < MAX_CONCURRENT:
            job = pending.pop(0)
            proc, log_f = launch(job)
            running.append((proc, log_f, job, time.time()))

        time.sleep(2)
        still_running = []
        for proc, log_f, job, start_time in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, log_f, job, start_time))
            else:
                log_f.close()
                wall = time.time() - start_time
                if ret != 0:
                    progress["failed"] += 1
                progress["completed"] += 1
                completed_jobs.append({**job, "wall_s": wall, "exit_code": ret})
                progress["last_completed"] = completed_jobs[-1]
                progress["elapsed_s"] = time.time() - progress["started_at"]
                with open(PROGRESS_PATH, "w") as f:
                    json.dump(progress, f, indent=2)
                print(f"[{progress['completed']}/{total}] seed {job['seed']} "
                      f"({job['combo_name']}) -> exit {ret}, {wall:.1f}s", flush=True)
        running = still_running

    progress["status"] = "done"
    progress["elapsed_s"] = time.time() - progress["started_at"]
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"BOUNDARY SWEEP DONE -- {progress['completed']} jobs, {progress['failed']} failed, "
          f"{progress['elapsed_s']:.1f}s total", flush=True)


if __name__ == "__main__":
    max_jobs_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_jobs_arg)

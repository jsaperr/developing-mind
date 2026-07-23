"""Follow-up to the N-scaling curve (step 3), per web's Gate 2 response: not a new gate-worthy
design change, just more data at the same 600s scale. Original curve found a non-monotonic
tier-legibility pattern with the highest variance at N=10 (std_top_tier_frac=0.33 on n=8 seeds) --
before treating that as real rather than n=8 sampling noise (the same trap strong_tight_gate's
1/8 anchor point was a reminder of), get more seeds at N=10, plus N=8 and N=9 to see whether the
reversal from N=7 is a sharp cliff right at 10 or a gradual creep starting earlier.

N=10 here ADDS to the original 8 seeds (different seed block), bringing N=10 to 16 total once
combined with n_scaling_data/analyze_n_scaling.py's existing loader (which just globs all
nscale_n10_seed*.json files, no changes needed there).
"""
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONDA_PYTHON = r"C:\Users\urawi\miniconda3\envs\developing-mind\python.exe"
SEED_SCRIPT = REPO_ROOT / "notebooks" / "brian2" / "n_scaling_data" / "run_n_scaling_seed.py"
OUT_DIR = Path(__file__).resolve().parent
PROGRESS_PATH = OUT_DIR / "n_scaling_followup_progress.json"

APRE_VAL = 0.005
DURATION_S = 600.0
REFERENCE_INHIB_MV = 13.0
GAP_SCALE = 1.5
N_VALUES = [8, 9, 10]   # N=10 here is ADDITIONAL seeds, not a replacement of the original 8
N_SEEDS_PER_POINT = 8
SEED_BASE = 23000  # fresh block, no overlap with 21000-21127 (sweep+TestA) or 22000-22031 (original N-curve)
MAX_CONCURRENT = 6


def build_jobs():
    jobs = []
    seed_val = SEED_BASE
    for n_post in N_VALUES:
        combo_name = f"n{n_post}"
        for _ in range(N_SEEDS_PER_POINT):
            out_path = OUT_DIR / f"nscale_{combo_name}_seed{seed_val}.json"
            jobs.append({
                "seed": seed_val, "n_post": n_post, "combo_name": combo_name,
                "out_path": str(out_path),
            })
            seed_val += 1
    return jobs


def launch(job):
    log_path = OUT_DIR / f"nscale_{job['combo_name']}_seed{job['seed']}.log"
    args = [CONDA_PYTHON, "-u", str(SEED_SCRIPT), str(job["seed"]), str(APRE_VAL),
            str(job["n_post"]), str(REFERENCE_INHIB_MV), str(GAP_SCALE), str(DURATION_S),
            job["combo_name"], job["out_path"]]
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
        "started_at": time.time(), "n_values": N_VALUES, "n_seeds_per_point": N_SEEDS_PER_POINT,
        "duration_s": DURATION_S, "reference_inhib_mV": REFERENCE_INHIB_MV, "gap_scale": GAP_SCALE,
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
    print(f"N-SCALING FOLLOWUP DONE -- {progress['completed']} jobs, {progress['failed']} failed, "
          f"{progress['elapsed_s']:.1f}s total", flush=True)


if __name__ == "__main__":
    max_jobs_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_jobs_arg)

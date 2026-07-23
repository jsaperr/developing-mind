"""Population-competition bistability sweep: maps inhib_strength x gap_scale around
strong_tight_gate (10mV, 1.0), the operating point that showed ~50/50 differentiate/converge
across 14 seeds. Tracks two things per grid point (reliability and richness), not just a
converge/differentiate tally -- see experiments_brian2.md for the full design and web's message.

Python-level orchestrator (concurrency-limited subprocess pool), not a bash loop -- this batch
is 128 jobs, too many to manage safely with shell `&`/`wait` backgrounding (and this project hit
a real bash subshell-variable-scoping bug earlier when trying exactly that at smaller scale).
Each job is a fresh `python -u run_competitive_seed.py ...` subprocess, invoked with the
developing-mind conda env's interpreter explicitly (not whatever `python` resolves to on PATH --
see the memory note this session added after nearly pip-installing brian2 into the wrong env).

Run with: python run_sweep.py [max_jobs]
  max_jobs: optional, limits to the first N jobs -- used for a concurrency/timing calibration
  batch before committing to the full 128-run sweep.

Per the standing "background long scripts, write structured intermediate progress" habit
(principles.md): writes sweep_progress.json after every single job completion, not just at
start/end.
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
PROGRESS_PATH = OUT_DIR / "sweep_progress.json"

APRE_VAL = 0.005  # matches strong_tight_gate's known-stable operating point; not swept here
DURATION_S = 600.0  # calibration scale, matching the original bistability characterization --
                      # full 5000s extension is a deliberate later follow-up, only for whichever
                      # region this sweep flags as promising, not run at every grid point
INHIB_VALUES = [6.0, 8.0, 10.0, 13.0]      # mV -- spans the known converged corner (6) up past
                                             # the known 50/50 point (10)
GAP_SCALE_VALUES = [0.5, 1.0, 1.5, 2.0]    # spans tighter-than-known (0.5) through the 50/50
                                             # point (1.0) up to the known converged corner (2.0)
N_SEEDS_PER_POINT = 8
SEED_BASE = 21000  # fresh block, no overlap with 5001-5004/6001-6010/20000-20001(calibration)
MAX_CONCURRENT = 6  # matches the physical-core count noted in the seed-expansion wall-clock entry


def build_jobs():
    jobs = []
    seed_val = SEED_BASE
    for inhib, gap_scale in product(INHIB_VALUES, GAP_SCALE_VALUES):
        combo_name = f"inhib{inhib:g}_gap{gap_scale:g}"
        for _ in range(N_SEEDS_PER_POINT):
            out_path = OUT_DIR / f"sweep_{combo_name}_seed{seed_val}.json"
            jobs.append({
                "seed": seed_val, "inhib": inhib, "gap_scale": gap_scale,
                "combo_name": combo_name, "out_path": str(out_path),
            })
            seed_val += 1
    return jobs


def launch(job):
    log_path = OUT_DIR / f"sweep_{job['combo_name']}_seed{job['seed']}.log"
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
    print(f"SWEEP DONE -- {progress['completed']} jobs, {progress['failed']} failed, "
          f"{progress['elapsed_s']:.1f}s total", flush=True)


if __name__ == "__main__":
    max_jobs_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_jobs_arg)

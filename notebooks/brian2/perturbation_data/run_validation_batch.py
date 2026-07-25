"""MANDATORY validation, per web's explicit instruction: before trusting the perturbation method
on any unknown (intermediate) point, confirm it reproduces the ALREADY-KNOWN contrast between the
two reference points. If it can't, the method is untrustworthy and gets thrown out -- same
decisive-check discipline that correctly killed the 600s reentry probe.

strong_tight_gate (10mV/1.0): known-marginal, known to show genuine spontaneous reorganization at
5000s. Expectation: at least some seeds should flip under small-to-moderate perturbation
magnitudes (a shallow basin).

13mV/1.5: confirmed locked-in and permanent at both N=3 (Test A) and N=7 (step 4), zero
spontaneous reentry in 8 seeds total across two structurally different detection attempts.
Expectation: structure should resist even the largest tested magnitude (a deep, or literally
unbreakable within this ladder's range, basin) in most/all seeds.

4 seeds per point (not 8-10) -- these runs are far more expensive than the 600s probe (settling
alone can take up to MAX_SETTLE_S=1200s, plus up to 5 ladder rungs at 250s each), so this
validation batch deliberately trades seed count for being affordable at all, per web's explicit
sizing guidance.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONDA_PYTHON = r"C:\Users\urawi\miniconda3\envs\developing-mind\python.exe"
SEED_SCRIPT = Path(__file__).resolve().parent / "run_perturbation_seed.py"
OUT_DIR = Path(__file__).resolve().parent
PROGRESS_PATH = OUT_DIR / "validation_progress.json"

APRE_VAL = 0.005
POINTS = [
    {"name": "strong_tight_gate", "inhib": 10.0, "gap_scale": 1.0, "seeds": [26000, 26001, 26002, 26003]},
    {"name": "reliable_13_1p5", "inhib": 13.0, "gap_scale": 1.5, "seeds": [26010, 26011, 26012, 26013]},
]
MAX_CONCURRENT = 6


def build_jobs():
    jobs = []
    for point in POINTS:
        for seed in point["seeds"]:
            out_path = OUT_DIR / f"perturb_{point['name']}_seed{seed}.json"
            jobs.append({"seed": seed, "inhib": point["inhib"], "gap_scale": point["gap_scale"],
                         "point_name": point["name"], "out_path": str(out_path)})
    return jobs


def launch(job):
    log_path = OUT_DIR / f"perturb_{job['point_name']}_seed{job['seed']}.log"
    args = [CONDA_PYTHON, "-u", str(SEED_SCRIPT), str(job["seed"]), str(job["inhib"]),
            str(job["gap_scale"]), job["out_path"]]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(args, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    return proc, log_f


def main():
    jobs = build_jobs()
    total = len(jobs)
    progress = {"status": "running", "total_jobs": total, "completed": 0, "failed": 0,
                "started_at": time.time(), "points": POINTS}
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"launching {total} validation jobs, max_concurrent={MAX_CONCURRENT}", flush=True)

    pending, running = list(jobs), []
    completed_jobs = []
    while pending or running:
        while pending and len(running) < MAX_CONCURRENT:
            job = pending.pop(0)
            proc, log_f = launch(job)
            running.append((proc, log_f, job, time.time()))

        time.sleep(3)
        still = []
        for proc, log_f, job, st in running:
            ret = proc.poll()
            if ret is None:
                still.append((proc, log_f, job, st))
            else:
                log_f.close()
                wall = time.time() - st
                if ret != 0:
                    progress["failed"] += 1
                progress["completed"] += 1
                completed_jobs.append({**job, "wall_s": wall, "exit_code": ret})
                progress["last_completed"] = completed_jobs[-1]
                progress["elapsed_s"] = time.time() - progress["started_at"]
                with open(PROGRESS_PATH, "w") as f:
                    json.dump(progress, f, indent=2)
                print(f"[{progress['completed']}/{total}] seed {job['seed']} "
                      f"({job['point_name']}) -> exit {ret}, {wall:.1f}s", flush=True)
        running = still

    progress["status"] = "done"
    progress["elapsed_s"] = time.time() - progress["started_at"]
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"VALIDATION BATCH DONE -- {progress['completed']} jobs, {progress['failed']} failed, "
          f"{progress['elapsed_s']:.1f}s total", flush=True)


if __name__ == "__main__":
    main()

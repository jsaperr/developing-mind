"""MANDATORY validation for perturbation-testing redesign #2 (weight-nudge, not I_pert), per
web's explicit instruction: same decisive check as round 1 -- confirm the method reproduces the
already-known contrast before touching any intermediate point. If it can't, stop and report;
don't proceed to a third redesign without checking in.

Same 4-seeds-per-point sizing as round 1 (not 8-10), same reasoning: these runs are expensive
(settling alone can take up to MAX_SETTLE_S=1200s).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONDA_PYTHON = r"C:\Users\urawi\miniconda3\envs\developing-mind\python.exe"
SEED_SCRIPT = Path(__file__).resolve().parent / "run_perturbation_seed_v2.py"
OUT_DIR = Path(__file__).resolve().parent
PROGRESS_PATH = OUT_DIR / "validation_v2_progress.json"

POINTS = [
    {"name": "strong_tight_gate", "inhib": 10.0, "gap_scale": 1.0, "seeds": [27000, 27001, 27002, 27003]},
    {"name": "reliable_13_1p5", "inhib": 13.0, "gap_scale": 1.5, "seeds": [27010, 27011, 27012, 27013]},
]
MAX_CONCURRENT = 6


def build_jobs():
    jobs = []
    for point in POINTS:
        for seed in point["seeds"]:
            out_path = OUT_DIR / f"perturbv2_{point['name']}_seed{seed}.json"
            jobs.append({"seed": seed, "inhib": point["inhib"], "gap_scale": point["gap_scale"],
                         "point_name": point["name"], "out_path": str(out_path)})
    return jobs


def launch(job):
    log_path = OUT_DIR / f"perturbv2_{job['point_name']}_seed{job['seed']}.log"
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
    print(f"launching {total} v2 validation jobs, max_concurrent={MAX_CONCURRENT}", flush=True)

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
    print(f"V2 VALIDATION BATCH DONE -- {progress['completed']} jobs, {progress['failed']} failed, "
          f"{progress['elapsed_s']:.1f}s total", flush=True)


if __name__ == "__main__":
    main()

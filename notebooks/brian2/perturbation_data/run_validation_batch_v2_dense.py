"""Denser-resolution rerun of the v2 (weight-nudge) validation, per web's explicit instruction
after the round-2 result came back with the contrast backwards (strong_tight_gate needed frac=1.0
to flip in 3/4 seeds; 13mV/1.5 flipped at the LOWER frac=0.75 in 3/4 seeds -- backwards from the
expected "reliable = deep basin" story).

Web's read: don't accept either "nonmonotonic" or "different quantity" story yet -- rule out the
cheap, boring explanation first. Candidate (1): the original ladder's upper range (0.75-1.0, i.e.
nudging most of the way to the literal weight ceiling) might be strong enough to overturn structure
almost everywhere, saturating the top of the ladder and swallowing any real separation that lives
lower down. Since dt=0.2ms cut the round-2 batch to ~5 min (vs round 1's ~14 min), finer sampling
is affordable now.

This is NOT a new mechanism redesign and NOT a move to intermediate points -- explicitly one of the
two things web said not to do yet. It's the same weight-nudge method (run_perturbation_seed_v2.py),
same two reference points, same n=4/point, with only the fraction ladder changed to denser coverage
of the LOWER range: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 (six rungs instead of four, none at 0.75/1.0).

If real separation shows up here that the coarse top swallowed -> candidate (1) confirmed, this was
a resolution problem. If the lower range still shows the same backwards pattern (13mV/1.5 flipping
at a lower fraction than strong_tight_gate) -> candidate (1) ruled out, leaves the "different
quantity" (weight-injection sensitivity vs spontaneous-drift resistance are just not the same
measurement) as the leading explanation. Report back before deciding anything further, per web's
explicit ask.
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
PROGRESS_PATH = OUT_DIR / "validation_v2_dense_progress.json"

DENSE_FRACTIONS = "0.1,0.2,0.3,0.4,0.5,0.6"
POINTS = [
    {"name": "strong_tight_gate", "inhib": 10.0, "gap_scale": 1.0, "seeds": [28000, 28001, 28002, 28003]},
    {"name": "reliable_13_1p5", "inhib": 13.0, "gap_scale": 1.5, "seeds": [28010, 28011, 28012, 28013]},
]
MAX_CONCURRENT = 6


def build_jobs():
    jobs = []
    for point in POINTS:
        for seed in point["seeds"]:
            out_path = OUT_DIR / f"perturbv2dense_{point['name']}_seed{seed}.json"
            jobs.append({"seed": seed, "inhib": point["inhib"], "gap_scale": point["gap_scale"],
                         "point_name": point["name"], "out_path": str(out_path)})
    return jobs


def launch(job):
    log_path = OUT_DIR / f"perturbv2dense_{job['point_name']}_seed{job['seed']}.log"
    args = [CONDA_PYTHON, "-u", str(SEED_SCRIPT), str(job["seed"]), str(job["inhib"]),
            str(job["gap_scale"]), job["out_path"], DENSE_FRACTIONS]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(args, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    return proc, log_f


def main():
    jobs = build_jobs()
    total = len(jobs)
    progress = {"status": "running", "total_jobs": total, "completed": 0, "failed": 0,
                "started_at": time.time(), "points": POINTS, "fractions": DENSE_FRACTIONS}
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"launching {total} v2-dense validation jobs, max_concurrent={MAX_CONCURRENT}", flush=True)

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
    print(f"V2-DENSE VALIDATION BATCH DONE -- {progress['completed']} jobs, {progress['failed']} failed, "
          f"{progress['elapsed_s']:.1f}s total", flush=True)


if __name__ == "__main__":
    main()

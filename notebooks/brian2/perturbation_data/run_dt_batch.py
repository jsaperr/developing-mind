"""Launches the coarse-dt arm of the dt-headroom check: 8 seeds at 13mV/1.5, 600s, dt=0.5ms.
Compared against the boundary sweep's existing 8 seeds at the same setting and default dt=0.1ms
(24120-24127) -- see analyze_dt_check.py. Same concurrency-limited subprocess pattern as every
other batch in this project.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONDA_PYTHON = r"C:\Users\urawi\miniconda3\envs\developing-mind\python.exe"
SEED_SCRIPT = Path(__file__).resolve().parent / "run_dt_check.py"
OUT_DIR = Path(__file__).resolve().parent
PROGRESS_PATH = OUT_DIR / "dt_check_progress.json"

# 0.2ms, not 0.5ms: measured directly rather than guessed. The presynaptic ISI distribution
# (600s, 20 neurons, 240k spikes) has a hard floor at exactly 0.2ms -- that's
# spikes.dedup_spike_times' min_gap. dt=0.2ms gives 0 same-timestep collisions; dt=0.3ms gives
# 469 (0.195% of gaps) and dt=0.5ms gives 1373 (0.572%), and Brian2 hard-errors on any of them
# ("some neurons spike more than once during a time step"). Going coarser than 0.2ms would
# require raising dedup's min_gap, which drops real spikes from the correlated group whose 2ms
# jitter IS the correlation structure STDP is meant to detect -- i.e. a different system, not a
# faster one, the same objection that ruled out time-rescaling. So the free headroom here is 2x,
# not 5x.
DT_MS = 0.2
INHIB_MV = 13.0
GAP_SCALE = 1.5
DURATION_S = 600.0
SEEDS = list(range(25000, 25008))
MAX_CONCURRENT = 6


def main():
    jobs = [{"seed": s, "out_path": str(OUT_DIR / f"dtcheck_dt{DT_MS:g}_seed{s}.json")} for s in SEEDS]
    total = len(jobs)
    progress = {"status": "running", "total_jobs": total, "completed": 0, "failed": 0,
                "started_at": time.time(), "dt_ms": DT_MS, "inhib_mV": INHIB_MV,
                "gap_scale": GAP_SCALE, "duration_s": DURATION_S}
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"launching {total} dt-check jobs at dt={DT_MS}ms", flush=True)

    pending, running = list(jobs), []
    while pending or running:
        while pending and len(running) < MAX_CONCURRENT:
            job = pending.pop(0)
            log_f = open(OUT_DIR / f"dtcheck_dt{DT_MS:g}_seed{job['seed']}.log", "w")
            proc = subprocess.Popen(
                [CONDA_PYTHON, "-u", str(SEED_SCRIPT), str(job["seed"]), str(DT_MS),
                 str(INHIB_MV), str(GAP_SCALE), str(DURATION_S), job["out_path"]],
                stdout=log_f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
            running.append((proc, log_f, job, time.time()))

        time.sleep(2)
        still = []
        for proc, log_f, job, st in running:
            ret = proc.poll()
            if ret is None:
                still.append((proc, log_f, job, st))
            else:
                log_f.close()
                if ret != 0:
                    progress["failed"] += 1
                progress["completed"] += 1
                progress["elapsed_s"] = time.time() - progress["started_at"]
                with open(PROGRESS_PATH, "w") as f:
                    json.dump(progress, f, indent=2)
                print(f"[{progress['completed']}/{total}] seed {job['seed']} -> exit {ret}, "
                      f"{time.time()-st:.1f}s", flush=True)
        running = still

    progress["status"] = "done"
    progress["elapsed_s"] = time.time() - progress["started_at"]
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"DT CHECK DONE -- {progress['completed']} jobs, {progress['failed']} failed, "
          f"{progress['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

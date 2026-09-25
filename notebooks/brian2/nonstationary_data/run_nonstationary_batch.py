"""Orchestrator for the non-stationary-correlation batch (see run_nonstationary_seed.py and
experiment_plan_nonstationary_stdp.md). 8 seeds at each of 13mV/1.5 and strong_tight_gate
(10mV/1.0), per the plan. MAX_CONCURRENT=8: ~6 physical cores, so beyond 6 the throughput gain is
modest (measured in the performance audit) while 8 leaves the machine usable. Progress JSON is
rewritten after every job completion."""
import json
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PY = r"C:\Users\urawi\miniconda3\envs\developing-mind\python.exe"
SEED_SCRIPT = Path(__file__).resolve().parent / "run_nonstationary_seed.py"
OUT_DIR = Path(__file__).resolve().parent
PROGRESS = OUT_DIR / "nonstationary_progress.json"
POINTS = [
    {"name": "reliable_13_1p5", "inhib": 13.0, "gap_scale": 1.5, "seeds": list(range(30000, 30008))},
    {"name": "strong_tight_gate", "inhib": 10.0, "gap_scale": 1.0, "seeds": list(range(30100, 30108))},
]
MAX_CONCURRENT = 8


def main():
    jobs = [{"seed": s, "inhib": p["inhib"], "gs": p["gap_scale"], "name": p["name"]}
            for p in POINTS for s in p["seeds"]]
    # interleave the two points so a partial batch still has both
    jobs = [j for pair in zip(jobs[:len(jobs) // 2], jobs[len(jobs) // 2:]) for j in pair]
    prog = {"status": "running", "total": len(jobs), "completed": 0, "failed": 0,
            "started_at": time.time()}
    PROGRESS.write_text(json.dumps(prog, indent=2))
    pending, running = list(jobs), []
    while pending or running:
        while pending and len(running) < MAX_CONCURRENT:
            j = pending.pop(0)
            out = OUT_DIR / f"nonstat_{j['name']}_seed{j['seed']}.json"
            logf = open(OUT_DIR / f"nonstat_{j['name']}_seed{j['seed']}.log", "w")
            p = subprocess.Popen([PY, "-u", str(SEED_SCRIPT), str(j["seed"]), str(j["inhib"]),
                                  str(j["gs"]), str(out)], stdout=logf, stderr=subprocess.STDOUT,
                                 cwd=str(REPO_ROOT))
            running.append((p, logf, j, time.time()))
        time.sleep(3)
        still = []
        for p, logf, j, st in running:
            if p.poll() is None:
                still.append((p, logf, j, st))
                continue
            logf.close()
            prog["completed"] += 1
            prog["failed"] += int(p.returncode != 0)
            prog["elapsed_s"] = time.time() - prog["started_at"]
            PROGRESS.write_text(json.dumps(prog, indent=2))
            print(f"[{prog['completed']}/{len(jobs)}] {j['name']} seed {j['seed']} "
                  f"exit {p.returncode} {time.time() - st:.0f}s", flush=True)
        running = still
    prog["status"] = "done"
    prog["elapsed_s"] = time.time() - prog["started_at"]
    PROGRESS.write_text(json.dumps(prog, indent=2))
    print("NONSTATIONARY BATCH DONE", flush=True)


if __name__ == "__main__":
    main()

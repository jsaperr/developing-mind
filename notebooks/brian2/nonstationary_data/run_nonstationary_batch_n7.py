"""N=7 follow-up to the non-stationary-correlation experiment (see run_nonstationary_seed.py).

Question: at N=3 exactly one neuron per swap keeps the old pattern (a "retainer") while the rest
re-learn. Does that stay ~one retainer at larger N, or scale with population size?

13mV/1.5 reference operating point only (the point the N-scaling experiment used at N=7; the
strong_tight_gate equivalent was never characterized at N=7). The reference 13mV is converted to
the per-connection strength for N=7 by scale_inhib_for_n, holding total inhibitory drive constant,
same as the N-scaling runs. 8 seeds, all concurrent."""
import json
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PY = r"C:\Users\urawi\miniconda3\envs\developing-mind\python.exe"
SEED_SCRIPT = Path(__file__).resolve().parent / "run_nonstationary_seed.py"
OUT_DIR = Path(__file__).resolve().parent
PROGRESS = OUT_DIR / "nonstationary_n7_progress.json"
N_POST = 7
REF_INHIB_MV, GAP_SCALE = 13.0, 1.5
SEEDS = list(range(31000, 31008))
MAX_CONCURRENT = 8


def main():
    prog = {"status": "running", "total": len(SEEDS), "completed": 0, "failed": 0,
            "started_at": time.time(), "n_post": N_POST}
    PROGRESS.write_text(json.dumps(prog, indent=2))
    pending, running = list(SEEDS), []
    while pending or running:
        while pending and len(running) < MAX_CONCURRENT:
            s = pending.pop(0)
            out = OUT_DIR / f"nonstat_n7_reliable_13_1p5_seed{s}.json"
            logf = open(OUT_DIR / f"nonstat_n7_reliable_13_1p5_seed{s}.log", "w")
            p = subprocess.Popen([PY, "-u", str(SEED_SCRIPT), str(s), str(REF_INHIB_MV), str(GAP_SCALE),
                                  str(out), str(N_POST)], stdout=logf, stderr=subprocess.STDOUT,
                                 cwd=str(REPO_ROOT))
            running.append((p, logf, s, time.time()))
        time.sleep(3)
        still = []
        for p, logf, s, st in running:
            if p.poll() is None:
                still.append((p, logf, s, st))
                continue
            logf.close()
            prog["completed"] += 1
            prog["failed"] += int(p.returncode != 0)
            prog["elapsed_s"] = time.time() - prog["started_at"]
            PROGRESS.write_text(json.dumps(prog, indent=2))
            print(f"[{prog['completed']}/{len(SEEDS)}] seed {s} exit {p.returncode} {time.time() - st:.0f}s",
                  flush=True)
        running = still
    prog["status"] = "done"
    prog["elapsed_s"] = time.time() - prog["started_at"]
    PROGRESS.write_text(json.dumps(prog, indent=2))
    print("NONSTATIONARY N7 BATCH DONE", flush=True)


if __name__ == "__main__":
    main()

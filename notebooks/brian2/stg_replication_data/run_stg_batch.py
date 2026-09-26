"""strong_tight_gate replication batch: v1b seeds 36000-36007 and v1c seeds 37000-37007, 16 jobs,
concurrency cap 8, a `python -u` process per job with its own .log, and a progress JSON after each
completed job."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = [("v1b", s) for s in range(36000, 36008)] + [("v1c", s) for s in range(37000, 37008)]
CAP = 8

if __name__ == '__main__':
    t0 = time.time()
    pending = list(JOBS); running = {}; done = []
    progress = HERE / "stg_progress.json"
    while pending or running:
        while pending and len(running) < CAP:
            sched, s = pending.pop(0)
            stem = f"stg_{sched}_n7_seed{s}"
            log = open(HERE / f"{stem}.log", "w")
            running[(sched, s)] = subprocess.Popen(
                [sys.executable, "-u", str(HERE / "run_stg_seed.py"), sched, str(s), str(HERE / f"{stem}.json.gz")],
                stdout=log, stderr=subprocess.STDOUT)
        for key, p in list(running.items()):
            if p.poll() is not None:
                done.append({'job': list(key), 'returncode': p.returncode, 'elapsed_s': round(time.time() - t0)})
                del running[key]
                json.dump({'done': done, 'running': [list(k) for k in running], 'pending': [list(k) for k in pending]},
                          open(progress, "w"), indent=1)
        time.sleep(5)
    print(f"BATCH DONE {len(done)} jobs in {time.time() - t0:.0f}s", flush=True)

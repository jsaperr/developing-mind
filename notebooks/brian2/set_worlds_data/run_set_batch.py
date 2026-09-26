"""Set-worlds batch: ov50 (38000-38007), ov70 (39000-39007), checkb (40000-40007); 24 jobs, cap 8,
`python -u` per job with its own .log, progress JSON after each completed job."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = ([("ov50", s) for s in range(38000, 38008)] + [("ov70", s) for s in range(39000, 39008)]
        + [("checkb", s) for s in range(40000, 40008)])
CAP = 8

if __name__ == '__main__':
    t0 = time.time(); pending = list(JOBS); running = {}; done = []
    progress = HERE / "set_progress.json"
    while pending or running:
        while pending and len(running) < CAP:
            w, s = pending.pop(0); stem = f"{w}_n7_seed{s}"
            log = open(HERE / f"{stem}.log", "w")
            running[(w, s)] = subprocess.Popen([sys.executable, "-u", str(HERE / "run_set_seed.py"), w, str(s),
                                                str(HERE / f"{stem}.json.gz")], stdout=log, stderr=subprocess.STDOUT)
        for key, p in list(running.items()):
            if p.poll() is not None:
                done.append({'job': list(key), 'returncode': p.returncode, 'elapsed_s': round(time.time() - t0)})
                del running[key]
                json.dump({'done': done, 'running': [list(k) for k in running], 'pending': [list(k) for k in pending]},
                          open(progress, "w"), indent=1)
        time.sleep(5)
    print(f"BATCH DONE {len(done)} jobs in {time.time() - t0:.0f}s", flush=True)

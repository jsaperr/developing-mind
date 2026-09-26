"""v1b batch orchestrator: seeds 34000-34007, concurrency cap 8, a `python -u` process per seed
with its own .log, and a progress JSON rewritten after each completed job. Same pattern as v1."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = list(range(34000, 34008))
CAP = 8
PREFIX = "v1b_n7_reliable_13_1p5"

if __name__ == '__main__':
    t0 = time.time()
    pending = list(SEEDS); running = {}; done = []
    progress = HERE / "v1b_progress.json"
    while pending or running:
        while pending and len(running) < CAP:
            s = pending.pop(0)
            log = open(HERE / f"{PREFIX}_seed{s}.log", "w")
            running[s] = subprocess.Popen(
                [sys.executable, "-u", str(HERE / "run_v1b_seed.py"), str(s),
                 str(HERE / f"{PREFIX}_seed{s}.json.gz")], stdout=log, stderr=subprocess.STDOUT)
        for s, p in list(running.items()):
            if p.poll() is not None:
                done.append({'seed': s, 'returncode': p.returncode, 'elapsed_s': round(time.time() - t0)})
                del running[s]
                json.dump({'done': done, 'running': list(running), 'pending': pending}, open(progress, "w"), indent=1)
        time.sleep(5)
    print(f"BATCH DONE {len(done)} seeds in {time.time() - t0:.0f}s", flush=True)

"""Rest batch (A B C REST B): seeds 46000-46007; 8 jobs, cap 8,
`python -u` per job with its own .log, progress JSON after each completed job."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = [("rest", s) for s in range(46000, 46008)]
CAP = 8

def already_completed(w, s):
    """Resume support: skip a job whose result file exists with status 'completed'. Runs are
    deterministic per seed, so rerunning an interrupted job reproduces it exactly."""
    f = HERE / f"{w}_n7_seed{s}.json.gz"
    if not f.exists():
        return False
    sys.path.insert(0, str(HERE.parents[2]))
    from src.brian2_stdp.results_io import load_result
    try:
        return load_result(f).get('status') == 'completed'
    except Exception:
        return False


if __name__ == '__main__':
    t0 = time.time(); running = {}; done = []
    pending = [j for j in JOBS if not already_completed(*j)]
    print(f"{len(JOBS) - len(pending)} jobs already completed, {len(pending)} to run", flush=True)
    progress = HERE / "rest_progress.json"
    while pending or running:
        while pending and len(running) < CAP:
            w, s = pending.pop(0); stem = f"{w}_n7_seed{s}"
            log = open(HERE / f"{stem}.log", "w")
            running[(w, s)] = subprocess.Popen([sys.executable, "-u", str(HERE / "run_rest_seed.py"), str(s),
                                                str(HERE / f"{stem}.json.gz")], stdout=log, stderr=subprocess.STDOUT)
        for key, p in list(running.items()):
            if p.poll() is not None:
                done.append({'job': list(key), 'returncode': p.returncode, 'elapsed_s': round(time.time() - t0)})
                del running[key]
                json.dump({'done': done, 'running': [list(k) for k in running], 'pending': [list(k) for k in pending]},
                          open(progress, "w"), indent=1)
        time.sleep(5)
    print(f"BATCH DONE {len(done)} jobs in {time.time() - t0:.0f}s", flush=True)

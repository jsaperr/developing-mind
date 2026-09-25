import subprocess, sys, time, os
PY = r"C:\Users\urawi\miniconda3\envs\developing-mind\python.exe"
SP = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(SP, "cmp"); os.makedirs(OUT, exist_ok=True)
jobs = []
for (inh, gs, name) in [(13.0, 1.5, "rel"), (10.0, 1.0, "stg")]:
    for k in range(32): jobs.append(("standalone", 40000 + k, inh, gs, name))
    for k in range(16): jobs.append(("cython", 41000 + k, inh, gs, name))
jobs.sort(key=lambda j: j[0] != "cython")   # start the slow cython jobs first
pending, running, t0 = list(jobs), [], time.time()
while pending or running:
    while pending and len(running) < 10:
        b, s, inh, gs, name = pending.pop(0)
        p = subprocess.Popen([PY, "-u", os.path.join(SP, "backend_compare.py"), b, str(s), str(inh), str(gs),
                              os.path.join(OUT, f"{b}_{name}_{s}.json")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running.append(p)
    time.sleep(2); running = [p for p in running if p.poll() is None]
open(os.path.join(SP, "cmp_done.txt"), "w").write(f"done in {time.time()-t0:.0f}s")

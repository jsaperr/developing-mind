import subprocess, sys, time, re
PY = r"C:\Users\urawi\miniconda3\envs\developing-mind\python.exe"
SCRIPT = sys.argv[0].replace("scale_test.py", "profile_run.py")
CWD = r"C:\Users\urawi\OneDrive\Desktop\a wide variety of vscode projects\developing-mind"
for k in [1, 6, 9, 12]:
    t = time.time()
    procs = [subprocess.Popen([PY, "-u", SCRIPT, str(100 + i), "3", "0.2"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, cwd=CWD, text=True) for i in range(k)]
    outs = [p.communicate()[0] for p in procs]
    wall = time.time() - t
    later = [float(re.search(r"later_chunks_mean=([\d.]+)s", o).group(1)) for o in outs if "later_chunks_mean" in o]
    print(f"k={k:2d} procs: batch wall={wall:5.1f}s  per-proc sim s/50s-chunk mean={sum(later)/len(later):.2f} "
          f"max={max(later):.2f}  aggregate throughput={k*50/ (sum(later)/len(later)):.0f} sim-s per wall-s", flush=True)

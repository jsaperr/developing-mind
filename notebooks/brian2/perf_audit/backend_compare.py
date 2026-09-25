"""Run one seed at 600s on a chosen backend, write final per-neuron gaps as JSON.
Usage: backend_compare.py <backend: cython|standalone> <seed> <inhib_mV> <gap_scale> <out.json>"""
import glob, json, os, shutil, subprocess, sys, time, tempfile
import numpy as np
REPO = r"C:\Users\urawi\OneDrive\Desktop\a wide variety of vscode projects\developing-mind"
sys.path.insert(0, REPO)
from brian2 import *
backend, seed, inhib, gs, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
DUR = 600.0
bdir = None
if backend == "standalone":
    bdir = tempfile.mkdtemp(prefix=f"sa_{seed}_")
    set_device('cpp_standalone', directory=bdir, build_on_run=False)
from src.brian2_stdp.network import build_competitive_population_network
from src.brian2_stdp.spikes import build_presynaptic_input
start_scope(); defaultclock.dt = 0.2 * ms
idx, tt = build_presynaptic_input(20.0, 0.9, DUR + 10, np.random.default_rng(seed))
pre, post, syn, inhibsyn = build_competitive_population_network(3, idx, tt, 0.005, inhib * mV, gs)
t0 = time.time()
if backend == "cython":
    run(DUR * second)
    w = np.array(syn.w[:]); si = np.array(syn.i[:]); sj = np.array(syn.j[:])
else:
    run(DUR * second)
    device.build(directory=bdir, compile=True, run=False, debug=False)
    t_comp = time.time() - t0
    r = subprocess.run([os.path.join(bdir, "main.exe")], cwd=bdir, capture_output=True)
    assert r.returncode == 0, r.stderr[-500:]
    rd = lambda p, dt: np.fromfile(glob.glob(os.path.join(bdir, "results", p))[0], dtype=dt)
    w = rd("_dynamic_array_synapses_w_*", np.float64)
    si = rd("_dynamic_array_synapses__synaptic_pre_*", np.int32)
    sj = rd("_dynamic_array_synapses__synaptic_post_*", np.int32)
    clk = rd("_array_defaultclock_t_*", np.float64)[0]
    assert abs(clk - DUR) < 1e-3, clk
gaps = []
for j in range(3):
    m = sj == j; wj = w[m][np.argsort(si[m])]
    gaps.append(float(wj[:10].mean() - wj[10:].mean()))
json.dump({"backend": backend, "seed": seed, "inhib": inhib, "gap_scale": gs, "gaps": gaps,
           "wall": time.time() - t0}, open(out, "w"))
if bdir: shutil.rmtree(bdir, ignore_errors=True)

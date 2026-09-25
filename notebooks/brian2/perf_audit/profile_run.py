"""Per-phase wall-clock profile of one representative competitive-population run (13mV/1.5, N=3,
dt=0.2ms), mirroring run_perturbation_seed_v2's setup. Usage: profile_run.py <seed> [chunks] [dt_ms]"""
import sys, time
from pathlib import Path
import numpy as np

T0 = time.time()
sys.path.insert(0, r"C:\Users\urawi\OneDrive\Desktop\a wide variety of vscode projects\developing-mind")
from brian2 import SpikeMonitor, defaultclock, mV, ms, run, second, start_scope, prefs
t_import = time.time() - T0

from src.brian2_stdp.network import build_competitive_population_network
from src.brian2_stdp.spikes import build_presynaptic_input
from notebooks.brian2.perturbation_data.run_perturbation_seed import _per_neuron_gap

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 4
dt_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
CHUNK_S = 50.0

t = time.time(); start_scope(); defaultclock.dt = dt_ms * ms
rng = np.random.default_rng(seed)
idx, tt = build_presynaptic_input(20.0, 0.9, chunks * CHUNK_S + 10, rng)
t_input = time.time() - t

t = time.time()
pre, post, syn, inhib = build_competitive_population_network(3, idx, tt, 0.005, 13.0 * mV, 1.5)
sm = SpikeMonitor(post)
t_build = time.time() - t
syn_i = np.array(syn.i[:]); syn_j = np.array(syn.j[:])

chunk_times, read_times = [], []
for _ in range(chunks):
    t = time.time(); run(CHUNK_S * second); chunk_times.append(time.time() - t)
    t = time.time(); _per_neuron_gap(np.array(syn.w[:]), syn_i, syn_j); read_times.append(time.time() - t)

print(f"RESULT seed={seed} dt={dt_ms} target_pref={prefs.codegen.target} "
      f"import={t_import:.2f}s input_gen={t_input:.2f}s build={t_build:.2f}s "
      f"chunk1(run+compile)={chunk_times[0]:.2f}s later_chunks_mean={np.mean(chunk_times[1:]):.2f}s "
      f"read_mean={np.mean(read_times)*1000:.1f}ms n_pre_spikes={len(idx)} total_wall={time.time()-T0:.1f}s")

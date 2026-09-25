import sys
import numpy as np
sys.path.insert(0, r"C:\Users\urawi\OneDrive\Desktop\a wide variety of vscode projects\developing-mind")
from brian2 import SpikeMonitor, defaultclock, mV, ms, run, second, start_scope, profiling_summary
from src.brian2_stdp.network import build_competitive_population_network
from src.brian2_stdp.spikes import build_presynaptic_input
start_scope(); defaultclock.dt = 0.2 * ms
idx, tt = build_presynaptic_input(20.0, 0.9, 70.0, np.random.default_rng(3))
pre, post, syn, inhib = build_competitive_population_network(3, idx, tt, 0.005, 13.0 * mV, 1.5)
run(5 * second)                      # warm-up / compile
run(30 * second, profile=True)
print(profiling_summary(show=10))
print("n_synapses:", len(syn.i), " n_inhib_synapses:", len(inhib.i) if hasattr(inhib, 'i') else 'n/a')
print("objects in network:")
import brian2
from brian2 import Network, magic_network
for o in magic_network.objects:
    print("  ", o.name, type(o).__name__, getattr(o, 'clock', None) and o.clock.dt)

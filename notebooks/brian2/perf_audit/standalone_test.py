import sys, time
import numpy as np
sys.path.insert(0, r"C:\Users\urawi\OneDrive\Desktop\a wide variety of vscode projects\developing-mind")
from brian2 import *
DUR = float(sys.argv[1]); threads = int(sys.argv[2])
set_device('cpp_standalone', directory=sys.argv[3], build_on_run=False)
if threads: prefs.devices.cpp_standalone.openmp_threads = threads
from src.brian2_stdp.network import build_competitive_population_network
from src.brian2_stdp.spikes import build_presynaptic_input
start_scope(); defaultclock.dt = 0.2 * ms
idx, tt = build_presynaptic_input(20.0, 0.9, DUR + 10, np.random.default_rng(3))
pre, post, syn, inhib = build_competitive_population_network(3, idx, tt, 0.005, 13.0 * mV, 1.5)
t = time.time(); run(DUR * second)
device.build(directory=sys.argv[3], compile=True, run=False, debug=False)
print("compiled in", round(time.time()-t,1), "s")

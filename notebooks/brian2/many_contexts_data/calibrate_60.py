"""Calibration for the 60-input, 6-block rig (generalization step 3) BEFORE any batch, as calibrate_novelc.py did for
30 inputs. Hold target_total = 10 (w_init = 10/60), so homeostatic scaling pins total synaptic drive to the same
value as the 20- and 30-input rigs. gmax unchanged. Block A correlated, 400 s, N = 7 and N = 40, Brian2 seeded.

Calibration bar (stated before running, not tuned after; the same bar as novel-C): per-neuron steady rate
(last 100 s) in 3-20 Hz, and w_total within [9.5, 10.5]. If it misses, report and stop; do not chase gmax.

Usage (repo root): python notebooks/brian2/many_contexts_data/calibrate_60.py
"""
import sys
from pathlib import Path

import numpy as np
from brian2 import SpikeMonitor, StateMonitor, defaultclock, mV, ms, run, second, seed as b2_seed, start_scope

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.brian2_stdp.network import build_competitive_population_network, scale_inhib_for_n
from src.brian2_stdp.spikes import build_multiblock_phase_input

for n in (7, 40):
    start_scope(); defaultclock.dt = 0.2 * ms; b2_seed(60001)
    rng = np.random.default_rng(60001)
    idx, t = build_multiblock_phase_input(20.0, 0.9, [400.0], [0], rng, n_blocks=6, block_size=10)
    per_conn = scale_inhib_for_n(n, reference_inhib_mV=13.0, reference_n_post=3)
    pre, post, syn, inhib = build_competitive_population_network(n, idx, t, 0.005, per_conn * mV, 1.5, n_pre=60,
                                                                 w_init=10.0 / 60, target_total=10.0)
    sp = SpikeMonitor(post); wt = StateMonitor(post, 'w_total', record=True, dt=1 * second)
    run(400 * second)
    st, si = np.array(sp.t / second), np.array(sp.i[:])
    rates = np.array([np.sum((si == j) & (st >= 300)) / 100.0 for j in range(n)])
    wtot = np.array(wt.w_total[:])[:, -1]
    ok = bool((rates.min() >= 3) and (rates.max() <= 20) and (wtot.min() >= 9.5) and (wtot.max() <= 10.5))
    print(f"N={n}: rates (last 100 s) {rates.min():.1f}-{rates.max():.1f} Hz, mean {rates.mean():.1f}; "
          f"w_total {wtot.min():.3f}-{wtot.max():.3f} -> {'PASS' if ok else 'MISS'}", flush=True)

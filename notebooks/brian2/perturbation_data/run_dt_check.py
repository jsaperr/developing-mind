"""Cheap side-task per web's message: is there headroom to coarsen the integration timestep?

`defaultclock.dt` is never set anywhere in this project, so every run so far has used Brian2's
default 0.1ms. If the dynamics tolerate a coarser dt, that's a straight speedup on every future
run at zero scientific cost.

Deliberately NOT adopted on "should be fine" reasoning -- there are two concrete reasons to be
skeptical here, both specific to this system rather than generic caution:
  1. STDP depends on precise spike timing (taupre/taupost = 20ms). A 0.5ms grid is 2.5% of that
     window -- probably tolerable, but "probably" isn't a validation.
  2. More worrying: SpikeGeneratorGroup quantizes input spike times to the dt grid, and the
     correlated group's shared-spike jitter is only 2ms (spikes.py, jitter_ms=2.0). At dt=0.5ms
     that's a 25% coarsening of the very jitter that defines the correlation structure STDP is
     supposed to detect. That could change the input statistics, not just the integration
     accuracy.

So this compares OUTCOMES at a known setting, distributionally. Exact seed-matched agreement is
impossible by construction (a different dt draws a different noise realization from `xi`), so the
comparison is: does the same setting produce the same classification distribution and similar
late-window gap values?

Baseline (dt=0.1ms) is NOT re-run -- the boundary sweep already has 8 seeds at 13mV/1.5
(24120-24127). This script only runs the coarse-dt arm, and analyze compares against those.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from brian2 import SpikeMonitor, StateMonitor, defaultclock, mV, ms, run, second, start_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.brian2_stdp.network import build_competitive_population_network
from src.brian2_stdp.spikes import build_presynaptic_input

TARGET_RATE = 20.0
P_SHARE = 0.9
N_POST = 3
WEIGHT_TRACE_DT = 1 * second


def run_dt_seed(seed_val, dt_ms, inhib_mV, gap_scale, duration_s, out_path, apre_val=0.005):
    result = {'seed': seed_val, 'status': 'started', 'dt_ms': dt_ms, 'duration_s': duration_s,
              'inhib_strength_mV': inhib_mV, 'gap_scale': gap_scale, 'n_post': N_POST}
    with open(out_path, 'w') as f:
        json.dump(result, f)

    try:
        start_scope()
        defaultclock.dt = dt_ms * ms
        rng = np.random.default_rng(seed_val)
        idx, t = build_presynaptic_input(TARGET_RATE, P_SHARE, duration_s, rng)
        pre, post, syn, inhib = build_competitive_population_network(
            N_POST, idx, t, apre_val, inhib_mV * mV, gap_scale)

        weight_trace = StateMonitor(syn, 'w', record=True, dt=WEIGHT_TRACE_DT)
        syn_i = np.array(syn.i[:])
        syn_j = np.array(syn.j[:])

        t0 = time.time()
        run(duration_s * second)
        wall = time.time() - t0

        result.update({
            'status': 'completed', 'wall_elapsed': wall,
            'weight_trace_t': (weight_trace.t / second).tolist(),
            'weight_trace': weight_trace.w[:].tolist(),
            'syn_i': syn_i.tolist(), 'syn_j': syn_j.tolist(),
        })
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    with open(out_path, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    run_dt_seed(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
                float(sys.argv[5]), sys.argv[6])
    print(f"dt-check seed {sys.argv[1]} dt={sys.argv[2]}ms -> {sys.argv[6]}")

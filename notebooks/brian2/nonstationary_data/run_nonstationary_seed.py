"""Non-stationary correlation structure, per experiment_plan_nonstationary_stdp.md.

One continuous run, substrate never reset: phase 1 correlated block = A (presynaptic [0, 10)),
phase 2 = B ([10, 20)), phase 3 = A again. Network, STDP, scaling and inhibition untouched;
I_pert stays at its default 0 mV. Only the input changes (spikes.build_phase_switching_input).

Exploratory by design: everything is saved (full per-synapse w(t), per-neuron r(t), w_total(t),
1s spike-count bins) and claims are made after inspecting the trajectories. The one thing on
record before running (from the plan, not a pass/fail bar): 13mV/1.5 (suppression-based) is
predicted to track the swap FAST, strong_tight_gate (dominance-based) to LAG -- opposite in
direction to how the two ranked under direct-injection perturbation.

Phases are 1000s each -- settling at these points has taken 250-750s, so phase 1 gets real room;
settling is confirmed per-seed from the trajectory afterward, not assumed from the cutoff.
dt=0.2ms (validated in the dt-headroom check); Cython runtime backend, not cpp_standalone (which
shifted outcomes at strong_tight_gate in the backend comparison).

Usage: run_nonstationary_seed.py <seed> <inhib_mV> <gap_scale> <out.json>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from brian2 import SpikeMonitor, StateMonitor, defaultclock, mV, ms, run, second, start_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.brian2_stdp.network import build_competitive_population_network, scale_inhib_for_n
from src.brian2_stdp.spikes import build_phase_switching_input

TARGET_RATE = 20.0
P_SHARE = 0.9
N_POST = 3
DT_MS = 0.2
PHASE_DURATIONS_S = [1000.0, 1000.0, 1000.0]
PHASE_CORR_BLOCKS = [0, 1, 0]
R_TRACE_DT = 200 * ms
WEIGHT_TRACE_DT = 1 * second
SPIKE_BIN_S = 1.0
PROGRESS_EVERY_S = 200.0


def run_nonstationary_seed(seed_val, inhib_mV, gap_scale, out_path, apre_val=0.005, n_post=N_POST):
    """inhib_mV is the N=3 reference value; for n_post != 3 it is converted to the per-connection
    strength via scale_inhib_for_n (holds total inhibitory drive constant, same as the N-scaling
    experiment). At n_post=3 the conversion is the identity, so N=3 runs are unchanged."""
    total_s = float(sum(PHASE_DURATIONS_S))
    swap_times = np.cumsum(PHASE_DURATIONS_S)[:-1].tolist()
    per_connection_mV = scale_inhib_for_n(n_post, reference_inhib_mV=inhib_mV, reference_n_post=3)
    result = {
        'seed': seed_val, 'status': 'started', 'inhib_strength_mV': per_connection_mV,
        'reference_inhib_mV': inhib_mV, 'gap_scale': gap_scale,
        'n_post': n_post, 'dt_ms': DT_MS, 'apre': apre_val, 'p_share': P_SHARE,
        'phase_durations_s': PHASE_DURATIONS_S, 'phase_corr_blocks': PHASE_CORR_BLOCKS,
        'swap_times_s': swap_times, 'total_s': total_s,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f)

    try:
        start_scope()
        defaultclock.dt = DT_MS * ms
        rng = np.random.default_rng(seed_val)
        idx, t = build_phase_switching_input(TARGET_RATE, P_SHARE, PHASE_DURATIONS_S,
                                             PHASE_CORR_BLOCKS, rng)
        pre, post, syn, inhib = build_competitive_population_network(
            n_post, idx, t, apre_val, per_connection_mV * mV, gap_scale)
        spikes = SpikeMonitor(post)
        r_trace = StateMonitor(post, 'r', record=True, dt=R_TRACE_DT)
        w_total_trace = StateMonitor(post, 'w_total', record=True, dt=WEIGHT_TRACE_DT)
        weight_trace = StateMonitor(syn, 'w', record=True, dt=WEIGHT_TRACE_DT)
        syn_i = np.array(syn.i[:])
        syn_j = np.array(syn.j[:])

        wall0 = time.time()
        elapsed = 0.0
        while elapsed < total_s - 1e-9:
            step = min(PROGRESS_EVERY_S, total_s - elapsed)
            run(step * second)
            elapsed += step
            print(f"[t={elapsed:.0f}s] wall={time.time() - wall0:.0f}s", flush=True)

        st = np.array(spikes.t / second)
        si = np.array(spikes.i[:])
        bin_edges = np.arange(int(np.ceil(total_s / SPIKE_BIN_S)) + 1) * SPIKE_BIN_S
        spike_rate_bins = np.stack([np.histogram(st[si == j], bins=bin_edges)[0] / SPIKE_BIN_S
                                    for j in range(n_post)])
        result.update({
            'status': 'completed', 'wall_elapsed': time.time() - wall0,
            'syn_i': syn_i.tolist(), 'syn_j': syn_j.tolist(),
            'weight_trace_t': (weight_trace.t / second).tolist(),
            'weight_trace': weight_trace.w[:].round(4).tolist(),      # (n_synapses, n_t)
            'r_trace_t': (r_trace.t / second).tolist(),
            'r_trace': r_trace.r[:].round(4).tolist(),                # (N_POST, n_t)
            'w_total_trace': w_total_trace.w_total[:].round(4).tolist(),
            'spike_bin_edges_s': bin_edges.tolist(),
            'spike_rate_bins': spike_rate_bins.tolist(),              # (N_POST, n_bins), Hz
            'overall_post_rate': (np.array(spikes.count[:]) / total_s).tolist(),
            'final_w': np.array(syn.w[:]).tolist(),
        })
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    with open(out_path, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    n_post_arg = int(sys.argv[5]) if len(sys.argv) > 5 else N_POST
    run_nonstationary_seed(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4],
                           n_post=n_post_arg)
    print(f"nonstationary seed {sys.argv[1]} -> {sys.argv[4]}")

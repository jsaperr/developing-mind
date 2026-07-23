"""Experiment B, step 3: population-competition N-scaling. Parametrized twin of
competitive_population_data/run_competitive_seed.py -- kept as a separate script rather than
adding n_post as a new CLI arg to the original, so the original's CLI signature (and therefore
the reproducibility of the already-published bistability-sweep results) doesn't change.

inhib_strength_mV passed on the CLI is the REFERENCE value (the known-good 13mV point at
n_post=3, the sweep's standout combo) -- this script converts it to the actual per-connection
value via scale_inhib_for_n before building the network, so (n_post-1)*per_connection stays
constant across n_post per Gate 1's normalization design.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
from brian2 import SpikeMonitor, StateMonitor, mV, ms, run, second, start_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.brian2_stdp.network import build_competitive_population_network, scale_inhib_for_n
from src.brian2_stdp.spikes import build_presynaptic_input

TARGET_RATE = 20.0
P_SHARE = 0.9

R_TRACE_DT = 200 * ms
WEIGHT_TRACE_DT = 1 * second
SPIKE_BIN_S = 1.0


def run_n_scaling_seed(seed_val, apre_val, n_post, reference_inhib_mV, gap_scale, duration_s,
                        combo_name, out_path):
    inhib_strength_mV = scale_inhib_for_n(n_post, reference_inhib_mV=reference_inhib_mV, reference_n_post=3)
    result = {
        'seed': seed_val, 'status': 'started', 'duration_s': duration_s,
        'p_share': P_SHARE, 'Apre': apre_val, 'n_post': n_post,
        'reference_inhib_mV': reference_inhib_mV, 'inhib_strength_mV': inhib_strength_mV,
        'gap_scale': gap_scale, 'combo_name': combo_name,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f)

    try:
        start_scope()
        rng = np.random.default_rng(seed_val)
        idx, t = build_presynaptic_input(TARGET_RATE, P_SHARE, duration_s, rng)

        inhib_strength = inhib_strength_mV * mV
        pre, post, syn, inhib = build_competitive_population_network(
            n_post, idx, t, apre_val, inhib_strength, gap_scale
        )

        post_spikes = SpikeMonitor(post)
        r_trace = StateMonitor(post, 'r', record=True, dt=R_TRACE_DT)
        weight_trace = StateMonitor(syn, 'w', record=True, dt=WEIGHT_TRACE_DT)

        syn_i = np.array(syn.i[:])
        syn_j = np.array(syn.j[:])

        wall_start = time.time()
        run(duration_s * second)
        wall_elapsed = time.time() - wall_start

        spike_times = np.array(post_spikes.t / second)
        spike_indices = np.array(post_spikes.i[:])
        n_bins = int(np.ceil(duration_s / SPIKE_BIN_S))
        bin_edges = np.arange(n_bins + 1) * SPIKE_BIN_S
        spike_rate_bins = np.zeros((n_post, n_bins))
        for j in range(n_post):
            counts, _ = np.histogram(spike_times[spike_indices == j], bins=bin_edges)
            spike_rate_bins[j] = counts / SPIKE_BIN_S

        final_w = np.array(syn.w[:])
        per_neuron_final = {}
        for j in range(n_post):
            mask = syn_j == j
            w_j = final_w[mask]
            i_j = syn_i[mask]
            order = np.argsort(i_j)
            w_j = w_j[order]
            per_neuron_final[str(j)] = {
                'corr_w_mean': float(w_j[:10].mean()),
                'uncorr_w_mean': float(w_j[10:].mean()),
                'gap': float(w_j[:10].mean() - w_j[10:].mean()),
            }

        result.update({
            'status': 'completed',
            'wall_elapsed': wall_elapsed,
            'overall_post_rate': (np.array(post_spikes.count[:]) / duration_s).tolist(),
            'r_trace_t': (r_trace.t / second).tolist(),
            'r_trace': r_trace.r[:].tolist(),
            'weight_trace_t': (weight_trace.t / second).tolist(),
            'weight_trace': weight_trace.w[:].tolist(),
            'syn_i': syn_i.tolist(),
            'syn_j': syn_j.tolist(),
            'spike_bin_edges_s': bin_edges.tolist(),
            'spike_rate_bins': spike_rate_bins.tolist(),
            'final_w': final_w.tolist(),
            'per_neuron_final': per_neuron_final,
        })
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    with open(out_path, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    seed_val = int(sys.argv[1])
    apre_val = float(sys.argv[2])
    n_post = int(sys.argv[3])
    reference_inhib_mV = float(sys.argv[4])
    gap_scale = float(sys.argv[5])
    duration_s = float(sys.argv[6])
    combo_name = sys.argv[7]
    out_path = sys.argv[8]
    run_n_scaling_seed(seed_val, apre_val, n_post, reference_inhib_mV, gap_scale, duration_s,
                        combo_name, out_path)
    print(f"seed {seed_val} (n_post={n_post}) -> {out_path}")

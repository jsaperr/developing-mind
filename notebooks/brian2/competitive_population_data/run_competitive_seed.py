import sys
import json
import time
from pathlib import Path

import numpy as np
from brian2 import SpikeMonitor, StateMonitor, mV, ms, run, second, start_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.brian2_stdp.network import build_competitive_population_network
from src.brian2_stdp.spikes import build_presynaptic_input

TARGET_RATE = 20.0
P_SHARE = 0.9
N_POST = 3

# Subsampling choices, documented rather than left implicit (per the "err heavily toward saving
# more than seems necessary" instruction, balanced against not bloating file size needlessly):
# - r (rate-trace) sampled at 200ms, matching tau_r itself -- fine enough to reconstruct g_ij(t)
#   at the timescale it actually varies.
# - weight traces sampled at 1s -- coarser than the r trace, but reversal frequency (~1 per
#   synapse per ~2s, from the population extension) means 1s resolution is still fine enough to
#   catch individual reversal events, not just smoothed trends.
# - spike activity saved as per-neuron per-1s-bin counts, not raw spike timestamps -- literal
#   spike times at ~15-20Hz x 3 neurons x multi-thousand seconds would be hundreds of thousands
#   of floats for no real analytical benefit over a binned rate; binned counts still answer any
#   "rate over time, not just final" question this might come back to.
R_TRACE_DT = 200 * ms
WEIGHT_TRACE_DT = 1 * second
SPIKE_BIN_S = 1.0


def run_competitive_seed(seed_val, apre_val, inhib_strength_mV, gap_scale, duration_s,
                          calibration_combo_name, out_path):
    result = {
        'seed': seed_val, 'status': 'started', 'duration_s': duration_s,
        'p_share': P_SHARE, 'Apre': apre_val, 'n_post': N_POST,
        'inhib_strength_mV': inhib_strength_mV, 'gap_scale': gap_scale,
        'calibration_combo_name': calibration_combo_name,
    }
    # write a "started" marker immediately so a hang is distinguishable from "never launched"
    with open(out_path, 'w') as f:
        json.dump(result, f)

    try:
        start_scope()
        rng = np.random.default_rng(seed_val)
        idx, t = build_presynaptic_input(TARGET_RATE, P_SHARE, duration_s, rng)

        inhib_strength = inhib_strength_mV * mV
        pre, post, syn, inhib = build_competitive_population_network(
            N_POST, idx, t, apre_val, inhib_strength, gap_scale
        )

        post_spikes = SpikeMonitor(post)
        r_trace = StateMonitor(post, 'r', record=True, dt=R_TRACE_DT)
        weight_trace = StateMonitor(syn, 'w', record=True, dt=WEIGHT_TRACE_DT)

        syn_i = np.array(syn.i[:])
        syn_j = np.array(syn.j[:])

        wall_start = time.time()
        run(duration_s * second)
        wall_elapsed = time.time() - wall_start

        # per-neuron per-1s-bin spike counts, from raw spike times/indices
        spike_times = np.array(post_spikes.t / second)
        spike_indices = np.array(post_spikes.i[:])
        n_bins = int(np.ceil(duration_s / SPIKE_BIN_S))
        bin_edges = np.arange(n_bins + 1) * SPIKE_BIN_S
        spike_rate_bins = np.zeros((N_POST, n_bins))
        for j in range(N_POST):
            counts, _ = np.histogram(spike_times[spike_indices == j], bins=bin_edges)
            spike_rate_bins[j] = counts / SPIKE_BIN_S  # Hz per bin

        final_w = np.array(syn.w[:])
        per_neuron_final = {}
        for j in range(N_POST):
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
            'r_trace': r_trace.r[:].tolist(),          # shape (N_POST, n_t)
            'weight_trace_t': (weight_trace.t / second).tolist(),
            'weight_trace': weight_trace.w[:].tolist(),  # shape (n_synapses, n_t)
            'syn_i': syn_i.tolist(),
            'syn_j': syn_j.tolist(),
            'spike_bin_edges_s': bin_edges.tolist(),
            'spike_rate_bins': spike_rate_bins.tolist(),  # shape (N_POST, n_bins), Hz
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
    inhib_strength_mV = float(sys.argv[3])
    gap_scale = float(sys.argv[4])
    duration_s = float(sys.argv[5])
    calibration_combo_name = sys.argv[6]
    out_path = sys.argv[7]
    run_competitive_seed(seed_val, apre_val, inhib_strength_mV, gap_scale, duration_s,
                          calibration_combo_name, out_path)
    print(f"seed {seed_val} -> {out_path}")

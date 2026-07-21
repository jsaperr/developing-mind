import sys
import json
import time
from pathlib import Path

import numpy as np
from brian2 import SpikeMonitor, StateMonitor, ms, run, second, start_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.brian2_stdp.metrics import compute_population_metrics
from src.brian2_stdp.network import build_population_network
from src.brian2_stdp.spikes import build_population_presynaptic_input

TARGET_RATE = 20.0
P_SHARE = 0.9
N_POST = 5


def run_population_seed(seed_val, apre_val, duration_s, out_path):
    result = {'seed': seed_val, 'status': 'started', 'duration_s': duration_s,
              'p_share': P_SHARE, 'Apre': apre_val, 'n_post': N_POST}
    # write a "started" marker immediately so a hang is distinguishable from "never launched"
    with open(out_path, 'w') as f:
        json.dump(result, f)

    try:
        start_scope()
        rng = np.random.default_rng(seed_val)
        idx, t, n_pre_per_neuron = build_population_presynaptic_input(N_POST, TARGET_RATE, P_SHARE, duration_s, rng)

        pre, post, syn = build_population_network(N_POST, idx, t, apre_val, n_pre_per_neuron=n_pre_per_neuron)

        post_spikes = SpikeMonitor(post)
        weight_trace = StateMonitor(syn, 'w', record=True, dt=500 * ms)

        syn_i = np.array(syn.i[:])
        syn_j = np.array(syn.j[:])

        wall_start = time.time()
        run(duration_s * second)
        wall_elapsed = time.time() - wall_start

        trace = weight_trace.w[:]
        trace_t = (weight_trace.t / second)

        pop_metrics = compute_population_metrics(trace, syn_i, syn_j, N_POST, n_corr=10)

        result.update({
            'status': 'completed',
            'wall_elapsed': wall_elapsed,
            'post_rate': (np.array(post_spikes.count[:]) / duration_s).tolist(),
            'trace_t': trace_t.tolist(),
            'per_neuron': {
                str(j): {
                    'group_mean_gap': pop_metrics[j]['group_mean_gap'].tolist(),
                    'mean_reversals': pop_metrics[j]['mean_reversals'],
                    'final_w': pop_metrics[j]['final_w'].tolist(),
                } for j in range(N_POST)
            },
        })
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    with open(out_path, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    seed_val = int(sys.argv[1])
    apre_val = float(sys.argv[2])
    duration_s = float(sys.argv[3])
    out_path = sys.argv[4]
    run_population_seed(seed_val, apre_val, duration_s, out_path)
    print(f"seed {seed_val} apre={apre_val} -> {out_path}")

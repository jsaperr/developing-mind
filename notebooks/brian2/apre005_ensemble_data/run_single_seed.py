import sys
import json
import time
from pathlib import Path

import numpy as np
from brian2 import SpikeMonitor, StateMonitor, ms, run, second, start_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.brian2_stdp.metrics import compute_group_metrics
from src.brian2_stdp.network import build_network
from src.brian2_stdp.spikes import build_presynaptic_input

TARGET_RATE = 20.0
APRE_VAL = 0.005
P_SHARE = 0.9
DURATION_S = 5000.0


def run_long_seed(seed_val, out_path):
    result = {'seed': seed_val, 'status': 'started', 'duration_s': DURATION_S,
              'p_share': P_SHARE, 'Apre': APRE_VAL}
    # write a "started" marker immediately so a hang is distinguishable from "never launched"
    with open(out_path, 'w') as f:
        json.dump(result, f)

    try:
        start_scope()
        rng = np.random.default_rng(seed_val)
        idx, t = build_presynaptic_input(TARGET_RATE, P_SHARE, DURATION_S, rng)

        pre, post, syn = build_network(idx, t, APRE_VAL)

        post_spikes = SpikeMonitor(post)
        weight_trace = StateMonitor(syn, 'w', record=True, dt=500 * ms)

        wall_start = time.time()
        run(DURATION_S * second)
        wall_elapsed = time.time() - wall_start

        trace = weight_trace.w[:]
        trace_t = (weight_trace.t / second)

        metrics = compute_group_metrics(trace, n_corr=10)
        group_mean_gap = metrics['group_mean_gap']
        group_median_gap = metrics['group_median_gap']
        overlap_fraction = metrics['overlap_fraction']

        final_w = np.array(syn.w[:])

        result.update({
            'status': 'completed',
            'wall_elapsed': wall_elapsed,
            'post_rate': post_spikes.count[0] / DURATION_S,
            'trace_t': trace_t.tolist(),
            'group_mean_gap': group_mean_gap.tolist(),
            'group_median_gap': group_median_gap.tolist(),
            'overlap_fraction': overlap_fraction.tolist(),
            'final_corr_w': final_w[:10].tolist(),
            'final_uncorr_w': final_w[10:].tolist(),
            'min_group_mean_gap': float(group_mean_gap.min()),
            'max_overlap_fraction': float(overlap_fraction.max()),
        })
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    with open(out_path, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    seed_val = int(sys.argv[1])
    out_path = sys.argv[2]
    run_long_seed(seed_val, out_path)
    print(f"seed {seed_val} -> {out_path}")

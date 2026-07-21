import sys
import json
import time
from pathlib import Path

import numpy as np
from brian2 import SpikeMonitor, StateMonitor, ms, run, second, start_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.brian2_stdp.network import build_network
from src.brian2_stdp.spikes import build_presynaptic_input

TARGET_RATE = 20.0
APRE_VAL = 0.005
P_SHARE = 0.9
DURATION_S = 1000.0  # matches the population extension's duration, for direct comparability


def run_positional_bias_replicate(seed_val, out_path):
    """Single-neuron (N=1) replicate at Apre=0.005, 1000s -- extending the sample for the
    positional-bias chi-square check from the population extension's EDA (p=0.068, n=20).
    Reuses build_network as-is, no new mechanism."""
    result = {'seed': seed_val, 'status': 'started', 'duration_s': DURATION_S,
              'p_share': P_SHARE, 'Apre': APRE_VAL}
    with open(out_path, 'w') as f:
        json.dump(result, f)

    try:
        start_scope()
        rng = np.random.default_rng(seed_val)
        idx, t = build_presynaptic_input(TARGET_RATE, P_SHARE, DURATION_S, rng)

        pre, post, syn = build_network(idx, t, APRE_VAL)

        wall_start = time.time()
        run(DURATION_S * second)
        wall_elapsed = time.time() - wall_start

        final_w = np.array(syn.w[:])

        result.update({
            'status': 'completed',
            'wall_elapsed': wall_elapsed,
            'final_corr_w': final_w[:10].tolist(),
            'final_uncorr_w': final_w[10:].tolist(),
        })
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    with open(out_path, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    seed_val = int(sys.argv[1])
    out_path = sys.argv[2]
    run_positional_bias_replicate(seed_val, out_path)
    print(f"seed {seed_val} -> {out_path}")

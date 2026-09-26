"""v1 world for system_contract.md: A->B->C->A->C over five 1000 s phases, one continuous run.

Why this schedule: swap 3 (C->A) returns to a context TWO back and swap 4 (A->C) returns to a
context ONE back. One-back retention (arc 05, novel-C) predicts recognition at swap 4 but not at
swap 3, so both halves are tested within the same seeds. The run also gives the same-rig lull test
for the H readout (arc 07, S1) and shows whether the lock-in residue accumulates over four changes.

Rig identical to novel-C (novelc_data/run_novelc_seed.py): 3 disjoint blocks of 10 (30 inputs), one
correlated per phase, target_total=10 (w_init=1/3), gmax unchanged, N=7 at the 13mV/1.5 reference
with scale_inhib_for_n, apre=0.005, dt=0.2 ms, Cython, I_pert=0. Input comes from
src.brian2_stdp.spikes.build_multiblock_phase_input, which is bit-identical to novel-C's local
builder (tested). Output: .json.gz via results_io (the convention for new runs).

Usage: run_v1_seed.py <seed> <out.json.gz>
"""
import sys
import time
from pathlib import Path

import numpy as np
from brian2 import SpikeMonitor, StateMonitor, defaultclock, mV, ms, run, second, start_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.brian2_stdp.network import build_competitive_population_network, scale_inhib_for_n
from src.brian2_stdp.results_io import save_result
from src.brian2_stdp.spikes import build_multiblock_phase_input

TARGET_RATE = 20.0
P_SHARE = 0.9
N_POST = 7
INHIB_REF_MV = 13.0
GAP_SCALE = 1.5
APRE = 0.005
DT_MS = 0.2
N_BLOCKS = 3
BLOCK = 10
N_PRE = N_BLOCKS * BLOCK
TARGET_TOTAL = 10.0
W_INIT = TARGET_TOTAL / N_PRE
PHASE_DURATIONS_S = [1000.0] * 5
PHASE_CORR_BLOCKS = [0, 1, 2, 0, 2]   # A, B, C, A (2 back), C (1 back)
R_TRACE_DT = 200 * ms
WEIGHT_TRACE_DT = 1 * second
PROGRESS_EVERY_S = 250.0


def run_v1_seed(seed_val, out_path):
    total_s = float(sum(PHASE_DURATIONS_S))
    per_conn = scale_inhib_for_n(N_POST, reference_inhib_mV=INHIB_REF_MV, reference_n_post=3)
    result = {
        'seed': seed_val, 'status': 'started', 'progress_s': 0.0, 'inhib_strength_mV': per_conn,
        'reference_inhib_mV': INHIB_REF_MV, 'gap_scale': GAP_SCALE, 'n_post': N_POST, 'dt_ms': DT_MS,
        'apre': APRE, 'p_share': P_SHARE, 'n_blocks': N_BLOCKS, 'block_size': BLOCK, 'n_pre': N_PRE,
        'w_init': W_INIT, 'target_total': TARGET_TOTAL, 'phase_durations_s': PHASE_DURATIONS_S,
        'phase_corr_blocks': PHASE_CORR_BLOCKS,
        'swap_times_s': np.cumsum(PHASE_DURATIONS_S)[:-1].tolist(), 'total_s': total_s,
    }
    save_result(out_path, result)
    try:
        start_scope()
        defaultclock.dt = DT_MS * ms
        rng = np.random.default_rng(seed_val)
        idx, t = build_multiblock_phase_input(TARGET_RATE, P_SHARE, PHASE_DURATIONS_S, PHASE_CORR_BLOCKS,
                                              rng, n_blocks=N_BLOCKS, block_size=BLOCK)
        pre, post, syn, inhib = build_competitive_population_network(
            N_POST, idx, t, APRE, per_conn * mV, GAP_SCALE, n_pre=N_PRE, w_init=W_INIT,
            target_total=TARGET_TOTAL)
        spikes = SpikeMonitor(post)
        r_trace = StateMonitor(post, 'r', record=True, dt=R_TRACE_DT)
        w_total_trace = StateMonitor(post, 'w_total', record=True, dt=WEIGHT_TRACE_DT)
        weight_trace = StateMonitor(syn, 'w', record=True, dt=WEIGHT_TRACE_DT)
        syn_i = np.array(syn.i[:]); syn_j = np.array(syn.j[:])

        wall0 = time.time(); elapsed = 0.0
        while elapsed < total_s - 1e-9:
            step = min(PROGRESS_EVERY_S, total_s - elapsed)
            run(step * second)
            elapsed += step
            print(f"[seed {seed_val} t={elapsed:.0f}s] wall={time.time() - wall0:.0f}s", flush=True)

        st = np.array(spikes.t / second); si = np.array(spikes.i[:])
        edges = np.arange(int(np.ceil(total_s)) + 1, dtype=float)
        rate_bins = np.stack([np.histogram(st[si == j], bins=edges)[0] for j in range(N_POST)])
        result.update({
            'status': 'completed', 'progress_s': total_s, 'wall_elapsed': time.time() - wall0,
            'syn_i': syn_i.tolist(), 'syn_j': syn_j.tolist(),
            'weight_trace_t': (weight_trace.t / second).tolist(),
            'weight_trace': weight_trace.w[:].round(4).tolist(),
            'r_trace_t': (r_trace.t / second).tolist(), 'r_trace': r_trace.r[:].round(4).tolist(),
            'w_total_trace': w_total_trace.w_total[:].round(4).tolist(),
            'spike_bin_edges_s': edges.tolist(), 'spike_rate_bins': rate_bins.tolist(),
            'overall_post_rate': (np.array(spikes.count[:]) / total_s).tolist(),
            'final_w': np.array(syn.w[:]).tolist(),
        })
    except Exception as e:
        result['status'] = 'failed'; result['error'] = str(e)
    save_result(out_path, result)


if __name__ == '__main__':
    run_v1_seed(int(sys.argv[1]), sys.argv[2])
    print(f"v1 seed {sys.argv[1]} -> {sys.argv[2]}")

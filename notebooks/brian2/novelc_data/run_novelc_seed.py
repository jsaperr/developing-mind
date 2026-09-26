"""Novel-C variant of the non-stationary experiment: phase 3 goes to a pattern the population has
NEVER seen, instead of returning to phase 1's. Adjudicates feature-vs-pathology for the retainer
result -- if ~1/3 of neurons keep holding a stale pattern (A or B) in phase 3 when the world has
moved to novel C, retention is a capacity tax (dead weight), not adaptive division of labour; if
the population commits to C, the earlier phase-3 "instant representation" was adaptive only because
the world happened to return.

Rig: 3 DISJOINT presynaptic blocks of 10 (30 inputs), one correlated per phase, corr blocks
[0,1,2] = A,B,C. target_total held at 10 (w_init=1/3) so total synaptic drive is matched to the
20-input runs -- calibration confirmed in scratchpad (per-neuron 15-17Hz, w_total ~10, gmax
unchanged) before this batch. Everything else (STDP, homeostatic scaling, ambiguity-gated lateral
inhibition, I_pert=0) identical to run_nonstationary_seed.py. Cython backend, dt=0.2ms.

build_phase_switching_multiblock lives here rather than in src/brian2_stdp/spikes.py deliberately:
this is exploratory and shouldn't touch the tested shared module until the experiment is worth
keeping. Promote it (generalizing build_phase_switching_input to n_blocks) if this proceeds.

Usage: run_novelc_seed.py <seed> <inhib_ref_mV> <gap_scale> <out.json> [n_post]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from brian2 import SpikeMonitor, StateMonitor, defaultclock, mV, ms, run, second, start_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.brian2_stdp.network import build_competitive_population_network, scale_inhib_for_n
from src.brian2_stdp.spikes import generate_correlated_group, generate_uncorrelated_group

TARGET_RATE = 20.0
P_SHARE = 0.9
N_POST = 7
DT_MS = 0.2
N_BLOCKS = 3
BLOCK = 10
N_PRE = N_BLOCKS * BLOCK
TARGET_TOTAL = 10.0            # matched to the 20-input rig; drive is set by this, not by n_pre
W_INIT = TARGET_TOTAL / N_PRE
PHASE_DURATIONS_S = [1000.0, 1000.0, 1000.0]
PHASE_CORR_BLOCKS = [0, 1, 2]  # A, B, novel C -- the whole point
R_TRACE_DT = 200 * ms
WEIGHT_TRACE_DT = 1 * second
SPIKE_BIN_S = 1.0
PROGRESS_EVERY_S = 250.0
BOUNDARY_GUARD_S = 0.001


def build_phase_switching_multiblock(phase_durations_s, phase_corr_blocks, rng):
    """One correlated block per phase among N_BLOCKS disjoint blocks of BLOCK; the rest independent
    Poisson. Each phase generated independently, times offset by phase start (+ a small guard so no
    neuron gets two spikes within one dt across a boundary). Rate-matched across all blocks/phases."""
    all_idx, all_t = [], []
    start = 0.0
    for dur, corr_block in zip(phase_durations_s, phase_corr_blocks):
        for b in range(N_BLOCKS):
            if b == corr_block:
                gi, gt = generate_correlated_group(BLOCK, TARGET_RATE, P_SHARE, dur, 2.0, rng)
            else:
                gi, gt = generate_uncorrelated_group(BLOCK, TARGET_RATE, dur, rng)
            guard = BOUNDARY_GUARD_S if start > 0 else 0.0
            all_idx.append(np.asarray(gi) + b * BLOCK)
            all_t.append(np.asarray(gt) + start + guard)
        start += dur
    idx = np.concatenate(all_idx).astype(int)
    t = np.concatenate(all_t)
    order = np.argsort(t)
    return idx[order], t[order] * second


def run_novelc_seed(seed_val, inhib_mV, gap_scale, out_path, n_post=N_POST, apre_val=0.005):
    total_s = float(sum(PHASE_DURATIONS_S))
    swap_times = np.cumsum(PHASE_DURATIONS_S)[:-1].tolist()
    per_connection_mV = scale_inhib_for_n(n_post, reference_inhib_mV=inhib_mV, reference_n_post=3)
    result = {
        'seed': seed_val, 'status': 'started', 'inhib_strength_mV': per_connection_mV,
        'reference_inhib_mV': inhib_mV, 'gap_scale': gap_scale, 'n_post': n_post, 'dt_ms': DT_MS,
        'apre': apre_val, 'p_share': P_SHARE, 'n_blocks': N_BLOCKS, 'block_size': BLOCK,
        'n_pre': N_PRE, 'w_init': W_INIT, 'target_total': TARGET_TOTAL,
        'phase_durations_s': PHASE_DURATIONS_S, 'phase_corr_blocks': PHASE_CORR_BLOCKS,
        'swap_times_s': swap_times, 'total_s': total_s,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f)

    try:
        start_scope()
        defaultclock.dt = DT_MS * ms
        rng = np.random.default_rng(seed_val)
        idx, t = build_phase_switching_multiblock(PHASE_DURATIONS_S, PHASE_CORR_BLOCKS, rng)
        pre, post, syn, inhib = build_competitive_population_network(
            n_post, idx, t, apre_val, per_connection_mV * mV, gap_scale,
            n_pre=N_PRE, w_init=W_INIT, target_total=TARGET_TOTAL)
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
        bin_edges = np.arange(int(np.ceil(total_s / SPIKE_BIN_S)) + 1) * SPIKE_BIN_S
        spike_rate_bins = np.stack([np.histogram(st[si == j], bins=bin_edges)[0] / SPIKE_BIN_S
                                    for j in range(n_post)])
        result.update({
            'status': 'completed', 'wall_elapsed': time.time() - wall0,
            'syn_i': syn_i.tolist(), 'syn_j': syn_j.tolist(),
            'weight_trace_t': (weight_trace.t / second).tolist(),
            'weight_trace': weight_trace.w[:].round(4).tolist(),
            'r_trace_t': (r_trace.t / second).tolist(),
            'r_trace': r_trace.r[:].round(4).tolist(),
            'w_total_trace': w_total_trace.w_total[:].round(4).tolist(),
            'spike_bin_edges_s': bin_edges.tolist(),
            'spike_rate_bins': spike_rate_bins.tolist(),
            'overall_post_rate': (np.array(spikes.count[:]) / total_s).tolist(),
            'final_w': np.array(syn.w[:]).tolist(),
        })
    except Exception as e:
        result['status'] = 'failed'; result['error'] = str(e)

    with open(out_path, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    n_post_arg = int(sys.argv[5]) if len(sys.argv) > 5 else N_POST
    run_novelc_seed(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4],
                    n_post=n_post_arg)
    print(f"novelc seed {sys.argv[1]} -> {sys.argv[4]}")

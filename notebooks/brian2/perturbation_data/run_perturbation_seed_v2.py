"""Perturbation testing, redesign #2, per web's message: direct weight nudge on the target's
CORRELATED synapses, not current injection (I_pert).

Why redesign #1 (I_pert, run_perturbation_seed.py) failed, root-caused not assumed: forcing the
target to fire via a resting-potential offset doesn't respect WHY it fires. STDP potentiates
whichever synapse recently had a presynaptic spike before the postsynaptic one, regardless of
whether that presynaptic spike came from the correlated group -- forced firing corrupts the
target's own selectivity rather than building it. Separately, since inhibition here fires per
postsynaptic spike (not per rate-difference alone), forced extra firing directly increases the
target's raw inhibitory output to competitors -- a different causal pathway than genuine
competitive strength. Confirmed on real data: every validation seed's target gap DECLINED
monotonically with I_pert magnitude, and the one seed that "flipped" did so via leader
suppression (leader's gap fell), not target strength (target's own gap stayed flat).

Fix: directly edit the target's correlated-group synaptic weights toward a "winner-like" value,
then release and let STDP/inhibition run completely unmodified. This sidesteps both failure
modes: any subsequent firing is a legitimate consequence of genuinely stronger correlated-input
weights (no artificial spike-selectivity corruption), and there's no forced-firing-volume
shortcut through the inhibition pathway either.

"Winner-like" calibrated from real data, not guessed: the post-hoc bimodal-distribution analysis
(experiments_brian2.md, 2026-07-20) found correlated synapses in a genuinely winning
configuration cluster near the weight ceiling (wmax=1.0, >0.9 counts as "near ceiling" there,
61% of winning-group synapses land there). Ladder magnitudes here are FRACTIONS of the way from
the target's current correlated-synapse weight toward that ceiling (wmax), not an arbitrary
absolute value: w_new = w_old + fraction * (wmax - w_old). fraction=1.0 sets the target's
correlated synapses directly to the ceiling; smaller fractions are gentler nudges.

The homeostatic scaling mechanism (already running, unmodified, dt=scaling_interval) will
naturally rebalance the target's OTHER (uncorrelated) synapses downward to preserve the
total-weight-sum constraint at its next scheduled correction -- this is not a new confound, it's
the same mechanism that already governs every run in this project, and it's a REALISTIC
consequence of one group of synapses suddenly being stronger, not a testing artifact.

Settling logic reused unchanged from run_perturbation_seed.py (the two-condition adaptive
criterion -- top-tier-set stability AND full-stretch value range under threshold -- already
validated and debugged there). Only the intervention step differs.

Uses dt=0.2ms per web's instruction (validated: real ISI floor, zero spike collisions, 4x
speedup over the previous default -- see the dt-headroom entry).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from brian2 import SpikeMonitor, defaultclock, mV, ms, run, second, start_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.brian2_stdp.metrics import compute_tiers
from src.brian2_stdp.network import build_competitive_population_network
from src.brian2_stdp.spikes import build_presynaptic_input
from notebooks.brian2.perturbation_data.run_perturbation_seed import _is_settled, _per_neuron_gap

TARGET_RATE = 20.0
P_SHARE = 0.9
N_POST = 3
N_CORR = 10
WMAX = 1.0  # matches network.py's stdp wmax clip

DT_MS = 0.2
CHUNK_S = 50.0
STABLE_WINDOWS = 3
THRESHOLD = 0.03
MAX_SETTLE_S = 1200.0
RECOVERY_S = 200.0
# Fractions of the way from current correlated-weight value toward the ceiling (wmax). 1.0 sets
# the target's correlated synapses directly to the ceiling -- the maximal "winner-like" boost.
FRACTIONS = [0.25, 0.5, 0.75, 1.0]


def run_perturbation_seed_v2(seed_val, inhib_mV, gap_scale, out_path, apre_val=0.005):
    result = {
        'seed': seed_val, 'status': 'started', 'inhib_strength_mV': inhib_mV,
        'gap_scale': gap_scale, 'n_post': N_POST, 'dt_ms': DT_MS, 'chunk_s': CHUNK_S,
        'stable_windows': STABLE_WINDOWS, 'threshold': THRESHOLD, 'recovery_s': RECOVERY_S,
        'fractions': FRACTIONS, 'max_settle_s': MAX_SETTLE_S, 'method': 'weight_nudge_v2',
    }
    with open(out_path, 'w') as f:
        json.dump(result, f)

    try:
        start_scope()
        defaultclock.dt = DT_MS * ms
        rng = np.random.default_rng(seed_val)
        total_budget_s = MAX_SETTLE_S + len(FRACTIONS) * RECOVERY_S + 4 * CHUNK_S
        idx, t = build_presynaptic_input(TARGET_RATE, P_SHARE, total_budget_s, rng)
        pre, post, syn, inhib = build_competitive_population_network(
            N_POST, idx, t, apre_val, inhib_mV * mV, gap_scale)
        spikes = SpikeMonitor(post)

        syn_i = np.array(syn.i[:])
        syn_j = np.array(syn.j[:])
        wall_start = time.time()

        # ---- Phase 1: settle (identical to run_perturbation_seed.py) ----
        # Live monitoring per web's ask: print tier state every chunk instead of silence until
        # done. python -u (see the background-script-logging habit in principles.md) means this
        # flushes immediately, so `tail -f` on the log shows it unfold in real time.
        gap_history, t_history = [], []
        elapsed = 0.0
        while elapsed < MAX_SETTLE_S:
            run(CHUNK_S * second)
            elapsed += CHUNK_S
            gap = _per_neuron_gap(np.array(syn.w[:]), syn_i, syn_j)
            gap_history.append(gap)
            t_history.append(elapsed)
            top_now = sorted(compute_tiers(gap, threshold=THRESHOLD)[0])
            print(f"[settle] t={elapsed:.0f}s gap={np.round(gap, 3).tolist()} top={top_now}", flush=True)
            if _is_settled(gap_history):
                break

        settled = _is_settled(gap_history)
        result['settled'] = bool(settled)
        result['settle_time_s'] = float(elapsed)
        result['settling_gap_trajectory'] = [g.tolist() for g in gap_history]
        result['settling_t'] = list(t_history)

        if not settled:
            result['status'] = 'completed'
            result['ladder'] = []
            result['threshold_frac'] = None
            result['censored_above'] = None
            result['skip_reason'] = 'never_settled_within_max_settle_s'
            result['wall_elapsed'] = time.time() - wall_start
            with open(out_path, 'w') as f:
                json.dump(result, f)
            return

        baseline_gap = gap_history[-1]
        baseline_top = sorted(compute_tiers(baseline_gap, threshold=THRESHOLD)[0])
        target = int(np.argmin(baseline_gap))
        result['baseline_top_tier'] = baseline_top
        result['baseline_gap'] = baseline_gap.tolist()
        result['perturb_target'] = target
        result['baseline_n_tiers'] = len(compute_tiers(baseline_gap, threshold=THRESHOLD))

        # Integer indices (not a boolean mask) for the target's CORRELATED synapses only
        # (presynaptic index < N_CORR). Boolean-mask assignment (`syn.w[mask] = values`) was
        # tried first and reproducibly raised "Provided values do not match the size of the
        # indices, 10 != 60" once enough simulated time had elapsed (settling regularly takes
        # 250-450s+) -- confirmed directly: the identical boolean mask assigns fine after a 1s
        # run but fails after 450s, while integer-index assignment (`syn.w[idxs] = values`)
        # works reliably at both durations. Root cause not fully chased down (plausibly an
        # interaction between Brian2's Cython codegen index caching and the `run_regularly`
        # scaling op that's been running the whole time), but the integer-index form is the
        # robust fix regardless of the exact mechanism.
        target_corr_mask = (syn_j == target) & (syn_i < N_CORR)
        target_corr_idxs = np.where(target_corr_mask)[0]
        n_target_corr = int(len(target_corr_idxs))
        result['n_target_corr_synapses'] = n_target_corr

        # ---- Phase 2: escalating weight-nudge ladder ----
        ladder = []
        threshold_frac = None
        for frac in FRACTIONS:
            w_before = np.array(syn.w[:])
            corr_w_before = w_before[target_corr_mask]
            w_new = corr_w_before + frac * (WMAX - corr_w_before)
            syn.w[target_corr_idxs] = w_new
            print(f"[nudge] target={target} fraction={frac} corr_w {corr_w_before.mean():.3f} -> "
                  f"{w_new.mean():.3f}", flush=True)

            rec_gaps, rec_t = [], []
            rec_elapsed = 0.0
            while rec_elapsed < RECOVERY_S:
                run(CHUNK_S * second)
                rec_elapsed += CHUNK_S
                gap = _per_neuron_gap(np.array(syn.w[:]), syn_i, syn_j)
                rec_gaps.append(gap)
                rec_t.append(rec_elapsed)
                top_now = sorted(compute_tiers(gap, threshold=THRESHOLD)[0])
                print(f"[recover frac={frac}] t+{rec_elapsed:.0f}s gap={np.round(gap, 3).tolist()} "
                      f"top={top_now}", flush=True)

            tops = [sorted(compute_tiers(g, threshold=THRESHOLD)[0]) for g in rec_gaps]
            final_top = tops[-1]
            stable_new = len(tops) >= 2 and tops[-1] == tops[-2]
            changed = bool(stable_new and final_top != baseline_top)

            ladder.append({
                'fraction': frac, 'corr_w_before_mean': float(corr_w_before.mean()),
                'corr_w_after_nudge_mean': float(w_new.mean()),
                'recovery_gaps': [g.tolist() for g in rec_gaps], 'recovery_t': rec_t,
                'recovery_top_tiers': tops, 'final_top_tier': final_top,
                'stable_new': bool(stable_new), 'changed': changed,
            })
            if changed:
                threshold_frac = frac
                break

        result['ladder'] = ladder
        result['threshold_frac'] = threshold_frac
        result['censored_above'] = None if threshold_frac is not None else FRACTIONS[-1]
        result['total_spikes'] = int(np.array(spikes.count[:]).sum())
        result['status'] = 'completed'
        result['wall_elapsed'] = time.time() - wall_start
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    with open(out_path, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    run_perturbation_seed_v2(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
    print(f"perturbation v2 seed {sys.argv[1]} -> {sys.argv[4]}")

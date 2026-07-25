"""Perturbation testing: measure BASIN DEPTH of a settled competitive hierarchy, instead of
passively waiting for a rare spontaneous reorganization.

Why this exists (per web's redesign): the 600s reentry probe failed for a confirmed reason --
this system's own settling can consume most of a 600s run, leaving no post-settled window to
observe reentry in. That was proven, not assumed, by the decisive check that even
strong_tight_gate (known-rich at 5000s) read reentry_rate=0 under that design. Running everything
at 5000s would work but is expensive and still passively waits for a rare event.

Instead: let the network settle, then actively push the excluded neuron and see how hard you have
to push before the hierarchy permanently changes rather than snapping back. This converts "did a
rare thing happen in my window?" into "how much force does this structure resist?" -- graded
rather than binary, and many measurements per expensive simulation rather than one.

Protocol per seed:
  1. SETTLE. Run in chunk_s-long chunks, reading per-neuron gap from syn.w at each boundary.
     Settled when the top tier is identical across `stable_windows` consecutive chunks AND every
     neuron's gap range across that stretch stays under `threshold`. This is the same two-condition
     criterion built and debugged for detect_tier_reentry -- top-tier-set stability alone was
     already proven insufficient (a slow continuous drift can hold the tier fixed while values
     still climb). Instantaneous end-of-chunk reads are noisier than window averages, which makes
     this criterion stricter, not looser -- the safe direction.
     If not settled by max_settle_s, the seed is flagged and the ladder is skipped: an unsettled
     network has no baseline for "permanently changed" to mean anything against.
  2. LADDER. Target = lowest-gap neuron at settling. For each magnitude in an escalating ladder:
     apply I_pert to the target (a sustained offset to its effective resting potential), hold for
     hold_s, release, then observe recovery_s in chunks. "Permanently changed" requires the new
     top tier to differ from baseline AND to be the same across the last two recovery chunks --
     a transient excursion during recovery doesn't count.
     Stop at the first magnitude that permanently changes the hierarchy; that magnitude is the
     basin-depth threshold. If none do, the threshold is right-censored (reported as None with
     `censored_above` set), which is itself the informative answer for a deeply locked-in setting.

A sustained I_pert offset is used rather than a one-time `v` nudge for the same reason sigma_v
exists at all: the LIF hard reset erases one-time perturbations at the next spike. That failure
mode is documented in network.py and was hit twice in this project's history.

Saves the full gap trajectory (settling + every ladder phase) so the run can be re-analyzed
without re-simulating -- the standing lesson from the post-hoc analysis that couldn't be done
because only aggregates had been kept.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from brian2 import SpikeMonitor, mV, run, second, start_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.brian2_stdp.metrics import compute_tiers
from src.brian2_stdp.network import build_competitive_population_network
from src.brian2_stdp.spikes import build_presynaptic_input

TARGET_RATE = 20.0
P_SHARE = 0.9
N_POST = 3
N_CORR = 10

CHUNK_S = 50.0
STABLE_WINDOWS = 3          # matches detect_tier_reentry's default -- 150s of stability required
THRESHOLD = 0.03            # matches every tier/classification threshold used in this project
MAX_SETTLE_S = 1200.0
HOLD_S = 50.0
RECOVERY_S = 200.0
# Escalating ladder, in mV of effective resting-potential offset. Scale reference: v_thresh -
# v_rest = 20mV, so 16mV puts the target 4mV from threshold (it will fire very readily), while
# 1mV is a gentle nudge. Spans "barely perceptible" to "should dominate if anything can."
MAGNITUDES_MV = [1.0, 2.0, 4.0, 8.0, 16.0]


def _per_neuron_gap(syn_w, syn_i, syn_j):
    gaps = np.zeros(N_POST)
    for j in range(N_POST):
        mask = syn_j == j
        w_j = syn_w[mask]
        order = np.argsort(syn_i[mask])
        w_j = w_j[order]
        gaps[j] = w_j[:N_CORR].mean() - w_j[N_CORR:].mean()
    return gaps


def _is_settled(gap_history):
    """Two-condition settling check (see module docstring): top tier identical across the last
    STABLE_WINDOWS snapshots AND every neuron's value range across them under THRESHOLD."""
    if len(gap_history) < STABLE_WINDOWS:
        return False
    stretch = np.stack(gap_history[-STABLE_WINDOWS:])
    tops = [frozenset(compute_tiers(g, threshold=THRESHOLD)[0]) for g in stretch]
    if not all(s == tops[0] for s in tops):
        return False
    return bool((stretch.max(axis=0) - stretch.min(axis=0)).max() < THRESHOLD)


def run_perturbation_seed(seed_val, inhib_mV, gap_scale, out_path, apre_val=0.005):
    result = {
        'seed': seed_val, 'status': 'started', 'inhib_strength_mV': inhib_mV,
        'gap_scale': gap_scale, 'n_post': N_POST, 'chunk_s': CHUNK_S,
        'stable_windows': STABLE_WINDOWS, 'threshold': THRESHOLD, 'hold_s': HOLD_S,
        'recovery_s': RECOVERY_S, 'magnitudes_mV': MAGNITUDES_MV, 'max_settle_s': MAX_SETTLE_S,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f)

    try:
        start_scope()
        rng = np.random.default_rng(seed_val)
        total_budget_s = MAX_SETTLE_S + len(MAGNITUDES_MV) * (HOLD_S + RECOVERY_S) + 4 * CHUNK_S
        idx, t = build_presynaptic_input(TARGET_RATE, P_SHARE, total_budget_s, rng)
        pre, post, syn, inhib = build_competitive_population_network(
            N_POST, idx, t, apre_val, inhib_mV * mV, gap_scale)
        spikes = SpikeMonitor(post)

        syn_i = np.array(syn.i[:])
        syn_j = np.array(syn.j[:])
        wall_start = time.time()

        # ---- Phase 1: settle ----
        gap_history, t_history = [], []
        elapsed = 0.0
        while elapsed < MAX_SETTLE_S:
            run(CHUNK_S * second)
            elapsed += CHUNK_S
            gap_history.append(_per_neuron_gap(np.array(syn.w[:]), syn_i, syn_j))
            t_history.append(elapsed)
            if _is_settled(gap_history):
                break

        settled = _is_settled(gap_history)
        result['settled'] = bool(settled)
        result['settle_time_s'] = float(elapsed)
        result['settling_gap_trajectory'] = [g.tolist() for g in gap_history]
        result['settling_t'] = list(t_history)

        if not settled:
            # No stable baseline -> "permanently changed" has nothing to mean. Flag, don't guess.
            result['status'] = 'completed'
            result['ladder'] = []
            result['threshold_mV'] = None
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

        # ---- Phase 2: escalating perturbation ladder ----
        ladder = []
        threshold_mV = None
        for m in MAGNITUDES_MV:
            post.I_pert[target] = m * mV
            run(HOLD_S * second)
            hold_gap = _per_neuron_gap(np.array(syn.w[:]), syn_i, syn_j)

            post.I_pert[target] = 0 * mV
            rec_gaps, rec_t = [], []
            rec_elapsed = 0.0
            while rec_elapsed < RECOVERY_S:
                run(CHUNK_S * second)
                rec_elapsed += CHUNK_S
                rec_gaps.append(_per_neuron_gap(np.array(syn.w[:]), syn_i, syn_j))
                rec_t.append(rec_elapsed)

            tops = [sorted(compute_tiers(g, threshold=THRESHOLD)[0]) for g in rec_gaps]
            final_top = tops[-1]
            # a transient excursion during recovery isn't a permanent change -- require the last
            # two recovery chunks to agree with each other AND to differ from baseline
            stable_new = len(tops) >= 2 and tops[-1] == tops[-2]
            changed = bool(stable_new and final_top != baseline_top)

            ladder.append({
                'magnitude_mV': m, 'hold_gap': hold_gap.tolist(),
                'recovery_gaps': [g.tolist() for g in rec_gaps], 'recovery_t': rec_t,
                'recovery_top_tiers': tops, 'final_top_tier': final_top,
                'stable_new': bool(stable_new), 'changed': changed,
            })
            if changed:
                threshold_mV = m
                break

        result['ladder'] = ladder
        result['threshold_mV'] = threshold_mV
        result['censored_above'] = None if threshold_mV is not None else MAGNITUDES_MV[-1]
        result['total_spikes'] = int(np.array(spikes.count[:]).sum())
        result['status'] = 'completed'
        result['wall_elapsed'] = time.time() - wall_start
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    with open(out_path, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    run_perturbation_seed(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
    print(f"perturbation seed {sys.argv[1]} -> {sys.argv[4]}")

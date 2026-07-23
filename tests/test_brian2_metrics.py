"""Regression checks for src/brian2_stdp/metrics.py. Pure numpy -- no brian2 import needed, so
these run under any Python environment, not just the developing-mind conda env."""
import numpy as np

from src.brian2_stdp.metrics import (
    classify_differentiation, classify_hierarchy, compute_competitive_metrics, compute_tiers,
    count_identity_swaps, count_reversals, compute_group_metrics, full_rank_swap_count,
)


def test_compute_group_metrics_gap_and_overlap():
    trace = np.zeros((4, 3))
    trace[:2] = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]   # correlated group, n_corr=2
    trace[2:] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]   # uncorrelated group
    result = compute_group_metrics(trace, n_corr=2)
    assert np.allclose(result["group_mean_gap"], [1.0, 1.0, 1.0])
    assert np.allclose(result["overlap_fraction"], [0.0, 0.0, 0.0])


def test_count_reversals_zero_for_monotonic_trace():
    assert count_reversals(np.linspace(0, 1, 20)) == 0


def test_count_reversals_detects_oscillation():
    trace = np.array([0, 1, 0, 1, 0, 1, 0, 1.0] * 3)
    assert count_reversals(trace, smooth_window=1) > 0


def test_compute_competitive_metrics_identifies_holder():
    # 2 synapses per neuron (n_corr=1), 2 postsynaptic neurons, 2 timepoints.
    # neuron 0: corr=1.0, uncorr=0.0 -> gap 1.0 at both timepoints (clear holder)
    # neuron 1: corr=0.0, uncorr=0.0 -> gap 0.0 at both timepoints
    weight_trace = np.array([
        [1.0, 1.0],  # syn (pre=0, post=0), corr
        [0.0, 0.0],  # syn (pre=1, post=0), uncorr
        [0.0, 0.0],  # syn (pre=0, post=1), corr
        [0.0, 0.0],  # syn (pre=1, post=1), uncorr
    ])
    syn_i = np.array([0, 1, 0, 1])
    syn_j = np.array([0, 0, 1, 1])
    result = compute_competitive_metrics(weight_trace, syn_i, syn_j, n_post=2, n_corr=1)
    assert np.allclose(result["per_neuron_gap"][0], [1.0, 1.0])
    assert np.allclose(result["per_neuron_gap"][1], [0.0, 0.0])
    assert np.array_equal(result["holder_identity"], [0, 0])
    assert result["population_max_gap"].tolist() == [1.0, 1.0]


def test_count_identity_swaps_counts_transitions():
    assert count_identity_swaps(np.array([0, 0, 0, 1, 1, 0, 0])) == 2
    assert count_identity_swaps(np.array([0, 0, 0, 0])) == 0


def test_classify_differentiation_converge_case():
    # 3 neurons, late window all near-identical gaps -> low cross-neuron std -> converge
    t = np.linspace(0, 600, 601)
    per_neuron_gap = np.tile(np.array([[0.58], [0.585], [0.582]]), (1, len(t)))
    label, std = classify_differentiation(per_neuron_gap, t, late_window_s=100, threshold=0.03)
    assert label == 'converge'
    assert std < 0.03


def test_classify_differentiation_differentiate_case():
    # 3 neurons, late window clearly split (one winner, two followers) -> differentiate
    t = np.linspace(0, 600, 601)
    per_neuron_gap = np.tile(np.array([[0.78], [0.40], [0.42]]), (1, len(t)))
    label, std = classify_differentiation(per_neuron_gap, t, late_window_s=100, threshold=0.03)
    assert label == 'differentiate'
    assert std > 0.03


def test_classify_differentiation_only_uses_late_window():
    # early window is a wild split, late window has fully settled to converged -- must classify
    # based on the late window only, not be thrown off by early transient divergence.
    t = np.linspace(0, 600, 601)
    early = np.array([[0.9], [0.1], [0.5]]) * np.ones((3, 400))
    late = np.array([[0.58], [0.585], [0.582]]) * np.ones((3, 201))
    per_neuron_gap = np.hstack([early, late])
    label, std = classify_differentiation(per_neuron_gap, t, late_window_s=100, threshold=0.03)
    assert label == 'converge'


def test_compute_tiers_single_tier_when_converged():
    tiers = compute_tiers(np.array([0.58, 0.585, 0.582]), threshold=0.03)
    assert len(tiers) == 1
    assert sorted(tiers[0]) == [0, 1, 2]


def test_compute_tiers_leader_plus_followers():
    tiers = compute_tiers(np.array([0.40, 0.78, 0.42]), threshold=0.03)  # neuron 1 leads
    assert len(tiers) == 2
    assert tiers[0] == [1]
    assert sorted(tiers[1]) == [0, 2]


def test_compute_tiers_fully_distinct_hierarchy():
    tiers = compute_tiers(np.array([0.9, 0.5, 0.1]), threshold=0.03)
    assert [t[0] for t in tiers] == [0, 1, 2]
    assert all(len(t) == 1 for t in tiers)


def test_full_rank_swap_count_zero_when_order_never_changes():
    per_neuron_gap = np.tile(np.array([[0.9], [0.5], [0.1]]), (1, 50))
    assert full_rank_swap_count(per_neuron_gap) == 0


def test_full_rank_swap_count_detects_non_leader_churn():
    # leader (neuron 0) never changes, but neurons 1 and 2 swap 2nd/3rd place repeatedly --
    # invisible to a holder-identity (top-1-only) swap count, must be caught here.
    n_t = 10
    per_neuron_gap = np.zeros((3, n_t))
    per_neuron_gap[0, :] = 0.9  # leader, constant
    for t_i in range(n_t):
        if t_i % 2 == 0:
            per_neuron_gap[1, t_i], per_neuron_gap[2, t_i] = 0.5, 0.4
        else:
            per_neuron_gap[1, t_i], per_neuron_gap[2, t_i] = 0.4, 0.5
    assert full_rank_swap_count(per_neuron_gap) == n_t - 1


def test_classify_hierarchy_converge_case():
    t = np.linspace(0, 600, 601)
    per_neuron_gap = np.tile(np.array([[0.58], [0.585], [0.582], [0.581]]), (1, len(t)))
    result = classify_hierarchy(per_neuron_gap, t, late_window_s=100)
    assert result['label'] == 'converge'
    assert len(result['tiers']) == 1


def test_classify_hierarchy_stable_differentiate_case():
    # 4 neurons, clean stable 2-tier split (1 leader, 3 followers) throughout the late window
    t = np.linspace(0, 600, 601)
    per_neuron_gap = np.tile(np.array([[0.80], [0.40], [0.42], [0.38]]), (1, len(t)))
    result = classify_hierarchy(per_neuron_gap, t, late_window_s=100)
    assert result['label'] == 'differentiate'
    assert result['n_distinct_top_sets'] == 1
    assert result['tiers'][0] == [0]


def test_classify_hierarchy_detects_disorder():
    # 4 neurons, real spread (not converged) but leadership keeps rotating across every
    # sub-window within the late window -- never settles into a consistent structure.
    t = np.linspace(0, 600, 601)
    late_mask = t >= 500
    per_neuron_gap = np.random.RandomState(0).uniform(0.3, 0.5, size=(4, len(t)))
    # force a different, clear leader in each of 5 sub-windows across the late window
    late_idx = np.where(late_mask)[0]
    sub_chunks = np.array_split(late_idx, 5)
    leaders = [0, 1, 2, 3, 0]
    for leader, chunk in zip(leaders, sub_chunks):
        per_neuron_gap[:, chunk] = 0.3
        per_neuron_gap[leader, chunk] = 0.9
    result = classify_hierarchy(per_neuron_gap, t, late_window_s=100)
    assert result['label'] == 'disorder'
    assert result['n_distinct_top_sets'] > 2

"""Metrics for the STDP correlated-vs-uncorrelated differentiation experiments.

group_mean_gap / overlap_fraction: from run_single_seed.py, the primary metrics for the
long-ensemble stability question (group-mean gap trajectory, not individual pairwise crossings
-- pairwise overlap alone was found contaminated by ordinary within-group spread, see
experiments_brian2.md).

count_reversals: from brian2_stdp_apre_sweep.ipynb -- the metric behind the "standing
explanation" that Apre controls excursion amplitude, not switching frequency.
"""
import numpy as np


def compute_group_metrics(trace, n_corr=10):
    """trace: array shape (n_synapses, n_timepoints); first n_corr rows are the correlated group.

    Returns group_mean_gap, group_median_gap (both shape (n_timepoints,)), and overlap_fraction
    (fraction of correlated x uncorrelated pairs where corr < uncorr at each timepoint).
    """
    corr_trace = trace[:n_corr]
    uncorr_trace = trace[n_corr:]
    group_mean_gap = corr_trace.mean(axis=0) - uncorr_trace.mean(axis=0)
    group_median_gap = np.median(corr_trace, axis=0) - np.median(uncorr_trace, axis=0)

    n_t = trace.shape[1]
    overlap_fraction = np.zeros(n_t)
    for ti in range(n_t):
        c = corr_trace[:, ti]
        u = uncorr_trace[:, ti]
        pairwise = c[:, None] - u[None, :]
        overlap_fraction[ti] = (pairwise < 0).mean()

    return {
        "group_mean_gap": group_mean_gap,
        "group_median_gap": group_median_gap,
        "overlap_fraction": overlap_fraction,
    }


def count_reversals(trace_row, smooth_window=5):
    """Mean direction-reversal count for one synapse's weight trajectory: sign changes in a
    lightly-smoothed derivative. Found to be essentially Apre-invariant across the tested range
    -- Apre controls excursion amplitude, not this."""
    if len(trace_row) < smooth_window + 2:
        return 0
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(trace_row, kernel, mode="valid")
    deriv = np.diff(smoothed)
    sign = np.sign(deriv)
    sign = sign[sign != 0]
    if len(sign) < 2:
        return 0
    return int(np.sum(sign[1:] != sign[:-1]))


def compute_population_metrics(trace, syn_i, syn_j, n_post, n_corr=10):
    """Per-postsynaptic-neuron group_mean_gap trajectory, reversal frequency, and final weight
    vector, for a population built with network.build_population_network (block-diagonal: each
    postsynaptic neuron has its own dedicated presynaptic block).

    trace: shape (n_synapses, n_timepoints), rows in synapse creation order.
    syn_i/syn_j: (global) presynaptic/postsynaptic index per synapse (same order as trace's
    rows) -- from `np.array(syn.i[:])` / `np.array(syn.j[:])` recorded at network-build time.
    Sorting the synapses belonging to one postsynaptic neuron by their (block-local, since
    global index within one block is just a constant offset from local) presynaptic index puts
    the correlated group first -- matches spikes.build_presynaptic_input's layout.

    Returns {post_index: {"group_mean_gap": ..., "mean_reversals": ..., "final_w": ...}}.
    """
    results = {}
    for j in range(n_post):
        mask = syn_j == j
        sub_trace = trace[mask]
        sub_i = syn_i[mask]
        order = np.argsort(sub_i)  # sort by presynaptic index so [:n_corr] is the correlated group
        sub_trace = sub_trace[order]

        corr_trace = sub_trace[:n_corr]
        uncorr_trace = sub_trace[n_corr:]
        group_mean_gap = corr_trace.mean(axis=0) - uncorr_trace.mean(axis=0)
        reversals = [count_reversals(row) for row in sub_trace]

        results[j] = {
            "group_mean_gap": group_mean_gap,
            "mean_reversals": float(np.mean(reversals)),
            "final_w": sub_trace[:, -1],
        }
    return results


def compute_competitive_metrics(weight_trace, syn_i, syn_j, n_post, n_corr=10):
    """Population-level metrics for the shared-input competitive population (built with
    network.build_competitive_population_network -- genuine all-to-all shared input, not
    block-diagonal, so syn_i's meaning (correlated < n_corr) is the same for every postsynaptic
    neuron rather than block-relative).

    weight_trace: shape (n_synapses, n_timepoints), rows in synapse creation order.
    syn_i/syn_j: presynaptic/postsynaptic index per synapse, same order as weight_trace's rows.

    Returns:
      per_neuron_gap: shape (n_post, n_timepoints) -- each neuron's own corr_w-uncorr_w gap.
      population_max_gap: shape (n_timepoints,) -- max per_neuron_gap across neurons at each
        timepoint. The primary "is the correlated pattern represented by *someone*" signal.
      holder_identity: shape (n_timepoints,) int -- argmax_j(per_neuron_gap) at each timepoint,
        i.e. which neuron currently "holds" the strongest representation. Track this over time
        (not just its final value) to see whether identity swaps during a run, independent of
        whether the population-level signal itself stays stable.
    """
    per_neuron_gap = np.zeros((n_post, weight_trace.shape[1]))
    for j in range(n_post):
        mask = syn_j == j
        w_j = weight_trace[mask]
        i_j = syn_i[mask]
        order = np.argsort(i_j)
        w_j = w_j[order]
        per_neuron_gap[j] = w_j[:n_corr].mean(axis=0) - w_j[n_corr:].mean(axis=0)

    population_max_gap = per_neuron_gap.max(axis=0)
    holder_identity = per_neuron_gap.argmax(axis=0)

    return {
        "per_neuron_gap": per_neuron_gap,
        "population_max_gap": population_max_gap,
        "holder_identity": holder_identity,
    }


def count_identity_swaps(holder_identity):
    """Number of times holder_identity (from compute_competitive_metrics) changes value --
    how many times the "who currently best represents the correlated pattern" role swapped
    over the run."""
    return int(np.sum(np.diff(holder_identity) != 0))


def classify_differentiation(per_neuron_gap, weight_trace_t, late_window_s=100.0, threshold=0.03):
    """Basin classification for the competitive-population network: 'differentiate' (a real
    hierarchy formed) vs. 'converge' (all postsynaptic neurons settled to the same shared
    representation). Reproduces the informal criterion used to sort the original
    strong_tight_gate calibration/seed-expansion runs (late-window cross-neuron gap std > 0.03),
    now as shared code rather than a one-off inline calculation -- see experiments_brian2.md's
    "Shared-input population competition" and "Seed expansion" entries for where this criterion
    came from.

    per_neuron_gap: shape (n_post, n_timepoints), from compute_competitive_metrics.
    weight_trace_t: shape (n_timepoints,), seconds -- used to select the late window rather than
    assuming a fixed sample count, so this works at any run duration/sampling rate.

    Returns (label, late_window_std) -- label is 'differentiate' or 'converge'; late_window_std
    is returned too so callers can inspect margin from the threshold, not just the binary call."""
    weight_trace_t = np.asarray(weight_trace_t)
    duration = weight_trace_t[-1]
    late_mask = weight_trace_t >= (duration - late_window_s)
    late_mean_per_neuron = per_neuron_gap[:, late_mask].mean(axis=1)
    late_window_std = float(late_mean_per_neuron.std())
    label = 'differentiate' if late_window_std > threshold else 'converge'
    return label, late_window_std

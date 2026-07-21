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

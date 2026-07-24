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


def detect_tier_reentry(per_neuron_gap, weight_trace_t, washout_s=100.0, window_s=100.0, threshold=0.03):
    """Does any neuron genuinely re-enter the top tier after an initial settling/washout period,
    despite NOT being in the top tier during that settled baseline? Catches actual reorganization
    (a previously-excluded neuron regaining contention) as distinct from noise-level swapping
    among an already-decided top tier -- the exact distinction Test A's step-4 analysis needed
    to build by hand; this is that check, generalized and reusable.

    Splits (washout_s, duration] into window_s-long windows, computes the top tier
    (compute_tiers) within each via the window-averaged gap. baseline_top = the top tier in the
    FIRST post-washout window. A neuron counts as a genuine reentrant only if it appears in the
    top tier of some LATER window while having been absent from baseline_top -- being in the top
    tier at every window (even a big, noisy one) is not reentry, it never left.

    Returns {'reentered': bool, 'reentrants': set of neuron indices, 'first_reentry_t': float or
    None, 'baseline_top': set}. Caveat, not solved here: this only detects reentry events that
    happen to occur within `duration` -- if genuine reorganization typically takes longer than
    the window tested (e.g. the ~1000-2600s range seen in the original strong_tight_gate
    typology, well past a 600s calibration-scale run), a short run will systematically undercount
    it. A 'reentered: False' result at short duration means "no reentry observed in this window,"
    not "this setting never reorganizes at any timescale" -- see the boundary-mapping sweep
    entry in experiments_brian2.md for how this gets interpreted in practice."""
    weight_trace_t = np.asarray(weight_trace_t)
    duration = weight_trace_t[-1]
    edges = np.arange(washout_s, duration + 1e-9, window_s)
    windows_top = []
    for i in range(len(edges) - 1):
        mask = (weight_trace_t >= edges[i]) & (weight_trace_t < edges[i + 1])
        if mask.sum() == 0:
            continue
        window_mean = per_neuron_gap[:, mask].mean(axis=1)
        tiers = compute_tiers(window_mean, threshold=threshold)
        windows_top.append((edges[i], frozenset(tiers[0])))

    if len(windows_top) < 2:
        return {'reentered': False, 'reentrants': set(), 'first_reentry_t': None, 'baseline_top': set()}

    baseline_top = windows_top[0][1]
    reentrants = set()
    first_reentry_t = None
    for t, top in windows_top[1:]:
        new_members = top - baseline_top
        if new_members:
            reentrants |= new_members
            if first_reentry_t is None:
                first_reentry_t = float(t)

    return {
        'reentered': len(reentrants) > 0, 'reentrants': reentrants,
        'first_reentry_t': first_reentry_t, 'baseline_top': set(baseline_top),
    }


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


def compute_tiers(late_mean_per_neuron, threshold=0.03):
    """Groups neurons into tiers by their late-window mean gap: sort descending, start a new
    tier wherever the gap to the next neuron exceeds `threshold` (same units/threshold as
    classify_differentiation, for consistency). Returns a list of tiers, each a list of neuron
    indices, highest tier first.

    Generalizes classify_differentiation beyond N=3's binary call: 1 tier containing every
    neuron is exactly the 'converge' case; N tiers of size 1 is a fully-distinct hierarchy;
    anything in between is a real intermediate shape (e.g. a leader tier + a followers tier, or
    2 co-leaders + N-2 followers) that the old binary classifier couldn't distinguish."""
    late_mean_per_neuron = np.asarray(late_mean_per_neuron)
    order = np.argsort(-late_mean_per_neuron)
    sorted_vals = late_mean_per_neuron[order]
    tiers = [[int(order[0])]]
    for k in range(1, len(order)):
        if sorted_vals[k - 1] - sorted_vals[k] > threshold:
            tiers.append([int(order[k])])
        else:
            tiers[-1].append(int(order[k]))
    return tiers


def full_rank_swap_count(per_neuron_gap):
    """Generalizes count_identity_swaps beyond just the top slot: counts how often the FULL
    descending rank order of all neurons changes over time, not just who's in 1st. At higher N,
    real churn can happen among 2nd/3rd/4th place without ever changing the leader -- invisible
    to holder-identity swaps alone, visible here."""
    ranks = np.argsort(-per_neuron_gap, axis=0)  # shape (n_post, n_t)
    changes = np.any(ranks[:, 1:] != ranks[:, :-1], axis=0)
    return int(changes.sum())


def classify_hierarchy(per_neuron_gap, weight_trace_t, late_window_s=100.0, n_subwindows=5,
                        tier_threshold=0.03, std_threshold=0.03):
    """Three-way generalization of classify_differentiation for N>3, where a real third outcome
    becomes possible: not just converged or a stable hierarchy, but genuine DISORDER -- real
    cross-neuron spread that never resolves into a consistent structure, a category that wasn't
    distinguishable from 'differentiate' at N=3.

    Splits the late window into n_subwindows equal pieces and computes each sub-window's TOP
    TIER SET (the neuron(s) within tier_threshold of that sub-window's max mean gap -- i.e. who's
    currently leading, allowing near-ties). If the overall late-window spread is near zero,
    that's 'converge' (unchanged from classify_differentiation). Otherwise: if the top-tier set
    stays the same (or changes at most once, allowing one legitimate settling-in transition
    within the window -- the same "laggard promoted late" pattern already validated as a real
    stable-hierarchy-with-a-transition case in the n=7 typology) across sub-windows, that's a
    genuine stable hierarchy -- 'differentiate'. If the top-tier set keeps changing across more
    than 2 distinct sets, leadership never actually settles within the late window -- 'disorder',
    a real spread that never becomes a consistent structure, not just a noisier version of a
    stable hierarchy.

    Returns dict: label ('converge'/'differentiate'/'disorder'), late_window_std, n_distinct_top_sets,
    tiers (from compute_tiers on the full late window), top_tier_sets_per_subwindow (list of sets,
    for inspection)."""
    weight_trace_t = np.asarray(weight_trace_t)
    duration = weight_trace_t[-1]
    late_start = duration - late_window_s
    late_mask = weight_trace_t >= late_start
    late_mean_per_neuron = per_neuron_gap[:, late_mask].mean(axis=1)
    late_window_std = float(late_mean_per_neuron.std())

    sub_edges = np.linspace(late_start, duration, n_subwindows + 1)
    top_tier_sets = []
    for k in range(n_subwindows):
        sub_mask = (weight_trace_t >= sub_edges[k]) & (weight_trace_t < sub_edges[k + 1] + 1e-9)
        if sub_mask.sum() == 0:
            continue
        sub_mean = per_neuron_gap[:, sub_mask].mean(axis=1)
        top_val = sub_mean.max()
        top_set = frozenset(np.where(sub_mean >= top_val - tier_threshold)[0].tolist())
        top_tier_sets.append(top_set)
    n_distinct_top_sets = len(set(top_tier_sets))

    if late_window_std <= std_threshold:
        label = 'converge'
    elif n_distinct_top_sets <= 2:
        label = 'differentiate'
    else:
        label = 'disorder'

    return {
        'label': label,
        'late_window_std': late_window_std,
        'n_distinct_top_sets': n_distinct_top_sets,
        'tiers': compute_tiers(late_mean_per_neuron, threshold=tier_threshold),
        'top_tier_sets_per_subwindow': top_tier_sets,
    }

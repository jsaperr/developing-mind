"""Correlated/uncorrelated presynaptic spike-train construction for the STDP experiment series.

Extracted from notebooks/brian2/apre005_ensemble_data/run_single_seed.py, confirmed
byte-for-byte identical to the simulation setup in brian2_stdp_apre_sweep.ipynb.
"""
import numpy as np
from brian2 import second


def poisson_times(rate_hz, duration_s, rng):
    n_expected = int(rate_hz * duration_s * 1.5) + 10
    isi = rng.exponential(1.0 / rate_hz, size=n_expected)
    times = np.cumsum(isi)
    return times[times < duration_s]


def dedup_spike_times(times, min_gap=0.0002):
    """Enforce a minimum gap between consecutive spikes for one neuron -- correlated-group
    generation can otherwise produce near-duplicate times within a single Brian2 dt."""
    if len(times) == 0:
        return times
    times = np.sort(times)
    keep = [times[0]]
    for t_val in times[1:]:
        if t_val - keep[-1] >= min_gap:
            keep.append(t_val)
    return np.array(keep)


def generate_correlated_group(n_neurons, target_rate_hz, p_share, duration_s, jitter_ms, rng):
    """Rate-matched correlated group: spikes copied from a shared master process with
    probability p_share, jittered, backfilled with independent spikes so mean rate stays equal
    to the uncorrelated group regardless of p_share."""
    master_times = poisson_times(target_rate_hz, duration_s, rng)
    all_times, all_indices = [], []
    for i in range(n_neurons):
        keep = rng.random(len(master_times)) < p_share
        shared = master_times[keep] + rng.normal(0, jitter_ms / 1000.0, size=keep.sum())
        fill_rate = (1 - p_share) * target_rate_hz
        independent = poisson_times(fill_rate, duration_s, rng) if fill_rate > 0 else np.array([])
        combined = np.clip(np.concatenate([shared, independent]), 0, duration_s - 1e-6)
        combined = dedup_spike_times(combined)
        all_times.append(combined)
        all_indices.append(np.full(len(combined), i))
    return np.concatenate(all_indices), np.concatenate(all_times)


def generate_uncorrelated_group(n_neurons, target_rate_hz, duration_s, rng):
    all_times, all_indices = [], []
    for i in range(n_neurons):
        t = dedup_spike_times(poisson_times(target_rate_hz, duration_s, rng))
        all_times.append(t)
        all_indices.append(np.full(len(t), i))
    return np.concatenate(all_indices), np.concatenate(all_times)


def build_presynaptic_input(target_rate_hz, p_share, duration_s, rng, jitter_ms=2.0,
                             n_correlated=10, n_uncorrelated=10):
    """Returns (indices, times_with_units) ready for SpikeGeneratorGroup(n_correlated +
    n_uncorrelated, indices, times) -- correlated group is indices [0, n_correlated), uncorrelated
    is [n_correlated, n_correlated + n_uncorrelated)."""
    corr_idx, corr_t = generate_correlated_group(n_correlated, target_rate_hz, p_share, duration_s, jitter_ms, rng)
    uncorr_idx, uncorr_t = generate_uncorrelated_group(n_uncorrelated, target_rate_hz, duration_s, rng)
    uncorr_idx = uncorr_idx + n_correlated
    all_idx = np.concatenate([corr_idx, uncorr_idx]).astype(int)
    all_t = np.concatenate([corr_t, uncorr_t])
    order = np.argsort(all_t)
    return all_idx[order], all_t[order] * second


def build_population_presynaptic_input(n_post, target_rate_hz, p_share, duration_s, rng,
                                        jitter_ms=2.0, n_correlated=10, n_uncorrelated=10):
    """n_post independent presynaptic blocks (each its own build_presynaptic_input draw, same
    generative process, independent randomness), concatenated into one global index space --
    block b occupies global presynaptic indices [b*n_pre_per_neuron, (b+1)*n_pre_per_neuron).

    This is the real symmetry-breaking for a population network: giving every postsynaptic
    neuron a literally different input, rather than nudging shared-input initial weights, which
    was tried first and found NOT to produce divergent postsynaptic neurons -- the LIF hard
    reset (`v = v_reset` on every spike) erases pre-spike membrane differences every interspike
    interval, and threshold-crossing timing turned out to be robust to small weight jitter, so
    all neurons converged to bit-identical trajectories despite different starting weights.

    Returns (indices, times_with_units) ready for SpikeGeneratorGroup(n_post * n_pre_per_neuron,
    indices, times), plus n_pre_per_neuron for use by build_network's block-diagonal connect.
    """
    n_pre_per_neuron = n_correlated + n_uncorrelated
    all_idx, all_t = [], []
    for b in range(n_post):
        idx, t = build_presynaptic_input(target_rate_hz, p_share, duration_s, rng,
                                          jitter_ms=jitter_ms, n_correlated=n_correlated,
                                          n_uncorrelated=n_uncorrelated)
        all_idx.append(idx + b * n_pre_per_neuron)
        all_t.append(t)
    combined_idx = np.concatenate(all_idx)
    combined_t = np.concatenate([arr / second for arr in all_t])
    order = np.argsort(combined_t)
    return combined_idx[order].astype(int), combined_t[order] * second, n_pre_per_neuron

"""Substrate -> memory interface signals (system_contract.md), promoted from the coupling toy.

Everything here is LABEL-FREE and CAUSAL: it uses only the substrate's own rates and weights and
their past, never knowledge of which input block is correlated or when the world changes.
Lineage: notebooks/integration/ (coupling_v0, content_consolidation, two_back_test) and
notebooks/brian2/interface_readout/ (S1). See experiments_integration.md and docs/log/brian2/07.

  h_readout             the query memory sees: activity projected through learned tuning,
                        sum_j (r_j - mean r) w_j, centered and unit-normalized. S1: rates alone say
                        "now vs just before", weights alone are a history code, and this carries
                        context identity. Weak before the first world change (cold start: the
                        whole population agrees, so there's no contrast).
  steady_flags          the query hasn't jumped: cosine to the previous query >= tau on two
                        consecutive steps. Catches the ABRUPT onset of a change.
  changing_per_second   the substrate is still re-learning: weight displacement over the last
                        L s above a causal trailing threshold. Catches the SLOW post-change drift
                        (about 200 s) that steady_flags misses. Per-second |dw| doesn't work,
                        since it's dominated by constant STDP churn (arc 01's standing
                        explanation); displacement over a window cancels churn and keeps
                        directional re-learning.
  window_any            reduce a per-second flag to memory steps (any second in the window).
"""
import numpy as np


def unit(v):
    """Center, then normalize to unit length (zero vector stays zero)."""
    v = np.asarray(v, dtype=float)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def h_readout(rates, weights):
    """rates: (n_post,) mean rates over a window. weights: (n_post, n_pre) mean weights over the same
    window, rows = postsynaptic neurons, columns = presynaptic inputs in index order."""
    rates = np.asarray(rates, dtype=float)
    return unit((rates - rates.mean()) @ np.asarray(weights, dtype=float))


def h_readout_rectified(rates, weights):
    """H+: what the DRIVEN neurons are tuned to, sum_j max(r_j - mean r, 0) w_j, centered and
    unit-normalized. Quieter neurons get weight 0 instead of negative weight, so it doesn't subtract
    the previous context the retainers hold. Candidate replacement for h_readout (the default switch
    is pending Jasper's call), from notebooks/integration/set_worlds/compare_readouts.py and
    run_rectified_memory.py:
      settled quality: 0.97 disjoint, 0.96 at 50% overlap (h_readout 0.87-0.90 / 0.46)
      previous-context leakage about 0 (h_readout about -0.7: it subtracts it)
      cold start fixed: phase-1 quality 0.96 (h_readout 0.2-0.34), first context recognized 8/8
    Weak spot left: 50% overlap at a 10 s clock (rehearsal survival 1/8; 5 absorption events)."""
    rates = np.asarray(rates, dtype=float)
    return unit(np.clip(rates - rates.mean(), 0, None) @ np.asarray(weights, dtype=float))


def steady_flags(queries, tau=0.9):
    """queries: (n_steps, dim), each already unit(). Step s is steady if s >= 2 and the two most
    recent consecutive cosines are both >= tau."""
    Q = np.asarray(queries, dtype=float)
    cs = np.r_[np.nan, np.sum(Q[1:] * Q[:-1], axis=1)]
    out = np.zeros(len(Q), bool)
    for s in range(2, len(Q)):
        out[s] = cs[s] >= tau and cs[s - 1] >= tau
    return out


def changing_per_second(weight_trace, L=60, trail=900, k=3.0, min_hist=100):
    """weight_trace: array with time as the LAST axis, sampled once per second (any leading shape,
    e.g. (n_synapses, n_t) or (n_post, n_pre, n_t)). Returns a bool per second: displacement
    D(t) = sum |w(t) - w(t-L)| exceeds median + k * 1.4826 * MAD of D over [t - trail, t - L].
    It's True while there's less than min_hist seconds of history (the substrate is still forming).
    Defaults are the values fixed a priori in the coupling toy (not tuned)."""
    w = np.asarray(weight_trace, dtype=float)
    n_t = w.shape[-1]
    D = np.full(n_t, np.nan)
    if n_t > L:
        D[L:] = np.abs(w[..., L:] - w[..., :-L]).reshape(-1, n_t - L).sum(axis=0)
    flag = np.ones(n_t, bool)
    for t in range(L, n_t):
        hist = D[max(L, t - trail):t - L]
        hist = hist[np.isfinite(hist)]
        if len(hist) < min_hist:
            continue
        med = np.median(hist)
        flag[t] = D[t] > med + k * 1.4826 * np.median(np.abs(hist - med))
    return flag


def window_any(flag_per_second, window_starts, W):
    """Per memory step: is any second in [start, start + W) flagged?"""
    f = np.asarray(flag_per_second, bool)
    return np.array([bool(f[a:min(a + W, len(f))].any()) for a in window_starts])

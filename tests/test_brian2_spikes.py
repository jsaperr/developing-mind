"""Checks for spikes.build_phase_switching_input (non-stationary input). Needs brian2 only for the
`second` unit the builders return; no simulation is run."""
import numpy as np
from brian2 import second

from src.brian2_stdp.spikes import build_phase_switching_input


def _coincidence(idx, t, a, b, lo, hi, window_s=0.005):
    """Fraction of neuron a's spikes in [lo, hi) with a neuron-b spike within window_s."""
    ta = t[(idx == a) & (t >= lo) & (t < hi)]
    tb = np.sort(t[(idx == b) & (t >= lo) & (t < hi)])
    pos = np.searchsorted(tb, ta)
    left = np.abs(ta - tb[np.clip(pos - 1, 0, len(tb) - 1)])
    right = np.abs(ta - tb[np.clip(pos, 0, len(tb) - 1)])
    return float((np.minimum(left, right) < window_s).mean())


def test_phase_switching_input_moves_the_correlated_block_and_keeps_rates_matched():
    durs, blocks = [60.0, 60.0, 60.0], [0, 1, 0]
    idx, t = build_phase_switching_input(20.0, 0.9, durs, blocks, np.random.default_rng(0))
    t = np.asarray(t / second)
    assert np.all(np.diff(t) >= 0)
    edges = [0.0, 60.0, 120.0, 180.0]
    for p, block in enumerate(blocks):
        lo, hi = edges[p] + 0.01, edges[p + 1]
        corr = (0, 1) if block == 0 else (10, 11)
        uncorr = (10, 11) if block == 0 else (0, 1)
        assert _coincidence(idx, t, *corr, lo, hi) > 0.5
        assert _coincidence(idx, t, *uncorr, lo, hi) < 0.35
        # rate-matched: every presynaptic neuron ~20Hz in every phase, whichever block is correlated
        for n in range(20):
            rate = ((idx == n) & (t >= lo) & (t < hi)).sum() / (hi - lo)
            assert 14 < rate < 26, (p, n, rate)


def test_phase_switching_input_never_puts_two_spikes_of_one_neuron_in_the_same_step():
    idx, t = build_phase_switching_input(20.0, 0.9, [30.0, 30.0], [0, 1], np.random.default_rng(1))
    t = np.asarray(t / second)
    for n in range(20):
        assert np.diff(np.sort(t[idx == n])).min() >= 0.0002 - 1e-12

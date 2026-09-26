"""Checks for spikes.build_phase_switching_input (non-stationary input). Needs brian2 only for the
`second` unit the builders return; no simulation is run."""
import numpy as np
from brian2 import second

from src.brian2_stdp.spikes import build_multiblock_phase_input, build_phase_switching_input


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


def test_multiblock_input_puts_the_correlation_on_the_named_block_each_phase_and_keeps_rates_matched():
    durs, blocks = [60.0, 60.0, 60.0, 60.0], [0, 1, 2, 0]
    idx, t = build_multiblock_phase_input(20.0, 0.9, durs, blocks, np.random.default_rng(0))
    t = np.asarray(t / second)
    assert np.all(np.diff(t) >= 0)
    assert set(np.unique(idx)) == set(range(30))
    for p, block in enumerate(blocks):
        lo, hi = 60.0 * p + 0.01, 60.0 * (p + 1)
        for b in range(3):
            c = _coincidence(idx, t, 10 * b, 10 * b + 1, lo, hi)
            assert (c > 0.5) if b == block else (c < 0.35), (p, b, c)
        for n in range(30):
            rate = ((idx == n) & (t >= lo) & (t < hi)).sum() / (hi - lo)
            assert 14 < rate < 26, (p, n, rate)


def test_multiblock_input_matches_the_novelc_local_builder_bit_for_bit():
    """The promoted builder must reproduce novel-C's inputs exactly (same rng stream)."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "notebooks/brian2/novelc_data/run_novelc_seed.py"
    spec = importlib.util.spec_from_file_location("run_novelc_seed", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    durs, blocks = [20.0, 20.0, 20.0], [0, 1, 2]
    a_idx, a_t = mod.build_phase_switching_multiblock(durs, blocks, np.random.default_rng(7))
    b_idx, b_t = build_multiblock_phase_input(20.0, 0.9, durs, blocks, np.random.default_rng(7))
    assert np.array_equal(a_idx, b_idx)
    assert np.array_equal(np.asarray(a_t / second), np.asarray(b_t / second))


def test_phase_switching_input_never_puts_two_spikes_of_one_neuron_in_the_same_step():
    idx, t = build_phase_switching_input(20.0, 0.9, [30.0, 30.0], [0, 1], np.random.default_rng(1))
    t = np.asarray(t / second)
    for n in range(20):
        assert np.diff(np.sort(t[idx == n])).min() >= 0.0002 - 1e-12

"""Regression checks for src/esn/ -- the reservoir builder, the memory-capacity task, and the
phase-cycling classification task."""
import numpy as np

from src.esn.memory_capacity import memory_capacity, ridge_fit
from src.esn.phase_task import classification_capacity, generate_phase_stream
from src.esn.reservoir import build_multiscale_leak_rates, build_reservoir, run_reservoir


def test_build_reservoir_matches_requested_spectral_radius():
    rng = np.random.default_rng(0)
    for sr in [0.1, 0.9, 1.5]:
        _, W = build_reservoir(n_units=50, spectral_radius=sr, rng=rng)
        actual_sr = np.max(np.abs(np.linalg.eigvals(W)))
        assert abs(actual_sr - sr) < 1e-6


def test_run_reservoir_produces_bounded_finite_states():
    rng = np.random.default_rng(0)
    W_in, W = build_reservoir(n_units=50, spectral_radius=0.9, rng=rng)
    u = rng.uniform(-1, 1, size=200)
    states = run_reservoir(W_in, W, u, leak_rate=0.3)

    assert states.shape == (200, 50)
    assert np.isfinite(states).all()
    assert (np.abs(states) <= 1.0).all()  # tanh-bounded


def test_zero_input_reservoir_relaxes_toward_zero():
    rng = np.random.default_rng(0)
    W_in, W = build_reservoir(n_units=50, spectral_radius=0.5, rng=rng)
    u = np.zeros(500)
    states = run_reservoir(W_in, W, u, leak_rate=0.3)
    assert np.abs(states[-1]).max() < 1e-6


def test_memory_capacity_is_nontrivial_and_decays_with_lag():
    rng = np.random.default_rng(0)
    W_in, W = build_reservoir(n_units=100, spectral_radius=0.9, rng=rng)
    mc_per_lag, total_mc = memory_capacity(W_in, W, leak_rate=0.3, duration=2000, max_lag=10, rng=rng)

    assert mc_per_lag.shape == (10,)
    assert (mc_per_lag >= 0).all()
    assert total_mc > 0.5  # a reasonable reservoir should recover at least short-lag inputs well
    assert mc_per_lag[0] > mc_per_lag[-1]  # capacity should decay with lag, not increase


def test_ridge_fit_recovers_a_known_linear_relationship():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5))
    true_w = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = X @ true_w
    w_hat = ridge_fit(X, y, alpha=1e-8)
    assert np.allclose(w_hat, true_w, atol=1e-3)


def test_multi_dim_input_matches_scalar_input_when_input_dim_one():
    """input_dim=1 (default) must reproduce the exact original scalar-input behavior -- same
    random draw sequence, just reshaped -- since the stage-1 notebook and its logged numbers
    depend on this."""
    rng1 = np.random.default_rng(0)
    W_in_a, W_a = build_reservoir(n_units=30, spectral_radius=0.8, rng=rng1)

    rng2 = np.random.default_rng(0)
    W_in_b, W_b = build_reservoir(n_units=30, spectral_radius=0.8, input_dim=1, rng=rng2)

    assert np.allclose(W_a, W_b)
    assert np.allclose(W_in_a.reshape(-1), W_in_b.reshape(-1))


def test_run_reservoir_accepts_multi_channel_input():
    rng = np.random.default_rng(0)
    W_in, W = build_reservoir(n_units=30, spectral_radius=0.9, input_dim=3, rng=rng)
    u = np.zeros((100, 3))
    u[np.arange(100), rng.integers(0, 3, size=100)] = 1.0  # one-hot stream
    states = run_reservoir(W_in, W, u, leak_rate=0.3)
    assert states.shape == (100, 30)
    assert np.isfinite(states).all()


def test_build_multiscale_leak_rates_respects_fast_fraction_and_values():
    rng = np.random.default_rng(0)
    rates = build_multiscale_leak_rates(1000, fast_rate=0.3, slow_rate=0.02, fast_fraction=0.5, rng=rng)
    assert rates.shape == (1000,)
    assert set(np.unique(rates)) == {0.3, 0.02}
    fast_frac_actual = (rates == 0.3).mean()
    assert abs(fast_frac_actual - 0.5) < 0.05  # random assignment, allow sampling slack


def test_run_reservoir_accepts_per_unit_leak_rate_array():
    rng = np.random.default_rng(0)
    n_units = 30
    W_in, W = build_reservoir(n_units, spectral_radius=0.8, rng=rng)
    leak_rates = build_multiscale_leak_rates(n_units, fast_rate=0.5, slow_rate=0.05, rng=rng)
    u = rng.uniform(-1, 1, size=200)
    states = run_reservoir(W_in, W, u, leak_rate=leak_rates)
    assert states.shape == (200, n_units)
    assert np.isfinite(states).all()


def test_generate_phase_stream_ground_truth_matches_one_hot_dominant_case():
    """With dominant_prob=1.0 (no retrieval noise), the one-hot input must always match the
    ground-truth phase exactly."""
    rng = np.random.default_rng(0)
    phase_id, one_hot = generate_phase_stream(3, phase_len=50, duration=500, dominant_prob=1.0, rng=rng)
    assert phase_id.shape == (500,)
    assert one_hot.shape == (500, 3)
    assert np.array_equal(np.argmax(one_hot, axis=1), phase_id)


def test_generate_phase_stream_is_not_strictly_periodic():
    """Regression test for the periodicity-leakage bug caught during development: phase order
    must not be a fixed 0,1,2,0,1,2,... cycle, and phase length must vary, or a linear readout
    can infer absolute time position instead of using genuine memory."""
    rng = np.random.default_rng(0)
    phase_id, _ = generate_phase_stream(3, phase_len=100, duration=5000, dominant_prob=1.0, rng=rng)
    boundaries = np.where(np.diff(phase_id) != 0)[0]
    segment_lengths = np.diff(boundaries)
    assert segment_lengths.std() > 5  # real variation in phase length, not a fixed clock

    phase_sequence = phase_id[boundaries]
    # a strictly-cycling 0,1,2,0,1,2,... schedule would make consecutive phase diffs constant
    assert len(set(np.diff(phase_sequence.astype(int)) % 3)) > 1 or len(np.unique(phase_sequence)) > 2


def test_classification_capacity_is_above_chance_and_decays_with_lag():
    rng = np.random.default_rng(0)
    W_in, W = build_reservoir(100, spectral_radius=0.9, input_dim=3, rng=rng)
    acc = classification_capacity(W_in, W, leak_rate=0.3, n_phases=3, phase_len=50, duration=3000,
                                   lags=[1, 20, 100], rng=rng)
    chance = 1 / 3
    assert acc[1] > chance + 0.1  # short lag should clearly beat chance
    assert acc[1] >= acc[100]  # short lag should not do worse than a much longer one

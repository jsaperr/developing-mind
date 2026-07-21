"""Regression checks for src/esn/ -- the reservoir builder and the memory-capacity task."""
import numpy as np

from src.esn.memory_capacity import memory_capacity, ridge_fit
from src.esn.reservoir import build_reservoir, run_reservoir


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

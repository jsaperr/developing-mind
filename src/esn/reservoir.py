"""Minimal Echo State Network -- a fixed random recurrent reservoir with spectral radius as the
one tunable knob, plus a leaky-integrator update. No training on reservoir weights ever; only a
linear readout gets fit, and only for characterization tasks (see memory_capacity.py,
phase_task.py).
"""
import numpy as np


def build_reservoir(n_units, spectral_radius, input_dim=1, input_scale=1.0, density=1.0, rng=None):
    """Fixed random input and recurrent weight matrices. W is rescaled so its largest-magnitude
    eigenvalue equals spectral_radius exactly (not approximately) -- this is the whole knob.

    input_dim>1 supports multi-channel input (e.g. one-hot categorical streams for the
    phase-cycling task); default 1 matches the original scalar-input stage-1 characterization
    exactly (same random draw sequence, just reshaped -- W_in was (n_units,), is now (n_units,1)).

    density<1.0 sparsifies W (a common ESN variant); left at 1.0 (dense) by default since
    nothing here calls for the extra parameter yet.
    """
    if rng is None:
        rng = np.random.default_rng()
    W = rng.normal(0, 1, size=(n_units, n_units))
    if density < 1.0:
        mask = rng.random((n_units, n_units)) < density
        W = W * mask
    current_radius = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (spectral_radius / current_radius)
    W_in = rng.normal(0, input_scale, size=(n_units, input_dim))
    return W_in, W


def run_reservoir(W_in, W, u, leak_rate=0.3, x0=None):
    """u: shape (T,) or (T, input_dim). Returns states, shape (T, n_units).

    Leaky-integrator update: x(t) = (1-leak_rate)*x(t-1) + leak_rate*tanh(W_in@u(t) + W@x(t-1)).
    leak_rate may be a scalar (uniform across units) or an array of shape (n_units,) for a
    heterogeneous/multi-timescale reservoir (see build_multiscale_leak_rates below).
    """
    n_units = W.shape[0]
    u = np.asarray(u)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    T = u.shape[0]
    x = np.zeros(n_units) if x0 is None else x0.copy()
    states = np.zeros((T, n_units))
    for t in range(T):
        pre = W_in @ u[t] + W @ x
        x = (1 - leak_rate) * x + leak_rate * np.tanh(pre)
        states[t] = x
    return states


def build_multiscale_leak_rates(n_units, fast_rate, slow_rate, fast_fraction=0.5, rng=None):
    """Per-unit leak-rate array: fast_fraction of units get fast_rate, the rest get slow_rate --
    the reservoir-world analog of the two-layer memory's fast/slow split. Assignment is randomly
    interleaved (not fast-block-then-slow-block), so both timescales contribute to every part of
    the recurrent dynamics rather than acting like two loosely-coupled sub-reservoirs."""
    if rng is None:
        rng = np.random.default_rng()
    is_fast = rng.random(n_units) < fast_fraction
    return np.where(is_fast, fast_rate, slow_rate)

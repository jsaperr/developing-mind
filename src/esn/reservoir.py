"""Minimal Echo State Network -- a fixed random recurrent reservoir with spectral radius as the
one tunable knob, plus a leaky-integrator update. No training on reservoir weights ever; only a
linear readout gets fit, and only for characterization tasks (see memory_capacity.py).
"""
import numpy as np


def build_reservoir(n_units, spectral_radius, input_scale=1.0, density=1.0, rng=None):
    """Fixed random input and recurrent weight matrices. W is rescaled so its largest-magnitude
    eigenvalue equals spectral_radius exactly (not approximately) -- this is the whole knob.

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
    W_in = rng.normal(0, input_scale, size=n_units)
    return W_in, W


def run_reservoir(W_in, W, u, leak_rate=0.3, x0=None):
    """u: shape (T,) scalar input sequence. Returns states, shape (T, n_units).

    Leaky-integrator update: x(t) = (1-leak_rate)*x(t-1) + leak_rate*tanh(W_in*u(t) + W@x(t-1)).
    """
    n_units = W.shape[0]
    u = np.asarray(u).reshape(-1)
    T = u.shape[0]
    x = np.zeros(n_units) if x0 is None else x0.copy()
    states = np.zeros((T, n_units))
    for t in range(T):
        pre = W_in * u[t] + W @ x
        x = (1 - leak_rate) * x + leak_rate * np.tanh(pre)
        states[t] = x
    return states

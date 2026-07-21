"""Standard ESN memory-capacity characterization (Jaeger 2001): for each lag k, fit a linear
readout mapping the current reservoir state to the input from k steps ago; per-lag capacity is
the held-out R^2 of that readout, total memory capacity is the sum across lags.
"""
import numpy as np

from .reservoir import run_reservoir


def ridge_fit(X, y, alpha=1e-6):
    """Closed-form ridge regression. X: (N, D) (bias column included if wanted), y: (N,)."""
    D = X.shape[1]
    A = X.T @ X + alpha * np.eye(D)
    b = X.T @ y
    return np.linalg.solve(A, b)


def memory_capacity(W_in, W, leak_rate, duration, max_lag=30, washout=200, train_frac=0.7,
                     ridge_alpha=1e-6, rng=None):
    """Returns (mc_per_lag: array shape (max_lag,), total_mc: float).

    Per-lag capacity is clipped at 0 (a readout worse than predicting the mean contributes
    nothing, standard practice -- otherwise near-zero-signal lags can go slightly negative from
    finite-sample noise and bias the sum).
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(-1, 1, size=duration)
    states = run_reservoir(W_in, W, u, leak_rate=leak_rate)

    valid_t = np.arange(washout + max_lag, duration)
    X_full = np.hstack([states[valid_t], np.ones((len(valid_t), 1))])

    n_train = int(len(valid_t) * train_frac)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, len(valid_t))

    mc_per_lag = np.zeros(max_lag)
    for k in range(1, max_lag + 1):
        y = u[valid_t - k]
        w = ridge_fit(X_full[train_idx], y[train_idx], alpha=ridge_alpha)
        y_pred_test = X_full[test_idx] @ w
        y_test = y[test_idx]
        ss_res = np.sum((y_test - y_pred_test) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mc_per_lag[k - 1] = max(r2, 0.0)

    return mc_per_lag, float(mc_per_lag.sum())

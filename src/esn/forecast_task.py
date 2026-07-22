"""Stage 2a testbed: does the reservoir carry a genuine forward-looking forecast signal, not
just the backward-looking memory tested by memory_capacity.py and phase_task.py?

Follow-ups 2/3 (see experiments_esn.md) answered "given current reservoir state, what happened
k steps ago" -- a memory question. Stage 2's actual need (episodic layer check (b): will a
currently-stale pattern become relevant again soon) is a forecasting question: "given current
reservoir state, will pattern p return within the next W steps." Those are not automatically the
same capability even in the same reservoir, so this module tests the forecasting direction
directly rather than assuming the memory result transfers.

Two things deliberately different from phase_task.py, per web's design:

1. generate_visitation_schedule is NOT periodic and is NOT a reshuffled-but-fixed-length cycle
   either -- core-pattern return times follow a genuine renewal process (gamma-distributed
   inter-visit gaps), so there is no fixed period for a readout to infer absolute time from, a
   fresh form of the same leakage class caught in phase_task.py's development. Between core
   visits, filler/distractor slots (one-off, shared channel) fill the gap, approximating real
   episodic traffic where most events never recur.
2. forecast_signal uses a CHRONOLOGICAL train/test split, not phase_task.py's shuffled split.
   phase_task.py shuffled specifically to break a train/test adjacency that could leak "when in
   the run is this" from slow reservoir drift. Here the label itself is a forward-looking window
   (t, t+W], so a shuffled split would let training examples sit immediately adjacent in time to
   test examples, and reservoir state is autocorrelated -- that adjacency would inflate held-out
   performance for reasons that have nothing to do with genuine forecasting. A chronological
   split (train on the first train_frac of the run, test on the rest) is the honest evaluation
   for a forecasting task, and the periodicity-style leakage is already blocked by the
   irregular schedule itself, not by shuffling.
"""
import numpy as np

from .reservoir import run_reservoir


def generate_visitation_schedule(n_core, mean_interval, gamma_shape, visit_len, duration, rng=None):
    """Returns (active_id: shape (duration,) int, -1 for filler/distractor slots, 0..n_core-1
    for core-pattern slots; one_hot_input: shape (duration, n_core+1), last channel is the
    shared filler/distractor channel).

    Each core pattern's return times form a renewal process: inter-visit gap ~
    Gamma(shape=gamma_shape, scale=mean_interval/gamma_shape), mean=mean_interval,
    coefficient of variation = 1/sqrt(gamma_shape) -- real irregularity, not a fixed period.
    When no core pattern is due, a filler slot occupies the gap (identity doesn't matter for
    this task, only presence -- modeled as a single shared channel rather than unique one-hot
    dims per filler, to keep input dimensionality bounded regardless of duration)."""
    if rng is None:
        rng = np.random.default_rng()
    next_visit = rng.uniform(0, mean_interval, size=n_core)
    active_id = np.full(duration, -1, dtype=int)
    t = 0
    while t < duration:
        due = np.where(next_visit <= t)[0]
        if len(due) > 0:
            p = due[np.argmin(next_visit[due])]
            end = min(t + visit_len, duration)
            active_id[t:end] = p
            gap = rng.gamma(shape=gamma_shape, scale=mean_interval / gamma_shape)
            next_visit[p] = end + gap
            t = end
        else:
            t = min(t + visit_len, duration)

    one_hot = np.zeros((duration, n_core + 1))
    is_filler = active_id == -1
    one_hot[np.arange(duration)[~is_filler], active_id[~is_filler]] = 1.0
    one_hot[is_filler, n_core] = 1.0
    return active_id, one_hot


def _time_since_last_visit(active_id, p, duration):
    """since[t] = t - (most recent time < t that p was active), or `duration` (max/sentinel,
    "maximally stale") if p has not appeared yet by time t."""
    since = np.full(duration, duration, dtype=float)
    last = -1
    for t in range(duration):
        since[t] = duration if last < 0 else t - last
        if active_id[t] == p:
            last = t
    return since


def _time_until_next_visit(active_id, p, duration):
    """until[t] = (nearest time >= t that p is active) - t, or `duration` (a true constant
    sentinel, "not returning within the observed horizon") if p never appears again after t.

    next_t starts as None (not `duration`) so the sentinel stays a fixed constant for every t
    before the last occurrence of p, rather than degrading to `duration - t` (which would
    shrink toward 0 for late t even when p genuinely never returns -- a bug caught by this
    module's own tests)."""
    until = np.full(duration, float(duration), dtype=float)
    next_t = None
    for t in range(duration - 1, -1, -1):
        if active_id[t] == p:
            next_t = t
        if next_t is not None:
            until[t] = next_t - t
    return until


def _ridge_probability_fit(X, y, alpha):
    D = X.shape[1]
    A = X.T @ X + alpha * np.eye(D)
    b = X.T @ y
    return np.linalg.solve(A, b)


def _auc(scores, labels):
    """Rank-based (Mann-Whitney U) AUC -- threshold-independent, appropriate here since the
    positive/negative balance varies a lot by lag and by pattern."""
    labels = np.asarray(labels)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None  # undefined -- degenerate split, caller should skip
    ranks = np.argsort(np.argsort(scores)) + 1  # average ties ignored (float scores, ties rare)
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def forecast_signal(W_in, W, leak_rate, n_core, mean_interval, gamma_shape, visit_len, duration,
                     w_forecast, washout=1000, train_frac=0.7, ridge_alpha=1e-2, rng=None):
    """For each core pattern p, at each valid timestep t (post-washout, p not currently active,
    at least w_forecast steps of runway remaining before duration): label = 1 if p is active
    again at some point in (t, t+w_forecast], else 0.

    Fits two competing predictors per pattern, both evaluated on the same chronological
    held-out split: 'reservoir' (ridge readout on reservoir state x(t)) and 'baseline' (ridge fit
    on the single scalar feature time-since-p-was-last-active(t), no reservoir at all). The
    reservoir's signal only matters if it beats this baseline -- raw staleness is already
    available to the real episodic-layer gate without any reservoir.

    Returns {p: {'reservoir_auc':, 'reservoir_acc':, 'baseline_auc':, 'baseline_acc':,
    'n_test':, 'base_rate':}}."""
    if rng is None:
        rng = np.random.default_rng()
    active_id, one_hot_input = generate_visitation_schedule(n_core, mean_interval, gamma_shape,
                                                              visit_len, duration, rng=rng)
    states = run_reservoir(W_in, W, one_hot_input, leak_rate=leak_rate)

    results = {}
    for p in range(n_core):
        until = _time_until_next_visit(active_id, p, duration)
        since = _time_since_last_visit(active_id, p, duration)
        label = (until <= w_forecast).astype(float)

        valid = (np.arange(duration) >= washout) & (np.arange(duration) < duration - w_forecast) \
            & (active_id != p)
        valid_t = np.where(valid)[0]
        if len(valid_t) < 50:
            continue

        n_train = int(len(valid_t) * train_frac)
        train_idx = valid_t[:n_train]
        test_idx = valid_t[n_train:]
        y_train, y_test = label[train_idx], label[test_idx]
        if y_train.sum() == 0 or y_train.sum() == len(y_train) or y_test.sum() == 0 or y_test.sum() == len(y_test):
            continue  # degenerate split for this pattern/seed, skip rather than report a meaningless AUC

        X_res_train = np.hstack([states[train_idx], np.ones((len(train_idx), 1))])
        X_res_test = np.hstack([states[test_idx], np.ones((len(test_idx), 1))])
        w_res = _ridge_probability_fit(X_res_train, y_train, ridge_alpha)
        score_res_test = X_res_test @ w_res

        X_base_train = np.column_stack([since[train_idx], np.ones(len(train_idx))])
        X_base_test = np.column_stack([since[test_idx], np.ones(len(test_idx))])
        w_base = _ridge_probability_fit(X_base_train, y_train, ridge_alpha)
        score_base_test = X_base_test @ w_base

        results[p] = {
            'reservoir_auc': _auc(score_res_test, y_test),
            'reservoir_acc': float(((score_res_test >= 0.5).astype(float) == y_test).mean()),
            'baseline_auc': _auc(score_base_test, y_test),
            'baseline_acc': float(((score_base_test >= 0.5).astype(float) == y_test).mean()),
            'n_test': int(len(test_idx)),
            'base_rate': float(y_test.mean()),
        }
    return results

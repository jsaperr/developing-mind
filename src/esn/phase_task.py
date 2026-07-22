"""Task-relevant capacity for the phase-cycling signal stage 2 actually needs: not exact
reconstruction (memory_capacity.py's task), but classification -- can a simple readout recover
*which phase was active* k steps ago from the reservoir's current state. A much lower-information
question than reconstruction, so its effective horizon can plausibly outlast the linear MC
horizon by a wide margin.

generate_phase_stream reconstructs the *shape* of the schedule used in the original fixed-X
phase-cycling Hopfield experiments (roughly phase_len-long stretches, cycling through n_phases
patterns, retrieval noisy around the dominant pattern with probability dominant_prob) -- not a
reuse of that experiment's exact generator code, which lives in a notebook this module doesn't
import from. Documented here as an approximation, not a byte-for-byte replication.

Deliberately NOT a strictly periodic clock, unlike the original: phase order is randomized and
phase length is jittered (see generate_phase_stream's docstring) specifically to avoid letting
a linear readout infer absolute time position instead of using genuine reservoir memory -- a
real bug caught via a sanity check during this module's development (see that docstring for
the diagnostic signature).
"""
import numpy as np

from .reservoir import run_reservoir


def generate_phase_stream(n_phases, phase_len, duration, dominant_prob=0.7, jitter_frac=0.5, rng=None):
    """Returns (phase_id: shape (duration,) int ground truth, one_hot_input: shape
    (duration, n_phases)) -- the noisy one-hot 'which pattern retrieved' signal fed to the
    reservoir, matching the two-layer memory's stochastic retrieval around the dominant
    pattern.

    Phase ORDER is randomized (no immediate repeats) and phase LENGTH is jittered by
    +/-jitter_frac, rather than a perfectly periodic 0,1,2,0,1,2,... clock at a fixed length.
    This matters: a strictly periodic schedule is a deterministic function of absolute time t
    alone, which lets a linear readout "cheat" by inferring t (mod period) from any
    time-correlated drift in reservoir state, rather than genuinely using memory of the input
    history -- caught via a sanity check where classification accuracy came out *higher* at
    lag k=400 than at k=200 with a strictly periodic schedule, the signature of exactly this
    aliasing, not real memory."""
    if rng is None:
        rng = np.random.default_rng()
    phase_id = np.empty(duration, dtype=int)
    t = 0
    last_phase = None
    while t < duration:
        choices = [p for p in range(n_phases) if p != last_phase]
        phase = choices[rng.integers(len(choices))]
        this_len = max(int(round(phase_len * (1 + rng.uniform(-jitter_frac, jitter_frac)))), 1)
        end = min(t + this_len, duration)
        phase_id[t:end] = phase
        last_phase = phase
        t = end

    retrieved = np.empty(duration, dtype=int)
    for t in range(duration):
        if rng.random() < dominant_prob:
            retrieved[t] = phase_id[t]
        else:
            others = [p for p in range(n_phases) if p != phase_id[t]]
            retrieved[t] = others[rng.integers(len(others))]
    one_hot = np.zeros((duration, n_phases))
    one_hot[np.arange(duration), retrieved] = 1.0
    return phase_id, one_hot


def classification_capacity(W_in, W, leak_rate, n_phases, phase_len, duration, lags,
                             dominant_prob=0.7, jitter_frac=0.5, washout=200, train_frac=0.7,
                             ridge_alpha=1e-3, rng=None):
    """For each lag k in `lags`: fit a ridge-regression readout on one-hot phase targets, take
    argmax as the predicted class, report held-out classification accuracy. Returns
    {k: accuracy}. Chance level is 1/n_phases."""
    if rng is None:
        rng = np.random.default_rng()
    phase_id, one_hot_input = generate_phase_stream(n_phases, phase_len, duration, dominant_prob,
                                                      jitter_frac=jitter_frac, rng=rng)
    states = run_reservoir(W_in, W, one_hot_input, leak_rate=leak_rate)

    max_lag = max(lags)
    valid_t = np.arange(washout + max_lag, duration)
    X_full = np.hstack([states[valid_t], np.ones((len(valid_t), 1))])

    # shuffle before splitting -- guards against any residual slow/chronological drift in
    # reservoir state leaking a "when in the run is this" cue into a contiguous split, on top
    # of the schedule-randomization fix in generate_phase_stream above.
    n_train = int(len(valid_t) * train_frac)
    shuffled = rng.permutation(len(valid_t))
    train_idx = shuffled[:n_train]
    test_idx = shuffled[n_train:]

    accuracy_per_lag = {}
    for k in lags:
        target_phase = phase_id[valid_t - k]
        Y_onehot = np.zeros((len(valid_t), n_phases))
        Y_onehot[np.arange(len(valid_t)), target_phase] = 1.0

        D = X_full.shape[1]
        A = X_full[train_idx].T @ X_full[train_idx] + ridge_alpha * np.eye(D)
        b = X_full[train_idx].T @ Y_onehot[train_idx]
        W_readout = np.linalg.solve(A, b)

        pred_class = np.argmax(X_full[test_idx] @ W_readout, axis=1)
        true_class = target_phase[test_idx]
        accuracy_per_lag[k] = float((pred_class == true_class).mean())

    return accuracy_per_lag

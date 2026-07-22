"""Regression checks for src/esn/forecast_task.py -- the stage-2a forward-looking forecast
testbed (irregular visitation schedule + reservoir-vs-staleness-baseline comparison)."""
import numpy as np

from src.esn.forecast_task import (
    _time_since_last_visit,
    _time_until_next_visit,
    forecast_signal,
    generate_visitation_schedule,
)
from src.esn.reservoir import build_multiscale_leak_rates, build_reservoir


def test_generate_visitation_schedule_shapes_and_channels():
    rng = np.random.default_rng(0)
    active_id, one_hot = generate_visitation_schedule(n_core=4, mean_interval=400, gamma_shape=4,
                                                        visit_len=40, duration=5000, rng=rng)
    assert active_id.shape == (5000,)
    assert one_hot.shape == (5000, 5)  # n_core + 1 filler channel
    assert set(np.unique(active_id)) <= {-1, 0, 1, 2, 3}
    assert np.allclose(one_hot.sum(axis=1), 1.0)  # exactly one active channel per timestep


def test_generate_visitation_schedule_is_not_a_fixed_period():
    """Same class of check as phase_task's periodicity regression test: a renewal process with
    gamma_shape=4 must produce real inter-visit variation, not a disguised fixed clock."""
    rng = np.random.default_rng(0)
    active_id, _ = generate_visitation_schedule(n_core=2, mean_interval=300, gamma_shape=4,
                                                  visit_len=30, duration=20000, rng=rng)
    visit_starts = np.where((active_id == 0) & (np.roll(active_id, 1) != 0))[0]
    gaps = np.diff(visit_starts)
    assert len(gaps) > 5
    assert gaps.std() / gaps.mean() > 0.15  # meaningful coefficient of variation, not ~fixed


def test_generate_visitation_schedule_mean_interval_is_approximately_respected():
    rng = np.random.default_rng(1)
    active_id, _ = generate_visitation_schedule(n_core=3, mean_interval=300, gamma_shape=4,
                                                  visit_len=30, duration=30000, rng=rng)
    visit_starts = np.where((active_id == 0) & (np.roll(active_id, 1) != 0))[0]
    gaps = np.diff(visit_starts)
    assert abs(gaps.mean() - 300) < 60  # within 20% of the target mean interval


def test_time_until_next_visit_is_zero_at_an_active_slot_and_counts_down():
    active_id = np.array([-1, -1, 0, 0, -1, -1, -1, 0, -1])
    until = _time_until_next_visit(active_id, p=0, duration=len(active_id))
    assert until[2] == 0 and until[3] == 0  # p active at t=2,3
    assert until[1] == 1  # one step until p becomes active
    assert until[6] == 1  # one step until the next occurrence at t=7
    assert until[8] == len(active_id)  # sentinel: no further occurrence before duration ends


def test_time_since_last_visit_resets_on_visit_and_is_sentinel_before_first():
    active_id = np.array([-1, -1, 0, 0, -1, -1])
    since = _time_since_last_visit(active_id, p=0, duration=len(active_id))
    # since[t] reflects visits strictly BEFORE t (no look-ahead into t itself), so at t=2 -- the
    # moment p first becomes active -- there is still no PRIOR visit to measure from: sentinel.
    assert since[0] == len(active_id) and since[1] == len(active_id) and since[2] == len(active_id)
    assert since[3] == 1  # one step since p's prior active slot at t=2
    assert since[4] == 1 and since[5] == 2  # counting up after p's last active slot at t=3


def test_forecast_signal_returns_valid_auc_range_and_includes_baseline():
    rng = np.random.default_rng(0)
    n_units = 150
    W_in, W = build_reservoir(n_units, spectral_radius=1.1, input_dim=5, rng=rng)  # n_core=4 -> 5 channels
    leak = build_multiscale_leak_rates(n_units, fast_rate=0.3, slow_rate=0.02, rng=rng)
    results = forecast_signal(W_in, W, leak_rate=leak, n_core=4, mean_interval=300, gamma_shape=4,
                               visit_len=30, duration=15000, w_forecast=150, washout=600, rng=rng)
    assert len(results) > 0
    for p, r in results.items():
        assert r['reservoir_auc'] is None or 0.0 <= r['reservoir_auc'] <= 1.0
        assert r['baseline_auc'] is None or 0.0 <= r['baseline_auc'] <= 1.0
        assert 0.0 < r['base_rate'] < 1.0  # a sane (non-degenerate) label balance

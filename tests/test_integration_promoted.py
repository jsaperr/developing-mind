"""The promoted modules (src/integration/interface.py, src/hopfield/episodic_consolidating.py) must
reproduce the coupling-toy code they were promoted from EXACTLY on real saved runs. They also
reduce to the validated EpisodicMemory path when every new feature is off.
Uses 2 seeds of the committed v1b data (13mV/1.5) at both memory clocks."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for sub in ("notebooks/integration/coupling_v0", "notebooks/integration/content_consolidation",
            "notebooks/brian2/interface_readout"):
    sys.path.insert(0, str(ROOT / sub))

import run_coupling_v0 as T                                    # noqa: E402
import run_content_consolidation as C                          # noqa: E402
import run_displacement_gate as G                              # noqa: E402
from run_creation_gates import stable_allow                    # noqa: E402

from src.hopfield.episodic_consolidating import TRANSITIONAL, GatedEpisodicMemory  # noqa: E402
from src.integration import interface as I                     # noqa: E402

DDIR = ROOT / "notebooks/brian2/v1b_schedule_data"
FILES = ["v1b_n7_reliable_13_1p5_seed34000.json.gz", "v1b_n7_reliable_13_1p5_seed34001.json.gz"]


@pytest.fixture(scope="module")
def runs():
    out = []
    for f in FILES:
        got = T.load_world(DDIR, f)
        assert got, f"missing test data {f}"
        out.append(got[0])
    return out


def window_starts(d, W):
    ps = [0.0] + list(d['swap_times_s']); starts = []
    for a in range(0, int(d['total_s']) - W + 1, W):
        ph = int(np.searchsorted(ps, a, side='right') - 1)
        if ph + 1 < len(ps) and a + W > ps[ph + 1]:
            continue
        starts.append(a)
    return starts


@pytest.mark.parametrize("W", [10, 50])
def test_interface_signals_match_the_toy(runs, W):
    rng = np.random.default_rng(0)
    for d, wm, r, n, bs in runs:
        s = T.build_streams(d, wm, r, n, bs, W, rng)
        flag = I.changing_per_second(wm)
        assert np.array_equal(flag, G.change_flag(wm))
        assert np.array_equal(I.window_any(flag, window_starts(d, W), W), G.step_flags(d, flag, W))
        allow = stable_allow(s['q']['substrate'], 0.9)
        steady = I.steady_flags(s['q']['substrate'], 0.9)
        assert all(bool(allow(i)) == bool(steady[i]) for i in range(len(steady)))


@pytest.mark.parametrize("W", [10, 50])
def test_gated_memory_reproduces_the_adopted_toy_configuration_exactly(runs, W):
    rng = np.random.default_rng(0); torch.manual_seed(0)
    streams = [T.build_streams(d, wm, r, n, bs, W, rng) for d, wm, r, n, bs in runs]
    gs = 0.3
    for (d, wm, r, n, bs), s in zip(runs, streams):
        changing = G.step_flags(d, G.change_flag(wm), W)
        steady = I.steady_flags(s['q']['substrate'], 0.9)
        allow = (lambda st, c: (lambda i: st[i] and not c[i]))(steady, changing)
        ref = C.run_cc(s, 'substrate', gs, 0.1, True, allow=allow, match_floor=0.5,
                       consolidate_when=(lambda c: (lambda i: not c[i]))(changing))
        m = GatedEpisodicMemory(dim=s['q']['substrate'].shape[1], gap_scale=gs)
        Q = torch.tensor(s['q']['substrate'], dtype=torch.float32)
        winners, reports, created = [], [], 0
        for i in range(len(Q)):
            o = m.step(Q[i], steady=bool(steady[i]), changing=bool(changing[i]))
            winners.append(o['winner']); reports.append(o['report']); created += o['created'] is not None
        assert winners == ref['winner_ids']
        assert created == ref['created']
        assert all((rep is TRANSITIONAL) == bool(changing[i]) for i, rep in enumerate(reports))


def test_all_new_features_off_reduces_to_the_validated_episodic_path(runs):
    rng = np.random.default_rng(0); torch.manual_seed(0)
    d, wm, r, n, bs = runs[0]
    s = T.build_streams(d, wm, r, n, bs, 10, rng)
    ref = T.run_memory(s, 'substrate', 0.3, 0.5)            # v0: src EpisodicMemory, THETA creation
    m = GatedEpisodicMemory(dim=s['q']['substrate'].shape[1], gap_scale=0.3, eta=0.0, match_floor=None)
    Q = torch.tensor(s['q']['substrate'], dtype=torch.float32)
    winners = [m.step(Q[i], steady=True, changing=False)['winner'] for i in range(len(Q))]
    assert winners == ref['winner_ids']


def test_rectified_readout_ignores_quieter_neurons_and_keeps_a_shared_tuning():
    w = np.zeros((4, 6)); w[:, :3] = 1.0                        # every neuron tuned to inputs 0-2 (cold start)
    r = np.array([10.0, 12.0, 14.0, 16.0])
    assert np.allclose(I.h_readout(r, w), 0.0)                  # contrast cancels a shared tuning entirely
    q = I.h_readout_rectified(r, w)
    assert q[:3].min() > 0 and np.allclose(q, I.unit(np.r_[np.ones(3), np.zeros(3)]))
    w2 = w.copy(); w2[:2] = 0; w2[:2, 3:] = 1.0                 # quieter neurons hold another context
    assert np.allclose(I.h_readout_rectified(r, w2), I.unit(np.r_[np.ones(3), np.zeros(3)]))


def test_single_pattern_and_transitional_output():
    torch.manual_seed(1)
    m = GatedEpisodicMemory(dim=8, gap_scale=0.3)
    q = torch.nn.functional.normalize(torch.randn(8) - 0.0, dim=0)
    o = m.step(q, steady=False, changing=True)                  # empty memory may always create
    assert o['created'] is not None and o['report'] is TRANSITIONAL and o['winner'] == o['created']
    o = m.step(q, steady=True, changing=False)                  # lone pattern retrieves without error
    assert o['report'] == o['winner'] and o['created'] is None

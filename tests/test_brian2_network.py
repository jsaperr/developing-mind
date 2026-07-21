"""Smoke test for the Brian2 STDP network builder.

Construction only -- no run() call, so importing/running this test does not execute a
simulation. NOT YET EXECUTED as of writing; run with `pytest tests/` before trusting it passes.
"""
import numpy as np
from brian2 import start_scope

from src.brian2_stdp.network import N_SYNAPSES, W_INIT, apre_to_apost, build_network, taupost, taupre
from src.brian2_stdp.spikes import build_presynaptic_input


def test_build_network_constructs_with_expected_shapes():
    start_scope()
    rng = np.random.default_rng(0)
    idx, t = build_presynaptic_input(target_rate_hz=20.0, p_share=0.9, duration_s=1.0, rng=rng)

    pre, post, syn = build_network(idx, t, apre_val=0.005)

    assert pre.N == N_SYNAPSES
    assert post.N == 1
    assert len(syn.w[:]) == N_SYNAPSES
    assert np.allclose(np.array(syn.w[:]), W_INIT)


def test_apre_to_apost_is_negative_and_scaled():
    apost = apre_to_apost(0.02)
    assert apost < 0
    assert abs(apost) == abs(-0.02 * (taupre / taupost) * 1.05)

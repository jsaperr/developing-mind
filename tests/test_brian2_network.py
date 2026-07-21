"""Smoke test for the Brian2 STDP network builder.

test_build_network_actually_runs actually calls run() (a couple of simulated seconds, past one
homeostatic-scaling event at 500ms) -- construction-only testing previously missed a real bug
here: Brian2 resolves free identifiers in equation strings from the *run()* call site's stack,
not the object-construction site, so network.py's constants were silently unresolvable the
moment network-building code moved into its own module. A construction-only test can't catch
that class of bug; this one exists specifically so it can't happen again undetected.
"""
import numpy as np
from brian2 import run, second, start_scope

from src.brian2_stdp.network import (
    N_SYNAPSES, TARGET_TOTAL, W_INIT, apre_to_apost, build_network,
    build_population_network, taupost, taupre,
)
from src.brian2_stdp.spikes import build_population_presynaptic_input, build_presynaptic_input


def test_build_network_constructs_with_expected_shapes():
    start_scope()
    rng = np.random.default_rng(0)
    idx, t = build_presynaptic_input(target_rate_hz=20.0, p_share=0.9, duration_s=1.0, rng=rng)

    pre, post, syn = build_network(idx, t, apre_val=0.005)

    assert pre.N == N_SYNAPSES
    assert post.N == 1
    assert len(syn.w[:]) == N_SYNAPSES
    assert np.allclose(np.array(syn.w[:]), W_INIT)


def test_build_network_actually_runs():
    start_scope()
    rng = np.random.default_rng(0)
    idx, t = build_presynaptic_input(target_rate_hz=20.0, p_share=0.9, duration_s=2.0, rng=rng)

    pre, post, syn = build_network(idx, t, apre_val=0.005)
    run(2 * second)

    w = np.array(syn.w[:])
    assert np.isfinite(w).all()
    assert (w >= 0).all() and (w <= 1.0).all()
    assert not np.allclose(w, W_INIT)  # STDP should have moved weights off their initial value
    assert abs(w.sum() - TARGET_TOTAL) < 1.0  # homeostatic scaling keeps the sum near target


def test_build_population_network_actually_runs():
    start_scope()
    rng = np.random.default_rng(0)
    n_post = 3
    idx, t, n_pre_per_neuron = build_population_presynaptic_input(n_post, target_rate_hz=20.0,
                                                                    p_share=0.9, duration_s=2.0, rng=rng)

    pre, post, syn = build_population_network(n_post, idx, t, apre_val=0.005, n_pre_per_neuron=n_pre_per_neuron)
    run(2 * second)

    w = np.array(syn.w[:])
    syn_j = np.array(syn.j[:])
    assert np.isfinite(w).all()
    for j in range(n_post):
        assert abs(w[syn_j == j].sum() - TARGET_TOTAL) < 1.0


def test_apre_to_apost_is_negative_and_scaled():
    apost = apre_to_apost(0.02)
    assert apost < 0
    assert abs(apost) == abs(-0.02 * (taupre / taupost) * 1.05)

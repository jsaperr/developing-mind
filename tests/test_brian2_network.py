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
    build_population_network, scale_inhib_for_n, taupost, taupre,
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


def test_scale_inhib_for_n_reproduces_reference_point():
    # n_post=3 must reproduce the exact known-good 13mV setting from the bistability sweep
    assert scale_inhib_for_n(3, reference_inhib_mV=13.0, reference_n_post=3) == 13.0


def test_scale_inhib_for_n_holds_total_drive_constant():
    # (n_post - 1) * inhib_strength should be the same constant at every n_post
    reference_total = 13.0 * (3 - 1)
    for n in [3, 5, 7, 10]:
        per_connection = scale_inhib_for_n(n, reference_inhib_mV=13.0, reference_n_post=3)
        assert abs(per_connection * (n - 1) - reference_total) < 1e-9


def test_scale_inhib_for_n_decreases_as_n_grows():
    values = [scale_inhib_for_n(n, reference_inhib_mV=13.0, reference_n_post=3) for n in [3, 5, 7, 10]]
    assert values == sorted(values, reverse=True)

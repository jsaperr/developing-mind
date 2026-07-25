"""Smoke test for the Brian2 STDP network builder.

test_build_network_actually_runs actually calls run() (a couple of simulated seconds, past one
homeostatic-scaling event at 500ms) -- construction-only testing previously missed a real bug
here: Brian2 resolves free identifiers in equation strings from the *run()* call site's stack,
not the object-construction site, so network.py's constants were silently unresolvable the
moment network-building code moved into its own module. A construction-only test can't catch
that class of bug; this one exists specifically so it can't happen again undetected.
"""
import numpy as np
from brian2 import SpikeMonitor, run, second, start_scope

from src.brian2_stdp.network import (
    N_SYNAPSES, TARGET_TOTAL, W_INIT, apre_to_apost, build_network,
    build_population_network, post_eqs_competitive, scale_inhib_for_n, taupost, taupre,
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


def _run_competitive_with_eqs(eqs, seed_val, duration_s, i_pert_mV=None, i_pert_target=0):
    """Build+run the competitive population network with a caller-supplied membrane equation, so
    the pre- and post-`I_pert` equations can be compared directly under an identical noise draw
    (brian2.seed fixes the xi stream). Mirrors build_competitive_population_network's construction
    exactly apart from the equation string itself."""
    from brian2 import NeuronGroup, SpikeGeneratorGroup, Synapses, mV, seed as b2_seed
    from src.brian2_stdp.network import (
        GMAX, SCALING_INTERVAL, TARGET_TOTAL, W_INIT, R_INC, SIGMA_V, TAU_R, _neuron_namespace,
        _synapse_namespace, scaling_op, stdp_model_homeo, stdp_on_post, stdp_on_pre, t_ref, v_rest,
    )

    start_scope()
    b2_seed(1234)
    rng = np.random.default_rng(seed_val)
    idx, t = build_presynaptic_input(target_rate_hz=20.0, p_share=0.9, duration_s=duration_s, rng=rng)

    pre = SpikeGeneratorGroup(20, idx, t)
    post = NeuronGroup(3, eqs, threshold="v>v_thresh", reset="v=v_reset; r += r_inc",
                        refractory=t_ref, method="euler",
                        namespace={**_neuron_namespace(), "tau_r": TAU_R, "r_inc": R_INC,
                                   "sigma_v": SIGMA_V})
    post.v = v_rest
    post.r = 0
    if "I_pert" in eqs:
        post.I_pert = 0 * mV
        if i_pert_mV is not None:
            post.I_pert[i_pert_target] = i_pert_mV * mV

    syn = Synapses(pre, post, model=stdp_model_homeo, on_pre=stdp_on_pre, on_post=stdp_on_post,
                    namespace=_synapse_namespace(0.005, apre_to_apost(0.005), GMAX, TARGET_TOTAL))
    syn.connect()
    syn.w = W_INIT
    syn.run_regularly(scaling_op, dt=SCALING_INTERVAL)

    inhib = Synapses(post, post,
                      on_pre="v_post -= inhib_strength / (1 + abs(r_pre - r_post) / gap_scale)",
                      namespace={"inhib_strength": 13.0 * mV, "gap_scale": 1.5})
    inhib.connect(condition="i != j")

    spikes = SpikeMonitor(post)
    run(duration_s * second)
    return np.array(syn.w[:]), np.array(spikes.count[:])


_EQS_ORIGINAL = """
dv/dt = (v_rest - v)/tau + sigma_v*xi*tau**-0.5 : volt (unless refractory)
dr/dt = -r/tau_r : 1
w_total : 1
"""


def test_i_pert_at_zero_is_bit_identical_to_the_original_equation():
    """The perturbation support added `+ I_pert` to the membrane equation. The claim that this is
    a no-op at I_pert=0 (so every prior result built on this network stays valid) is asserted
    here against the actual pre-change equation under an identical noise draw, rather than
    trusted as an IEEE754 argument."""
    w_new, counts_new = _run_competitive_with_eqs(post_eqs_competitive, seed_val=7, duration_s=5.0)
    w_old, counts_old = _run_competitive_with_eqs(_EQS_ORIGINAL, seed_val=7, duration_s=5.0)

    assert np.array_equal(counts_new, counts_old)  # identical spike counts, not just similar
    assert np.array_equal(w_new, w_old)            # bit-identical final weights


def test_i_pert_actually_drives_the_targeted_neuron():
    """The perturbation must do something -- a boost to one neuron's effective resting potential
    should measurably raise its firing rate relative to the unperturbed run."""
    _, counts_base = _run_competitive_with_eqs(post_eqs_competitive, seed_val=7, duration_s=5.0)
    _, counts_pert = _run_competitive_with_eqs(post_eqs_competitive, seed_val=7, duration_s=5.0,
                                                i_pert_mV=10.0, i_pert_target=0)

    assert counts_pert[0] > counts_base[0]  # targeted neuron fires more
    assert counts_pert[0] > counts_pert[1]  # and outfires its unperturbed peers


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

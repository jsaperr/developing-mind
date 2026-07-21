"""Postsynaptic LIF neuron + STDP synapses with homeostatic synaptic scaling.

Canonical rig confirmed byte-for-byte identical between
notebooks/brian2/apre005_ensemble_data/run_single_seed.py (the 8-seed/5000s ensemble) and
notebooks/brian2/brian2_stdp_apre_sweep.ipynb (the Apre sweep that established the standing
amplitude-vs-frequency explanation) -- see experiments_brian2.md for the experiment history
that produced these parameters (runaway -> homeostatic scaling fix -> interval/jump-cap
hypotheses rejected -> amplitude-vs-frequency reframe).

Variable names below (tau, v_rest, taupre, wmax, ...) are deliberately lowercase, matching the
original notebooks. Unlike those notebooks, every free identifier used in an equation/threshold/
reset string is passed explicitly via `namespace=` here, rather than relying on Brian2's
stack-based auto-resolution -- that auto-resolution looks at the frame *calling run()*, not the
frame that constructed the object, so it silently breaks the moment network-building code lives
in a different module than the run() call (exactly the situation this extraction creates).
Caught by the population-network calibration run failing with `KeyError: "tau" could not be
resolved` -- fixed here for build_network() too, which had the same latent bug.
"""
from brian2 import (
    NeuronGroup, SpikeGeneratorGroup, Synapses, ms, mV, prefs,
)

prefs.codegen.target = "cython"

# Neuron
tau = 10 * ms
v_rest = -70 * mV
v_thresh = -50 * mV
v_reset = -65 * mV
t_ref = 5 * ms

post_eqs = """
dv/dt = (v_rest - v)/tau : volt (unless refractory)
w_total : 1
"""

# STDP / homeostatic scaling
GMAX = 6 * mV
W_INIT = 0.5
N_SYNAPSES = 20  # 10 correlated + 10 uncorrelated, per spikes.build_presynaptic_input's default split
TARGET_TOTAL = N_SYNAPSES * W_INIT
SCALING_INTERVAL = 500 * ms
taupre = 20 * ms
taupost = 20 * ms
wmax = 1.0
DEPRESSION_BIAS = 1.05  # Apost = -Apre * (taupre/taupost) * DEPRESSION_BIAS

stdp_model_homeo = """
w : 1
w_total_post = w : 1 (summed)
dapre/dt = -apre/taupre : 1 (event-driven)
dapost/dt = -apost/taupost : 1 (event-driven)
"""
stdp_on_pre = """
v_post += w*gmax
apre += Apre
w = clip(w+apost, 0, wmax)
"""
stdp_on_post = """
apost += Apost
w = clip(w+apre, 0, wmax)
"""
scaling_op = "w = w * target_total / (w_total_post + 1e-9)"


def apre_to_apost(apre, depression_bias=DEPRESSION_BIAS):
    return -apre * (taupre / taupost) * depression_bias


def _neuron_namespace():
    return {"tau": tau, "v_rest": v_rest, "v_thresh": v_thresh, "v_reset": v_reset}


def _synapse_namespace(apre_val, apost_val, gmax, target_total):
    return {"Apre": apre_val, "Apost": apost_val, "gmax": gmax, "target_total": target_total,
            "taupre": taupre, "taupost": taupost, "wmax": wmax}


def build_network(idx, t, apre_val, gmax=GMAX, w_init=W_INIT, target_total=TARGET_TOTAL,
                   scaling_interval=SCALING_INTERVAL, n_synapses=N_SYNAPSES):
    """Build (pre, post, syn) for one run. Caller owns start_scope() before and run() after."""
    apost_val = apre_to_apost(apre_val)

    pre = SpikeGeneratorGroup(n_synapses, idx, t)
    post = NeuronGroup(1, post_eqs, threshold="v>v_thresh", reset="v=v_reset",
                        refractory=t_ref, method="exact", namespace=_neuron_namespace())
    post.v = v_rest

    syn = Synapses(pre, post, model=stdp_model_homeo, on_pre=stdp_on_pre, on_post=stdp_on_post,
                    namespace=_synapse_namespace(apre_val, apost_val, gmax, target_total))
    syn.connect()
    syn.w = w_init
    syn.run_regularly(scaling_op, dt=scaling_interval)

    return pre, post, syn


def build_population_network(n_post, idx, t, apre_val, n_pre_per_neuron=N_SYNAPSES, gmax=GMAX,
                              w_init=W_INIT, target_total=TARGET_TOTAL,
                              scaling_interval=SCALING_INTERVAL):
    """n_post postsynaptic neurons, each block-diagonally connected to its own dedicated
    n_pre_per_neuron-neuron presynaptic block (build with
    spikes.build_population_presynaptic_input, which returns matching idx/t/n_pre_per_neuron).
    Each neuron gets its own independent homeostatic scaling -- Brian2's `w_total_post`
    (summed) variable groups by postsynaptic index automatically.

    An earlier version shared one presynaptic pool across all postsynaptic neurons (all-to-all)
    and relied on small initial-weight jitter to break the symmetry between them. That was
    found NOT to work: the LIF hard reset erases pre-spike membrane differences every
    interspike interval, and postsynaptic spike timing turned out to be robust to the jitter
    magnitude tried, so every neuron converged to a bit-identical trajectory regardless of
    starting weights. Giving each neuron a genuinely different input (this version) is the more
    robust way to get real divergence, at the cost of not testing literal input-sharing/
    competition between neurons -- see experiments_brian2.md for the full reasoning.

    Returns (pre, post, syn); syn.i[:]/syn.j[:] (global presynaptic index, postsynaptic index)
    are needed to map weight-trace rows back to (correlated-or-not within its own block, which
    neuron) -- see metrics.compute_population_metrics.
    """
    apost_val = apre_to_apost(apre_val)
    n_pre_total = n_post * n_pre_per_neuron

    pre = SpikeGeneratorGroup(n_pre_total, idx, t)
    post = NeuronGroup(n_post, post_eqs, threshold="v>v_thresh", reset="v=v_reset",
                        refractory=t_ref, method="exact", namespace=_neuron_namespace())
    post.v = v_rest

    syn = Synapses(pre, post, model=stdp_model_homeo, on_pre=stdp_on_pre, on_post=stdp_on_post,
                    namespace=_synapse_namespace(apre_val, apost_val, gmax, target_total))
    syn.connect(j=f"i // {n_pre_per_neuron}")  # block-diagonal: presynaptic block b -> postsynaptic neuron b
    syn.w = w_init
    syn.run_regularly(scaling_op, dt=scaling_interval)

    return pre, post, syn

"""Postsynaptic LIF neuron + STDP synapses with homeostatic synaptic scaling.

Canonical rig confirmed byte-for-byte identical between
notebooks/brian2/apre005_ensemble_data/run_single_seed.py (the 8-seed/5000s ensemble) and
notebooks/brian2/brian2_stdp_apre_sweep.ipynb (the Apre sweep that established the standing
amplitude-vs-frequency explanation) -- see experiments_brian2.md for the experiment history
that produced these parameters (runaway -> homeostatic scaling fix -> interval/jump-cap
hypotheses rejected -> amplitude-vs-frequency reframe).

Variable names below (tau, v_rest, taupre, wmax, ...) are deliberately lowercase and
module-level, matching the original notebooks: Brian2 resolves free identifiers in equation
strings from the enclosing Python namespace by stack inspection, so build_network() must be
called from a scope where these names are visible (true here, since it's this module's globals)
rather than passed explicitly except where they vary per-call (Apre/Apost/gmax/target_total).
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


def build_network(idx, t, apre_val, gmax=GMAX, w_init=W_INIT, target_total=TARGET_TOTAL,
                   scaling_interval=SCALING_INTERVAL, n_synapses=N_SYNAPSES):
    """Build (pre, post, syn) for one run. Caller owns start_scope() before and run() after."""
    apost_val = apre_to_apost(apre_val)

    pre = SpikeGeneratorGroup(n_synapses, idx, t)
    post = NeuronGroup(1, post_eqs, threshold="v>v_thresh", reset="v=v_reset",
                        refractory=t_ref, method="exact")
    post.v = v_rest

    syn = Synapses(pre, post, model=stdp_model_homeo, on_pre=stdp_on_pre, on_post=stdp_on_post,
                    namespace={"Apre": apre_val, "Apost": apost_val, "gmax": gmax,
                               "target_total": target_total})
    syn.connect()
    syn.w = w_init
    syn.run_regularly(scaling_op, dt=scaling_interval)

    return pre, post, syn

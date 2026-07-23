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

Design note for future long runs: existing ensemble/population run scripts only save group
aggregates (group_mean_gap etc.), final-snapshot weights, and reversal counts -- not full
per-synapse time traces. That's fine for the questions asked so far, but it means anything
about *when* an individual synapse settled, co-movement between synapses, or reversal-interval
timing is unanswerable from saved data without a full rerun (found the hard way via a post-hoc
analysis, see experiments_brian2.md). Future long runs should save at least a subsampled
per-synapse trace (e.g. every Nth StateMonitor sample, not every 500ms point) so trace-level
questions like this are answerable after the fact instead of requiring new compute.
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


# Rate-trace timescale for lateral inhibition's ambiguity gate -- deliberately slower than the
# STDP traces (taupre/taupost=20ms) since this needs to reflect *recent firing rate*, not
# spike-to-spike proximity. r_inc=1.0 per own spike with tau_r=200ms gives r's steady state
# roughly on the order of rate_Hz * 0.2 (e.g. ~4 at a 20Hz baseline) -- gap_scale is calibrated
# against that scale, not an independent absolute unit.
TAU_R = 200 * ms
R_INC = 1.0

# A one-time initial-v jitter alone was found NOT to be enough: LIF's hard reset to a fixed
# v_reset after the first spike wipes it out, and (as with the earlier failed shared-input+
# weight-jitter attempt) synchronized correlated-group bursts are large enough that a small
# one-time offset doesn't even reliably change which discrete timestep the first threshold
# crossing lands in -- confirmed directly, all 3 neurons still came back bit-identical after
# 30s with a 0.5mV initial jitter. Fixed with a small continuous noise term in the membrane
# equation instead (standard practice for exactly this reason) -- this supplies an ongoing
# source of tiny, independent asymmetry between neurons at every timestep for the inhibition
# feedback loop to amplify, rather than a single nudge that a hard reset can erase. The
# inhibition coupling still does the actual differentiating; this only gives it something
# to work with, matching the same "coupling, not noise, does the differentiating" intent.
SIGMA_V = 0.3 * mV

post_eqs_competitive = """
dv/dt = (v_rest - v)/tau + sigma_v*xi*tau**-0.5 : volt (unless refractory)
dr/dt = -r/tau_r : 1
w_total : 1
"""


REFERENCE_N_POST = 3          # the already-characterized bistability-sweep scale
REFERENCE_INHIB_STRENGTH = 13.0  # mV, per-connection -- the sweep's standout point


def scale_inhib_for_n(n_post, reference_inhib_mV=REFERENCE_INHIB_STRENGTH,
                       reference_n_post=REFERENCE_N_POST):
    """Per-connection inhib_strength (mV) for build_competitive_population_network's all-to-all
    inhibition, normalized so TOTAL incoming inhibitory drive per neuron stays comparable as
    n_post changes.

    Each neuron receives an inhibitory kick from every OTHER neuron's spike -- (n_post - 1)
    independent sources, not n_post^2 (that count is the total number of directed inhibitory
    synapses across the whole population, not what any single neuron actually receives). Holding
    (n_post - 1) * inhib_strength constant as n_post grows is what isolates n_post as the
    variable under test, rather than confounding it with "more total inhibition." A static,
    one-time division at build time is the right-sized fix here -- unlike the STDP layer's
    homeostatic synaptic scaling (a `run_regularly` correction applied every 500ms), n_post and
    the resulting connection count are fixed at construction, not a learned quantity that drifts
    over a run and needs continuous re-normalization.

    Calibrated so n_post=reference_n_post reproduces reference_inhib_mV exactly (the
    already-characterized standout point from the original bistability sweep, inhib=13mV at
    n_post=3), so this is a strict generalization, not a re-tuning of the known-good setting.

    Known limitation, not solved here: this matches the MEAN total inhibitory drive across
    n_post, not its variance -- (n_post-1) independent smaller kicks average out smoother than 2
    larger ones even at matched mean (a real, second-order confound), but chasing higher-moment
    matching is out of scope for a first normalization pass."""
    return reference_inhib_mV * (reference_n_post - 1) / (n_post - 1)


def build_competitive_population_network(n_post, idx, t, apre_val, inhib_strength, gap_scale,
                                          n_pre=N_SYNAPSES, gmax=GMAX, w_init=W_INIT,
                                          target_total=TARGET_TOTAL,
                                          scaling_interval=SCALING_INTERVAL,
                                          tau_r=TAU_R, r_inc=R_INC, sigma_v=SIGMA_V):
    """n_post postsynaptic neurons genuinely sharing one presynaptic pool, all-to-all (unlike
    build_population_network's block-diagonal independent replicates) -- each still gets its
    own independent STDP + homeostatic scaling on its own n_pre incoming synapses (the
    stdp_model_homeo/stdp_on_pre/stdp_on_post/scaling_op strings are reused completely
    unmodified from build_network/build_population_network).

    New here: post-to-post lateral inhibition, ambiguity-gated by a per-neuron exponential
    recent-firing-rate trace `r` (own spike -> r += r_inc; decays with tau_r otherwise). On a
    postsynaptic-population spike from neuron i, neuron j gets an inhibitory kick of
    `inhib_strength * g_ij`, where `g_ij = 1 / (1 + |r_i - r_j| / gap_scale)` -- the same
    ambiguity-gate shape as the Hopfield retrieval bias (principles.md), just measuring the gap
    between two neurons' recent rates instead of a top1-top2 content-similarity gap. Two
    neurons with clearly different recent rates get weak inhibition (the gap can keep growing);
    two neurons with near-equal recent rates get strong inhibition (forced differentiation).
    inhib_strength: a Brian2 voltage Quantity (e.g. 5*mV), NOT a bare float, matching gmax's
    convention. gap_scale: a bare float, compared against |r_i - r_j| (see TAU_R note above).

    An earlier shared-input attempt (no lateral inhibition, just initial-weight jitter) failed
    to produce any divergence at all -- LIF's hard reset erases pre-spike membrane differences
    every interspike interval, and spike timing was robust to the jitter tried. This adds actual
    coupling between neurons instead of relying on noise to break symmetry.

    A perfectly symmetric starting point still needs *something* to seed a direction, though.
    Caught by pre-calibration smoke tests, in two stages: (1) with identical shared input,
    identical initial weights, and symmetric inhibition, all n_post neurons stayed bit-identical
    the entire run -- a deterministic system with zero asymmetry anywhere has no direction to
    differentiate toward, no matter how strong the inhibition is; inhibition amplifies a
    difference, it doesn't create one from nothing. (2) A one-time initial-v jitter (tried
    first) did NOT fix it either -- the hard reset to a fixed v_reset after the first spike
    wipes out a one-time nudge, and (matching the earlier failed weight-jitter attempt) large
    synchronized correlated-group bursts are robust enough that a small one-time offset doesn't
    even reliably change which discrete timestep the first threshold crossing lands in; all 3
    neurons still came back bit-identical after 30s. Fixed with `sigma_v`, a small *continuous*
    noise term in the membrane equation (standard practice for exactly this reason) -- an
    ongoing source of tiny independent per-neuron asymmetry every timestep, for the inhibition
    feedback loop to amplify at any point, not a single erasable nudge. The inhibition coupling
    still does the actual differentiating; this only gives it something to work with. Requires
    `method="euler"` instead of `"exact"` -- Brian2's exact/analytic integrator doesn't support
    stochastic (`xi`) terms.

    Returns (pre, post, syn, inhib). `g_ij(t)` is deliberately NOT stored as its own monitored
    variable -- it's fully reconstructable post-hoc from a saved `r` trace (`g_ij(t) = 1 / (1 +
    abs(r_i(t) - r_j(t)) / gap_scale)`) plus the known `gap_scale`, so record `r` instead of
    duplicating that computation into the simulation.
    """
    apost_val = apre_to_apost(apre_val)

    pre = SpikeGeneratorGroup(n_pre, idx, t)
    post = NeuronGroup(n_post, post_eqs_competitive, threshold="v>v_thresh",
                        reset="v=v_reset; r += r_inc", refractory=t_ref, method="euler",
                        namespace={**_neuron_namespace(), "tau_r": tau_r, "r_inc": r_inc,
                                   "sigma_v": sigma_v})
    post.v = v_rest
    post.r = 0

    syn = Synapses(pre, post, model=stdp_model_homeo, on_pre=stdp_on_pre, on_post=stdp_on_post,
                    namespace=_synapse_namespace(apre_val, apost_val, gmax, target_total))
    syn.connect()  # genuine all-to-all: every presynaptic neuron feeds every postsynaptic neuron
    syn.w = w_init
    syn.run_regularly(scaling_op, dt=scaling_interval)

    inhib = Synapses(post, post,
                      on_pre="v_post -= inhib_strength / (1 + abs(r_pre - r_post) / gap_scale)",
                      namespace={"inhib_strength": inhib_strength, "gap_scale": gap_scale})
    inhib.connect(condition="i != j")

    return pre, post, syn, inhib

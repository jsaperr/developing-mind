"""MNIST pilot simulation core (generalization plan step 4a), shared by the local calibration and the Modal runs.

The substrate is unchanged: src.brian2_stdp.network.build_competitive_population_network (LIF, STDP, homeostatic
weight budget, ambiguity-gated lateral inhibition), 13 mV/1.5 reference inhibition normalized with
scale_inhib_for_n, Apre 0.005, dt 0.2 ms, Brian2 seeded. Only the input is new:
  rate coding (Diehl & Cook 2015): pixel i fires Poisson at x_i/255 * MAX_RATE during a 350 ms presentation, then
  150 ms of silence. Spike times are snapped to the 0.2 ms grid and de-duplicated per input.
No adaptive threshold (it's added only if the pilot shows monopolization; plan step 4).
Plasticity is never switched off (the framework has no train/test split). The splits only label time periods.

Timeline per seed: train (1200 images, classes 0-3, shuffled) -> label (200 images, classes 0-3) -> test (600
images: 100 each of 0-5, where 4 and 5 were never seen).
Labels are used only for scoring, afterwards.
"""
import time

import numpy as np

MAX_RATE = 63.75
PRESENT_S, REST_S = 0.35, 0.15
DT_S = 0.0002


def build_input(images, max_rate, rng, t0=0.0):
    """images (n, 196) uint8 -> (idx, t) arrays for SpikeGeneratorGroup, plus per-image start times."""
    period = PRESENT_S + REST_S
    starts = t0 + period * np.arange(len(images))
    rates = np.asarray(images, dtype=float) / 255.0 * max_rate          # (n, 196)
    counts = rng.poisson(rates * PRESENT_S)                              # (n, 196)
    img_of = np.repeat(np.arange(len(images)), counts.sum(1))
    pix = np.concatenate([np.repeat(np.arange(images.shape[1]), c) for c in counts]) if counts.sum() else np.zeros(0, int)
    t = starts[img_of] + rng.uniform(0, PRESENT_S, size=len(img_of))
    step = np.floor(t / DT_S).astype(np.int64)
    key = np.unique(step * 1000 + pix)                                   # one spike per input per time step
    step, pix = key // 1000, key % 1000
    return pix.astype(np.int64), step * DT_S, starts


def run_pilot(seed, subset, target_total, n_post=40, n_train=None, max_rate=MAX_RATE, normalize=False, adaptive=None):
    """subset: dict of arrays from prep_mnist.py's npz. Returns a compact result dict (all small).
    normalize (v2): per-image gain control. Every image delivers the same total input rate, equal to the TRAINING
      set's mean total (mean summed intensity x max_rate), so total drive no longer encodes ink.
    adaptive (v2): None, or dict(theta_plus_mV, tau_theta_s). Diehl & Cook's adaptive threshold: threshold
      v > v_thresh + theta; each spike adds theta_plus; theta decays with tau_theta. Network otherwise identical to
      src build_competitive_population_network (built here as a notebook variant; src is untouched)."""
    from brian2 import SpikeMonitor, defaultclock, mV, ms, prefs, run, second, seed as b2_seed, start_scope
    from src.brian2_stdp.network import build_competitive_population_network, scale_inhib_for_n

    rng = np.random.default_rng(seed)
    order = {}
    for split in ("train", "label", "test"):
        n = len(subset[f"{split}_y"]) if (split != "train" or n_train is None) else n_train
        order[split] = rng.permutation(len(subset[f"{split}_y"]))[:n]
    seq_x = np.concatenate([subset[f"{s}_x"][order[s]].reshape(len(order[s]), -1) for s in ("train", "label", "test")])
    seq_y = np.concatenate([subset[f"{s}_y"][order[s]] for s in ("train", "label", "test")])
    seq_split = np.concatenate([[s] * len(order[s]) for s in ("train", "label", "test")])
    if normalize:
        tr = subset["train_x"].reshape(len(subset["train_y"]), -1).astype(float)
        total = tr.sum(1).mean()                                         # the training set's own mean total intensity
        sums = seq_x.astype(float).sum(1, keepdims=True)
        seq_in = seq_x.astype(float) / np.maximum(sums, 1e-9) * total    # same total for every image
    else:
        seq_in = seq_x
    idx, t, starts = build_input(seq_in, max_rate, rng)
    n_pre = seq_x.shape[1]

    start_scope()
    defaultclock.dt = 0.2 * ms
    b2_seed(seed)
    per_conn = scale_inhib_for_n(n_post, reference_inhib_mV=13.0, reference_n_post=3)
    builder = build_competitive_population_network if adaptive is None else build_adaptive
    kw = {} if adaptive is None else dict(theta_plus=adaptive["theta_plus_mV"] * mV, tau_theta=adaptive["tau_theta_s"] * second)
    pre, post, syn, inhib = builder(n_post, idx, t * second, 0.005, per_conn * mV, 1.5, n_pre=n_pre,
                                    w_init=target_total / n_pre, target_total=target_total, **kw)
    sp = SpikeMonitor(post)
    si, sj = np.array(syn.i[:]), np.array(syn.j[:])

    def weights():
        w = np.zeros((n_post, n_pre)); w[sj, si] = np.array(syn.w[:]); return np.round(w, 4).tolist()

    t0 = time.time(); snaps = {}
    period = PRESENT_S + REST_S
    for split in ("train", "label", "test"):
        run(len(order[split]) * period * second)
        snaps[split] = weights()
    st, sidx = np.array(sp.t / second), np.array(sp.i[:])
    img = np.floor(st / period).astype(int)
    in_present = (st - starts[np.clip(img, 0, len(starts) - 1)]) < PRESENT_S
    counts = np.zeros((n_post, len(seq_y)), dtype=np.int32)
    np.add.at(counts, (sidx[in_present], img[in_present]), 1)
    theta_final = (np.asarray(post.theta / mV).round(4).tolist() if adaptive is not None else None)
    return dict(status="completed", seed=int(seed), n_post=n_post, normalize=bool(normalize), adaptive=adaptive,
                theta_final_mV=theta_final, n_pre=n_pre, target_total=target_total,
                max_rate=max_rate, present_s=PRESENT_S, rest_s=REST_S, inhib_strength_mV=float(per_conn),
                gap_scale=1.5, apre=0.005, dt_ms=0.2, brian2_seeded=True, target=prefs.codegen.target,
                wall_elapsed=time.time() - t0, labels=seq_y.tolist(), split=seq_split.tolist(),
                image_order={k: v.tolist() for k, v in order.items()},
                counts=counts.tolist(), weights_after=snaps)


def build_adaptive(n_post, idx, t, apre_val, inhib_strength, gap_scale, n_pre, w_init, target_total, theta_plus, tau_theta):
    """src build_competitive_population_network plus Diehl & Cook's adaptive threshold (notebook variant, src
    untouched). Everything else (LIF + sigma_v noise, STDP, homeostatic weight budget, ambiguity-gated lateral
    inhibition) is the same code and the same constants."""
    from brian2 import NeuronGroup, SpikeGeneratorGroup, Synapses, mV
    from src.brian2_stdp import network as N
    eqs = N.post_eqs_competitive + "dtheta/dt = -theta/tau_theta : volt\n"
    pre = SpikeGeneratorGroup(n_pre, idx, t)
    post = NeuronGroup(n_post, eqs, threshold="v>v_thresh + theta", reset="v=v_reset; r += r_inc; theta += theta_plus",
                       refractory=N.t_ref, method="euler",
                       namespace={**N._neuron_namespace(), "tau_r": N.TAU_R, "r_inc": N.R_INC, "sigma_v": N.SIGMA_V,
                                  "theta_plus": theta_plus, "tau_theta": tau_theta})
    post.v = N.v_rest; post.r = 0; post.I_pert = 0 * mV; post.theta = 0 * mV
    syn = Synapses(pre, post, model=N.stdp_model_homeo, on_pre=N.stdp_on_pre, on_post=N.stdp_on_post,
                   namespace=N._synapse_namespace(apre_val, N.apre_to_apost(apre_val), N.GMAX, target_total))
    syn.connect(); syn.w = w_init
    syn.run_regularly(N.scaling_op, dt=N.SCALING_INTERVAL)
    inhib = Synapses(post, post, on_pre="v_post -= inhib_strength / (1 + abs(r_pre - r_post) / gap_scale)",
                     namespace={"inhib_strength": inhib_strength, "gap_scale": gap_scale})
    inhib.connect(condition="i != j")
    return pre, post, syn, inhib

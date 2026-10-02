"""Gain map: is the leaky fair-share window set by TWO quantities, memory length (tau) and loop gain (step x tau)?

Leak test (modal_leak.py): at Diehl & Cook's step theta_plus = 0.05 mV, tau 50-100 s passes both worlds. Below that, MNIST
loses diversity; above it, the click world flickers and then habituates. Only tau was moved, so two things changed at once:
  - memory length (tau): how far back the threshold remembers who has been winning. MNIST needs it to span enough images
    to sample every digit.
  - loop gain G ~ theta_plus * tau * (rate slope, ~1 Hz/mV): the steady state shrinks rate differences by 1/(1+G). A high
    gain erases a sustained context (and, through adaptation, makes neurons take turns: flicker).
Hypothesis: click-world survival and flicker depend on G (the product), MNIST diversity mostly on tau. If so, a long memory
with a gentle gain passes both comfortably, and the hand-picked tau becomes two self-describable targets.

Cells (winner-take-all 20 mV, gate off + leaky fair-share; MNIST with causal gain control off, i.e. v4-style per-image,
for comparability with the leak test). Reference cell tp 0.05 / tau 100 = leak_tau100 / mnist_leak_tau100 (already run).
  equal gain (tp*tau = 5 mV.s):   tp 0.2 / tau 25,   tp 0.0125 / tau 400
  low gain   (tp*tau = 1 mV.s):   tp 0.01 / tau 100, tp 0.0025 / tau 400
Click: modal_unify.py's world, N=40, seeds 53100-53107 (paired). MNIST: v4 conditions, seeds 70100-70107 (paired).
Files: gmap_tp<x>_tau<y>_n40_seed<S>.json.gz here (compact); ../mnist_pilot/mnist_gmap_tp<x>_tau<y>_seed<S>.json.gz.
Scoring: analyze_factorial.score + the flicker measure (analyze_gain_map.py); MNIST via analyze_mnist_pilot.py <prefix> v3.

PREDICTIONS ON RECORD (written before launch):
  GM-P1 at equal gain, click flicker is about the same across tau 25 / 100 / 400 (each within 2x of tau 100's 14%), and the
        click world passes (readout >= 0.9, memory >= 7/8) at all three.
  GM-P2 at equal gain, MNIST improves with tau: tau 25 < tau 100 (74.8%) <= tau 400 in accuracy, and tau 25 loses diversity
        (every digit >= 3 neurons in < 6/8 seeds).
  GM-P3 at low gain, click flicker is < 3% at both tau (no turn-taking).
  GM-P4 at low gain and tau 400 s, MNIST keeps diversity (every digit >= 3 in >= 6/8) with accuracy >= 65%.
  => GM-P1..P4 together = the product law: tau = "long enough to sample the inputs", gain = "about 1".

Cost: 32 click runs (~20 min) + 32 MNIST (~5 min) in parallel, about $1.5.
Run from the repo root:  python -m modal run notebooks/brian2/unified_substrate/modal_gain_map.py
"""
import gzip
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MN = HERE.parent / "mnist_pilot"
sys.path.insert(0, str(HERE.parent))
from modal_common import IMAGE, RAW_MOUNT, RAW_VOLUME  # noqa: E402

image = (IMAGE.add_local_file(str(HERE.parent / "modal_common.py"), "/root/modal_common.py")
         .add_local_file(str(MN / "mnist_sim.py"), "/root/mnist_sim.py")
         .add_local_file(str(MN / "data" / "pilot_subset.npz"), "/root/pilot_subset.npz"))
app = modal.App("developing-mind-gain-map")
CELLS = [(0.2, 25.0), (0.0125, 400.0), (0.01, 100.0), (0.0025, 400.0)]
CLICK_SEEDS = list(range(53100, 53108))
MNIST_SEEDS = list(range(70100, 70108))
SCHEDULE, DURS = [0, 1, 2, 0, 1], [1000.0] * 5


def tag(tp, tau):
    return f"tp{tp:g}_tau{tau:g}".replace(".", "p")


@app.function(image=image, cpu=1.0, memory=4096, timeout=3 * 3600, max_containers=64,
              volumes={RAW_MOUNT: RAW_VOLUME}, retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_click(tp: float, tau: float, seed_val: int) -> dict:
    import time
    import numpy as np
    from brian2 import SpikeMonitor, StateMonitor, defaultclock, mV, ms, prefs, run, second, start_scope, seed as b2_seed
    import mnist_sim as S
    import modal_common as MC
    from src.brian2_stdp import network as N
    from src.brian2_stdp.spikes import build_multiblock_phase_input

    n_post = 40
    start_scope()
    defaultclock.dt = 0.2 * ms
    b2_seed(seed_val)
    rng = np.random.default_rng(seed_val)
    idx, t = build_multiblock_phase_input(20.0, 0.9, DURS, SCHEDULE, rng, n_blocks=3, block_size=10)
    inh = float((N.v_thresh - N.v_rest) / mV)
    pre, post, syn, inhib, share = S.build_adaptive(n_post, idx, t, 0.005, inh * mV, 1e9, n_pre=30, w_init=1 / 3,
                                                    target_total=10.0, theta_plus=tp * mV, tau_theta=tau * second,
                                                    fair_share=True)
    spikes = SpikeMonitor(post)
    wt = StateMonitor(syn, "w", record=True, dt=1 * second)
    t0 = time.time()
    run(sum(DURS) * second)
    total = sum(DURS)
    st = np.array(spikes.t / second); si = np.array(spikes.i[:])
    edges = np.arange(int(total) + 1, dtype=float)
    rate_bins = np.stack([np.histogram(st[si == j], bins=edges)[0] for j in range(n_post)])
    full = dict(status="completed", seed=seed_val, n_post=n_post, cell=tag(tp, tau), tp=tp, tau=tau, target=prefs.codegen.target,
                wall_elapsed=time.time() - t0, world="unify_v1b", inhib_strength_mV=inh, gap_scale=1e9, theta_plus_mV=tp,
                tau_theta_s=tau, fair_share=True, apre=0.005, p_share=0.9, dt_ms=0.2, brian2_seeded=True,
                phase_durations_s=DURS, phase_corr_blocks=SCHEDULE, swap_times_s=[1000.0, 2000.0, 3000.0, 4000.0], total_s=total,
                context_sets=[list(range(0, 10)), list(range(10, 20)), list(range(20, 30))],
                syn_i=np.array(syn.i[:]).tolist(), syn_j=np.array(syn.j[:]).tolist(),
                weight_trace=wt.w[:].round(4).tolist(), spike_rate_bins=rate_bins.tolist(),
                theta_final_mV=np.asarray(post.theta / mV).round(4).tolist())
    raw_path = f"unified_substrate/gmap/gmap_{tag(tp, tau)}_n40_seed{seed_val}.json.gz"
    MC.write_raw(raw_path, full)
    out = MC.compact(full, raw_path)
    out.update(tp=tp, tau=tau)
    return out


@app.function(image=image, cpu=1.0, memory=4096, timeout=2 * 3600, max_containers=64,
              retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_mnist(tp: float, tau: float, seed: int) -> dict:
    import numpy as np
    import mnist_sim as S
    subset = dict(np.load("/root/pilot_subset.npz"))
    r = S.run_pilot(seed, subset, 30.0, n_post=40, normalize=True, wta=True,
                    adaptive=dict(theta_plus_mV=tp, tau_theta_s=tau, fair_share=True))
    r.update(tp=tp, tau=tau)
    return r


@app.local_entrypoint()
def main():
    clicks = [(tp, tau, s) for tp, tau in CELLS for s in CLICK_SEEDS if not (HERE / f"gmap_{tag(tp, tau)}_n40_seed{s}.json.gz").exists()]
    mn = [(tp, tau, s) for tp, tau in CELLS for s in MNIST_SEEDS if not (MN / f"mnist_gmap_{tag(tp, tau)}_seed{s}.json.gz").exists()]
    print(f"{len(clicks)} click + {len(mn)} MNIST runs to do", flush=True)
    calls = [("mnist", run_mnist.spawn(*x)) for x in mn] + [("click", run_click.spawn(*x)) for x in clicks]
    ok = 0
    for kind, c in calls:
        try:
            r = c.get()
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {kind} {e!r}", flush=True); continue
        p = (MN / f"mnist_gmap_{tag(r['tp'], r['tau'])}_seed{r['seed']}.json.gz" if kind == "mnist"
             else HERE / f"gmap_{tag(r['tp'], r['tau'])}_n40_seed{r['seed']}.json.gz")
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(r, f)
        ok += 1
        print(f"saved {p.name}", flush=True)
    print(f"== {ok}/{len(calls)} completed", flush=True)

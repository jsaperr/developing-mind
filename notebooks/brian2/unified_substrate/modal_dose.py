"""Dose test: is there a fair-share speed at which ONE substrate passes both worlds?

The factorial (modal_factorial.py, 2026-10-02) found the fair-share threshold alone breaks the click world, and the
mechanism is HABITUATION. Fair-share equalizes firing rates across neurons. Over a sustained context it erases "who
responds": the readout peaks at about 0.6 around 60 s after a switch, then falls to about 0 by 300 s, as the rate CV
collapses from 0.29 to 0.04. Winner-take-all alone is harmless. MNIST still needs both rules (alone: 29% and 47%).
On MNIST (0.35 s images) equalization works ACROSS images, giving diversity; on the click world (1000 s contexts) it
works WITHIN a context and erases it.
Hypothesis: homeostasis must be slower than the experiences it should represent. The equalization rate scales with
theta_plus (0.05 mV, Diehl & Cook's). So slow it 10x and 100x, and give MNIST 10 passes over its training images so the
total homeostatic work is matched at 10x.
NOTE (thalamus-tripwire spirit): theta_plus is a substrate constant, and this is a hand-chosen dose, a DIAGNOSTIC for
whether a window exists, not a value to adopt. If one exists, the principled version is a homeostasis timescale set
relative to the timescale of experience.

Cells (all the unified substrate: winner-take-all 20 mV, gate off, fair-share threshold):
  click world (modal_unify.py's world, N in {7, 40}, the same seeds 53000-53007 / 53100-53107): theta_plus 0.005, 0.0005 mV
  MNIST (v4's: gain control on, N=40, budget 30, classes 0-3, seeds 70100-70107), 10 training passes: theta_plus 0.05
    (control: does more experience alone change v4?), 0.005, 0.0005.
Click files: dose_tp<x>_n<N>_seed<S>.json.gz here (compact; raw on the Volume under unified_substrate/dose/).
MNIST: ../mnist_pilot/mnist_dose_tp<x>_e10_seed<S>.json.gz. Scoring: analyze_factorial.py's metrics (via analyze_dose.py),
and analyze_mnist_pilot.py <prefix> v3.

PREDICTIONS ON RECORD (written before launch):
  D-P1 click, theta_plus 0.005: passes (settled readout >= 0.9 and B named at two-back W=50 >= 7/8, both N). Habituation is
       ~10x slower (~3000 s), longer than a 1000 s phase. The readout 900 s into a phase stays >= 0.8.
  D-P2 click, theta_plus 0.0005: passes, both N.
  D-P3 MNIST, theta_plus 0.005 x 10 passes: keeps v4's result: vote accuracy >= 70%, every class >= 3 neurons in >= 6/8.
  D-P4 MNIST, theta_plus 0.0005 x 10 passes: diversity doesn't fully develop (a tenth of v4's homeostatic work): accuracy
       < 65%.
  D-P5 MNIST, theta_plus 0.05 x 10 passes: >= 75% (more experience doesn't hurt v4).
  => D-P1 and D-P3 together = a window where one substrate passes both, with longer development as the price. The scaling
     caveat, stated now: any fixed homeostasis speed will habituate to a context that lasts long enough.

Cost estimate: 32 click runs (~20 min) + 24 MNIST runs (~7000 s simulated, ~35 min wall), about $1.5.
Run from the repo root:  python -m modal run notebooks/brian2/unified_substrate/modal_dose.py
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
app = modal.App("developing-mind-dose")
NS = [7, 40]
SEED_BASE = {7: 53000, 40: 53100}
CLICK_TP = [0.005, 0.0005]
MNIST_TP = [0.05, 0.005, 0.0005]
EPOCHS = 10
MNIST_SEEDS = list(range(70100, 70108))
SCHEDULE, DURS = [0, 1, 2, 0, 1], [1000.0] * 5


def tag(tp):
    return f"{tp:g}".replace(".", "p")


@app.function(image=image, cpu=1.0, memory=4096, timeout=4 * 3600, max_containers=64,
              volumes={RAW_MOUNT: RAW_VOLUME}, retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_click(tp: float, n_post: int, seed_val: int) -> dict:
    import time
    import numpy as np
    from brian2 import SpikeMonitor, StateMonitor, defaultclock, mV, ms, prefs, run, second, start_scope, seed as b2_seed
    import mnist_sim as S
    import modal_common as MC
    from src.brian2_stdp import network as N
    from src.brian2_stdp.spikes import build_multiblock_phase_input

    start_scope()
    defaultclock.dt = 0.2 * ms
    b2_seed(seed_val)
    rng = np.random.default_rng(seed_val)
    idx, t = build_multiblock_phase_input(20.0, 0.9, DURS, SCHEDULE, rng, n_blocks=3, block_size=10)
    inh = float((N.v_thresh - N.v_rest) / mV)
    pre, post, syn, inhib, share = S.build_adaptive(n_post, idx, t, 0.005, inh * mV, 1e9, n_pre=30, w_init=1 / 3,
                                                    target_total=10.0, theta_plus=tp * mV, tau_theta=1e12 * second,
                                                    fair_share=True)
    spikes = SpikeMonitor(post)
    wt = StateMonitor(syn, "w", record=True, dt=1 * second)
    th = StateMonitor(post, "theta", record=True, dt=10 * second)
    t0 = time.time()
    run(sum(DURS) * second)
    total = sum(DURS)
    st = np.array(spikes.t / second); si = np.array(spikes.i[:])
    edges = np.arange(int(total) + 1, dtype=float)
    rate_bins = np.stack([np.histogram(st[si == j], bins=edges)[0] for j in range(n_post)])
    full = dict(status="completed", seed=seed_val, n_post=n_post, cell=f"uni_tp{tag(tp)}", target=prefs.codegen.target,
                wall_elapsed=time.time() - t0, world="unify_v1b", inhib_strength_mV=inh, gap_scale=1e9, theta_plus_mV=tp,
                fair_share=True, apre=0.005, p_share=0.9, dt_ms=0.2, brian2_seeded=True,
                phase_durations_s=DURS, phase_corr_blocks=SCHEDULE, swap_times_s=[1000.0, 2000.0, 3000.0, 4000.0], total_s=total,
                context_sets=[list(range(0, 10)), list(range(10, 20)), list(range(20, 30))],
                syn_i=np.array(syn.i[:]).tolist(), syn_j=np.array(syn.j[:]).tolist(),
                weight_trace=wt.w[:].round(4).tolist(), spike_rate_bins=rate_bins.tolist(),
                theta_trace_10s_mV=np.asarray(th.theta / mV).round(4).tolist(),
                theta_final_mV=np.asarray(post.theta / mV).round(4).tolist())
    raw_path = f"unified_substrate/dose/dose_tp{tag(tp)}_n{n_post}_seed{seed_val}.json.gz"
    MC.write_raw(raw_path, full)
    out = MC.compact(full, raw_path)
    out.update(tp=tp, theta_trace_10s_mV=full["theta_trace_10s_mV"])
    return out


@app.function(image=image, cpu=1.0, memory=4096, timeout=6 * 3600, max_containers=64,
              retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_mnist(tp: float, seed: int) -> dict:
    import numpy as np
    import mnist_sim as S
    subset = dict(np.load("/root/pilot_subset.npz"))
    fair = dict(theta_plus_mV=tp, tau_theta_s=1e12, fair_share=True)
    r = S.run_pilot(seed, subset, 30.0, n_post=40, normalize=True, adaptive=fair, wta=True, epochs=EPOCHS)
    r["tp"] = tp
    return r


@app.local_entrypoint()
def main():
    clicks = [(tp, n, SEED_BASE[n] + k) for tp in CLICK_TP for n in NS for k in range(8)
              if not (HERE / f"dose_tp{tag(tp)}_n{n}_seed{SEED_BASE[n] + k}.json.gz").exists()]
    mn = [(tp, s) for tp in MNIST_TP for s in MNIST_SEEDS if not (MN / f"mnist_dose_tp{tag(tp)}_e{EPOCHS}_seed{s}.json.gz").exists()]
    print(f"{len(clicks)} click + {len(mn)} MNIST runs to do", flush=True)
    calls = [("click", run_click.spawn(*x)) for x in clicks] + [("mnist", run_mnist.spawn(*x)) for x in mn]
    ok = 0
    for kind, c in calls:                      # all jobs already running; blocking get in order only delays saving
        try:
            r = c.get()
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {kind} {e!r}", flush=True); continue
        if kind == "mnist":
            p = MN / f"mnist_dose_tp{tag(r['tp'])}_e{EPOCHS}_seed{r['seed']}.json.gz"
        else:
            p = HERE / f"dose_tp{tag(r['tp'])}_n{r['n_post']}_seed{r['seed']}.json.gz"
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(r, f)
        ok += 1
        print(f"saved {p.name} (wall {r.get('wall_elapsed', 0):.0f}s)", flush=True)
    print(f"== {ok}/{len(calls)} completed", flush=True)

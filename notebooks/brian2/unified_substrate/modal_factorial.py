"""Which rule breaks the click world? A 2x2 factorial of the two substrate rules MNIST needed, on both worlds.

The unified-substrate check (modal_unify.py, 2026-10-01) changed two rules at once: winner-take-all inhibition (20 mV =
v_thresh - v_rest per competitor spike, ambiguity gate off) and the fair-share threshold (+0.05 mV own spike, -0.05/N all).
Together they learn digits (MNIST v4, 79%) but break click-world integration (readout -0.18/0.06, memory 0/8). Each rule
alone has never run on either world. Cells (inhibition x threshold):
                     no threshold                      fair-share threshold
  gated 13 mV/1.5    old substrate (click: passes;      "fair"  (NEW on both worlds)
                     MNIST+gain: 47%, no specialization)
  WTA 20 mV, gate off "wta"  (NEW on both worlds)        unified (click: FAILS; MNIST+gain: 79%)
Click world: exactly modal_unify.py's (v1b A B C A B, 5 x 1000 s, disjoint 30-input rig, target_total 10, w_init 1/3, dt
0.2 ms, Brian2 seeded, N in {7, 40}, the same seeds 53000-53007 / 53100-53107, so every cell is paired with the unified one).
All click cells use mnist_sim.build_adaptive, including a "ctrl" cell (gated, theta_plus = 0, no fair-share), which should
behave as the old substrate; it's the code control for the builder.
MNIST: exactly v4's (classes 0-3, gain control on, N=40, budget 30, seeds 70100-70107), arms "norm_fair" (gated + fair) and
"wta_norm" (WTA, no threshold). Scored with analyze_mnist_pilot.py <prefix> v3.
Click files: fact_<cell>_n<N>_seed<S>.json.gz here (compact; raw on the Volume under unified_substrate/fact/). MNIST files:
../mnist_pilot/mnist_fact_<arm>_seed<S>.json.gz. Click scoring: analyze_factorial.py (U-P metrics, from analyze_unify.py).

PREDICTIONS ON RECORD (written before launch). Pass = U-P4's bar: settled readout >= 0.9 to its prototype AND B named at
two-back (W=50) in >= 7/8 seeds, at both N.
  F-P1 ctrl passes (it's the old substrate through the notebook builder).
  F-P2 fair-share alone passes the click world: with gentle gated inhibition the threshold only evens out firing.
  F-P3 WTA alone passes the click world's readout and memory bar (the winners are tuned to the current context). Its
       retention (incoming share before the two-back returns) isn't predicted.
  => if F-P2 and F-P3 both hold, the failure is the INTERACTION of the two rules (the standing hypothesis: fair-share lowers
     the thresholds of quiet, stale neurons and winner-take-all then lets them win), not either rule alone.
  F-P4 MNIST, WTA alone: one or a few neurons win every image (no threshold to spread the work): fewer than 3 classes with
       >= 3 neurons in >= 6/8 seeds, vote accuracy < 60%.
  F-P5 MNIST, fair-share alone (gated): no sharp specialization (specialized < 60%), vote accuracy < 65%.
  => if F-P4 and F-P5 hold, MNIST needs BOTH rules and the click world can't take both: the fix has to change how they
     interact, not drop one.

Cost estimate: 48 click runs (~15-25 min each) + 16 MNIST runs (~6 min) in parallel, ~25 min wall, about $1.
Run from the repo root:  python -m modal run notebooks/brian2/unified_substrate/modal_factorial.py
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
app = modal.App("developing-mind-factorial")
NS = [7, 40]
SEED_BASE = {7: 53000, 40: 53100}
CELLS = ["ctrl", "fair", "wta"]
MNIST_ARMS = ["norm_fair", "wta_norm"]
MNIST_SEEDS = list(range(70100, 70108))
SCHEDULE, DURS = [0, 1, 2, 0, 1], [1000.0] * 5


@app.function(image=image, cpu=1.0, memory=4096, timeout=4 * 3600, max_containers=64,
              volumes={RAW_MOUNT: RAW_VOLUME}, retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_click(cell: str, n_post: int, seed_val: int) -> dict:
    import time
    import numpy as np
    from brian2 import SpikeMonitor, StateMonitor, defaultclock, mV, ms, prefs, run, second, start_scope, seed as b2_seed
    import mnist_sim as S
    import modal_common as MC
    from src.brian2_stdp import network as N
    from src.brian2_stdp.network import scale_inhib_for_n
    from src.brian2_stdp.spikes import build_multiblock_phase_input

    start_scope()
    defaultclock.dt = 0.2 * ms
    b2_seed(seed_val)
    rng = np.random.default_rng(seed_val)
    idx, t = build_multiblock_phase_input(20.0, 0.9, DURS, SCHEDULE, rng, n_blocks=3, block_size=10)
    if cell == "wta":
        inh, gap = float((N.v_thresh - N.v_rest) / mV), 1e9
    else:
        inh, gap = scale_inhib_for_n(n_post, reference_inhib_mV=13.0, reference_n_post=3), 1.5
    fair = cell == "fair"
    built = S.build_adaptive(n_post, idx, t, 0.005, inh * mV, gap, n_pre=30, w_init=1 / 3, target_total=10.0,
                             theta_plus=(0.05 if fair else 0.0) * mV, tau_theta=1e12 * second, fair_share=fair)
    pre, post, syn, inhib = built[:4]
    share = built[4] if len(built) > 4 else None    # plain local so run() collects it
    spikes = SpikeMonitor(post)
    wt = StateMonitor(syn, "w", record=True, dt=1 * second)
    t0 = time.time()
    run(sum(DURS) * second)
    total = sum(DURS)
    st = np.array(spikes.t / second); si = np.array(spikes.i[:])
    edges = np.arange(int(total) + 1, dtype=float)
    rate_bins = np.stack([np.histogram(st[si == j], bins=edges)[0] for j in range(n_post)])
    full = dict(status="completed", seed=seed_val, n_post=n_post, cell=cell, target=prefs.codegen.target,
                wall_elapsed=time.time() - t0, world="unify_v1b", inhib_strength_mV=inh, gap_scale=gap,
                theta_plus_mV=0.05 if fair else 0.0, fair_share=fair, apre=0.005, p_share=0.9, dt_ms=0.2, brian2_seeded=True,
                phase_durations_s=DURS, phase_corr_blocks=SCHEDULE, swap_times_s=[1000.0, 2000.0, 3000.0, 4000.0], total_s=total,
                context_sets=[list(range(0, 10)), list(range(10, 20)), list(range(20, 30))],
                syn_i=np.array(syn.i[:]).tolist(), syn_j=np.array(syn.j[:]).tolist(),
                weight_trace=wt.w[:].round(4).tolist(), spike_rate_bins=rate_bins.tolist(),
                theta_final_mV=np.asarray(post.theta / mV).round(4).tolist())
    raw_path = f"unified_substrate/fact/fact_{cell}_n{n_post}_seed{seed_val}.json.gz"
    MC.write_raw(raw_path, full)
    out = MC.compact(full, raw_path)
    out["cell"] = cell
    return out


@app.function(image=image, cpu=1.0, memory=4096, timeout=4 * 3600, max_containers=64,
              retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_mnist(arm: str, seed: int) -> dict:
    import numpy as np
    import mnist_sim as S
    subset = dict(np.load("/root/pilot_subset.npz"))
    fair = dict(theta_plus_mV=0.05, tau_theta_s=1e12, fair_share=True)
    kw = dict(norm_fair=dict(normalize=True, adaptive=fair, wta=False), wta_norm=dict(normalize=True, adaptive=None, wta=True))[arm]
    r = S.run_pilot(seed, subset, 30.0, n_post=40, **kw)
    r["arm"] = arm
    return r


@app.local_entrypoint()
def main():
    clicks = [(c, n, SEED_BASE[n] + k) for c in CELLS for n in NS for k in range(8)
              if not (HERE / f"fact_{c}_n{n}_seed{SEED_BASE[n] + k}.json.gz").exists()]
    mn = [(a, s) for a in MNIST_ARMS for s in MNIST_SEEDS if not (MN / f"mnist_fact_{a}_seed{s}.json.gz").exists()]
    print(f"{len(clicks)} click + {len(mn)} MNIST runs to do", flush=True)
    calls = [("mnist", run_mnist.spawn(a, s)) for a, s in mn] + [("click", run_click.spawn(*x)) for x in clicks]
    ok = 0
    for kind, c in calls:                      # all jobs already running; blocking get in order only delays saving
        try:
            r = c.get()
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {kind} {e!r}", flush=True); continue
        if kind == "mnist":
            p = MN / f"mnist_fact_{r['arm']}_seed{r['seed']}.json.gz"
        else:
            p = HERE / f"fact_{r['cell']}_n{r['n_post']}_seed{r['seed']}.json.gz"
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(r, f)
        ok += 1
        print(f"saved {p.name} (wall {r.get('wall_elapsed', 0):.0f}s)", flush=True)
    print(f"== {ok}/{len(calls)} completed", flush=True)

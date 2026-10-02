"""Leaky fair-share: proportional instead of integral homeostasis. Can ONE substrate pass both worlds?

The dose test (modal_dose.py, 2026-10-02) showed slowing the fair-share threshold only DELAYS habituation. At 10x slower
the click-world readout peaks ~300 s into a phase and fades by 900 s; at 100x slower N=7 passes but N=40 still fades
(0.84 -> 0.64 within a phase). Reason: fair-share with no decay is an INTEGRAL controller. Theta accumulates (own spikes
minus the mean) forever, so any sustained rate difference is eventually driven to zero, and the readout is exactly
that difference.
Fix to test: put the decay back (Diehl & Cook's theta had one; v4 removed it with the ratchet). The leaky fair-share
theta_j relaxes to tp * tau * (r_j - r_mean): PROPORTIONAL control. Differences shrink by a loop gain but keep their
rank and direction, and the readout is a direction (cosine). A short tau makes it spike-frequency adaptation with fair
sharing: whoever has been winning lately sits out a bit, which could still spread digits across neurons. The sum of theta
stays zero (uniform decay of a zero-sum quantity).
Doses (a DIAGNOSTIC, not values to adopt): tau_theta in {10, 100, 1000} s at Diehl & Cook's theta_plus 0.05 mV. Estimated
loop gain ~ tp * tau * (rate slope ~1 Hz/mV): ~0.5 / 5 / 50.

Cells (all winner-take-all 20 mV, gate off, plus the leaky fair-share threshold):
  click world (modal_unify.py's world, N in {7, 40}, seeds 53000-53007 / 53100-53107)
  MNIST (v4's exact conditions: gain control, N=40, budget 30, 1 pass, seeds 70100-70107)
Files: leak_tau<x>_n<N>_seed<S>.json.gz here (compact; raw on the Volume under unified_substrate/leak/);
../mnist_pilot/mnist_leak_tau<x>_seed<S>.json.gz. Scoring: analyze_leak.py (analyze_factorial.py metrics plus the
habituation trace) and analyze_mnist_pilot.py <prefix> v3.

PREDICTIONS ON RECORD (written before launch; the MNIST half of the dose test was still running, unseen):
  L-P1 click, tau 10 s: passes at both N (settled readout >= 0.9, B named at two-back W=50 >= 7/8); the readout 900 s into a
       phase is >= 0.9 (no habituation).
  L-P2 click, tau 1000 s: fails at >= 1 N (gain ~50, near-integral).
  L-P3 MNIST, tau 10 s: diversity survives (every class >= 3 neurons in >= 6/8) and vote accuracy >= 65%. This is the risky
       one: a 10 s memory spans only ~20 images.
  L-P4 MNIST, tau 1000 s: ~v4 (accuracy >= 75%).
  tau 100 s: reported, not predicted.
  => L-P1 + L-P3 = one substrate passing both worlds, with no ratchet, no erasure, and only a leak added back.

FOLLOW-UP (written after the tau 10/100/1000 results, before running 30/50 s):
  Results so far: tau 10 s passes the click world cleanly but MNIST loses diversity (45%, one digit takes ~35 neurons);
  tau 100 s passes BOTH (click 0.98/0.97 median, 8/8; MNIST 74.8%, every digit), but its click readout FLICKERS (14-21% of
  settled 10 s windows < 0.5, ~8-12 dips per phase: adaptation-driven alternation); tau 1000 s fails the click world.
  Mapping the window with tau 30 and 50 s (run with --taus 30,50):
  L-P5 tau 30 s: click passes with < 5% flicker windows at both N; MNIST every class >= 3 in >= 6/8 seeds, accuracy >= 65%.
  L-P6 tau 50 s: both worlds pass; flicker between tau 30's and tau 100's.
  Flicker = share of settled 10 s windows whose rectified readout has cosine < 0.5 to the current prototype
  (analyze_leak_flicker.py).

Cost estimate: 48 click runs (~20 min) + 24 MNIST runs (~5 min), about $1.5.
Run from the repo root:  python -m modal run notebooks/brian2/unified_substrate/modal_leak.py
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
app = modal.App("developing-mind-leak")
NS = [7, 40]
SEED_BASE = {7: 53000, 40: 53100}
TAUS = [10.0, 100.0, 1000.0]
TP = 0.05
MNIST_SEEDS = list(range(70100, 70108))
SCHEDULE, DURS = [0, 1, 2, 0, 1], [1000.0] * 5


def tag(tp):
    return f"{tp:g}".replace(".", "p")


@app.function(image=image, cpu=1.0, memory=4096, timeout=4 * 3600, max_containers=64,
              volumes={RAW_MOUNT: RAW_VOLUME}, retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_click(tau: float, n_post: int, seed_val: int) -> dict:
    tp = TP
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
                                                    target_total=10.0, theta_plus=tp * mV, tau_theta=tau * second,
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
    full = dict(status="completed", seed=seed_val, n_post=n_post, cell=f"leak_tau{tag(tau)}", tau_theta_s=tau, target=prefs.codegen.target,
                wall_elapsed=time.time() - t0, world="unify_v1b", inhib_strength_mV=inh, gap_scale=1e9, theta_plus_mV=tp,
                fair_share=True, apre=0.005, p_share=0.9, dt_ms=0.2, brian2_seeded=True,
                phase_durations_s=DURS, phase_corr_blocks=SCHEDULE, swap_times_s=[1000.0, 2000.0, 3000.0, 4000.0], total_s=total,
                context_sets=[list(range(0, 10)), list(range(10, 20)), list(range(20, 30))],
                syn_i=np.array(syn.i[:]).tolist(), syn_j=np.array(syn.j[:]).tolist(),
                weight_trace=wt.w[:].round(4).tolist(), spike_rate_bins=rate_bins.tolist(),
                theta_trace_10s_mV=np.asarray(th.theta / mV).round(4).tolist(),
                theta_final_mV=np.asarray(post.theta / mV).round(4).tolist())
    raw_path = f"unified_substrate/leak/leak_tau{tag(tau)}_n{n_post}_seed{seed_val}.json.gz"
    MC.write_raw(raw_path, full)
    out = MC.compact(full, raw_path)
    out.update(tau=tau, theta_trace_10s_mV=full["theta_trace_10s_mV"])
    return out


@app.function(image=image, cpu=1.0, memory=4096, timeout=6 * 3600, max_containers=64,
              retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_mnist(tau: float, seed: int) -> dict:
    import numpy as np
    import mnist_sim as S
    subset = dict(np.load("/root/pilot_subset.npz"))
    fair = dict(theta_plus_mV=TP, tau_theta_s=tau, fair_share=True)
    r = S.run_pilot(seed, subset, 30.0, n_post=40, normalize=True, adaptive=fair, wta=True)
    r["tau"] = tau
    return r


@app.local_entrypoint()
def main(taus: str = ""):
    global TAUS
    if taus:
        TAUS = [float(x) for x in taus.split(",")]
    clicks = [(tau, n, SEED_BASE[n] + k) for tau in TAUS for n in NS for k in range(8)
              if not (HERE / f"leak_tau{tag(tau)}_n{n}_seed{SEED_BASE[n] + k}.json.gz").exists()]
    mn = [(tau, s) for tau in TAUS for s in MNIST_SEEDS if not (MN / f"mnist_leak_tau{tag(tau)}_seed{s}.json.gz").exists()]
    print(f"{len(clicks)} click + {len(mn)} MNIST runs to do", flush=True)
    calls = [("mnist", run_mnist.spawn(*x)) for x in mn] + [("click", run_click.spawn(*x)) for x in clicks]
    ok = 0
    for kind, c in calls:                      # all jobs already running; blocking get in order only delays saving
        try:
            r = c.get()
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {kind} {e!r}", flush=True); continue
        if kind == "mnist":
            p = MN / f"mnist_leak_tau{tag(r['tau'])}_seed{r['seed']}.json.gz"
        else:
            p = HERE / f"leak_tau{tag(r['tau'])}_n{r['n_post']}_seed{r['seed']}.json.gz"
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(r, f)
        ok += 1
        print(f"saved {p.name} (wall {r.get('wall_elapsed', 0):.0f}s)", flush=True)
    print(f"== {ok}/{len(calls)} completed", flush=True)

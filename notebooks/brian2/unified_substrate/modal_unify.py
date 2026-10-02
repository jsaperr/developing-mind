"""Unified-substrate check: the integration-critical properties (one-back retention, the change signal at its hand-set
defaults, the readout, memory recognition) re-tested on the substrate MNIST needed. Every earlier synthetic result used the old
substrate (gentle normalized inhibition, ambiguity-gated, no threshold adaptation).

New substrate: winner-take-all inhibition (20 mV = v_thresh - v_rest per competitor spike, gate off) + fair-share threshold
(+0.05 mV own spike, -0.05/N all, sum conserved). Gain control is irrelevant here (inputs are rate-matched by design).
World: v1b A B C A B (5 x 1000 s), disjoint 30-input rig, target_total 10, w_init 1/3, dt 0.2 ms, Brian2 seeded. N in {7, 40},
8 seeds each, on Modal. Data policy: full 1 s results go to the Volume developing-mind-raw (unified_substrate/...); compact
files (modal_common.compact) come back here as unify_n<N>_seed<S>.json.gz.

PREDICTIONS ON RECORD (written before launch; scored by analyze_unify.py):
  U-P1 no silencing: over the last 500 s, the mean rate is >= 1 Hz with >= 80% of neurons active (rate > 0.1 Hz), both N.
  U-P2 one back survives the new substrate: B keeps >= 60% of its holders at C's arrival (+150 s), and the incoming context
       holds <= 15% of the population just before each two-back return, both N. Recorded risk: fair-share LOWERS the
       threshold of quiet neurons (the retainers), which could pull them into the current context and erode retention.
  U-P3 the change signal at its hand-set defaults (tripwire #3): fires within 300 s after >= 90% of swaps, on for < 5% of
       settled seconds, both N.
  U-P4 readout and memory: settled rectified fingerprint >= 0.9 to its prototype, and the default memory names B in more than
       half of its settled two-back checks at W=50 in >= 7/8 seeds, both N.

Run from the repo root:  python -m modal run notebooks/brian2/unified_substrate/modal_unify.py
"""
import gzip
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from modal_common import IMAGE, RAW_MOUNT, RAW_VOLUME  # noqa: E402

image = (IMAGE.add_local_file(str(HERE.parent / "modal_common.py"), "/root/modal_common.py")
         .add_local_file(str(HERE.parent / "mnist_pilot" / "mnist_sim.py"), "/root/mnist_sim.py"))
app = modal.App("developing-mind-unify")
NS = [7, 40]
SEED_BASE = {7: 53000, 40: 53100}
SCHEDULE, DURS = [0, 1, 2, 0, 1], [1000.0] * 5


@app.function(image=image, cpu=1.0, memory=4096, timeout=4 * 3600, max_containers=40,
              volumes={RAW_MOUNT: RAW_VOLUME}, retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_n(n_post: int, seed_val: int) -> dict:
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
    wta = float((N.v_thresh - N.v_rest) / mV)
    pre, post, syn, inhib, share = S.build_adaptive(n_post, idx, t, 0.005, wta * mV, 1e9, n_pre=30, w_init=1 / 3,
                                                    target_total=10.0, theta_plus=0.05 * mV, tau_theta=1e12 * second,
                                                    fair_share=True)
    spikes = SpikeMonitor(post)
    wt = StateMonitor(syn, "w", record=True, dt=1 * second)
    t0 = time.time()
    run(sum(DURS) * second)
    total = sum(DURS)
    st = np.array(spikes.t / second); si = np.array(spikes.i[:])
    edges = np.arange(int(total) + 1, dtype=float)
    rate_bins = np.stack([np.histogram(st[si == j], bins=edges)[0] for j in range(n_post)])
    full = dict(status="completed", seed=seed_val, n_post=n_post, target=prefs.codegen.target, wall_elapsed=time.time() - t0,
                world="unify_v1b", substrate="wta20_fairshare", inhib_strength_mV=wta, gap_scale=1e9, theta_plus_mV=0.05,
                fair_share=True, apre=0.005, p_share=0.9, dt_ms=0.2, brian2_seeded=True, phase_durations_s=DURS,
                phase_corr_blocks=SCHEDULE, swap_times_s=[1000.0, 2000.0, 3000.0, 4000.0], total_s=total,
                context_sets=[list(range(0, 10)), list(range(10, 20)), list(range(20, 30))],
                syn_i=np.array(syn.i[:]).tolist(), syn_j=np.array(syn.j[:]).tolist(),
                weight_trace=wt.w[:].round(4).tolist(), spike_rate_bins=rate_bins.tolist(),
                theta_final_mV=np.asarray(post.theta / mV).round(4).tolist())
    raw_path = f"unified_substrate/unify_n{n_post}_seed{seed_val}.json.gz"
    MC.write_raw(raw_path, full)
    return MC.compact(full, raw_path)


@app.local_entrypoint()
def main():
    todo = [(n, SEED_BASE[n] + k) for n in NS for k in range(8) if not (HERE / f"unify_n{n}_seed{SEED_BASE[n] + k}.json.gz").exists()]
    print(f"{len(todo)} runs to do", flush=True)
    ok = 0
    for r in run_n.starmap(todo, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print(f"FAILED {r!r}", flush=True); continue
        with gzip.open(HERE / f"unify_n{r['n_post']}_seed{r['seed']}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(r, f)
        ok += 1
        print(f"N={r['n_post']} seed {r['seed']} saved (wall {r['wall_elapsed']:.0f}s)", flush=True)
    print(f"== {ok}/{len(todo)} completed", flush=True)

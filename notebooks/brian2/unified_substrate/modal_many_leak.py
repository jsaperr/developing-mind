"""Generalization check for the reconciled substrate: step 3's six-context world on the leaky fair-share substrate.

The leak test (modal_leak.py, 2026-10-02) found ONE substrate that passes both worlds: winner-take-all (20 mV, gate off)
plus a leaky fair-share threshold (+0.05 mV own spike, -0.05/N all, decay tau 50-100 s). It was tested on the 3-context
v1b world. Step 3's world is harder: 6 contexts on 60 inputs, 12 x 1000 s, A B C D E F | E C F B F A, with returns from
depths 1-5. On the old substrate it gave graded retention (~30% / 20% / < 7% at 1 / 2 / 3+ back), a working change signal
and memory that remembered every return (step 3, arc 05).
World and rig exactly step 3's (modal_many.py): target_total 10, w_init 1/6, dt 0.2 ms, Brian2 seeded, N in {7, 40}, the
same seeds 52000-52007 / 52100-52107 (paired). tau in {50, 100} s. Detached: each job writes its compact file to the Volume
(unified_substrate/many_leak/), raw beside it. Collect with --collect. Scored by analyze_many_leak.py, which runs step 3's
analyze_many.py metrics unchanged on the compact files, plus the flicker measure.

PREDICTIONS ON RECORD (written before launch):
  ML-P1 memory remembers every return: each return is remembered by an old memory in >= 7/8 seeds at W=50, both N, both tau
        (step 3: 8/8 everywhere).
  ML-P2 the readout holds: settled median >= 0.9 at both N and tau; flicker at tau 100 >= flicker at tau 50.
  ML-P3 older contexts are still released: the incoming share before the 3-, 4- and 5-back returns is <= 15% on average (step
        3: 0-7%). Winner-take-all kept more two back in v1b (22-56%), so the 1- and 2-back levels are reported, not predicted.
  Change-signal coverage is reported, not predicted (winner-take-all lowered it to 31-81% in v1b, by re-wiring fewer neurons).

Cost estimate: 32 runs x ~60-75 min (1 core, 4 GiB), about $2.5. Detached, results land in ~75 min.
Launch (repo root):  python -m modal run --detach notebooks/brian2/unified_substrate/modal_many_leak.py
Collect:             python -m modal run notebooks/brian2/unified_substrate/modal_many_leak.py --collect
"""
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MN = HERE.parent / "mnist_pilot"
sys.path.insert(0, str(HERE.parent))
from modal_common import IMAGE, RAW_MOUNT, RAW_VOLUME  # noqa: E402

image = (IMAGE.add_local_file(str(HERE.parent / "modal_common.py"), "/root/modal_common.py")
         .add_local_file(str(MN / "mnist_sim.py"), "/root/mnist_sim.py"))
app = modal.App("developing-mind-many-leak")
NS = [7, 40]
SEED_BASE = {7: 52000, 40: 52100}
TAUS = [50.0, 100.0]
SCHEDULE = [0, 1, 2, 3, 4, 5, 4, 2, 5, 1, 5, 0]
DURS = [1000.0] * 12
OUTDIR = HERE / "many_leak"


def rel(tau, n, s):
    return f"unified_substrate/many_leak/tau{int(tau)}/many6_n{n}_seed{s}.json.gz"


@app.function(image=image, cpu=1.0, memory=4096, timeout=6 * 3600, max_containers=40, volumes={RAW_MOUNT: RAW_VOLUME},
              retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_n(tau: float, n_post: int, seed_val: int) -> str:
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
    idx, t = build_multiblock_phase_input(20.0, 0.9, DURS, SCHEDULE, rng, n_blocks=6, block_size=10)
    inh = float((N.v_thresh - N.v_rest) / mV)
    pre, post, syn, inhib, share = S.build_adaptive(n_post, idx, t, 0.005, inh * mV, 1e9, n_pre=60, w_init=10.0 / 60,
                                                    target_total=10.0, theta_plus=0.05 * mV, tau_theta=tau * second,
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
                world="many6", substrate="wta20_leaky_fairshare", tau_theta_s=tau, theta_plus_mV=0.05, inhib_strength_mV=inh,
                gap_scale=1e9, apre=0.005, p_share=0.9, dt_ms=0.2, brian2_seeded=True, phase_durations_s=DURS,
                phase_corr_blocks=SCHEDULE, swap_times_s=[1000.0 * k for k in range(1, 12)], total_s=total,
                context_sets=[list(range(b * 10, b * 10 + 10)) for b in range(6)],
                syn_i=np.array(syn.i[:]).tolist(), syn_j=np.array(syn.j[:]).tolist(),
                weight_trace=wt.w[:].round(4).tolist(), spike_rate_bins=rate_bins.tolist(),
                theta_final_mV=np.asarray(post.theta / mV).round(4).tolist())
    raw = rel(tau, n_post, seed_val).replace(".json.gz", "_raw.json.gz")
    MC.write_raw(raw, full)
    out = MC.compact(full, raw)
    p = Path(RAW_MOUNT) / rel(tau, n_post, seed_val); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(MC.gz_bytes(out))
    RAW_VOLUME.commit()
    return rel(tau, n_post, seed_val)


@app.local_entrypoint()
def main(collect: bool = False):
    import os
    import subprocess
    jobs = [(tau, n, SEED_BASE[n] + k) for tau in TAUS for n in NS for k in range(8)]
    if collect:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        got = 0
        for tau, n, s in jobs:
            dst = OUTDIR / f"tau{int(tau)}" / f"many6_n{n}_seed{s}.json.gz"
            if dst.exists():
                got += 1; continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run([sys.executable, "-m", "modal", "volume", "get", "developing-mind-raw", rel(tau, n, s), str(dst)],
                               capture_output=True, text=True, env=env)
            got += r.returncode == 0
        print(f"== {got}/{len(jobs)} collected", flush=True)
        return
    calls = [run_n.spawn(*j) for j in jobs]
    print("spawned", len(calls), "detached runs", flush=True)

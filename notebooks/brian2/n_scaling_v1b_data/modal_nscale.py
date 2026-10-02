"""Generalization plan, step 2: network size. The v1b world A->B->C->A->B (5 x 1000 s), the disjoint 30-input
3-block rig, 13 mV/1.5 reference inhibition normalized with scale_inhib_for_n, N in {5, 7, 10, 15, 40}, 8 seeds
each, Brian2 seeded. Run on Modal (cleared 2026-10-01: seeded Modal and laptop runs match seed for seed).
Mirrors run_v1_seed.py's setup (dt 0.2 ms, Apre 0.005, p_share 0.9, w_init 1/3, target_total 10, gap 1.5,
1 s weight trace, 1 s spike-count bins) with N as the only variable. N=7 is included so every size runs under the
identical seeded protocol (the older v1b N=7 runs were unseeded).

Top size N=40 (Jasper's go, 2026-10-01): it equals the MNIST pilot size at Diehl & Cook's ~10 neurons per class
x 4 classes, which is this step's bridge role. 30 was a loose earlier guess.

PREDICTIONS ON RECORD (written before launch; scored by analyze_nscale.py). The pass criteria are the plan's.
  NS-P1 one back holds at every N: at C's arrival B keeps >= 60% of its holders (+150 s), and just before each
        two-back return (swaps 3 and 4) the incoming context holds <= 15% of the population (seed mean).
  NS-P2 the retainer fraction is roughly size-independent: the share of the population still holding A at the end
        of B's phase is within 0.35-0.65 at every N (N=7 has been ~0.5).
  NS-P3 the change signal generalizes WITHOUT re-tuning (tripwire): interface.changing_per_second at its fixed
        defaults (L 60 s, trail 900 s, k 3) fires within 300 s after >= 90% of swaps and is on for < 5% of settled
        seconds (+300 s to the next swap), at every N.
  NS-P4 the readout generalizes: the settled rectified fingerprint's cosine to its true prototype is >= 0.9 at
        every N, and between-context cosines stay below the 0.8 radius.
  NS-P5 end to end: the default DormantGatedMemory (radius 0.8, W=50) names B in more than half of its settled
        two-back return checks in >= 7/8 seeds at every N.
  Leaning (not a scored prediction): more neurons means more retainers per context but still one back.

Run from the repo root:  python -m modal run notebooks/brian2/n_scaling_v1b_data/modal_nscale.py
Results: notebooks/brian2/n_scaling_v1b_data/nscale_v1b_n<N>_seed<S>.json.gz (format readable by
notebooks/integration/set_worlds/run_rectified_memory.load). Resume-safe: completed files are skipped.
"""
import gzip
import io
import json
from pathlib import Path

import modal

app = modal.App("developing-mind-nscale")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    # versions match the local developing-mind conda env (see CLAUDE.md, Modal)
    .pip_install("brian2==2.9.0", "cython==3.2.8", "numpy==2.0.1", "setuptools==82.0.1")
    .add_local_python_source("src")
)

NS = [5, 7, 10, 15, 40]
SEED_BASE = {5: 51000, 7: 51100, 10: 51200, 15: 51300, 40: 51400}
N_SEEDS = 8
OUT = Path(__file__).resolve().parent
SCHEDULE = [0, 1, 2, 0, 1]
DURS = [1000.0] * 5


def fname(n, s):
    return OUT / f"nscale_v1b_n{n}_seed{s}.json.gz"


@app.function(image=image, cpu=1.0, memory=4096, timeout=6 * 3600, max_containers=60,
              retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_n(n_post: int, seed_val: int) -> bytes:
    import time

    import numpy as np
    from brian2 import SpikeMonitor, StateMonitor, defaultclock, mV, ms, prefs, run, second, start_scope, seed as b2_seed
    from src.brian2_stdp.network import build_competitive_population_network, scale_inhib_for_n
    from src.brian2_stdp.spikes import build_multiblock_phase_input

    start_scope()
    defaultclock.dt = 0.2 * ms
    b2_seed(seed_val)
    rng = np.random.default_rng(seed_val)
    idx, t = build_multiblock_phase_input(20.0, 0.9, DURS, SCHEDULE, rng, n_blocks=3, block_size=10)
    per_conn = scale_inhib_for_n(n_post, reference_inhib_mV=13.0, reference_n_post=3)
    pre, post, syn, inhib = build_competitive_population_network(n_post, idx, t, 0.005, per_conn * mV, 1.5, n_pre=30,
                                                                 w_init=10.0 / 30, target_total=10.0)
    spikes = SpikeMonitor(post)
    wt = StateMonitor(syn, "w", record=True, dt=1 * second)
    t0 = time.time()
    run(sum(DURS) * second)
    total = sum(DURS)
    st = np.array(spikes.t / second); si = np.array(spikes.i[:])
    edges = np.arange(int(total) + 1, dtype=float)
    rate_bins = np.stack([np.histogram(st[si == j], bins=edges)[0] for j in range(n_post)])
    result = dict(status="completed", seed=seed_val, n_post=n_post, target=prefs.codegen.target,
                  wall_elapsed=time.time() - t0, world="nscale_v1b", inhib_strength_mV=float(per_conn),
                  reference_inhib_mV=13.0, gap_scale=1.5, apre=0.005, p_share=0.9, dt_ms=0.2, brian2_seeded=True,
                  phase_durations_s=DURS, phase_corr_blocks=SCHEDULE, swap_times_s=[1000.0, 2000.0, 3000.0, 4000.0],
                  total_s=total, context_sets=[list(range(0, 10)), list(range(10, 20)), list(range(20, 30))],
                  syn_i=np.array(syn.i[:]).tolist(), syn_j=np.array(syn.j[:]).tolist(),
                  weight_trace=wt.w[:].round(4).tolist(), spike_rate_bins=rate_bins.tolist())
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(json.dumps(result).encode("utf-8"))
    return buf.getvalue()


def done(n, s):
    f = fname(n, s)
    if not f.exists():
        return False
    try:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            return json.load(fh).get("status") == "completed"
    except Exception:
        return False


@app.local_entrypoint()
def main():
    jobs = [(n, SEED_BASE[n] + k) for n in NS for k in range(N_SEEDS) if not done(n, SEED_BASE[n] + k)]
    print(f"{len(jobs)} jobs to run", flush=True)
    n_ok = 0
    for (n, s), r in zip(jobs, run_n.starmap(jobs, return_exceptions=True, order_outputs=True)):
        if isinstance(r, Exception):
            print(f"N={n} seed {s}: FAILED {r!r}", flush=True)
            continue
        fname(n, s).write_bytes(r); n_ok += 1
        print(f"N={n} seed {s}: saved ({len(r) / 1e6:.1f} MB gz)", flush=True)
    print(f"== {n_ok}/{len(jobs)} completed", flush=True)

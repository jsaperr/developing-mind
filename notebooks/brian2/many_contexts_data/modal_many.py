"""Generalization plan, step 3: more contexts than a small network can give each its own neurons. Six disjoint
contexts on a 60-input, 6-block rig (calibrated first: calibrate_60.py, PASS at N=7 and N=40 with target_total
10, w_init 1/6, gmax unchanged). 13 mV/1.5 normalized with scale_inhib_for_n, N in {7, 40}, 8 seeds each, Brian2
seeded, run on Modal (cleared by the arc-06 fidelity checks). Otherwise mirrors run_v1_seed.py (dt 0.2 ms, Apre
0.005, p_share 0.9, 1 s weight trace, 1 s spike bins).

World (12 x 1000 s): A B C D E F | E C F B F A. The second half returns contexts from different depths of the
recency list: E 1 back, C 3 back, F 2 back, B 4 back, F 1 back, A 5 back. It's the synthetic control for MNIST
(step 4): if MNIST breaks and this doesn't, the cause is real-data features, not the number of contexts.

PREDICTIONS ON RECORD (written before launch; scored by analyze_many.py):
  MC-P1 one back holds with six contexts: at every change, the just-departed context keeps >= 60% of its holders
        (+150 s), at both N.
  MC-P2 retention is still only one deep: just before each return from 2+ back, the incoming context holds <= 15% of
        the population; before each 1-back return it holds >= 30% (seed means), at both N.
  MC-P3 the change signal works at its hand-set defaults (tripwire): fires within 300 s after >= 90% of changes, on
        for < 5% of settled seconds, at both N.
  MC-P4 the readout separates six contexts: settled cosine to the true prototype >= 0.9, between-context cosine
        < 0.8, at both N.
  MC-P5 memory remembers every return: the default DormantGatedMemory (radius 0.8) names the returning context in
        more than half of its settled checks, AND the entry that first names it is old (live from before or
        reawakened from dormant, not newly learned), for every return in >= 7/8 seeds, at W=50 and W=10, at both N.

Run from the repo root:  python -m modal run notebooks/brian2/many_contexts_data/modal_many.py
Results: notebooks/brian2/many_contexts_data/many6_n<N>_seed<S>.json.gz. Resume-safe.
"""
import gzip
import io
import json
from pathlib import Path

import modal

app = modal.App("developing-mind-many-contexts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    # versions match the local developing-mind conda env (see CLAUDE.md, Modal)
    .pip_install("brian2==2.9.0", "cython==3.2.8", "numpy==2.0.1", "setuptools==82.0.1")
    .add_local_python_source("src")
)

NS = [7, 40]
SEED_BASE = {7: 52000, 40: 52100}
N_SEEDS = 8
OUT = Path(__file__).resolve().parent
SCHEDULE = [0, 1, 2, 3, 4, 5, 4, 2, 5, 1, 5, 0]
DURS = [1000.0] * 12


def fname(n, s):
    return OUT / f"many6_n{n}_seed{s}.json.gz"


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
    idx, t = build_multiblock_phase_input(20.0, 0.9, DURS, SCHEDULE, rng, n_blocks=6, block_size=10)
    per_conn = scale_inhib_for_n(n_post, reference_inhib_mV=13.0, reference_n_post=3)
    pre, post, syn, inhib = build_competitive_population_network(n_post, idx, t, 0.005, per_conn * mV, 1.5, n_pre=60,
                                                                 w_init=10.0 / 60, target_total=10.0)
    spikes = SpikeMonitor(post)
    wt = StateMonitor(syn, "w", record=True, dt=1 * second)
    t0 = time.time()
    run(sum(DURS) * second)
    total = sum(DURS)
    st = np.array(spikes.t / second); si = np.array(spikes.i[:])
    edges = np.arange(int(total) + 1, dtype=float)
    rate_bins = np.stack([np.histogram(st[si == j], bins=edges)[0] for j in range(n_post)])
    result = dict(status="completed", seed=seed_val, n_post=n_post, target=prefs.codegen.target,
                  wall_elapsed=time.time() - t0, world="many6", inhib_strength_mV=float(per_conn),
                  reference_inhib_mV=13.0, gap_scale=1.5, apre=0.005, p_share=0.9, dt_ms=0.2, brian2_seeded=True,
                  phase_durations_s=DURS, phase_corr_blocks=SCHEDULE, swap_times_s=[1000.0 * k for k in range(1, 12)],
                  total_s=total, context_sets=[list(range(b * 10, b * 10 + 10)) for b in range(6)],
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

"""Modal fidelity check, part 2: the competitive network (every remaining planned experiment uses it).
The single-neuron check (arc 06, 2026-10-01) showed Modal matches the laptop to rounding error, but that rig
is noise-free and stable. The competitive network has per-step membrane noise (sigma_v) and can amplify tiny
differences (the fork showed small differences compound in wiring; strong_tight_gate is bistable).

Design: MATCHED seeds on both platforms, Brian2 seeded (brian2.seed) so both draw the same noise. The same
function bodies run on Modal (`.map`) and on the laptop (`.local()` via local_run.py), so the only difference
is the platform. Arms:
  probe  strong_tight_gate (10 mV, gap 1.0), N=3, 5 s, seeds 49100-49102: is Brian2's seeded noise stream the
         same on Windows and Linux? Full-precision weights every 1 s.
  stg    strong_tight_gate, N=3, 600 s, seeds 49200-49231 (32): the bistable canary. Mirrors
         competitive_population_data/run_competitive_seed.py (dt left at Brian2's default, as there).
  op     the operating point every integration result uses: 13 mV/1.5 (scale_inhib_for_n), N=7, the v1b
         schedule A B C A B with 300 s phases (short_phase_data's world), disjoint 30-input rig, dt 0.2 ms,
         seeds 49300-49315 (16). Mirrors run_v1_seed.py's setup.

PREDICTIONS ON RECORD (written before any run):
  CC-P0 same noise stream: at t = 1 s the laptop-vs-Modal max |dw| is <= 1e-12 for all 3 probe seeds. If not,
        "matched seeds" aren't matched, and only the statistical comparisons below mean anything.
  CC-P1 the competitive network amplifies rounding: for stg at 600 s, the median per-seed max |dw| between
        platforms is >= 0.05 (unlike the single-neuron rig, where it stayed ~1e-15).
  CC-P2 but the statistics match (stg): the 'differentiate' counts differ by <= 4 of 32, and the late-window
        std distributions are not distinguishable (Mann-Whitney p > 0.05).
  CC-P3 the operating point is robust (op): per-seed preferred-context assignments at the end of each phase
        are identical on both platforms in >= 12/16 seeds, and the one-back statistics (B keeps >= 60% of its
        holders at C's arrival; two-back incoming holders <= 1) hold on both platforms.
Decision rule: if CC-P2 and CC-P3 hold, Modal is usable for new experiments and for STATISTICAL comparison
with earlier laptop results; never for per-seed exact comparisons across platforms.

Run (from the repo root):  python -m modal run notebooks/brian2/modal_competitive_check/modal_competitive_check.py
Laptop side:               python notebooks/brian2/modal_competitive_check/local_batch.py
Scoring:                   python notebooks/brian2/modal_competitive_check/score.py
"""
import gzip
import json
from pathlib import Path

import modal

app = modal.App("developing-mind-competitive-check")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    # versions match the local developing-mind conda env (see CLAUDE.md, Modal)
    .pip_install("brian2==2.9.0", "cython==3.2.8", "numpy==2.0.1", "setuptools==82.0.1")
    .add_local_python_source("src")
)

PROBE_SEEDS = [49100, 49101, 49102]
STG_SEEDS = list(range(49200, 49232))
OP_SEEDS = list(range(49300, 49316))
OUT = Path(__file__).resolve().parent / "out"


def _stg_body(seed_val: int, duration_s: float, trace_dt_s: float):
    """strong_tight_gate N=3 competitive run, as run_competitive_seed.py builds it, Brian2 seeded."""
    import time
    import numpy as np
    from brian2 import StateMonitor, mV, prefs, run, second, start_scope, seed as b2_seed
    from src.brian2_stdp.network import build_competitive_population_network
    from src.brian2_stdp.spikes import build_presynaptic_input

    start_scope()
    b2_seed(seed_val)
    rng = np.random.default_rng(seed_val)
    idx, t = build_presynaptic_input(20.0, 0.9, duration_s, rng)
    pre, post, syn, inhib = build_competitive_population_network(3, idx, t, 0.005, 10.0 * mV, 1.0)
    wt = StateMonitor(syn, "w", record=True, dt=trace_dt_s * second)
    t0 = time.time()
    run(duration_s * second)
    return dict(status="completed", seed=seed_val, target=prefs.codegen.target, wall_elapsed=time.time() - t0,
                syn_i=np.array(syn.i[:]).tolist(), syn_j=np.array(syn.j[:]).tolist(),
                weight_trace_t=(wt.t / second).tolist(), weight_trace=wt.w[:].tolist())


def _op_body(seed_val: int):
    """13 mV/1.5, N=7, v1b schedule with 300 s phases, as run_v1_seed.py builds it, Brian2 seeded."""
    import time
    import numpy as np
    from brian2 import StateMonitor, defaultclock, mV, ms, prefs, run, second, start_scope, seed as b2_seed
    from src.brian2_stdp.network import build_competitive_population_network, scale_inhib_for_n
    from src.brian2_stdp.spikes import build_multiblock_phase_input

    start_scope()
    defaultclock.dt = 0.2 * ms
    b2_seed(seed_val)
    durs, sched = [300.0] * 5, [0, 1, 2, 0, 1]
    rng = np.random.default_rng(seed_val)
    idx, t = build_multiblock_phase_input(20.0, 0.9, durs, sched, rng, n_blocks=3, block_size=10)
    per_conn = scale_inhib_for_n(7, reference_inhib_mV=13.0, reference_n_post=3)
    pre, post, syn, inhib = build_competitive_population_network(7, idx, t, 0.005, per_conn * mV, 1.5, n_pre=30,
                                                                 w_init=10.0 / 30, target_total=10.0)
    wt = StateMonitor(syn, "w", record=True, dt=1 * second)
    t0 = time.time()
    run(sum(durs) * second)
    return dict(status="completed", seed=seed_val, target=prefs.codegen.target, wall_elapsed=time.time() - t0,
                swap_times_s=[300.0, 600.0, 900.0, 1200.0], schedule=sched,
                syn_i=np.array(syn.i[:]).tolist(), syn_j=np.array(syn.j[:]).tolist(),
                weight_trace=np.round(wt.w[:], 6).tolist())


@app.function(image=image, cpu=1.0, memory=2048, timeout=900, max_containers=60)
def run_probe(seed_val: int):
    return _stg_body(seed_val, 5.0, 1.0)


@app.function(image=image, cpu=1.0, memory=2048, timeout=1800, max_containers=60)
def run_stg(seed_val: int):
    return _stg_body(seed_val, 600.0, 1.0)


@app.function(image=image, cpu=1.0, memory=2048, timeout=2400, max_containers=60)
def run_op(seed_val: int):
    return _op_body(seed_val)


def save(platform: str, arm: str, r: dict):
    OUT.mkdir(exist_ok=True)
    with gzip.open(OUT / f"{platform}_{arm}_seed{r['seed']}.json.gz", "wt") as f:
        json.dump(r, f)


@app.local_entrypoint()
def main():
    jobs = [("probe", run_probe, PROBE_SEEDS), ("stg", run_stg, STG_SEEDS), ("op", run_op, OP_SEEDS)]
    for arm, fn, seeds in jobs:
        n = 0
        for r in fn.map(seeds, return_exceptions=True):
            if isinstance(r, Exception):
                print(f"{arm}: FAILED {r!r}", flush=True)
                continue
            save("modal", arm, r); n += 1
            print(f"{arm} seed {r['seed']} [{r['target']}] wall {r['wall_elapsed']:.0f}s", flush=True)
        print(f"== {arm}: {n}/{len(seeds)} completed", flush=True)

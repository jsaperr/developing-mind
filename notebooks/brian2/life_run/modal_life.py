"""A life in eras: one long, structured run on the substrate where integration works (the original competitive substrate,
13 mV/1.5 normalized, ambiguity-gated inhibition, no threshold adaptation; NOT the MNIST substrate, which failed the
unified-substrate check). Its purpose is the questions only a long life can answer: does character track a life, do rare old
contexts come back after hours, does rest keep memories alive over a whole life, does the dormant store stay bounded, do
representations drift over hours, does residue pile up over dozens of one-off experiences.

Rig: 60 inputs in 6 disjoint blocks (calibrated, calibrate_60.py), N=40, target_total 10, w_init 1/6, dt 0.2 ms, Brian2 seeded.
Contexts: home A-F = the 6 blocks; G = half of A + half of C (50% overlap with each); novel one-offs = random 10-subsets that
never return; REST = every input independent at the same rate (no synchrony).
Life (fixed for all seeds, rng 2027; the seeds vary the noise and the input realizations): 200 phases x 500 s = 100,000 s
(~28 h simulated). Every 10th phase is rest; otherwise 15% novel one-offs, and the rest drawn by era:
  era 1 (phases 1-60)    A .45  B .45  C .10
  era 2 (phases 61-140)  C .38  D .38  G .12  A .06  B .06
  era 3 (phases 141-200) E .38  F .38  A .06  B .06  C .06  D .06
(no immediate repeats). Phase length is not a variable for this substrate (generalization step 1), so 500 s phases buy more events.

Recording (data policy). EVERYTHING is kept for later analysis, raw on the Volume (free up to 1 TiB):
- A network_operation every 1 s keeps a 60 s ring buffer and records D(t) = sum |w(t) - w(t-60)|, the change signal's
  displacement. Change flags are then computed with interface.changing_per_second's exact rule and defaults (flags_from_D).
- 10 s block means of all weights (exact for W = 10 / 50 readout windows and 50 s holder windows, which all align to 10 s).
- RAW on the Volume developing-mind-raw (life_run/raw/life_seed<seed>.npz): the FULL 1 s weight trace (float32, n_sec x
  n_syn, ~1 GB uncompressed), every spike (neuron index + time), 10 s weight blocks, 10 s rates, D. Fetch it with
  `python -m modal volume get developing-mind-raw <raw_path> <local_path>` (large).
- COMPACT (life_run/compact/<seed>.json.gz on the Volume, pulled into this folder):
  - rectified and contrast fingerprints at W = 10 and 50;
  - 1 s change flags and D;
  - per-neuron nearest-context assignment every 50 s;
  - 10 s rates; the schedule.
Each seed is spawned DETACHED (tested 2026-10-01: spawned jobs finish and write to the Volume after the local driver exits).
Collect with collect_life.py.

PREDICTIONS ON RECORD (written before launch; scored by analyze_life.py, written before results are read):
  LL-P1 character tracks the life, recency-weighted at character's own timescale. At the end (default DormantGatedMemory,
        W=10, character = max w_char over live and dormant entries per home context), era 3's contexts (E, F) outrank
        era 2's (C, D), which outrank era 1's (A, B), in >= 6/8 seeds. Early experience fades: with ~7000 memory steps since
        era 1 against a ~2000-step decay, A/B character should sit near baseline unless refreshed by their rare visits.
  LL-P2 rare old contexts come back: A's and B's rare returns in eras 2-3 are named by an OLD memory (live, or reawakened
        from dormant) in >= 80% of returns, pooled over seeds (W=10).
  LL-P3 rest over a lifetime: during rest phases, memory names a home context in >= 30% of checks (ghosts), and the context
        named most during a rest is most often the one visited just before it (>= 50% of rests).
  LL-P4 the dormant store stays bounded: the count of dormant entries at the end is <= 1.3x the count at the midpoint
        (pruning keeps pace with ~25 one-off experiences).
  LL-P5 no drift over the life: the settled rectified fingerprint of A at its last visit vs its first has cosine >= 0.9
        (seed mean); and residue doesn't pile up: the share of neurons assigned to any PAST novel context, at the end of each
        era, is <= 10% and doesn't rise era to era.
  LL-P6 the gates at their defaults over a whole life (tripwire): the change signal fires within 300 s after >= 90% of
        changes INTO a structured context, and is on for < 5% of settled seconds.

Launch (repo root):  python -m modal run --detach notebooks/brian2/life_run/modal_life.py
"""
import gzip
import io
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from modal_common import IMAGE, RAW_MOUNT, RAW_VOLUME  # noqa: E402

image = IMAGE.add_local_file(str(HERE.parent / "modal_common.py"), "/root/modal_common.py")
app = modal.App("developing-mind-life")
SEEDS = list(range(54000, 54008))
N_POST, N_PRE, PHASE_S, N_PHASES = 40, 60, 500.0, 200
HOME = ["A", "B", "C", "D", "E", "F"]


def build_life(rng_seed=2027, n_phases=N_PHASES):
    """-> (names, sets): one entry per phase; sets[i] is the list of synchronized inputs ([] for rest)."""
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    home = {h: list(range(b * 10, b * 10 + 10)) for b, h in enumerate(HOME)}
    home["G"] = list(range(0, 5)) + list(range(20, 25))
    eras = [(0, 60, {"A": .45, "B": .45, "C": .10}),
            (60, 140, {"C": .38, "D": .38, "G": .12, "A": .06, "B": .06}),
            (140, 200, {"E": .38, "F": .38, "A": .06, "B": .06, "C": .06, "D": .06})]
    names, sets, prev, k_novel = [], [], None, 0
    for p in range(n_phases):
        if p % 10 == 9:
            names.append("REST"); sets.append([]); prev = "REST"; continue
        lo, hi, w = next(e for e in eras if e[0] <= p < e[1])
        if rng.random() < 0.15:
            k_novel += 1
            names.append(f"N{k_novel}"); sets.append(sorted(int(x) for x in rng.choice(N_PRE, 10, replace=False)))
            prev = names[-1]; continue
        keys = [k for k in w if k != prev]
        pr = np.array([w[k] for k in keys]); pr /= pr.sum()
        c = str(rng.choice(keys, p=pr))
        names.append(c); sets.append(home[c]); prev = c
    return names, sets


def flags_from_D(D, L=60, trail=900, k=3.0, min_hist=100):
    """interface.changing_per_second's rule, applied to a precomputed displacement series D (NaN for t < L)."""
    import numpy as np
    n_t = len(D)
    flag = np.ones(n_t, bool)
    for t in range(L, n_t):
        hist = D[max(L, t - trail):t - L]
        hist = hist[np.isfinite(hist)]
        if len(hist) < min_hist:
            continue
        med = np.median(hist)
        flag[t] = D[t] > med + k * 1.4826 * np.median(np.abs(hist - med))
    return flag


@app.function(image=image, cpu=1.0, memory=8192, timeout=24 * 3600, volumes={RAW_MOUNT: RAW_VOLUME},
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def run_life(seed_val: int, phase_s: float = PHASE_S, n_phases: int = N_PHASES) -> str:
    import time
    import numpy as np
    from brian2 import NetworkOperation, SpikeMonitor, defaultclock, mV, ms, network_operation, prefs, run, second, \
        start_scope, seed as b2_seed
    from brian2 import second as s_
    from src.brian2_stdp.network import build_competitive_population_network, scale_inhib_for_n
    from src.brian2_stdp.spikes import build_set_phase_input, generate_uncorrelated_group
    from src.integration.interface import unit
    import modal_common as MC

    names, sets = build_life(n_phases=n_phases)
    total = phase_s * n_phases
    rng = np.random.default_rng(seed_val)
    all_i, all_t, start = [], [], 0.0
    for st in sets:
        guard = 0.001 if start > 0 else 0.0
        if st:
            i, t = build_set_phase_input(20.0, 0.9, [phase_s], [st], N_PRE, rng)
            t = np.asarray(t / second)
        else:
            i, t = generate_uncorrelated_group(N_PRE, 20.0, phase_s, rng)
        all_i.append(np.asarray(i, int)); all_t.append(np.asarray(t) + start + guard); start += phase_s
    idx = np.concatenate(all_i); tt = np.concatenate(all_t); o = np.argsort(tt, kind="stable")

    start_scope()
    defaultclock.dt = 0.2 * ms
    b2_seed(seed_val)
    per_conn = scale_inhib_for_n(N_POST, reference_inhib_mV=13.0, reference_n_post=3)
    pre, post, syn, inhib = build_competitive_population_network(N_POST, idx[o], tt[o] * second, 0.005, per_conn * mV, 1.5,
                                                                 n_pre=N_PRE, w_init=10.0 / N_PRE, target_total=10.0)
    si, sj = np.array(syn.i[:]), np.array(syn.j[:])
    n_syn, n_sec, n_blk = len(si), int(total), int(total // 10)
    ring = np.zeros((61, n_syn), np.float32)
    trace = np.zeros((n_sec, n_syn), np.float32)                         # full 1 s weight trace, kept for later analysis
    D = np.full(n_sec, np.nan, np.float32)
    blocks = np.zeros((n_blk, n_syn), np.float32)
    acc = np.zeros(n_syn, np.float64)
    state = {"s": 0}

    @network_operation(dt=1 * second, when="start")
    def sample():
        s = state["s"]
        if s >= n_sec:
            return
        w = np.asarray(syn.w[:], np.float32)
        ring[s % 61] = w
        trace[s] = w
        if s >= 60:
            D[s] = float(np.abs(w - ring[(s - 60) % 61]).sum())
        acc[:] += w
        if s % 10 == 9:
            blocks[s // 10] = acc / 10.0; acc[:] = 0.0
        state["s"] = s + 1

    sp = SpikeMonitor(post)
    t0 = time.time()
    run(total * second)
    stimes, sidx = np.array(sp.t / second), np.array(sp.i[:])
    rates10 = np.zeros((N_POST, n_blk), np.float32)
    np.add.at(rates10, (sidx, np.minimum((stimes // 10).astype(int), n_blk - 1)), 1.0)
    rates10 /= 10.0
    flags = flags_from_D(D.astype(float))

    # weight blocks as (n_post, n_pre, n_blk) views for readout/assignments
    def wm_block(b0, b1):
        w = blocks[b0:b1].mean(0)
        m = np.zeros((N_POST, N_PRE), np.float32); m[sj, si] = w; return m
    protos = {n: unit(np.isin(np.arange(N_PRE), s).astype(float)) for n, s in zip(names, sets) if s}
    pnames = list(protos); P = np.array([protos[n] for n in pnames])
    fp = {}
    for W in (10, 50):
        nb = W // 10
        qh, qc = [], []
        for b in range(0, n_blk - nb + 1, nb):
            r = rates10[:, b:b + nb].mean(1).astype(float); m = wm_block(b, b + nb).astype(float)
            qh.append(unit(np.clip(r - r.mean(), 0, None) @ m)); qc.append(unit((r - r.mean()) @ m))
        fp[f"W{W}"] = dict(rect=np.round(np.array(qh), 4).tolist(), contrast=np.round(np.array(qc), 4).tolist())
    assign = []
    for b in range(0, n_blk, 5):
        m = wm_block(b, b + 5).astype(float)
        assign.append([int(np.argmax(P @ unit(m[j]))) for j in range(N_POST)])
    raw_rel = f"life_run/raw/life_seed{seed_val}.npz"
    buf = io.BytesIO()
    np.savez_compressed(buf, trace=trace, spike_t=stimes.astype(np.float32), spike_i=sidx.astype(np.int16), blocks=blocks,
                        rates10=rates10, D=D, syn_i=si, syn_j=sj)
    p = Path(RAW_MOUNT) / raw_rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(buf.getvalue())
    compact = dict(status="completed", seed=seed_val, n_post=N_POST, n_pre=N_PRE, phase_s=phase_s, n_phases=n_phases,
                   total_s=total, phase_names=names, phase_sets=sets, proto_names=pnames, target=prefs.codegen.target,
                   wall_elapsed=time.time() - t0, brian2_seeded=True, substrate="original 13mV/1.5 gated",
                   fingerprints=fp, change_flags=flags.astype(int).tolist(), D=np.round(np.nan_to_num(D, nan=-1), 3).tolist(),
                   assign_50s=assign, rates10=np.round(rates10, 2).tolist(), raw_volume="developing-mind-raw", raw_path=raw_rel)
    crel = f"life_run/compact/life_seed{seed_val}.json.gz"
    q = Path(RAW_MOUNT) / crel; q.parent.mkdir(parents=True, exist_ok=True); q.write_bytes(MC.gz_bytes(compact))
    RAW_VOLUME.commit()
    return crel


@app.local_entrypoint()
def main():
    calls = [run_life.spawn(s) for s in SEEDS]
    print("spawned", len(calls), "life runs (detached):", [c.object_id for c in calls], flush=True)

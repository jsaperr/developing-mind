"""Creation gates: follow-up to coupling toy v0's two-back capture.

v0 finding (strength-independent): after a NOVEL change, the substrate's first signal is the old
context's disappearance ("not-B"), not the new one's appearance. Memory's provisional creation rule
(create when the best similarity is below THETA) commits an entry from that transitional query.
Later queries keep matching it through the shared "not-B" component, so the system reports the
two-back context, in one seed (32005) for a whole phase. Two label-free gates on CREATION only.
Retrieval, update and prune are unchanged, and so are THETA and the grounded gap_scale.

  G-stable  create only if the query has been steady: cos(q_s, q_s-1) >= TAU and
            cos(q_s-1, q_s-2) >= TAU, with TAU=0.9 fixed a priori. The substrate's settled
            consecutive cosine is about 0.967, so 0.9 is clearly below settled jitter. Uses only the
            query stream. Sensitivity at TAU 0.8 and 0.95.
  G-dip     no creation while the substrate's population rate (mean over neurons in the window) is
            below its trailing baseline (median of the previous 30 windows) by more than DELTA =
            3 x 1.4826 x MAD of the rate stream's first differences. This is the contract's S4/P3 idea:
            the context-blind swap dip used as "transition in progress". Label-free.
  An empty memory may always create (otherwise nothing could ever be retrieved).

PREDICTIONS ON RECORD (written before this script was first run; substrate arm, W=10 s):
  GATE-P1: G-stable cuts total two-back ("other") reports after novel swaps in A->B->C from 124 to
           <= 40, and seed 32005's lock from 100 steps to <= 20.
  GATE-P2: G-dip also cuts them (<= 60 total), but RAISES median tracking latency at novel swaps
           above the ungated arm, because the dip recovers slowly (~300 s) and blocks creation
           that long.
  GATE-P3: neither gate changes A->B->A accuracy by more than 0.01.
Everything else is descriptive. Worlds include v1 (A->B->C->A->C) when its runs are complete.

Usage: run_creation_gates.py   (conda env; appends nothing to v0's outputs; writes creation_gates_summary.json)
"""
import json
from pathlib import Path

import numpy as np
import torch

import run_coupling_v0 as T
from analyze_s1_readout import windows

HERE = Path(__file__).resolve().parent
TAU = 0.9
TAU_SENS = [0.8, 0.95]
BASELINE_WINDOWS = 30


def pop_rate_stream(d, wm, r, W):
    return np.array([x[2].mean() for x in windows(d, wm, r, [0.0] + list(d['swap_times_s']), W)])


def dip_flags(rate):
    diffs = np.diff(rate)
    delta = 3 * 1.4826 * np.median(np.abs(diffs - np.median(diffs)))
    flags = np.zeros(len(rate), bool)
    for s in range(len(rate)):
        base = rate[max(0, s - BASELINE_WINDOWS):s]
        flags[s] = len(base) >= 3 and rate[s] < np.median(base) - delta
    return flags, delta


def run_memory_gated(stream, arm, gap_scale, allow):
    """As T.run_memory with THETA, but creation additionally requires allow(s)."""
    Q = torch.tensor(stream['q'][arm], dtype=torch.float32)
    mem = T.EpisodicMemory(dim=Q.shape[1], gap_scale=gap_scale)
    tag, born_phase, reported, winner_ids, alive = {}, {}, [], [], []
    created = 0
    for s in range(len(Q)):
        q = Q[s]
        novel = not mem.patterns or float((torch.stack(mem.patterns) @ q).max()) < T.THETA
        if novel and (not mem.patterns or allow(s)):
            mem.add_pattern(q.clone(), s)
            tag[mem.ids[-1]] = int(np.argmax(stream['protos'] @ q.numpy()))
            born_phase[mem.ids[-1]] = int(stream['ph'][s])
            created += 1
        wi, _ = T.step_memory(mem, q)
        wid = mem.ids[wi]
        winner_ids.append(wid); reported.append(tag[wid])
        mem.prune_step(s)
        alive.append(len(mem.patterns))
    return dict(reported=np.array(reported), winner_ids=winner_ids, born_phase=born_phase,
                alive=np.array(alive), gaps=[], created=created, evicted=len(mem.eviction_log))


def stable_allow(Qn, tau):
    cs = np.r_[np.nan, np.sum(Qn[1:] * Qn[:-1], axis=1)]
    return lambda s: s >= 2 and cs[s] >= tau and cs[s - 1] >= tau


def summarize(streams, results):
    sc = [T.score(s, x) for s, x in zip(streams, results)]
    novel_other, novel_lat, per_seed_other = [], [], []
    for x in sc:
        o = 0
        for sw in x['swaps']:
            if not sw['returning']:
                o += sw['other']
                novel_lat.append(sw['latency'] if sw['latency'] is not None else np.nan)
        per_seed_other.append(o)
    return dict(acc=float(np.mean([x['acc'] for x in sc])), other_total=int(sum(per_seed_other)),
                other_per_seed=per_seed_other, novel_latency_median=float(np.nanmedian(novel_lat)),
                novel_latency_never=int(np.sum(np.isnan(novel_lat))),
                created=float(np.mean([x['created'] for x in results])),
                swaps=[x['swaps'] for x in sc])


def main():
    v0 = json.load(open(HERE / "coupling_v0_summary.json"))
    out = {}
    for world, ddir, pat in T.WORLDS:
        runs = T.load_world(ddir, pat)
        key = f"{world}_W10"
        if not runs or key not in v0:
            print(f"[{world}] skipped (no completed runs or no v0 grounding yet)"); continue
        rng = np.random.default_rng(0); torch.manual_seed(0)       # same streams as v0
        streams = [T.build_streams(d, wm, r, n, bs, 10, rng) for d, wm, r, n, bs in runs]
        rates = [pop_rate_stream(d, wm, r, 10) for d, wm, r, n, bs in runs]
        gs = v0[key]['gap_scale']
        arms = {}
        arms['ungated'] = summarize(streams, [T.run_memory(s, 'substrate', gs, T.THETA) for s in streams])
        for tau in [TAU] + TAU_SENS:
            arms[f'G-stable tau={tau}'] = summarize(streams, [
                run_memory_gated(s, 'substrate', gs, stable_allow(s['q']['substrate'], tau)) for s in streams])
        dips = [dip_flags(rt) for rt in rates]
        arms['G-dip'] = summarize(streams, [
            run_memory_gated(s, 'substrate', gs, (lambda f: (lambda i: not f[i]))(fl)) for s, (fl, _) in zip(streams, dips)])
        arms['G-dip']['flagged_fraction'] = float(np.mean([fl.mean() for fl, _ in dips]))
        out[key] = arms
        print(f"\n===== {key} (substrate arm, gap_scale {gs:.4f}, THETA {T.THETA}) =====")
        for name, a in arms.items():
            extra = f"  flagged {a['flagged_fraction']:.2f} of steps" if 'flagged_fraction' in a else ""
            print(f"  {name:18s} acc {a['acc']:.3f}  two-back total {a['other_total']:4d} per seed {a['other_per_seed']}  "
                  f"novel-swap latency median {a['novel_latency_median']:.1f} (never: {a['novel_latency_never']})  "
                  f"created {a['created']:.1f}{extra}")
            for k, sw in enumerate(a['swaps'][0]):
                if sw['returning']:
                    reuse = [x[k]['reuse'] for x in a['swaps'] if x[k]['reuse'] is not None]
                    lats = [x[k]['latency'] for x in a['swaps']]
                    print(f"      return swap {k + 1} ({T.NAMES[sw['frm']]}->{T.NAMES[sw['to']]}): latency {lats} reuse {sum(reuse)}/{len(reuse)}")
    json.dump(out, open(HERE / "creation_gates_summary.json", "w"), indent=1, default=float)


if __name__ == '__main__':
    main()

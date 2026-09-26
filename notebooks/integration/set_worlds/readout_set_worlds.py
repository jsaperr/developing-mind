"""Readout for the set worlds (notebooks/brian2/set_worlds_data): overlapping contexts (ov50, ov70) and
the integrated episodic check (b) (checkb). It uses the PROMOTED modules only, as their first
real use: src/integration/interface.py for readout and signals, and
src/hopfield/episodic_consolidating.GatedEpisodicMemory with the adopted rule and transitional
output.

Arms, per world and clock (W = 10, 50 s; gap_scale grounded per world/clock by the original
procedure, 0.6 x the top1-top2 gap median of the substrate queries under plain memory):
  clean    memory with every new feature off, fed the true-context prototype plus jitter matched to
           the substrate's step-to-step jitter (the memory layer's own baseline / clock control)
  v0       memory with every new feature off, fed substrate queries (equals the validated
           EpisodicMemory path; tested)
  adopted  GatedEpisodicMemory with the steady and changing signals: commit only when steady and
           not re-learning, rehearse always, report TRANSITIONAL while re-learning
Scoring uses labels only here. Entries are content-tagged (nearest context prototype, re-tagged
when content moves). Genuine recognition = the entry born in the returning context's original
phase wins more than half of the return phase. Merged entry = one entry that wins at least 10
settled steps under each of two different true contexts.

PREDICTIONS ON RECORD (written while the batch ran, before any set-world result existed):
 Overlap (v1b schedule A->B->C->A->B; the prototype cosine between contexts is 0.25 at 50%
 overlap and 0.55 at 70%, against memory's novelty threshold of 0.5):
  OV-P1 substrate stays one-back: holders of the incoming context at the two-back returns
        (swaps 3, 4) average <= 1 per seed, at both overlaps.
  OV-P2 readout: settled query cosine to its true prototype (phases >= 2) is >= 0.8 at ov50,
        and lower at ov70 than at ov50.
  OV-P3 THE STRESS TEST, adopted memory at W=50 s, B two-back genuine recognition:
        ov50 >= 5/8 (still works); ov70 <= 3/8 (contexts more similar than the novelty threshold,
        so memory can't keep them apart).
  OV-P4 merged entries (adopted, W=50 s): present in >= 5/8 seeds at ov70 and <= 2/8 at ov50.
 Integrated check (b) (A F1 B F2 C F3 A; cores disjoint, fillers never return):
  CB-P1 the original failure doesn't recur: adopted, W=50 s, all three core entries (born in
        their own first dominant phase) are alive at the end in 8/8, and A's original entry is
        recognized at its return in >= 6/8.
  CB-P2 the original question, graded primacy/recency: adopted, W=50 s, final w_char of the core
        entries has A (first + returned) highest in >= 6/8 seeds and C (recent) > B in >= 5/8.
  CB-P3 horizon-rule baseline: adopted, W=10 s, A's original entry recognized at its return in
        <= 3/8. Its last reliable rehearsal is ~2600 s before the return, beyond the 1500 s horizon.
        Caveat stated now: the fillers add extra changes, and so extra rehearsal windows, which
        could keep A alive longer.
  CB-P4 bounded memory (adopted, W=10 s): alive entries at the end <= 6 per seed on average.

Usage: readout_set_worlds.py   (conda env; writes set_worlds_summary.json)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.brian2_stdp.results_io import load_result
from src.hopfield.episodic import EpisodicMemory
from src.hopfield.episodic_consolidating import TRANSITIONAL, GatedEpisodicMemory
from src.integration import interface as I

DDIR = ROOT / "notebooks" / "brian2" / "set_worlds_data"
CLOCKS = [10, 50]
SETTLE = 300


def load(world):
    runs = []
    for f in sorted(glob.glob(str(DDIR / f"{world}_n7_seed*.json.gz"))):
        d = load_result(f)
        if d.get('status') != 'completed':
            continue
        si, sj = np.array(d['syn_i']), np.array(d['syn_j'])
        w = np.array(d['weight_trace'])
        wm = np.zeros((d['n_post'], 30, w.shape[1])); wm[sj, si] = w
        runs.append((d, wm, np.array(d['spike_rate_bins'], dtype=float)))
    return runs


def stream(d, wm, r, W, rng):
    ps = [0.0] + list(d['swap_times_s']); ids = d['phase_corr_blocks']
    protos = np.array([I.unit(np.isin(np.arange(30), s).astype(float)) for s in d['context_sets']])
    starts, ph, q = [], [], []
    for a in range(0, int(d['total_s']) - W + 1, W):
        p = int(np.searchsorted(ps, a, side='right') - 1)
        if p + 1 < len(ps) and a + W > ps[p + 1]:
            continue
        starts.append(a); ph.append(p)
        q.append(I.h_readout(r[:, a:a + W].mean(axis=1), wm[:, :, a:a + W].mean(axis=2)))
    ph = np.array(ph); q = np.array(q)
    true = np.array([ids[p] for p in ph]); prev = np.array([ids[p - 1] if p else -1 for p in ph])
    settled = np.array([a - ps[p] >= SETTLE for a, p in zip(starts, ph)])
    t1 = np.mean([q[i] @ q[i + 1] for i in range(len(q) - 1) if settled[i] and settled[i + 1] and ph[i] == ph[i + 1]])
    sigma = np.sqrt(max(1 / max(t1, 1e-3) - 1, 0) / q.shape[1])
    clean = np.array([I.unit(protos[true[i]] + sigma * rng.standard_normal(q.shape[1])) for i in range(len(q))])
    return dict(starts=starts, ph=ph, true=true, prev=prev, settled=settled, protos=protos, q=q, clean=clean,
                steady=I.steady_flags(q, 0.9), changing=I.window_any(I.changing_per_second(wm), starts, W))


def ground_gap_scale(streams):
    """Exactly the grounding used for every earlier world (coupling_v0's run_memory, pass 1)."""
    sys.path.insert(0, str(HERE.parent / "coupling_v0"))
    sys.path.insert(0, str(ROOT / "notebooks" / "brian2" / "interface_readout"))
    import run_coupling_v0 as T
    gaps = [g for s in streams for g in T.run_memory(
        dict(q={'substrate': s['q']}, true=s['true'], ph=s['ph'], protos=s['protos']),
        'substrate', T.OLD_GAP_SCALE, T.THETA)['gaps']]
    return 0.6 * float(np.median(gaps))


def run_arm(s, arm, gs):
    queries = s['clean'] if arm == 'clean' else s['q']
    kw = dict(eta=0.0, match_floor=None) if arm in ('clean', 'v0') else {}
    m = GatedEpisodicMemory(dim=queries.shape[1], gap_scale=gs, **kw)
    tag, born, wins, est, absorbed = {}, {}, {}, {}, set()
    winners, reports, alive = [], [], []
    for i, x in enumerate(torch.tensor(queries, dtype=torch.float32)):
        if arm == 'adopted':
            o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
        else:
            o = m.step(x)
        if o['created'] is not None:
            tag[o['created']] = int(np.argmax(s['protos'] @ x.numpy())); born[o['created']] = int(s['ph'][i])
        wid = o['winner']
        rep = TRANSITIONAL if o['report'] is TRANSITIONAL else tag[wid]
        if o['consolidation_rate'] > 0:
            wi = m.mem.ids.index(wid) if wid in m.mem.ids else None
            if wi is not None:
                tag[wid] = int(np.argmax(s['protos'] @ m.mem.patterns[wi].numpy()))
        if s['settled'][i] and rep is not TRANSITIONAL:
            c = wins.setdefault(wid, {}); c[int(s['true'][i])] = c.get(int(s['true'][i]), 0) + 1
            if rep == s['true'][i] and wid not in est and c[int(s['true'][i])] >= 10:
                est[wid] = int(s['true'][i])
        if wid in est and tag[wid] != est[wid]:
            absorbed.add(wid)
        winners.append(wid); reports.append(rep); alive.append(tuple(m.mem.ids))
    merged = sum(1 for c in wins.values() if sum(v >= 10 for v in c.values()) >= 2)
    final_wchar = {i: float(wc) for i, wc in zip(m.mem.ids, m.mem.w_char)}
    return dict(winners=winners, reports=reports, alive=alive, born=born, absorbed=len(absorbed), merged=merged,
                final_wchar=final_wchar, n_alive_end=len(m.mem.ids))


def score(s, res):
    rep, true = res['reports'], s['true']
    committed = [i for i in range(len(rep)) if rep[i] is not TRANSITIONAL]
    return dict(acc_committed=float(np.mean([rep[i] == true[i] for i in committed])) if committed else np.nan,
                transitional=1 - len(committed) / len(rep))


def genuine(s, res, ret_phase, orig_phase):
    m = np.where(s['ph'] == ret_phase)[0]
    return float(np.mean([res['born'].get(res['winners'][i]) == orig_phase for i in m]))


def alive_at(s, res, phase, orig_phase):
    i0 = int(np.where(s['ph'] == phase)[0][0])
    return any(res['born'].get(x) == orig_phase for x in res['alive'][i0 - 1])


def substrate_part(runs):
    d0 = runs[0][0]; ids = d0['phase_corr_blocks']
    protos = np.array([I.unit(np.isin(np.arange(30), s).astype(float)) for s in d0['context_sets']])
    line = "  substrate holders of the incoming context at each swap (mean per seed):"
    for k, s_t in enumerate(d0['swap_times_s']):
        inc = ids[k + 1]; h = []
        for d, wm, r in runs:
            s_t = int(d['swap_times_s'][k])
            wbar = wm[:, :, s_t - 50:s_t].mean(axis=2)
            fav = [int(np.argmax(protos @ I.unit(wbar[j]))) for j in range(d['n_post'])]
            h.append(sum(f == inc for f in fav))
        line += f"  s{k + 1}->{inc}: {np.mean(h):.2f}"
    print(line)


def main():
    out = {}
    for world in ("ov50", "ov70", "checkb"):
        runs = load(world)
        print(f"\n################ {world}: {len(runs)} completed seeds")
        if not runs:
            continue
        substrate_part(runs)
        out[world] = {}
        for W in CLOCKS:
            rng = np.random.default_rng(0); torch.manual_seed(0)
            streams = [stream(d, wm, r, W, rng) for d, wm, r in runs]
            if W == 10:
                nph = max(s['ph'].max() for s in streams) + 1
                qual = [[np.mean([s['q'][i] @ s['protos'][s['true'][i]] for i in range(len(s['ph']))
                                  if s['ph'][i] == p and s['settled'][i]]) for p in range(nph)] for s in streams]
                qual = np.array(qual)
                print("  readout quality by phase: " + " ".join(f"ph{p + 1} {qual[:, p].mean():.2f}" for p in range(nph)))
                out[world]['readout_quality'] = qual.mean(0).tolist()
            gs = ground_gap_scale(streams)
            print(f"  --- W={W} s (gap_scale {gs:.4f}, horizon {150 * W} s)")
            out[world][f"W{W}"] = {}
            for arm in ("clean", "v0", "adopted"):
                res = [run_arm(s, arm, gs) for s in streams]
                sc = [score(s, x) for s, x in zip(streams, res)]
                row = dict(acc_committed=float(np.nanmean([x['acc_committed'] for x in sc])),
                           transitional=float(np.mean([x['transitional'] for x in sc])),
                           absorbed=int(sum(x['absorbed'] for x in res)),
                           merged_seeds=int(sum(x['merged'] > 0 for x in res)),
                           n_alive_end=float(np.mean([x['n_alive_end'] for x in res])))
                txt = (f"    {arm:8s} acc(committed) {row['acc_committed']:.3f}  transitional {row['transitional']:.2f}  "
                       f"absorbed {row['absorbed']}  merged-seeds {row['merged_seeds']}/8  alive-at-end {row['n_alive_end']:.1f}")
                if world in ("ov50", "ov70"):
                    g = int(sum(genuine(s, x, 4, 1) > 0.5 for s, x in zip(streams, res)))
                    a = int(sum(alive_at(s, x, 4, 1) for s, x in zip(streams, res)))
                    row['B_two_back'] = dict(genuine=g, alive=a)
                    txt += f"  | B two-back: alive {a}/8 genuine {g}/8"
                else:
                    g = int(sum(genuine(s, x, 6, 0) > 0.5 for s, x in zip(streams, res)))
                    a = int(sum(alive_at(s, x, 6, 0) for s, x in zip(streams, res)))
                    order_A, order_CB, cores_alive = 0, 0, 0
                    for s, x in zip(streams, res):
                        first = {c: int(np.where(s['true'] == c)[0][0]) for c in (0, 1, 2)}
                        orig = {c: [i for i, b in x['born'].items() if b == s['ph'][first[c]]] for c in (0, 1, 2)}
                        wc = {c: max([x['final_wchar'][i] for i in orig[c] if i in x['final_wchar']], default=None)
                              for c in (0, 1, 2)}
                        cores_alive += all(wc[c] is not None for c in (0, 1, 2))
                        if all(wc[c] is not None for c in (0, 1, 2)):
                            order_A += wc[0] > max(wc[1], wc[2]); order_CB += wc[2] > wc[1]
                    row.update(A_return=dict(genuine=g, alive=a), cores_alive_end=cores_alive,
                               A_highest=order_A, C_over_B=order_CB)
                    txt += (f"  | A return: alive {a}/8 genuine {g}/8 | all 3 cores alive at end {cores_alive}/8 | "
                            f"w_char A highest {order_A}/8, C>B {order_CB}/8")
                out[world][f"W{W}"][arm] = row
                print(txt)
    json.dump(out, open(HERE / "set_worlds_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

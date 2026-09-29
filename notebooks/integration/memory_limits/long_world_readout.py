"""Readout of the long world (notebooks/brian2/long_world_data: 20 phases x 1000 s, contexts A B C
disjoint, D 70% overlap with A, E 50% with A and B; each visited 4 times; seeds 42000-42007).
Substrate side (one-back over 20 phases, drift) and the dormant character store on real input,
rectified readout, adopted commit rule at radius 0.8, both clocks. Store: notebook-only, as in
dormant_long_toy.py (link radius 0.8, pruning at baseline).

PREDICTIONS ON RECORD (written while the batch ran, before any long-world result existed; LW-P1/P2
are also in run_long_seed.py):
 Substrate
  LW-P1 one-back holds: at returns more than one back, holders of the incoming context average
        <= 1 per seed, and the first-half mean is not below the second-half mean by more than 0.5
        (no accumulating residue).
  LW-P2 no drift: cosine between a context's settled mean readout at its 4th visit and at its 1st
        is >= 0.9 (seed mean, every context).
 Store on real input
  LR-P1 misattribution stays inside memory's resolution: 0 wrong links at W=50, <= 2 total at W=10
        (only A-D is close: prototype 0.55, readout up to ~0.75).
  LR-P2 the store is exercised at the fast clock: >= 5 links per seed at W=10.
  LR-P3 missed links <= 1 per seed at W=10.
  LR-P4 no behavioural cost: committed accuracy with the store within 0.01 of without it, both
        clocks.
  LR-P5 character recency with equal exposure (each context visited 4 times): at W=10 with the store,
        the Spearman correlation between a record's final w_char and its context's last-visit phase
        is > 0 in >= 6/8 seeds. Without the store, fewer than 5 contexts have a live entry at the
        end in most seeds, so the ranking can't be taken.

Usage: long_world_readout.py   (conda env; writes long_world_readout_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SW = ROOT / "notebooks" / "integration" / "set_worlds"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(SW)); sys.path.insert(0, str(HERE))
import readout_set_worlds as RS
import run_rectified_memory as R
from src.hopfield.episodic_consolidating import GatedEpisodicMemory
from src.hopfield.two_layer import CONSTANTS
from src.integration import interface as I

PAT = R.B2 / "long_world_data" / "long20_n7_seed*.json.gz"
RADIUS, LINK, PRUNE_EPS = 0.8, 0.8, 0.05


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def substrate(runs):
    d0 = runs[0][0]; ids = d0['phase_corr_blocks']
    protos = np.array([I.unit(np.isin(np.arange(30), s).astype(float)) for s in d0['context_sets']])
    rows = []
    for k in range(len(ids) - 1):
        inc = ids[k + 1]
        back = next((j for j in range(1, k + 2) if ids[k + 1 - j] == inc), None)
        h = []
        for d, wm, r, chg in runs:
            s_t = int(d['swap_times_s'][k]); wbar = wm[:, :, s_t - 50:s_t].mean(axis=2)
            h.append(sum(int(np.argmax(protos @ I.unit(wbar[j]))) == inc for j in range(d['n_post'])))
        rows.append(dict(swap=k + 1, incoming=inc, back=back, holders=float(np.mean(h))))
    far = [x for x in rows if x['back'] is not None and x['back'] > 1]
    half = len(rows) // 2
    first = [x['holders'] for x in far if x['swap'] <= half]; second = [x['holders'] for x in far if x['swap'] > half]
    print(f"  substrate: holders at >1-back returns mean {np.mean([x['holders'] for x in far]):.2f} "
          f"(first half {np.mean(first):.2f}, second half {np.mean(second):.2f}); "
          f"at 1-back returns {np.mean([x['holders'] for x in rows if x['back'] == 1] or [np.nan]):.2f}")
    return rows, float(np.mean([x['holders'] for x in far])), float(np.mean(first)), float(np.mean(second))


def drift(streams):
    out = {}
    for c in range(5):
        sims = []
        for s in streams:
            visits = [p for p in np.unique(s['ph']) if s['true'][s['ph'] == p][0] == c]
            m = [s['q'][(s['ph'] == p) & s['settled']].mean(0) for p in visits]
            sims.append(float(m[-1] @ m[0] / np.linalg.norm(m[-1]) / np.linalg.norm(m[0])))
        out["ABCDE"[c]] = float(np.mean(sims))
    return out


def run(s, gs, store):
    m = GatedEpisodicMemory(dim=s['q'].shape[1], gap_scale=gs, theta=RADIUS, match_floor=RADIUS)
    mem = m.mem
    records, link, reports = [], {}, []
    links = wrong = missed = 0
    tag, pairs = {}, []
    for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
        o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
        true = int(s['true'][i])
        if o['created'] is not None:
            tag[o['created']] = int(np.argmax(s['protos'] @ x.numpy()))
            if store:
                k = mem.ids.index(o['created']); p = mem.patterns[k]
                sims = [float(r['pattern'] @ p) if r is not None else -9 for r in records]
                if sims and max(sims) >= LINK:
                    ri = int(np.argmax(sims)); mem.w_char[k] = records[ri]['w_char']; links += 1
                    wrong += records[ri]['label'] != true
                    if records[ri]['label'] != true:
                        pairs.append(("ABCDE"[records[ri]['label']], "ABCDE"[true], round(max(sims), 3)))
                else:
                    missed += any(r is not None and r['label'] == true for r in records)
                    records.append(dict(pattern=p.clone(), w_char=1.0, label=true)); ri = len(records) - 1
                link[o['created']] = ri
        if store:
            alive = {}
            for k, e in enumerate(mem.ids):
                if e in link and (link[e] not in alive or mem.w_char[k] > mem.w_char[alive[link[e]]]):
                    alive[link[e]] = k
            for ri, r in enumerate(records):
                if r is None:
                    continue
                if ri in alive:
                    r['w_char'] = mem.w_char[alive[ri]]; r['pattern'] = mem.patterns[alive[ri]].clone()
                else:
                    r['w_char'] += CONSTANTS.decay_char * (1 - r['w_char'])
                    if r['w_char'] - 1 < PRUNE_EPS:
                        records[ri] = None
        tag_w = tag.get(o['winner'])
        reports.append(None if o['report'] is None else tag_w)
    committed = [i for i, r in enumerate(reports) if r is not None]
    acc = float(np.mean([reports[i] == s['true'][i] for i in committed]))
    if store:
        final = {}
        for r in records:
            if r is not None:
                final[r['label']] = max(final.get(r['label'], 0), r['w_char'])
    else:
        final = {}
        for k, e in enumerate(mem.ids):
            final[tag[e]] = max(final.get(tag[e], 0), mem.w_char[k])
    relaxed = {}
    if store:
        wf, wc = torch.tensor(mem.w_fast), torch.tensor(mem.w_char)
        for _ in range(300):
            wf, wc = mem._update_fn(wf, wc, torch.zeros(len(mem.ids)))
        alive = {link[e]: k for k, e in enumerate(mem.ids) if e in link}
        for ri, r in enumerate(records):
            if r is not None:
                v = float(wc[alive[ri]]) if ri in alive else 1 + (r['w_char'] - 1) * (1 - CONSTANTS.decay_char) ** 300
                relaxed[r['label']] = max(relaxed.get(r['label'], 0), v)
    return dict(acc=acc, links=links, wrong=wrong, missed=missed, final=final, pairs=pairs, relaxed=relaxed)


def main():
    runs = R.load(PAT)
    print(f"long world: {len(runs)} completed seeds")
    rows, far, h1, h2 = substrate(runs)
    out = dict(substrate=dict(rows=rows, far_mean=far, first_half=h1, second_half=h2))
    rng = np.random.default_rng(0)
    dr = drift([R.build(d, wm, r, chg, 10, "H+", rng) for d, wm, r, chg in runs])
    print("  drift (4th visit vs 1st, settled H+ cosine): " + "  ".join(f"{k} {v:.3f}" for k, v in dr.items()))
    out['drift'] = dr
    ids = runs[0][0]['phase_corr_blocks']
    last_visit = {c: max(i for i, x in enumerate(ids) if x == c) for c in range(5)}
    for W in (10, 50):
        rng = np.random.default_rng(0); torch.manual_seed(0)
        streams = [R.build(d, wm, r, chg, W, "H+", rng) for d, wm, r, chg in runs]
        gs = RS.ground_gap_scale(streams)
        for store in (False, True):
            res = [run(s, gs, store) for s in streams]
            rho, full = [], 0
            for x in res:
                if all(c in x['final'] for c in range(5)):
                    full += 1
                    rho.append(spearman([x['final'][c] for c in range(5)], [last_visit[c] for c in range(5)]))
            row = dict(acc=float(np.mean([x['acc'] for x in res])), links=int(sum(x['links'] for x in res)),
                       wrong=int(sum(x['wrong'] for x in res)), missed_per_seed=float(np.mean([x['missed'] for x in res])),
                       seeds_all5=full, rho_pos=int(sum(v > 0 for v in rho)), rho_mean=float(np.mean(rho)) if rho else None)
            key = f"W{W} {'store' if store else 'no store'}"
            out[key] = row
            print(f"  {key:14s} acc {row['acc']:.3f}  links {row['links']}  WRONG {row['wrong']}  missed/seed "
                  f"{row['missed_per_seed']:.2f} | all 5 contexts have character at end {full}/8, "
                  f"recency rho>0 {row['rho_pos']}/{full} (mean {row['rho_mean'] if rho else float('nan'):+.2f})")
    json.dump(out, open(HERE / "long_world_readout_summary.json", "w"), indent=1)


def post_hoc():
    """POST-HOC (added after seeing the results; not predicted): which pairs the wrong links were, the
    final record w_char values, and the recency correlation after 300 steps of retrieval-free relaxation
    (removes consolidation lag, as in recency_toy.py)."""
    runs = R.load(PAT)
    ids = runs[0][0]['phase_corr_blocks']
    last_visit = [max(i for i, x in enumerate(ids) if x == c) for c in range(5)]
    print("POST-HOC: last-visit phase per context A-E:", last_visit)
    for W in (10, 50):
        rng = np.random.default_rng(0); torch.manual_seed(0)
        streams = [R.build(d, wm, r, chg, W, "H+", rng) for d, wm, r, chg in runs]
        gs = RS.ground_gap_scale(streams)
        res = [run(s, gs, True) for s in streams]
        print(f"  W={W}: wrong-link pairs (record, true, link cosine): {[p for x in res for p in x['pairs']]}")
        rel = []
        for x in res:
            print("    final w_char " + " ".join(f"{'ABCDE'[c]} {x['final'].get(c, float('nan')):.2f}" for c in range(5))
                  + " | relaxed " + " ".join(f"{'ABCDE'[c]} {x['relaxed'].get(c, float('nan')):.2f}" for c in range(5)))
            if all(c in x['relaxed'] for c in range(5)):
                rel.append(spearman([x['relaxed'][c] for c in range(5)], last_visit))
        print(f"  W={W}: relaxed recency rho>0 {sum(v > 0 for v in rel)}/{len(rel)} (mean {np.mean(rel):+.2f})")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'posthoc':
        post_hoc()
    else:
        main()

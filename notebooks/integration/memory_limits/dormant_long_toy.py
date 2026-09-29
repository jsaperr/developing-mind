"""Dormant entries under heavy exposure, memory only (no substrate): does the persistent character
store misattribute when contexts return many times after being forgotten, and does pruning at
baseline keep it bounded? Notebook-only store, src untouched.

World (30 inputs; synthetic queries = prototype + jitter): seven returning contexts with graded
similarity, plus never-returning fillers.
  A 0-9, B 10-19, C 20-29 (disjoint, cosine -0.5)
  D = 0-6 + 20-22    (vs A 0.55: the 70% overlap)
  E = 5-14           (vs A, B 0.25: 50%)
  F = 10-17 + 25-26  (vs B 0.70: 80%, just under the 0.8 radius)
  G = 20-28 + 9      (vs C 0.85: 90%, ABOVE the 0.8 radius, a resolution probe)
150 phases x 100 steps (W=10 s scale; memory's horizon is ~150 steps, so contexts are forgotten
between visits). Each phase is a returning context (random, no immediate repeat) or, with
probability 0.3, a fresh filler (random 10-subset). The first 20 steps of each phase ramp linearly
from the previous prototype to the new one and are flagged changing (a crude stand-in for the
substrate's re-learning window).

Memory: promoted GatedEpisodicMemory with the adopted rule at radius 0.8 (theta = match_floor).
Store (as character_store_real.py): records mirror their live episode's pattern and w_char, freeze
and decay (decay_char) after eviction, and a new episode matching a record (cosine >= 0.8) links to
it and inherits its w_char. Arms: jitter 0.032 (the substrate's settled level) and 0.06 (noisier);
pruning off vs on (drop a dormant record once w_char - 1 < 0.05, when it carries no character).

PREDICTIONS ON RECORD (written before this script was first run):
  LM-P1 no misattribution below the radius: 0 wrong links between contexts with prototype cosine
        <= 0.55 (A-D, A-E, B-E, ...) at jitter 0.032; at most a handful (<= 3 total over 8 seeds)
        at 0.06.
  LM-P2 the 0.70 pair (B-F), just under the radius: <= 2 wrong links total at 0.032, more at 0.06.
  LM-P3 the 0.85 pair (C-G) is one thing at this resolution: C and G share a record (G links to C's
        record or never gets its own) in >= 6/8 seeds. A resolution limit shared by episodes and
        records, not a store-specific error.
  LM-P4 pruning bounds the store: without pruning, records grow with the fillers (~45); with
        pruning the final count is <= half of that, and the growth flattens in the second half.
  LM-P5 missed links (a returning context starting a second record) stay rare at 0.032 (<= 1 per
        seed on average).

Usage: dormant_long_toy.py   (any env with torch; writes dormant_long_toy_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.hopfield.episodic_consolidating import GatedEpisodicMemory
from src.hopfield.two_layer import CONSTANTS
from src.integration.interface import unit

N_IN = 30
SETS = {'A': list(range(0, 10)), 'B': list(range(10, 20)), 'C': list(range(20, 30)),
        'D': list(range(0, 7)) + [20, 21, 22], 'E': list(range(5, 15)),
        'F': list(range(10, 18)) + [25, 26], 'G': list(range(20, 29)) + [9]}
N_PHASES, L, RAMP, P_FILL = 150, 100, 20, 0.3
RADIUS, LINK, PRUNE_EPS = 0.8, 0.8, 0.05
SEEDS = range(8)


def proto(s):
    return unit(np.isin(np.arange(N_IN), s).astype(float))


def world(rng, sigma):
    names, prev = [], None
    for _ in range(N_PHASES):
        if rng.random() < P_FILL:
            names.append(f"fill{len(names)}")
        else:
            names.append(str(rng.choice([k for k in SETS if k != prev])))
        prev = names[-1]
    protos = {n: proto(SETS[n]) if n in SETS else proto(rng.choice(N_IN, 10, replace=False)) for n in names}
    q, lab, chg = [], [], []
    last = None
    for n in names:
        for t in range(L):
            a = min(1.0, (t + 1) / RAMP) if last is not None else 1.0
            base = protos[n] if a >= 1.0 else unit((1 - a) * protos[last] + a * protos[n])
            q.append(unit(base + sigma * rng.standard_normal(N_IN))); lab.append(n); chg.append(a < 1.0)
        last = n
    return np.array(q), lab, np.array(chg)


def run(q, lab, chg, gs, prune):
    m = GatedEpisodicMemory(dim=N_IN, gap_scale=gs, theta=RADIUS, match_floor=RADIUS)
    mem = m.mem
    records, link = [], {}
    links, wrong, missed, wrong_pairs, counts = 0, 0, 0, {}, []
    for i, x in enumerate(torch.tensor(q, dtype=torch.float32)):
        o = m.step(x, steady=True, changing=bool(chg[i]))
        if o['created'] is not None:
            k = mem.ids.index(o['created']); p = mem.patterns[k]
            sims = [float(r['pattern'] @ p) if r is not None else -9 for r in records]
            if sims and max(sims) >= LINK:
                ri = int(np.argmax(sims)); mem.w_char[k] = records[ri]['w_char']; links += 1
                if records[ri]['label'] != lab[i]:
                    wrong += 1; key = tuple(sorted((records[ri]['label'][:4], lab[i][:4])))
                    wrong_pairs[key] = wrong_pairs.get(key, 0) + 1
            else:
                missed += any(r is not None and r['label'] == lab[i] for r in records)
                records.append(dict(pattern=p.clone(), w_char=1.0, label=lab[i])); ri = len(records) - 1
            link[o['created']] = ri
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
                if prune and r['w_char'] - 1 < PRUNE_EPS:
                    records[ri] = None
        if (i + 1) % 1500 == 0:
            counts.append(sum(r is not None for r in records))
    shared_CG = any(r is not None and r['label'] == 'C' for r in records) and not any(
        r is not None and r['label'] == 'G' for r in records) or wrong_pairs.get(('C', 'G'), 0) > 0
    return dict(links=links, wrong=wrong, missed=missed, wrong_pairs=wrong_pairs, counts=counts, shared_CG=shared_CG)


def main():
    out = {}
    for sigma in (0.032, 0.06):
        rng0 = np.random.default_rng(7)
        g = []
        for sd in range(2):
            q, lab, chg = world(np.random.default_rng(500 + sd), sigma)
            m = GatedEpisodicMemory(dim=N_IN, gap_scale=0.1534, theta=RADIUS, match_floor=RADIUS)
            for i, x in enumerate(torch.tensor(q, dtype=torch.float32)):
                if len(m.mem.patterns) >= 2:
                    s = np.sort((torch.stack(m.mem.patterns) @ x).numpy()); g.append(float(s[-1] - s[-2]))
                m.step(x, steady=True, changing=bool(chg[i]))
        gs = 0.6 * float(np.median(g))
        for prune in (False, True):
            res = []
            for sd in SEEDS:
                q, lab, chg = world(np.random.default_rng(1000 + sd), sigma)
                res.append(run(q, lab, chg, gs, prune))
            pairs = {}
            for r in res:
                for k, v in r['wrong_pairs'].items():
                    pairs['-'.join(k)] = pairs.get('-'.join(k), 0) + v
            counts = np.mean([r['counts'] for r in res], axis=0)
            row = dict(gap_scale=gs, links=int(sum(r['links'] for r in res)), wrong=int(sum(r['wrong'] for r in res)),
                       wrong_pairs=pairs, missed_per_seed=float(np.mean([r['missed'] for r in res])),
                       shared_CG_seeds=int(sum(r['shared_CG'] for r in res)), records_over_time=counts.tolist())
            key = f"sigma{sigma} {'prune' if prune else 'no-prune'}"
            out[key] = row
            print(f"{key:20s} links {row['links']:4d}  WRONG {row['wrong']:3d} {pairs}  missed/seed {row['missed_per_seed']:.2f}  "
                  f"C&G shared {row['shared_CG_seeds']}/8  records every 1500 steps: {np.round(counts, 1).tolist()}")
    json.dump(out, open(HERE / "dormant_long_toy_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

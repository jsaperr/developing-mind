"""The persistent character store (option 2, recency_carriers.py) on REAL substrate runs, where
linking isn't trivial: overlapping contexts (ov50, ov70), the integrated check (b) with filler
contexts, and disjoint controls. Replay only, rectified readout, adopted commit rule, both clocks,
memory radius 0.5 (adopted) and 0.8 (recommended). Notebook-only toy; src untouched.

Store (as in recency_carriers.py, one change): every created episode that matches no record
(cosine < LINK = 0.8) starts a record. A new episode matching a record links to it and starts at
the record's w_char. While any linked episode is alive the record mirrors the largest linked
w_char AND the pattern of that episode (content consolidation moves patterns, so the record
follows while remembered). After the last linked episode is evicted the record is frozen and
decays (decay_char toward 1).

Scoring (labels used only here): a record's label = the true context when it was started. A link
is WRONG if the new episode's true context differs from the record's label (misattribution:
inheriting another context's history). A DUPLICATE record = a record started for a context that
already had a record (the opposite error: failing to find your own history, e.g. from drift).
Also: key genuine recognitions with the store on vs off (same code path, store disabled).

PREDICTIONS ON RECORD (written before this script was first run):
  MA-P1 disjoint worlds (v1b, v1c): 0 wrong links, 0 duplicate records.
  MA-P2 ov50 (between-context cosine <= 0.39): 0 wrong links.
  MA-P3 ov70 (between-context cosine up to 0.75, within 5th pct 0.96 at W=10): wrong links appear
        at W=10 in >= 2/8 seeds under at least one memory radius. The 0.8 link radius sits only
        0.05 above the largest between-context similarity.
  MA-P4 links mostly happen at W=10 (eviction within the run); at W=50 there are few or none, so the
        store is barely exercised at the slow clock.
  MA-P5 behaviour: with the store on, no key recognition drops by more than 1 seed vs store off.

Usage: character_store_real.py   (conda env; writes character_store_real_summary.json)
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
import similarity_radius as SR
from src.hopfield.episodic_consolidating import GatedEpisodicMemory
from src.hopfield.two_layer import CONSTANTS

LINK = 0.8
WORLDS = [w for w in SR.WORLDS if w[0] in ("13mV v1b", "13mV v1c", "ov50", "ov70", "checkb")]


def run(s, gs, radius, store):
    m = GatedEpisodicMemory(dim=s['q'].shape[1], gap_scale=gs, theta=radius, match_floor=radius)
    mem = m.mem
    records, link, born, winners = [], {}, {}, []
    links = wrong = dup = 0
    for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
        o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
        true = int(s['true'][i])
        if o['created'] is not None:
            born[o['created']] = int(s['ph'][i])
            if store:
                k = mem.ids.index(o['created']); p = mem.patterns[k]
                sims = [float(r['pattern'] @ p) for r in records]
                if sims and max(sims) >= LINK:
                    ri = int(np.argmax(sims)); mem.w_char[k] = records[ri]['w_char']
                    links += 1; wrong += records[ri]['label'] != true
                else:
                    dup += any(r['label'] == true for r in records)
                    records.append(dict(pattern=p.clone(), w_char=1.0, label=true)); ri = len(records) - 1
                link[o['created']] = ri
        if store:
            alive = {}
            for k, e in enumerate(mem.ids):
                if e in link and (link[e] not in alive or mem.w_char[k] > mem.w_char[alive[link[e]]]):
                    alive[link[e]] = k
            for ri, r in enumerate(records):
                if ri in alive:
                    r['w_char'] = mem.w_char[alive[ri]]; r['pattern'] = mem.patterns[alive[ri]].clone()
                else:
                    r['w_char'] += CONSTANTS.decay_char * (1 - r['w_char'])
        winners.append(o['winner'])
    return dict(winners=winners, born=born, links=links, wrong=wrong, dup=dup, records=len(records),
                contexts=len(np.unique(s['true'])))


def main():
    out = {}
    for name, pat, returns in WORLDS:
        runs = R.load(pat)
        print(f"\n##### {name}")
        out[name] = {}
        for W in (10, 50):
            rng = np.random.default_rng(0); torch.manual_seed(0)
            streams = [R.build(d, wm, r, chg, W, "H+", rng) for d, wm, r, chg in runs]
            gs = RS.ground_gap_scale(streams)
            for radius in (0.5, 0.8):
                row = {}
                for store in (False, True):
                    res = [run(s, gs, radius, store) for s in streams]
                    rec = {lab: int(sum(RS.genuine(s, x, rp, op) > 0.5 for s, x in zip(streams, res)))
                           for rp, op, lab in returns}
                    row["store" if store else "no store"] = rec
                    if store:
                        row.update(links=int(sum(x['links'] for x in res)), wrong=int(sum(x['wrong'] for x in res)),
                                   wrong_seeds=int(sum(x['wrong'] > 0 for x in res)),
                                   dup=int(sum(x['dup'] for x in res)),
                                   records_per_context=float(np.mean([x['records'] / x['contexts'] for x in res])))
                out[name][f"W{W} r{radius}"] = row
                print(f"  W={W:2d} r={radius}  links {row['links']:3d}  WRONG {row['wrong']} ({row['wrong_seeds']}/8 seeds)  "
                      f"dup records {row['dup']}  records/context {row['records_per_context']:.2f} | recognition "
                      + "  ".join(f"{k}: {row['no store'][k]}->{row['store'][k]}" for k in row['store']))
    json.dump(out, open(HERE / "character_store_real_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

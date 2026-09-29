"""Anchored consolidation: can a memory's content sharpen but never wander away from what it was at
birth? Replay only, existing substrate runs, rectified readout, radius 0.8, both clocks.
Notebook-only memory variant; src untouched.

Problem (three sightings: ov70 at W=10, the long world, the dormant store's wrong links). Under
heavy overlap at the fast clock an entry's content slides, query by query, into a neighbouring
context. Each step passes the 0.8 floor against the entry's *current* content, so nothing stops a
chain. In the long world, 25% of settled D steps were won by an A-born entry, and the store
carried A's history onto D content.

Anchored rule (one change to content consolidation, nothing else): each entry remembers its birth
pattern p0. It consolidates toward a query only if (1) the query matches p0 at >= the radius, and
(2) the moved pattern still matches p0 at >= the radius (otherwise the move is skipped). Identity
is fixed at birth; content can only refine within the radius around it.

Arms: "r0.8" = adopted rule at radius 0.8 (the similarity_radius.py recommendation) vs "r0.8 anchored".
Worlds: the 7 of similarity_radius.py plus the long world (with the dormant store for wrong links).

PREDICTIONS ON RECORD (written before this script was first run):
  AN-P1 anchoring stops the slide: ov70 W=10 merged seeds <= 2/8 (r0.8: 6/8), absorbed 0 (r0.8: 6).
  AN-P2 long world W=10: settled D steps won by an A-born entry <= 5% (r0.8: 25%); dormant-store
        wrong links <= 1 (r0.8: 7).
  AN-P3 no cost elsewhere: every key recognition within 1 seed of r0.8, committed accuracy no more
        than 0.01 lower, in all other world/clock cells.
  AN-P4 small price: entries created per context occurrence rise by <= 0.1 in any cell (entries can't
        follow real drift, but the readout doesn't drift: 0.96 over 4 visits).

Usage: anchored_consolidation.py   (conda env; writes anchored_consolidation_summary.json)
"""
import functools
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
import long_world_readout as LW
from src.hopfield import episodic_consolidating as EC

RADIUS = 0.8


class AnchoredMemory(EC.ConsolidatingEpisodicMemory):
    def __init__(self, dim, anchor=RADIUS, **kwargs):
        super().__init__(dim, **kwargs)
        self.anchor = anchor
        self.birth = {}

    def add_pattern(self, vec, step):
        idx = super().add_pattern(vec, step)
        self.birth[self.ids[idx]] = vec.clone()
        return idx

    def consolidate(self, winner, query, g):
        p0 = self.birth[self.ids[winner]]
        if float(p0 @ query) < self.anchor:
            return 0.0
        old = self.patterns[winner].clone()
        rate = super().consolidate(winner, query, g)
        if rate > 0 and float(self.patterns[winner] @ p0) < self.anchor:
            self.patterns[winner] = old
            return 0.0
        return rate


class AnchoredGated(EC.GatedEpisodicMemory):
    def __init__(self, dim, gap_scale, theta=RADIUS, eta=EC.ADOPTED['eta'], gate_content=EC.ADOPTED['gate_content'],
                 match_floor=RADIUS, **kw):
        super().__init__(dim, gap_scale, theta=theta, eta=eta, gate_content=gate_content, match_floor=match_floor, **kw)
        self.mem = AnchoredMemory(dim, anchor=RADIUS, eta=eta, gate_content=gate_content, match_floor=match_floor,
                                  gap_scale=gap_scale, **kw)


ARMS = {"r0.8": functools.partial(EC.GatedEpisodicMemory, theta=RADIUS, match_floor=RADIUS),
        "r0.8 anchored": AnchoredGated}


def main():
    out = {}
    for name, pat, returns in SR.WORLDS:
        runs = R.load(pat)
        print(f"\n##### {name}")
        out[name] = {}
        for W in (10, 50):
            rng = np.random.default_rng(0); torch.manual_seed(0)
            streams = [R.build(d, wm, r, chg, W, "H+", rng) for d, wm, r, chg in runs]
            gs = RS.ground_gap_scale(streams)
            for arm, cls in ARMS.items():
                RS.GatedEpisodicMemory = cls
                res = [RS.run_arm(s, 'adopted', gs) for s in streams]
                sc = [RS.score(s, x) for s, x in zip(streams, res)]
                row = dict(acc=float(np.nanmean([x['acc_committed'] for x in sc])),
                           absorbed=int(sum(x['absorbed'] for x in res)),
                           merged_seeds=int(sum(x['merged'] > 0 for x in res)),
                           created_per_occ=float(np.mean([SR.created_per_occurrence(s, x) for s, x in zip(streams, res)])))
                for rp, op, label in returns:
                    row[label] = int(sum(RS.genuine(s, x, rp, op) > 0.5 for s, x in zip(streams, res)))
                out[name][f"W{W} {arm}"] = row
                print(f"  W={W:2d} {arm:14s} acc {row['acc']:.3f} abs {row['absorbed']} merged {row['merged_seeds']}/8 "
                      f"created/occ {row['created_per_occ']:.2f} | "
                      + " | ".join(f"{lab} {row[lab]}/8" for _, _, lab in returns))
    RS.GatedEpisodicMemory = EC.GatedEpisodicMemory

    print("\n##### long world")
    runs = R.load(LW.PAT)
    out['long'] = {}
    for W in (10, 50):
        rng = np.random.default_rng(0); torch.manual_seed(0)
        streams = [R.build(d, wm, r, chg, W, "H+", rng) for d, wm, r, chg in runs]
        gs = RS.ground_gap_scale(streams)
        for arm, cls in ARMS.items():
            RS.GatedEpisodicMemory = cls
            res = [RS.run_arm(s, 'adopted', gs) for s in streams]
            d_idx = [(s, x) for s, x in zip(streams, res)]
            num = den = 0
            for s, x in d_idx:
                ids = np.array(runs[0][0]['phase_corr_blocks'])
                for i in np.where((s['true'] == 3) & s['settled'])[0]:
                    den += 1
                    b = x['born'].get(x['winners'][i])
                    num += b is not None and ids[b] == 0
            LW.GatedEpisodicMemory = cls
            st = [LW.run(s, gs, True) for s in streams]
            row = dict(acc=float(np.nanmean([RS.score(s, x)['acc_committed'] for s, x in zip(streams, res)])),
                       absorbed=int(sum(x['absorbed'] for x in res)), D_won_by_A_born=num / den,
                       store_links=int(sum(x['links'] for x in st)), store_wrong=int(sum(x['wrong'] for x in st)),
                       created_per_occ=float(np.mean([SR.created_per_occurrence(s, x) for s, x in zip(streams, res)])))
            out['long'][f"W{W} {arm}"] = row
            print(f"  W={W:2d} {arm:14s} acc {row['acc']:.3f} abs {row['absorbed']} D steps won by A-born "
                  f"{row['D_won_by_A_born']:.1%} | store links {row['store_links']} WRONG {row['store_wrong']} | "
                  f"created/occ {row['created_per_occ']:.2f}")
    RS.GatedEpisodicMemory = EC.GatedEpisodicMemory
    json.dump(out, open(HERE / "anchored_consolidation_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

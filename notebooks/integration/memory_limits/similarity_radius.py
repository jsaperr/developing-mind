"""Memory's overlap limit and consolidation merging: is it the 0.5 threshold, and is there a working
radius? Replay only, on existing substrate runs, rectified readout (the default), adopted memory
(promoted GatedEpisodicMemory), both clocks.

Two open items share one number: memory's novelty threshold theta (create a new entry if the best
match is below it) and the consolidation match_floor (only move an entry's content toward a query
that matches at least this well). Both are 0.5. At 70% input overlap the contexts' prototype
cosine is 0.55, above theta, so memory merges them even on perfect input. At 50% overlap and the
10 s clock, content consolidation merges contexts (4/8), which is a query close enough to pass the
0.5 floor pulling an entry toward the other context.

Step 1 (geometry, no memory): settled H+ query cosines within a context (consecutive steady
queries) vs between contexts (phase means), per world.
Step 2: sweep a single similarity radius r, theta = match_floor = r, r in {0.5 (adopted), 0.6, 0.7,
0.8, 0.9}; plus two split arms at 0.8 (theta only, floor only) to see which knob does what.

PREDICTIONS ON RECORD (written before this script was first run):
  SR-P1 geometry: in ov70, H+ between-context cosine sits well below the within-context cosine
        (between <= 0.75, within >= 0.9). The substrate/readout separates the contexts; only the
        threshold doesn't.
  SR-P2 at r = 0.8, ov70 W=50: merged seeds <= 2/8 (adopted 0.5: 8/8) and B two-back genuine >= 5/8
        (0.5: 0/8). ov50 W=10 merged seeds <= 1/8 (0.5: 4/8).
  SR-P3 no cost in the other worlds at r = 0.8: every key recognition within 1 seed of r = 0.5,
        absorption 0, committed accuracy >= 0.95.
  SR-P4 the radius has a ceiling: at r = 0.9, duplicates appear (more entries created per context
        occurrence than at 0.8) somewhere, because 0.9 sits near the within-context spread.
  SR-P5 the split arms: theta alone fixes ov70 (the creation side) but not ov50 W=10 merging;
        the floor alone does the reverse.

Usage: similarity_radius.py   (conda env; writes similarity_radius_summary.json)
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(SW))
import readout_set_worlds as RS
import run_rectified_memory as R
from src.hopfield import episodic_consolidating as EC

B2 = R.B2
WORLDS = [
    ("13mV v1b", B2 / "v1b_schedule_data" / "v1b_n7_reliable_13_1p5_seed*.json.gz", [(3, 0, "A two-back"), (4, 1, "B two-back")]),
    ("13mV v1c", B2 / "v1c_schedule_data" / "v1c_n7_reliable_13_1p5_seed*.json.gz", [(5, 1, "B 2000s")]),
    ("stg v1b", B2 / "stg_replication_data" / "stg_v1b_n7_seed*.json.gz", [(3, 0, "A two-back"), (4, 1, "B two-back")]),
    ("stg v1c", B2 / "stg_replication_data" / "stg_v1c_n7_seed*.json.gz", [(5, 1, "B 2000s")]),
    ("ov50", B2 / "set_worlds_data" / "ov50_n7_seed*.json.gz", [(3, 0, "A two-back"), (4, 1, "B two-back")]),
    ("ov70", B2 / "set_worlds_data" / "ov70_n7_seed*.json.gz", [(3, 0, "A two-back"), (4, 1, "B two-back")]),
    ("checkb", B2 / "set_worlds_data" / "checkb_n7_seed*.json.gz", [(6, 0, "A return")]),
]
ARMS = [("r0.5", 0.5, 0.5), ("r0.6", 0.6, 0.6), ("r0.7", 0.7, 0.7), ("r0.8", 0.8, 0.8), ("r0.9", 0.9, 0.9),
        ("theta0.8 only", 0.8, 0.5), ("floor0.8 only", 0.5, 0.8)]


def geometry(streams):
    within, between = [], []
    for s in streams:
        q, ph, st = s['q'], s['ph'], s['settled']
        within += [q[i] @ q[i + 1] for i in range(len(q) - 1) if st[i] and st[i + 1] and ph[i] == ph[i + 1]]
        means = {}
        for p in np.unique(ph):
            m = q[(ph == p) & st]
            if len(m):
                means.setdefault(int(s['true'][ph == p][0]), []).append(RS.I.unit(m.mean(0)))
        ctx = {c: RS.I.unit(np.mean(v, 0)) for c, v in means.items()}
        cs = sorted(ctx)
        between += [ctx[a] @ ctx[b] for i, a in enumerate(cs) for b in cs[i + 1:]]
    return float(np.median(within)), float(np.percentile(within, 5)), float(np.mean(between)), float(np.max(between))


def created_per_occurrence(s, res):
    occ = len(np.unique(s['ph']))
    return len(res['born']) / occ


def main():
    out = {}
    for name, pat, returns in WORLDS:
        runs = R.load(pat)
        rng = np.random.default_rng(0)
        g = geometry([R.build(d, wm, r, chg, 10, "H+", rng) for d, wm, r, chg in runs])
        print(f"\n##### {name} ({len(runs)} seeds) geometry H+: within median {g[0]:.3f} (5th pct {g[1]:.3f}) | "
              f"between mean {g[2]:.3f} max {g[3]:.3f}")
        out[name] = {"geometry": dict(within_median=g[0], within_p5=g[1], between_mean=g[2], between_max=g[3])}
        for W in (10, 50):
            rng = np.random.default_rng(0); torch.manual_seed(0)
            streams = [R.build(d, wm, r, chg, W, "H+", rng) for d, wm, r, chg in runs]
            gs = RS.ground_gap_scale(streams)
            out[name][f"W{W}"] = {}
            for arm, theta, floor in ARMS:
                RS.GatedEpisodicMemory = functools.partial(EC.GatedEpisodicMemory, theta=theta, match_floor=floor)
                res = [RS.run_arm(s, 'adopted', gs) for s in streams]
                sc = [RS.score(s, x) for s, x in zip(streams, res)]
                row = dict(acc=float(np.nanmean([x['acc_committed'] for x in sc])),
                           absorbed=int(sum(x['absorbed'] for x in res)),
                           merged_seeds=int(sum(x['merged'] > 0 for x in res)),
                           created_per_occ=float(np.mean([created_per_occurrence(s, x) for s, x in zip(streams, res)])),
                           alive_end=float(np.mean([x['n_alive_end'] for x in res])))
                txt = []
                for rp, op, label in returns:
                    gn = int(sum(RS.genuine(s, x, rp, op) > 0.5 for s, x in zip(streams, res)))
                    row[label] = gn; txt.append(f"{label} {gn}/8")
                out[name][f"W{W}"][arm] = row
                print(f"  W={W:2d} {arm:14s} acc {row['acc']:.3f} abs {row['absorbed']} merged {row['merged_seeds']}/8 "
                      f"created/occ {row['created_per_occ']:.2f} alive-end {row['alive_end']:.1f} | " + " | ".join(txt))
    RS.GatedEpisodicMemory = EC.GatedEpisodicMemory
    json.dump(out, open(HERE / "similarity_radius_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

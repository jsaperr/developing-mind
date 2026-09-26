"""Stability-gated creation + content consolidation, combined (still the forked memory; src/ untouched).

Motivation, from two specific results:
  - Content consolidation alone (run_content_consolidation.py) fixes captured entries (two-back
    reports down) but NOT recognition. At a return, the first transitional query spawns a new
    entry before the query settles, and that duplicate wins the return phase (8/8 at A's return,
    W=10 s; 7/8 at W=50 s, where the original entry is still alive).
  - The stability gate alone (coupling_v0/run_creation_gates.py) delays creation, but a frozen
    original entry still doesn't match the settled return query, and the capture persists.
Combined: creation waits for a steady query (G-stable at the pre-registered TAU=0.9, unchanged),
and meanwhile the retrieved entry's content consolidates (gated, eta=0.1, the pre-registered
primary). The hypothesis is that during a return's first steps memory must retrieve an existing
entry, which is the original if still alive. Consolidation pulls it toward the returning query, so
by the time creation is allowed the query already matches it, and no duplicate forms.

Genuine recognition = the ORIGINAL-phase entry wins MORE THAN HALF the return phase. That rules
out the 2-step forced-retrieval artifact flagged in the creation-gates entry.

PREDICTIONS ON RECORD (written before this script was first run; substrate arm):
  COMBO-P1: genuine recognition rises where memory still holds the original entry:
            v1 W=50 s: A two-back >= 5/8 and C one-back >= 6/8 (from 1/8 and 3/8 with consolidation
            alone); A->B->A W=10 s A return >= 5/8 (0/8 with consolidation alone).
  COMBO-P2: two-back reports after novel swaps are no worse than consolidation alone
            (A->B->C W=10 s <= 77, v1 W=10 s <= 34).
  COMBO-P3: v1 W=10 s A two-back stays <= 2/8. The original is evicted first, and no creation
            rule can bring it back.

Usage: run_combined.py   (conda env; writes combined_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "coupling_v0"))
import run_coupling_v0 as T
import run_content_consolidation as C
from run_creation_gates import stable_allow

TAU = 0.9
ETA = 0.1


def main():
    v0 = json.load(open(HERE.parent / "coupling_v0" / "coupling_v0_summary.json"))
    out = {}
    for world, ddir, pat in T.WORLDS:
        runs = T.load_world(ddir, pat)
        for W in T.CLOCKS:
            key = f"{world}_W{W}"
            rng = np.random.default_rng(0); torch.manual_seed(0)
            streams = [T.build_streams(d, wm, r, n, bs, W, rng) for d, wm, r, n, bs in runs]
            gs = v0[key]['gap_scale']
            variants = {
                "consolidation only": lambda s: C.run_cc(s, 'substrate', gs, ETA, True),
                "stability gate only": lambda s: C.run_cc(s, 'substrate', gs, 0.0, True,
                                                          allow=stable_allow(s['q']['substrate'], TAU)),
                "combined": lambda s: C.run_cc(s, 'substrate', gs, ETA, True,
                                               allow=stable_allow(s['q']['substrate'], TAU)),
            }
            out[key] = {}
            print(f"\n===== {key} =====")
            for name, fn in variants.items():
                res = [fn(s) for s in streams]
                sc = [T.score(s, x) for s, x in zip(streams, res)]
                two_back = int(sum(sw['other'] for x in sc for sw in x['swaps'] if not sw['returning']))
                gen = {f"ret{rp}": int(sum(C.genuine(s, x, rp, op) > 0.5 for s, x in zip(streams, res)))
                       for rp, op in C.RETURNS.get(world, [])}
                row = dict(acc=float(np.mean([x['acc'] for x in sc])), two_back=two_back, genuine=gen,
                           created=float(np.mean([x['created'] for x in res])),
                           absorbed=int(sum(x['absorbed'] for x in res)))
                out[key][name] = row
                print(f"  {name:20s} acc {row['acc']:.3f}  two-back {two_back:4d}  created {row['created']:5.1f}  "
                      f"absorbed {row['absorbed']}  " + "  ".join(f"genuine {k}: {v}/8" for k, v in gen.items()))
    json.dump(out, open(HERE / "combined_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

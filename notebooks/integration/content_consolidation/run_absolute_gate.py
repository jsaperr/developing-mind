"""Absolute-match gate on content consolidation (forked memory; src/ untouched).

Follows the absorption finding (experiments_integration.md): the top1-top2 ambiguity gate is
relative and can't see novelty, so a query that matches nothing well was consolidated into the
best of bad matches at full rate. Fix under test: consolidate only if the winner's current content
has cosine >= MATCH_FLOOR with the query, and MATCH_FLOOR = THETA = 0.5. That's the SAME line the
creation rule already uses for "this query is novel", so no new parameter is introduced. Everything
else is as in run_combined.py (stability gate TAU=0.9, eta=0.1 gated by g, v0's grounded gap_scale).

Variants (substrate arm): combined (the reference, as before); combined + absolute gate;
consolidation only + absolute gate (no creation gate).

PREDICTIONS ON RECORD (written before this script was first run):
  AMG-P1: absorption goes to 0 for combined + absolute gate in every world and clock (combined
          alone had 1 / 6 / 3 at W=10 s).
  AMG-P2: the recognition gain survives: v1 C one-back genuine >= 7/8 at W=50 s and >= 5/8 at
          W=10 s.
  AMG-P3: first-context recognition in A->B->A (W=10 s) comes back to at least the gate-only level,
          >= 5/8 (combined alone: 1/8). The lone A entry is no longer consolidated toward B queries.
  AMG-P4: two-back reports after novel swaps are no worse than 1.5x combined. The captured
          "not-old" entries were captured precisely because later queries match them at >= THETA,
          so they can still consolidate and drift.
Everything else is descriptive, including v1 A two-back (first context + two back + eviction at
W=10 s).

Usage: run_absolute_gate.py   (conda env; writes absolute_gate_summary.json)
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

TAU, ETA = 0.9, 0.1
MATCH_FLOOR = T.THETA


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
            st = lambda s: stable_allow(s['q']['substrate'], TAU)
            variants = {
                "combined": lambda s: C.run_cc(s, 'substrate', gs, ETA, True, allow=st(s)),
                "combined + abs gate": lambda s: C.run_cc(s, 'substrate', gs, ETA, True, allow=st(s),
                                                          match_floor=MATCH_FLOOR),
                "consol only + abs gate": lambda s: C.run_cc(s, 'substrate', gs, ETA, True,
                                                             match_floor=MATCH_FLOOR),
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
                print(f"  {name:24s} acc {row['acc']:.3f}  two-back {two_back:4d}  created {row['created']:5.1f}  "
                      f"absorbed {row['absorbed']}  " + "  ".join(f"genuine {k}: {v}/8" for k, v in gen.items()))
    json.dump(out, open(HERE / "absolute_gate_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

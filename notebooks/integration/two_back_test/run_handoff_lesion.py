"""Causal test of the substrate -> memory "handoff" (rehearsal at transitions).

Observation (v1b, best configuration, W=10 s; see the two-back entry in experiments_integration.md):
B's original memory survives a 2000 s lull, past memory's nominal 150-step (1500 s) horizon, in
6/8 seeds. All 33 of its lull-period wins fall within 100 s after a world change, and survival
matches exactly (8/8) whether it was reactivated at the C->A change. The proposed mechanism: when
the substrate is about to release its one-back hold (B), its query briefly points at B, and memory,
blocked from creating by the stability gate, retrieves and so refreshes B's entry.

Lesion: identical run, except that wins on TRANSITIONAL steps (where the stability gate is closed,
i.e. the query isn't steady) don't reset the winner's staleness. Label-free: it uses the same
gate that decides creation. Settled-step wins still refresh normally.

PREDICTIONS ON RECORD (written before this script was first run):
  HO-P1: W=10 s: B's original entry alive at its return drops from 6/8 to <= 1/8, and genuine
         recognition of B from 6/8 to <= 1/8.
  HO-P2: W=50 s (horizon 7500 s, no eviction pressure): B alive stays 8/8 and genuine recognition
         stays >= 7/8, so the lesion only matters where eviction is in play.
  HO-P3: the lesion doesn't change accuracy by more than 0.02 at either clock. It only changes
         staleness bookkeeping, not what's retrieved.

Usage: run_handoff_lesion.py   (conda env; writes handoff_lesion_summary.json)
"""
import json
from pathlib import Path

import numpy as np
import torch

import run_two_back as TB

T, C = TB.T, TB.C
HERE = Path(__file__).resolve().parent


def main():
    runs = T.load_world(TB.DDIR, TB.PATTERN)
    out = {}
    for W in T.CLOCKS:
        rng = np.random.default_rng(0); torch.manual_seed(0)
        streams = [T.build_streams(d, wm, r, n, bs, W, rng) for d, wm, r, n, bs in runs]
        g1 = np.concatenate([T.run_memory(s, "substrate", T.OLD_GAP_SCALE, T.THETA)['gaps'] for s in streams])
        gs = 0.6 * float(np.median(g1))
        out[f"W{W}"] = {}
        print(f"\n===== v1b W={W} s (gap_scale {gs:.4f}) =====")
        for name, lesion in (("best", False), ("best + handoff lesion", True)):
            res = []
            for s in streams:
                allow = TB.stable_allow(s['q']['substrate'], TB.TAU)
                res.append(C.run_cc(s, 'substrate', gs, TB.ETA, True, allow=allow, match_floor=T.THETA,
                                    no_refresh=(lambda a: (lambda i: not a(i)))(allow) if lesion else None))
            sc = [T.score(s, x) for s, x in zip(streams, res)]
            row = dict(acc=float(np.mean([x['acc'] for x in sc])))
            line = f"  {name:22s} acc {row['acc']:.3f}"
            for rp, op, label in TB.RETURNS:
                gen = int(sum(C.genuine(s, x, rp, op) > 0.5 for s, x in zip(streams, res)))
                alive = int(sum(TB.alive_at(x, s, rp, op) for s, x in zip(streams, res)))
                row[label] = dict(genuine=gen, alive=alive)
                line += f"  | {label}: alive {alive}/8, genuine {gen}/8"
            out[f"W{W}"][name] = row
            print(line)
    json.dump(out, open(HERE / "handoff_lesion_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

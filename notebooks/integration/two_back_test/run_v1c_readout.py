"""Readout for v1c (A->B->C->A->C->B, seeds 35000-35007): the handoff horizon rule.

Rule under test (from the v1b handoff lesion): memory has to bridge only the time from the
substrate RELEASING a context to the context's return, because the release refreshes the context's
memory entry. In v1b, B's release-to-return was 1000 s and B survived at W=10 s (6/8). In v1c, B is
released at C->A and returns only after A and C, so release-to-return is 2000 s, beyond the
1500 s horizon at W=10 s. Nothing in between should refresh B: during phase A the substrate holds
C, and the A->C change returns C, which is held, not B.

Same analysis code as run_two_back.py (best configuration, v0, clean), plus the handoff lesion.

PREDICTIONS ON RECORD (written while the v1c batch was running, before any v1c result existed):
  V1C-P1: substrate one-back: at the final swap (C->B), holders of B average <= 1 per seed.
  V1C-P2 (the rule): best configuration, W=10 s: B's original entry alive at its return <= 2/8,
          genuine recognition <= 2/8. That's in contrast to v1b's 6/8, and the only difference is
          the release-to-return interval.
  V1C-P3: best configuration, W=50 s (horizon 7500 s): B alive 8/8, genuine >= 6/8.
  V1C-P4: at W=10 s the handoff lesion makes little difference for B (<= 1 seed change), because by
          the rule the refresh at release happens too early to save it anyway.
Descriptive: A (first context, two back, at swap 3) and C (one back, at swap 4).

Usage: run_v1c_readout.py   (conda env; writes v1c_readout_summary.json)
"""
import json
from pathlib import Path

import numpy as np
import torch

import run_two_back as TB

T, C = TB.T, TB.C
HERE = Path(__file__).resolve().parent
DDIR = TB.ROOT / "notebooks" / "brian2" / "v1c_schedule_data"
PATTERN = "v1c_n7_reliable_13_1p5_seed*.json.gz"
RETURNS = [(3, 0, "A two-back (first ctx)"), (4, 2, "C one-back"), (5, 1, "B release-to-return 2000 s")]


def main():
    runs = T.load_world(DDIR, PATTERN)
    print(f"v1c: {len(runs)} completed seeds")
    TB.substrate_part(runs)
    out = {}
    for W in T.CLOCKS:
        rng = np.random.default_rng(0); torch.manual_seed(0)
        streams = [T.build_streams(d, wm, r, n, bs, W, rng) for d, wm, r, n, bs in runs]
        g1 = np.concatenate([T.run_memory(s, "substrate", T.OLD_GAP_SCALE, T.THETA)['gaps'] for s in streams])
        gs = 0.6 * float(np.median(g1))
        best = lambda s, lesion=False: C.run_cc(
            s, 'substrate', gs, TB.ETA, True, allow=TB.stable_allow(s['q']['substrate'], TB.TAU), match_floor=T.THETA,
            no_refresh=(lambda a: (lambda i: not a(i)))(TB.stable_allow(s['q']['substrate'], TB.TAU)) if lesion else None)
        variants = {"clean": lambda s: C.run_cc(s, 'clean', gs, 0.0, True),
                    "substrate v0": lambda s: C.run_cc(s, 'substrate', gs, 0.0, True),
                    "substrate best": lambda s: best(s),
                    "best + handoff lesion": lambda s: best(s, True)}
        print(f"\nmemory replay, W={W} s (gap_scale {gs:.4f}; horizon {150 * W} s)")
        out[f"W{W}"] = {}
        for name, fn in variants.items():
            res = [fn(s) for s in streams]
            sc = [T.score(s, x) for s, x in zip(streams, res)]
            row = {"acc": float(np.mean([x['acc'] for x in sc]))}
            line = f"  {name:22s} acc {row['acc']:.3f}"
            for rp, op, label in RETURNS:
                gen = int(sum(C.genuine(s, x, rp, op) > 0.5 for s, x in zip(streams, res)))
                alive = int(sum(TB.alive_at(x, s, rp, op) for s, x in zip(streams, res)))
                row[label] = dict(genuine=gen, alive=alive)
                line += f"\n      {label}: alive {alive}/8, genuine {gen}/8"
            out[f"W{W}"][name] = row
            print(line)
    json.dump(out, open(HERE / "v1c_readout_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

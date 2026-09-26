"""Is the two-back pass a knife-edge? One-at-a-time neighbours of the best configuration on v1b.

Best config: stability gate TAU=0.9, gated content consolidation ETA=0.1, absolute floor = THETA
(0.5). Varied one at a time: TAU in {0.8, 0.95}, ETA in {0.03, 0.3}, floor in {0.4, 0.6}. Creation
THETA stays 0.5 throughout. The floor is varied separately, since it was defined as THETA.

PREDICTIONS ON RECORD (written before this script was first run):
  SENS-P1: genuine recognition of B (two back, not first) at W=50 s stays >= 6/8 at every neighbour.
  SENS-P2: at W=10 s, survival of B (which depends on the handoff) varies more across neighbours
           than at W=50 s. No bar; descriptive.

Usage: run_two_back_sensitivity.py   (conda env; writes two_back_sensitivity_summary.json)
"""
import json
from pathlib import Path

import numpy as np
import torch

import run_two_back as TB

T, C = TB.T, TB.C
HERE = Path(__file__).resolve().parent
NEIGHBOURS = [("best", 0.9, 0.1, 0.5), ("tau 0.8", 0.8, 0.1, 0.5), ("tau 0.95", 0.95, 0.1, 0.5),
              ("eta 0.03", 0.9, 0.03, 0.5), ("eta 0.3", 0.9, 0.3, 0.5),
              ("floor 0.4", 0.9, 0.1, 0.4), ("floor 0.6", 0.9, 0.1, 0.6)]


def main():
    runs = T.load_world(TB.DDIR, TB.PATTERN)
    out = {}
    for W in T.CLOCKS:
        rng = np.random.default_rng(0); torch.manual_seed(0)
        streams = [T.build_streams(d, wm, r, n, bs, W, rng) for d, wm, r, n, bs in runs]
        g1 = np.concatenate([T.run_memory(s, "substrate", T.OLD_GAP_SCALE, T.THETA)['gaps'] for s in streams])
        gs = 0.6 * float(np.median(g1))
        print(f"\n===== v1b W={W} s =====")
        out[f"W{W}"] = {}
        for name, tau, eta, floor in NEIGHBOURS:
            res = [C.run_cc(s, 'substrate', gs, eta, True, allow=TB.stable_allow(s['q']['substrate'], tau),
                            match_floor=floor) for s in streams]
            sc = [T.score(s, x) for s, x in zip(streams, res)]
            row = {"acc": float(np.mean([x['acc'] for x in sc])), "absorbed": int(sum(x['absorbed'] for x in res))}
            line = f"  {name:10s} acc {row['acc']:.3f} absorbed {row['absorbed']}"
            for rp, op, label in TB.RETURNS:
                gen = int(sum(C.genuine(s, x, rp, op) > 0.5 for s, x in zip(streams, res)))
                alive = int(sum(TB.alive_at(x, s, rp, op) for s, x in zip(streams, res)))
                row[label] = dict(genuine=gen, alive=alive)
                line += f"  | {label.split(' (')[0]}: alive {alive}/8 genuine {gen}/8"
            out[f"W{W}"][name] = row
            print(line)
    json.dump(out, open(HERE / "two_back_sensitivity_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

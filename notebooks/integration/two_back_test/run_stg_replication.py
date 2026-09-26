"""Replication at strong_tight_gate (10mV/1.0): do one-back retention, the two-back pass, the
handoff and the horizon rule hold at the second operating point?

Data: notebooks/brian2/stg_replication_data/ (v1b schedule, seeds 36000-36007; v1c schedule,
seeds 37000-37007). Same 30-input rig, N=7, calibration-checked. Analysis code is identical to
run_two_back.py / run_handoff_lesion.py / run_v1c_readout.py (best configuration = stability gate
TAU 0.9 + gated content consolidation eta 0.1 + absolute floor THETA). gap_scale is re-grounded per
world and clock by the v0 procedure.

Why it's a hard test: strong_tight_gate is bistable at N=3 (about half the seeds never
differentiate) and the one point with genuine ongoing identity churn. Known risk, stated up front:
if some seeds don't differentiate, the contrast readout degrades for a reason that has nothing to
do with memory. So readout quality per phase (settled query cosine to its true prototype) is
reported alongside, to keep a failure attributable.

PREDICTIONS ON RECORD (written while the batch was running, before any result existed):
  STG-P1 (one-back replicates): holders of the incoming pattern average <= 1 per seed at every
         two-back return (v1b swaps 3 and 4; v1c swaps 3 and 5), and >= 2 at v1c's one-back return
         (swap 4, A->C).
  STG-P2 (two-back pass replicates; same decision rule): v1b, best config, W=50 s, B genuine >= 5/8.
  STG-P3 (handoff replicates): v1b, W=10 s, best-config B survival >= 4/8, and the handoff lesion
         lowers it by >= 3 seeds.
  STG-P4 (horizon rule replicates): v1c, best config, B (release-to-return 2000 s) survival <= 2/8
         at W=10 s and genuine >= 6/8 at W=50 s.

Usage: run_stg_replication.py   (conda env; writes stg_replication_summary.json)
"""
import json
from pathlib import Path

import numpy as np
import torch

import run_two_back as TB

T, C = TB.T, TB.C
HERE = Path(__file__).resolve().parent
DDIR = TB.ROOT / "notebooks" / "brian2" / "stg_replication_data"
WORLDS = {
    "v1b": ("stg_v1b_n7_seed*.json.gz", [(3, 0, "A two-back (first ctx)"), (4, 1, "B two-back (not first)")]),
    "v1c": ("stg_v1c_n7_seed*.json.gz", [(3, 0, "A two-back (first ctx)"), (4, 2, "C one-back"),
                                         (5, 1, "B release-to-return 2000 s")]),
}


def readout_quality(streams):
    nph = max(s['ph'].max() for s in streams) + 1
    per = np.array([[np.mean([s['q']['substrate'][i] @ s['protos'][s['true'][i]]
                              for i in range(len(s['ph'])) if s['ph'][i] == p and s['settled'][i]])
                     for p in range(nph)] for s in streams])
    return per


def main():
    out = {}
    for world, (pattern, returns) in WORLDS.items():
        runs = T.load_world(DDIR, pattern)
        print(f"\n################ strong_tight_gate {world}: {len(runs)} completed seeds")
        if not runs:
            continue
        TB.substrate_part(runs)
        out[world] = {}
        for W in T.CLOCKS:
            rng = np.random.default_rng(0); torch.manual_seed(0)
            streams = [T.build_streams(d, wm, r, n, bs, W, rng) for d, wm, r, n, bs in runs]
            if W == 10:
                q = readout_quality(streams)
                print("readout quality (settled query cos to true prototype) by phase, mean [min..max]: " +
                      "  ".join(f"ph{p + 1} {q[:, p].mean():.2f} [{q[:, p].min():.2f}..{q[:, p].max():.2f}]"
                                for p in range(q.shape[1])))
                out[world]['readout_quality'] = q.tolist()
            g1 = np.concatenate([T.run_memory(s, "substrate", T.OLD_GAP_SCALE, T.THETA)['gaps'] for s in streams])
            gs = 0.6 * float(np.median(g1))
            st = lambda s: TB.stable_allow(s['q']['substrate'], TB.TAU)
            variants = {
                "clean": lambda s: C.run_cc(s, 'clean', gs, 0.0, True),
                "substrate v0": lambda s: C.run_cc(s, 'substrate', gs, 0.0, True),
                "substrate best": lambda s: C.run_cc(s, 'substrate', gs, TB.ETA, True, allow=st(s), match_floor=T.THETA),
                "best + handoff lesion": lambda s: C.run_cc(s, 'substrate', gs, TB.ETA, True, allow=st(s),
                                                            match_floor=T.THETA,
                                                            no_refresh=(lambda a: (lambda i: not a(i)))(st(s))),
            }
            print(f"memory replay W={W} s (gap_scale {gs:.4f}, horizon {150 * W} s)")
            out[world][f"W{W}"] = {}
            for name, fn in variants.items():
                res = [fn(s) for s in streams]
                sc = [T.score(s, x) for s, x in zip(streams, res)]
                row = {"acc": float(np.mean([x['acc'] for x in sc])), "absorbed": int(sum(x['absorbed'] for x in res))}
                line = f"  {name:22s} acc {row['acc']:.3f} absorbed {row['absorbed']}"
                for rp, op, label in returns:
                    gen = int(sum(C.genuine(s, x, rp, op) > 0.5 for s, x in zip(streams, res)))
                    alive = int(sum(TB.alive_at(x, s, rp, op) for s, x in zip(streams, res)))
                    row[label] = dict(genuine=gen, alive=alive)
                    line += f"\n      {label}: alive {alive}/8, genuine {gen}/8"
                out[world][f"W{W}"][name] = row
                print(line)
    json.dump(out, open(HERE / "stg_replication_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

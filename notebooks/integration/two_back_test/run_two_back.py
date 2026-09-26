"""The clean two-back test of the complementary-systems split (v1b world A->B->C->A->B).

The architecture's central bet (principles.md, named finding): the substrate holds only ONE context
back, so memory must carry anything older. v1 could only test this on A, the first context, which
is weakly represented for its own reasons (arc 07) and, at a 10 s clock, evicted before it returns.
Here the final return (swap 4, A->B) is to B, which is two back (C and A intervened) and NOT the
first context. It's analysis only, over notebooks/brian2/v1b_schedule_data (seeds 34000-34007).

Part 1, substrate: at each swap, how many neurons already hold the incoming pattern, and do they
speed up (same definitions as brian2/v1_schedule_data/analyze_v1.py).
Part 2, memory replay, W in {10, 50} s. gap_scale is grounded by v0's procedure on this world
(pass 1, substrate arm, 0.6 x the top1-top2 gap median).
  clean            v0 memory, clean-prototype queries (the clock control)
  substrate v0     v0 memory, substrate queries (THETA creation, no gates, no consolidation)
  substrate best   forked memory: stability-gated creation (TAU 0.9) + ambiguity-gated content
                   consolidation (eta 0.1) + absolute floor (THETA). The best configuration so far
                   (experiments_integration.md).
Genuine recognition = the entry born in the returning context's ORIGINAL phase wins more than half
of the return phase. Also reported: whether that original entry is still alive at the swap, which
separates "evicted" from "alive but not retrieved".

PREDICTIONS ON RECORD (written before any v1b data existed):
  TB-P1 (substrate is one-back): holders of the incoming pattern average <= 1 per seed at swap 4
        (B, two back) and at swap 3 (A, two back); only lock-in residue.
  TB-P2 (clock control): clean arm, genuine recognition of B at swap 4 is 8/8 at W=50 s and <= 1/8
        at W=10 s (B is absent 2000 s, memory's horizon is 150 steps = 1500 s at 10 s).
  TB-P3 (THE QUESTION): substrate best, W=50 s, genuine recognition of B at swap 4 >= 5/8.
  TB-P4: substrate best beats substrate v0 on that measure at W=50 s.
DECISION RULE (stated in advance):
  >= 5/8 -> the split works in principle; the next step is Jasper's Q2 requirement (how many
            contexts back, over how long), which sets clock x staleness.
  <= 3/8 -> something deeper is wrong with substrate-fed memory; diagnose that before deciding
            Q2 or Q4.
     4/8 -> ambiguous; report as such, no conclusion.

Usage: run_two_back.py   (conda env; writes two_back_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "coupling_v0"))
sys.path.insert(0, str(HERE.parent / "content_consolidation"))
sys.path.insert(0, str(ROOT / "notebooks" / "brian2" / "v1_schedule_data"))
import run_coupling_v0 as T
import run_content_consolidation as C
from run_creation_gates import stable_allow
from analyze_v1 import fav_at, prep

DDIR = ROOT / "notebooks" / "brian2" / "v1b_schedule_data"
PATTERN = "v1b_n7_reliable_13_1p5_seed*.json.gz"
RETURNS = [(3, 0, "A two-back (first context)"), (4, 1, "B two-back (not first)")]
TAU, ETA = 0.9, 0.1


def substrate_part(runs):
    print("PART 1, substrate: holders of the incoming pattern at each swap (per seed), and their first-10 s rate change")
    blocks = runs[0][0]['phase_corr_blocks']
    for k in range(len(blocks) - 1):
        hold, speed = [], []
        for d, *_ in runs:
            _, bm, r, t = prep(d)
            s = int(d['swap_times_s'][k]); inc = blocks[k + 1]
            h = [j for j in range(d['n_post']) if fav_at(bm, t, s)[j] == inc]
            hold.append(len(h))
            if h:
                speed.append(np.mean([r[j, s:s + 10].mean() - r[j, s - 100:s].mean() for j in h]))
        print(f"  swap {k + 1} ({T.NAMES[blocks[k]]}->{T.NAMES[blocks[k + 1]]}): holders {hold} mean {np.mean(hold):.2f}"
              + (f"  holder rate change mean {np.mean(speed):+.2f} Hz (n={len(speed)} seeds)" if speed else ""))


def alive_at(res, stream, ret_phase, orig_phase):
    """Was any entry born in orig_phase still in memory at the last step before the return swap?
    Exact: read from run_cc's per-step alive-id snapshots (after that step's prune)."""
    i0 = int(np.where(stream['ph'] == ret_phase)[0][0])
    return any(res['born_phase'][x] == orig_phase for x in res['alive_ids'][i0 - 1])


def main():
    rows = [T.load_world(DDIR, PATTERN)][0]
    if not rows:
        print("no completed v1b runs"); return
    runs = [(d, wm, r, n, bs) for d, wm, r, n, bs in rows]
    print(f"v1b: {len(runs)} completed seeds")
    substrate_part(runs)
    out = {}
    for W in T.CLOCKS:
        rng = np.random.default_rng(0); torch.manual_seed(0)
        streams = [T.build_streams(d, wm, r, n, bs, W, rng) for d, wm, r, n, bs in runs]
        g1 = np.concatenate([T.run_memory(s, "substrate", T.OLD_GAP_SCALE, T.THETA)['gaps'] for s in streams])
        gs = 0.6 * float(np.median(g1))
        variants = {
            "clean": lambda s: C.run_cc(s, 'clean', gs, 0.0, True),
            "substrate v0": lambda s: C.run_cc(s, 'substrate', gs, 0.0, True),
            "substrate best": lambda s: C.run_cc(s, 'substrate', gs, ETA, True,
                                                 allow=stable_allow(s['q']['substrate'], TAU), match_floor=T.THETA),
        }
        print(f"\nPART 2, memory replay, W={W} s (grounded gap_scale {gs:.4f}; horizon 150 steps = {150 * W} s)")
        out[f"W{W}"] = {"gap_scale": gs}
        for name, fn in variants.items():
            res = [fn(s) for s in streams]
            sc = [T.score(s, x) for s, x in zip(streams, res)]
            line = f"  {name:15s} acc {np.mean([x['acc'] for x in sc]):.3f}  absorbed {sum(x['absorbed'] for x in res)}"
            row = {}
            for rp, op, label in RETURNS:
                gen = [C.genuine(s, x, rp, op) for s, x in zip(streams, res)]
                alive = [alive_at(x, s, rp, op) for s, x in zip(streams, res)]
                lat = [x['swaps'][rp - 1]['latency'] for x in sc]
                row[label] = dict(genuine=int(sum(v > 0.5 for v in gen)), alive=int(sum(alive)), latency=lat)
                line += (f"\n      {label}: genuine {row[label]['genuine']}/8  original entry alive at swap "
                         f"{row[label]['alive']}/8  latency {lat}")
            out[f"W{W}"][name] = row
            print(line)
    json.dump(out, open(HERE / "two_back_summary.json", "w"), indent=1, default=str)


if __name__ == '__main__':
    main()

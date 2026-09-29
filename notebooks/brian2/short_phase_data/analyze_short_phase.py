"""Scores SP-P1/SP-P2 (predictions in run_short_seed.py, written before launch) on the short-phase arm
(300 s phases, disjoint, v1b schedule A B C A B), with the same holder metric as step 0
(set_worlds_data/analyze_release_timing.py). Prints the who-holds-what table next to v1b's (1000 s)
for comparison.

Usage: analyze_short_phase.py   (conda env)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "notebooks" / "brian2" / "set_worlds_data"))
import analyze_release_timing as RT

R = RT.R


def table(runs, label):
    d0 = runs[0][0]
    print(f"\n{label}: mean holders per context (A B C)")
    rows = {}
    for k, st in enumerate(d0['swap_times_s']):
        st = int(st)
        for off, lab in ((0, "before"), (150, "+150")):
            t = st + off
            v = [np.mean([RT.holders(d, wm, c, t) for d, wm, r, chg in runs]) for c in range(3)]
            rows[(k + 1, lab)] = v
            print(f"  swap {k + 1} ->{'ABC'[d0['phase_corr_blocks'][k + 1]]} {lab:7s} " + "  ".join(f"{x:4.2f}" for x in v))
    end = int(d0['total_s'])
    v = [np.mean([RT.holders(d, wm, c, end) for d, wm, r, chg in runs]) for c in range(3)]
    rows[('end', '')] = v
    print(f"  end of run         " + "  ".join(f"{x:4.2f}" for x in v))
    return rows


def main():
    sp = R.load(HERE / "short300_n7_seed*.json.gz")
    vb = R.load(R.B2 / "v1b_schedule_data" / "v1b_n7_reliable_13_1p5_seed*.json.gz")
    print(f"short-phase seeds: {len(sp)}")
    t = table(sp, "SHORT 300 s disjoint")
    table(vb, "v1b 1000 s disjoint (reference)")
    a_after = t[(2, "+150")][0]
    b_keep = t[(2, "+150")][1] / t[(2, "before")][1] if t[(2, "before")][1] else float('nan')
    inc3, inc4 = t[(3, "before")][0], t[(4, "before")][1]
    print(f"\nSP-P1: at C's arrival, A holders at +150 = {a_after:.2f} (<= 0.5?), B keeps {b_keep:.0%} (>= 60%?); "
          f"incoming holders just before two-back returns: swap 3 (A) {inc3:.2f}, swap 4 (B) {inc4:.2f} (<= 1?)")
    d0 = sp[0][0]; ends = [int(x) for x in d0['swap_times_s']] + [int(d0['total_s'])]
    own = [np.mean([RT.holders(d, wm, d0['phase_corr_blocks'][p], ends[p]) for d, wm, r, chg in sp])
           for p in range(len(ends))]
    print("SP-P2: own-context holders at the end of each phase: " + "  ".join(f"{x:.2f}" for x in own) + "  (all >= 4?)")


if __name__ == '__main__':
    main()

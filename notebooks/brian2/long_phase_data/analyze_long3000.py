"""Scores LP-P1/LP-P2 (predictions in run_long3000_seed.py, written before launch) on the 3000 s disjoint
arm, with step 0's holder metric (set_worlds_data/analyze_release_timing.py). Prints the who-holds-what
table next to the 300 s and 1000 s disjoint arms for comparison.

Usage: analyze_long3000.py   (conda env)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "notebooks" / "brian2" / "set_worlds_data"))
import analyze_release_timing as RT

R = RT.R


def holders(runs, c, t):
    return float(np.mean([RT.holders(d, wm, c, t) for d, wm, r, chg in runs]))


def table(runs, label):
    d0 = runs[0][0]
    print(f"\n{label}: mean holders per context (A B C)")
    rows = {}
    for k, st in enumerate(d0['swap_times_s']):
        st = int(st)
        for off, lab in ((0, "before"), (150, "+150"), (300, "+300")):
            v = [holders(runs, c, st + off) for c in range(3)]
            rows[(k + 1, lab)] = v
            print(f"  swap {k + 1} ->{'ABC'[d0['phase_corr_blocks'][k + 1]]} {lab:7s} " + "  ".join(f"{x:4.2f}" for x in v))
    return rows


def main():
    lp = R.load(HERE / "long3000_n7_seed*.json.gz")
    print(f"3000 s seeds: {len(lp)}")
    t = table(lp, "3000 s disjoint")
    table(R.load(R.B2 / "v1b_schedule_data" / "v1b_n7_reliable_13_1p5_seed*.json.gz"), "1000 s disjoint (v1b, reference)")
    table(R.load(R.B2 / "short_phase_data" / "short300_n7_seed*.json.gz"), "300 s disjoint (reference)")

    b_keep = t[(2, "+150")][1] / t[(2, "before")][1] if t[(2, "before")][1] else float('nan')
    print(f"\nLP-P1: at C's arrival B keeps {b_keep:.0%} at +150 (>= 60%?); A holders at +300 = {t[(2, '+300')][0]:.2f} (<= 1?); "
          f"incoming just before two-back returns: swap 3 (A) {t[(3, 'before')][0]:.2f}, swap 4 (B) {t[(4, 'before')][1]:.2f} (<= 1?)")
    d0 = lp[0][0]; sw = [int(x) for x in d0['swap_times_s']] + [int(d0['total_s'])]
    worst = 0.0
    for k in range(len(sw) - 1):
        a, b = sw[k] + 300, sw[k + 1]
        for c in range(3):
            worst = max(worst, abs(holders(lp, c, b) - holders(lp, c, a)))
    print(f"LP-P2: largest change in any context's mean holders between +300 s after a swap and the next swap: {worst:.2f} (<= 0.5?)")


if __name__ == '__main__':
    main()

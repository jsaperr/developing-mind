"""Scores SO-P1/SO-P2 (predictions in run_short_ov50_seed.py, written before launch) on ov50 at 300 s
(seeds 44000-44007) AND on the existing ov50 at 1000 s (set_worlds_data, seeds 38000-38007), with step
0's holder metric (nearest of the world's prototypes, 50 s mean weights). Disjoint references are
printed alongside (short 300 s and v1b 1000 s).

Usage: analyze_short_ov50.py   (conda env)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "notebooks" / "brian2" / "set_worlds_data"))
import analyze_release_timing as RT

R = RT.R


def score(runs, label):
    d0 = runs[0][0]; ids = d0['phase_corr_blocks']
    inc, keep = [], []
    line = f"{label:24s}"
    for k in (0, 1):
        st = int(d0['swap_times_s'][k])
        new, dep = ids[k + 1], ids[k]
        before = np.mean([RT.holders(d, wm, dep, st) for d, wm, r, c in runs])
        after_new = np.mean([RT.holders(d, wm, new, st + 150) for d, wm, r, c in runs])
        inc.append(after_new)
        line += f" | swap {k + 1}: incoming holds {after_new:.2f} at +150"
        if k == 1:
            before_dep = before
            after_dep = np.mean([RT.holders(d, wm, dep, st + 150) for d, wm, r, c in runs])
            keep.append(after_dep / before_dep if before_dep else float('nan'))
            line += f", just-departed keeps {keep[-1]:.0%} ({before_dep:.2f} -> {after_dep:.2f})"
    print(line)
    return inc, keep


def main():
    so = R.load(HERE / "short300ov50_n7_seed*.json.gz")
    lo = R.load(R.B2 / "set_worlds_data" / "ov50_n7_seed*.json.gz")
    sd = R.load(HERE / "short300_n7_seed*.json.gz")
    vb = R.load(R.B2 / "v1b_schedule_data" / "v1b_n7_reliable_13_1p5_seed*.json.gz")
    print(f"seeds: ov50 300 s {len(so)}, ov50 1000 s {len(lo)}, disjoint 300 s {len(sd)}, disjoint 1000 s {len(vb)}")
    res = {}
    for runs, label in ((so, "ov50, 300 s"), (lo, "ov50, 1000 s"), (sd, "disjoint, 300 s (ref)"), (vb, "disjoint, 1000 s (ref)")):
        res[label] = score(runs, label)
    for label in ("ov50, 300 s", "ov50, 1000 s"):
        inc, keep = res[label]
        print(f"{label}: SO-P1 incoming >= 5 at both swaps: {all(x >= 5 for x in inc)}; "
              f"SO-P2 just-departed keeps < 60%: {keep[0] < 0.6}")


if __name__ == '__main__':
    main()

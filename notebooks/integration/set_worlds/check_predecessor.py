"""Is the readout predecessor-dependent? Does the SAME context read the same when it arrives after a
DIFFERENT context? Analysis only, on existing runs, for both readouts (contrast H and rectified H+).

Motivation: H = sum_j (r_j - mean r) w_j subtracts what the quieter neurons (the previous context's
retainers) are tuned to, so it could encode "what's different from the last context". That would
make one context read differently depending on its predecessor, and memory could fail to recognize
it after an unfamiliar transition. ov50's positive result couldn't expose this, because B returned
after the same predecessor (A) it first followed.

Pairs (settled phase-mean readouts of two occurrences of one context, cosine per seed):
  different predecessor: 13mV v1 C3 (after B) vs C5 (after A); 13mV v1c C3 (after B) vs C5 (after A)
                         and B2 (after A) vs B6 (after C); stg v1c, the same two; ov50 A1 (first)
                         vs A4 (after C), which is also a cold-start pair for H
  same predecessor (control): ov50 B2 (after A) vs B5 (after A)

PREDICTIONS ON RECORD (written before this script was first run):
  PD-P1 H+: same-context similarity >= 0.9 (seed mean) for every different-predecessor pair.
  PD-P2 H is predecessor-dependent: its different-predecessor pairs sit below its same-predecessor
        control (ov50 B2-B5), while H+ shows no such gap (difference <= 0.05).

Usage: check_predecessor.py   (conda env)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_rectified_memory as R

B2 = R.B2
CASES = [
    ("13mV v1", B2 / "v1_schedule_data" / "v1_n7_reliable_13_1p5_seed*.json.gz", [((2, 4), "C after B vs after A", "diff")]),
    ("13mV v1c", B2 / "v1c_schedule_data" / "v1c_n7_reliable_13_1p5_seed*.json.gz",
     [((2, 4), "C after B vs after A", "diff"), ((1, 5), "B after A vs after C", "diff")]),
    ("stg v1c", B2 / "stg_replication_data" / "stg_v1c_n7_seed*.json.gz",
     [((2, 4), "C after B vs after A", "diff"), ((1, 5), "B after A vs after C", "diff")]),
    ("ov50", B2 / "set_worlds_data" / "ov50_n7_seed*.json.gz",
     [((0, 3), "A first vs after C", "diff"), ((1, 4), "B after A vs after A", "same")]),
]


def main():
    summary = {"H": {"diff": [], "same": []}, "H+": {"diff": [], "same": []}}
    for name, pat, pairs in CASES:
        runs = R.load(pat)
        for ro in R.READOUTS:
            rng = np.random.default_rng(0)
            streams = [R.build(d, wm, r, chg, 10, ro, rng) for d, wm, r, chg in runs]
            for (p1, p2), label, kind in pairs:
                sims = []
                for s in streams:
                    m1 = s['q'][(s['ph'] == p1) & s['settled']].mean(0)
                    m2 = s['q'][(s['ph'] == p2) & s['settled']].mean(0)
                    sims.append(float(m1 @ m2 / (np.linalg.norm(m1) * np.linalg.norm(m2))))
                summary[ro][kind].append(np.mean(sims))
                print(f"  {name:9s} {ro:2s} {label:24s} ({kind}): mean {np.mean(sims):.2f}  [min {min(sims):.2f}]")
    for ro, v in summary.items():
        print(f"{ro}: different-predecessor mean {np.mean(v['diff']):.2f} (min over pairs {min(v['diff']):.2f}) | "
              f"same-predecessor control {np.mean(v['same']):.2f} | gap {np.mean(v['same']) - np.mean(v['diff']):+.2f}")


if __name__ == '__main__':
    main()

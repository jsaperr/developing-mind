"""Generalization plan, step 0 (experiment_plan_generalization.md): is one-back release TRIGGERED
BY THE CHANGE, or just SLOW (~1000 s after departure)? Analysis only, on existing runs.

Every earlier one-back measurement used 1000 s phases, where "the next change" and "1000 s after
departure" coincide. checkb breaks the tie: its fillers are 300 s long, so each core becomes two
back only 300 s after it departed.
  A departs at 1000 (F1 300 s), two back at 1300 when B arrives
  B departs at 2300 (F2 300 s), two back at 2600 when C arrives
  C departs at 3600 (F3 300 s), two back at 3900 when A returns
Reference: v1b (1000 s phases), A departs at 1000 and is two back at 2000.

Holders of context X at time t = neurons whose 50 s mean weight vector is closest (cosine to the
centered prototypes of all the world's contexts, fillers included) to X.

PREDICTIONS ON RECORD (written before this script was first run):
  RT-P1 release is change-triggered: in checkb, a core's holders 300 s after it becomes two back
        (i.e. 600 s after departure) are <= 30% of its holders just before it became two back,
        seed-pooled, for B and C (A too, though A's value is the first swap and can differ).
  RT-P2 the one-back part holds over a short phase: a core's holders at the end of the 300 s filler
        (just before it becomes two back) are >= 60% of its holders at departure.
  (Alternative, "slow": holders 600 s after departure stay >= 70% of the departure level, and drop only
   ~1000 s after departure.)

Usage: analyze_release_timing.py   (conda env)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "notebooks" / "integration" / "set_worlds"))
import run_rectified_memory as R
from src.integration import interface as I

OFFSETS = [-50, 0, 150, 300, 450, 600, 800, 1000, 1300]   # seconds relative to departure (window end)


def holders(d, wm, ctx, t):
    protos = np.array([I.unit(np.isin(np.arange(30), s).astype(float)) for s in d['context_sets']])
    wbar = wm[:, :, max(t - 50, 0):t].mean(axis=2)
    return sum(int(np.argmax(protos @ I.unit(wbar[j]))) == ctx for j in range(d['n_post']))


def curve(runs, ctx, depart):
    rows = []
    for off in OFFSETS:
        t = depart + off
        rows.append(np.mean([holders(d, wm, ctx, t) for d, wm, r, chg in runs if t <= d['total_s']]))
    return rows


def show(label, runs, ctx, depart, two_back_at):
    c = curve(runs, ctx, depart)
    print(f"  {label:28s} (two back at +{two_back_at - depart} s) " +
          "  ".join(f"{o:+5d}:{v:4.2f}" for o, v in zip(OFFSETS, c)))
    return dict(zip(OFFSETS, c))


def main():
    cb = R.load(R.B2 / "set_worlds_data" / "checkb_n7_seed*.json.gz")
    vb = R.load(R.B2 / "v1b_schedule_data" / "v1b_n7_reliable_13_1p5_seed*.json.gz")
    print(f"holders of a core context after it departs (seed mean; checkb {len(cb)} seeds, v1b {len(vb)} seeds)")
    res = {}
    res['cb A'] = show("checkb A (F1 300 s)", cb, 0, 1000, 1300)
    res['cb B'] = show("checkb B (F2 300 s)", cb, 1, 2300, 2600)
    res['cb C'] = show("checkb C (F3 300 s)", cb, 2, 3600, 3900)
    res['v1b A'] = show("v1b A (B 1000 s)", vb, 0, 1000, 2000)
    res['v1b B'] = show("v1b B (C 1000 s)", vb, 1, 2000, 3000)
    print("\nRT-P1 (holders 600 s after departure / holders just before becoming two back at +300):")
    for k in ('cb A', 'cb B', 'cb C'):
        pre = res[k][300]; post = res[k][600]
        print(f"  {k}: {post:.2f} / {pre:.2f} = {post / pre if pre else float('nan'):.0%}")
    print("RT-P2 (holders at +300, end of filler / at departure):")
    for k in ('cb A', 'cb B', 'cb C'):
        print(f"  {k}: {res[k][300]:.2f} / {res[k][0]:.2f} = {res[k][300] / res[k][0] if res[k][0] else float('nan'):.0%}")


def post_hoc():
    """POST-HOC (added after seeing RT-P1/P2 results, not predicted): who holds what at each change.
    Mean holders per context (A B C F1 F2 F3 in checkb; A B C in v1b) just before each swap and 150 s /
    500 s after it, to see where the neurons a new context recruits come from."""
    for name, pat, names in (("checkb", "set_worlds_data/checkb_n7_seed*.json.gz", "A B C F1 F2 F3".split()),
                             ("v1b", "v1b_schedule_data/v1b_n7_reliable_13_1p5_seed*.json.gz", "A B C".split())):
        runs = R.load(R.B2 / pat)
        d0 = runs[0][0]
        print()
        print(f"POST-HOC {name}: mean holders per context")
        print("  " + " " * 22 + "  ".join(f"{n:>4s}" for n in names))
        for k, st in enumerate(d0['swap_times_s']):
            st = int(st)
            for off, lab in ((0, "before"), (150, "+150"), (500, "+500")):
                t = st + off
                v = [np.mean([holders(d, wm, c, t) for d, wm, r, chg in runs]) for c in range(len(names))]
                inc = names[d0['phase_corr_blocks'][k + 1]]
                print(f"  swap {k + 1} ->{inc:3s} {lab:7s}  " + "  ".join(f"{x:4.2f}" for x in v))


if __name__ == '__main__':
    main()
    post_hoc()

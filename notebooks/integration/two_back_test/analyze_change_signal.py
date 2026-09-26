"""Is there a label-free substrate signal that the world is STILL changing?

Motivation: the residual absorption and the two-back capture both happen during the slow drift
after a change. The swap dip marks only the onset (about 2% of steps), and the stability gate
misses slow drift. Candidate: the substrate's own plasticity, i.e. weights moving while neurons
re-learn.

Two measures, per run, around every change, with settled mid-phase windows as a null:
  1. population |dw| per second (10 s smoothing). PREDICTED elevated >= 60 s after changes.
     REFUTED: flat, because it's dominated by constant STDP churn. Individual synapses reverse
     direction at a setting-independent rate (the standing explanation in arc 01), so per-second
     |dw| is the same whether the network is re-learning or settled.
  2. displacement over the last 60 s, sum |w(t) - w(t-60)|. Churn reverses and cancels over the
     window while directional re-learning accumulates, a fix grounded in that same explanation.
     Same prediction. The first duration metric was MIS-SPECIFIED: it took the first time below
     threshold after 5 s, which returns about 6 s because the lagging window hasn't risen yet at
     the change. The corrected metric is onset (first crossing) plus duration above threshold
     from onset.

Threshold: 3 x robust sd above the median pre-change baseline (300 to 10 s before each change).

Usage: analyze_change_signal.py   (conda env)
"""
import glob
from pathlib import Path

import numpy as np

import run_two_back as TB
from src.brian2_stdp.results_io import load_result

B2 = TB.ROOT / "notebooks" / "brian2"
WORLDS = [("13mV v1b", B2 / "v1b_schedule_data" / "v1b_n7_reliable_13_1p5_seed*.json.gz"),
          ("stg v1c", B2 / "stg_replication_data" / "stg_v1c_n7_seed*.json.gz")]
L = 60


def episode(row, thr):
    up = np.where(row > thr)[0]
    if len(up) == 0:
        return None, 0
    on = up[0]
    down = np.where(row[on:] <= thr)[0]
    return int(on), int(down[0]) if len(down) else len(row) - on


def main():
    for label, pat in WORLDS:
        rate_curves, disp_curves, rate_base, disp_base, disp_null = [], [], [], [], []
        for f in sorted(glob.glob(str(pat))):
            d = load_result(f)
            w = np.array(d['weight_trace'])
            rate = np.convolve(np.abs(np.diff(w, axis=1)).sum(0), np.ones(10) / 10, 'same')
            disp = np.r_[np.full(L, np.nan), np.abs(w[:, L:] - w[:, :-L]).sum(0)]
            for s in (int(x) for x in d['swap_times_s']):
                rate_curves.append(rate[s:s + 400]); rate_base.append(np.median(rate[s - 300:s - 10]))
                disp_curves.append(disp[s:s + 400]); disp_base.append(np.nanmedian(disp[s - 300:s - 10]))
                disp_null.append(disp[s + 500:s + 900])
        for name, curves, base in (("|dw|/s (10 s smoothed)", rate_curves, rate_base),
                                   ("60 s displacement", disp_curves, disp_base)):
            b = np.median(base); thr = b + 3 * np.median(np.abs(np.array(base) - b)) * 1.4826
            ep = [episode(r, thr) for r in curves]
            dur = np.array([e[1] for e in ep]); n_on = sum(e[0] is not None for e in ep)
            m = np.nanmean(np.array(curves), 0)
            print(f"{label} | {name}: baseline {b:.2f}, mean at +0/+60/+120/+200/+300 s: "
                  + " ".join(f"{m[k]:.2f}" for k in (0, 60, 120, 200, 299))
                  + f" | episodes {n_on}/{len(curves)}, duration median {np.median(dur):.0f} s, >=60 s in {int(np.sum(dur >= 60))}/{len(dur)}")
            if name.startswith("60"):
                nd = np.array([episode(r, thr)[1] for r in disp_null])
                print(f"{'':>10} null (settled mid-phase): >=60 s in {int(np.sum(nd >= 60))}/{len(nd)}, longest {nd.max()} s")


if __name__ == '__main__':
    main()

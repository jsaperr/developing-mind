"""Scores CC-P0..P3 (predictions in modal_competitive_check.py, written before any run) from
out/{laptop,modal}_{probe,stg,op}_seed<N>.json.gz. Written before any result was read.

  P0 probe: laptop-vs-modal max |dw| at t = 1 s (and over the whole 5 s), per seed.
  P1 stg:   per-seed max |dw| between platforms at the final sample (600 s).
  P2 stg:   'differentiate' counts per platform (metrics.classify_differentiation on
            compute_competitive_metrics, as analyze_sweep.py does), Mann-Whitney on late-window std;
            per-seed label agreement reported too.
  P3 op:    per-seed preferred-context assignment of all 7 neurons at the end of each phase (nearest centered
            block prototype of the 50 s mean weights, step 0's metric), identical across platforms?;
            one-back statistics on each platform.

Usage: score.py   (conda env)
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from src.brian2_stdp.metrics import classify_differentiation, compute_competitive_metrics
from modal_competitive_check import OP_SEEDS, PROBE_SEEDS, STG_SEEDS

OUT = HERE / "out"


def load(platform, arm, seed):
    f = OUT / f"{platform}_{arm}_seed{seed}.json.gz"
    if not f.exists():
        return None
    with gzip.open(f, "rt") as fh:
        r = json.load(fh)
    return r if r.get("status") == "completed" else None


def pairs(arm, seeds):
    out = []
    for s in seeds:
        a, b = load("laptop", arm, s), load("modal", arm, s)
        if a is None or b is None:
            print(f"  {arm} seed {s}: missing or failed (laptop {a is not None}, modal {b is not None})")
            continue
        out.append((s, a, b))
    return out


def prefs(r, t):
    wt = np.array(r["weight_trace"]); si, sj = np.array(r["syn_i"]), np.array(r["syn_j"])
    wm = np.zeros((7, 30, wt.shape[1])); wm[sj, si] = wt
    P = []
    for b in range(3):
        v = np.isin(np.arange(30), range(b * 10, b * 10 + 10)).astype(float); v -= v.mean(); P.append(v / np.linalg.norm(v))
    P = np.array(P)
    wbar = wm[:, :, t - 50:t].mean(axis=2)
    out = []
    for j in range(7):
        u = wbar[j] - wbar[j].mean(); u /= np.linalg.norm(u) or 1
        out.append(int(np.argmax(P @ u)))
    return out


def main():
    print("== P0 probe (same noise stream?)")
    for s, a, b in pairs("probe", PROBE_SEEDS):
        A, B = np.array(a["weight_trace"]), np.array(b["weight_trace"])
        print(f"  seed {s}: max|dw| at t=1 s {np.abs(A[:, 1] - B[:, 1]).max():.3g}, over 0-4 s {np.abs(A - B).max():.3g}")

    print("== P1/P2 stg (strong_tight_gate, N=3, 600 s)")
    rows = []
    for s, a, b in pairs("stg", STG_SEEDS):
        res = []
        for r in (a, b):
            comp = compute_competitive_metrics(np.array(r["weight_trace"]), np.array(r["syn_i"]), np.array(r["syn_j"]), 3, n_corr=10)
            res.append(classify_differentiation(comp["per_neuron_gap"], np.array(r["weight_trace_t"])))
        dw = float(np.abs(np.array(a["weight_trace"])[:, -1] - np.array(b["weight_trace"])[:, -1]).max())
        rows.append((s, res[0], res[1], dw))
    if rows:
        dws = [x[3] for x in rows]
        nl = sum(x[1][0] == "differentiate" for x in rows); nm = sum(x[2][0] == "differentiate" for x in rows)
        agree = sum(x[1][0] == x[2][0] for x in rows)
        la, lm = [x[1][1] for x in rows], [x[2][1] for x in rows]
        p = mannwhitneyu(la, lm).pvalue
        print(f"  n={len(rows)} | per-seed final max|dw| median {np.median(dws):.3g} (min {min(dws):.3g}, max {max(dws):.3g})")
        print(f"  differentiate: laptop {nl}/{len(rows)}, modal {nm}/{len(rows)} | per-seed label agreement {agree}/{len(rows)}")
        print(f"  late-window std: laptop median {np.median(la):.4f}, modal {np.median(lm):.4f}, Mann-Whitney p {p:.3f}; "
              f"per-seed correlation {np.corrcoef(la, lm)[0, 1]:.2f}")

    print("== P3 op (13 mV/1.5, N=7, 300 s phases A B C A B)")
    ends = [300, 600, 900, 1200, 1500]
    same, stats = 0, {"laptop": [], "modal": []}
    ps = pairs("op", OP_SEEDS)
    for s, a, b in ps:
        pa = [prefs(a, t) for t in ends]; pb = [prefs(b, t) for t in ends]
        same += pa == pb
        for name, r in (("laptop", a), ("modal", b)):
            before = prefs(r, 600).count(1); after = prefs(r, 750).count(1)
            stats[name].append((after / before if before else np.nan, prefs(r, 900).count(0), prefs(r, 1200).count(1)))
    print(f"  n={len(ps)} | per-seed assignments identical at all 5 phase ends: {same}/{len(ps)}")
    for name, v in stats.items():
        v = np.array(v)
        print(f"  {name}: B keeps {np.nanmean(v[:, 0]):.0%} of its holders at C's arrival (+150 s); "
              f"incoming holders before two-back returns: A {v[:, 1].mean():.2f}, B {v[:, 2].mean():.2f}")


if __name__ == "__main__":
    main()

"""Offline readout comparison: does a RECTIFIED contrast fix the overlap problem? Analysis only.

Finding that motivates this (ov50, 50% overlapping contexts): settled readout quality of H dropped
to ~0.46, from ~0.9 with disjoint contexts. H = sum_j (r_j - mean r) w_j subtracts what the quieter
neurons (the retainers of the previous context) are tuned to. Under overlap the SHARED inputs are
strong in both groups and cancel, leaving roughly "what's different from the last context". That
also makes the readout predecessor-dependent.

Readouts, per settled window (all label-free, then centered and unit-normalized):
  H    sum_j (r_j - mean r) w_j           current readout (contrast)
  H+   sum_j max(r_j - mean r, 0) w_j     rectified contrast: what the DRIVEN neurons are tuned to,
                                          without subtracting what the quiet ones hold. No new parameter.
  U    sum_j r_j w_j                      absolute, rate-weighted tuning (no contrast), as reference
Metrics (labels used only to score):
  quality   cosine to the true current-context prototype (phases >= 2; phase 1 reported separately,
            since it's the cold start)
  leakage   least-squares fit q ~ b_cur * p_cur + b_prev * p_prev; leakage = b_prev / b_cur. The
            previous context's signed contribution: negative means it's subtracted, positive means
            it leaks in.
Worlds: 13mV v1b and strong_tight_gate v1b (disjoint contexts), ov50 (50% overlap).

PREDICTIONS ON RECORD (written before this script was first run):
  RD-P1 H+ restores overlap quality: ov50 phases >= 2 quality >= 0.70 for H+ (H: ~0.46), while the
        disjoint worlds stay >= 0.85.
  RD-P2 leakage: H strongly negative (it subtracts the previous context), U positive (retainers
        leak in), H+ within [-0.2, +0.2] in every world.
  RD-P3 the cold start is NOT fixed by H+: phase-1 quality stays <= 0.45 for H and H+, while U is
        high there (nothing to contrast, but the absolute tuning is clean).

Usage: compare_readouts.py   (conda env)
"""
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.brian2_stdp.results_io import load_result
from src.integration.interface import unit

B2 = ROOT / "notebooks" / "brian2"
WORLDS = [("13mV v1b (disjoint)", B2 / "v1b_schedule_data" / "v1b_n7_reliable_13_1p5_seed*.json.gz"),
          ("stg v1b (disjoint)", B2 / "stg_replication_data" / "stg_v1b_n7_seed*.json.gz"),
          ("ov50 (50% overlap)", B2 / "set_worlds_data" / "ov50_n7_seed*.json.gz")]
W, SETTLE = 10, 300


def readouts(r, wbar):
    dev = r - r.mean()
    return {"H": unit(dev @ wbar), "H+": unit(np.clip(dev, 0, None) @ wbar), "U": unit(r @ wbar)}


def main():
    for name, pat in WORLDS:
        acc = {k: {"q1": [], "q": [], "leak": []} for k in ("H", "H+", "U")}
        for f in sorted(glob.glob(str(pat))):
            d = load_result(f)
            if d.get('status') != 'completed':
                continue
            n_pre = int(max(d['syn_i'])) + 1
            sets = d.get('context_sets') or [list(range(b * 10, b * 10 + 10)) for b in range(n_pre // 10)]
            protos = np.array([unit(np.isin(np.arange(n_pre), s).astype(float)) for s in sets])
            w = np.array(d['weight_trace']); wm = np.zeros((d['n_post'], n_pre, w.shape[1]))
            wm[np.array(d['syn_j']), np.array(d['syn_i'])] = w
            r = np.array(d['spike_rate_bins'], dtype=float)
            ps = [0.0] + list(d['swap_times_s']); ids = d['phase_corr_blocks']
            for a in range(0, int(d['total_s']) - W + 1, W):
                p = int(np.searchsorted(ps, a, side='right') - 1)
                if (p + 1 < len(ps) and a + W > ps[p + 1]) or a - ps[p] < SETTLE:
                    continue
                ro = readouts(r[:, a:a + W].mean(1), wm[:, :, a:a + W].mean(2))
                cur = protos[ids[p]]
                for k, q in ro.items():
                    if p == 0:
                        acc[k]["q1"].append(q @ cur)
                    else:
                        acc[k]["q"].append(q @ cur)
                        X = np.stack([cur, protos[ids[p - 1]]], axis=1)
                        b = np.linalg.lstsq(X, q, rcond=None)[0]
                        acc[k]["leak"].append(b[1] / b[0] if abs(b[0]) > 1e-9 else np.nan)
        print(f"{name}:")
        for k, v in acc.items():
            print(f"   {k:3s} quality phases>=2 {np.mean(v['q']):.2f}  | phase 1 (cold start) {np.mean(v['q1']):.2f}"
                  f"  | leakage of previous context {np.nanmedian(v['leak']):+.2f}")


if __name__ == '__main__':
    main()

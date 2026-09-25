"""Retainer/tracker analysis for the non-stationary-correlation runs, general in N.

For each swap k (1, 2) and each neuron: at the end of the new phase the neuron is a RETAINER if its
phase-aligned gap is still negative (it holds the previous phase's pattern) and a TRACKER otherwise.
Reports, per swap: number of retainers, whether retainers came from the previous phase's top tier
(compute_tiers, threshold 0.03) or how their previous-phase gaps rank against the trackers',
re-learning latency of trackers (first time aligned gap >= 0.3 after the swap; neurons already
positive at the swap are "already there", i.e. they held the returning pattern), and late-phase
firing rates. Use: analyze_retainers.py <glob prefix e.g. nonstat_n7_reliable_13_1p5>.
"""
import glob
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_nonstationary import load
from src.brian2_stdp.metrics import compute_tiers

HERE = Path(__file__).resolve().parent


def analyze(prefix, latency_level=0.3):
    files = sorted(glob.glob(str(HERE / f"{prefix}_seed*.json")))
    n_retainers, retainer_prev, tracker_prev, lat_relearn, already_there = [], [], [], [], []
    retainer_prev1, tracker_prev1 = [], []
    ret_rates, trk_rates = [], []
    single_slots = single_in_top = 0
    for f in files:
        got = load(f)
        if got is None:
            print("NOT COMPLETED:", f)
            continue
        d, t, gap, phase = got
        n = d['n_post']
        rates = np.array(d['spike_rate_bins'])
        ends = [0.0] + d['swap_times_s'] + [d['total_s']]
        line = f"seed {d['seed']} (N={n}):"
        for k in (1, 2):
            s = ends[k]
            prev = gap[:, (t > s - 50) & (t < s)].mean(axis=1)
            end = gap[:, (t > ends[k + 1] - 50) & (t <= ends[k + 1])].mean(axis=1)
            ret = [j for j in range(n) if end[j] < 0]
            trk = [j for j in range(n) if end[j] >= 0]
            n_retainers.append(len(ret))
            retainer_prev += prev[ret].tolist()
            tracker_prev += prev[trk].tolist()
            if k == 1:      # swap 1 only: every neuron starts with a positive gap, so the comparison is clean
                retainer_prev1 += prev[ret].tolist()
                tracker_prev1 += prev[trk].tolist()
            if len(ret) == 1:
                single_slots += 1
                single_in_top += int(ret[0] in compute_tiers(prev, threshold=0.03)[0])
            lats = {}
            for j in trk:
                mm = (t >= s) & (gap[j] >= latency_level)
                lats[j] = float(t[mm][0] - s) if mm.any() else np.nan
            already_there.append(sum(1 for v in lats.values() if v == 0.0))
            lat_relearn += [v for v in lats.values() if v > 0.0]
            m0, m1 = int(s) + 800, int(ends[k + 1])
            ret_rates += [rates[j, m0:m1].mean() for j in ret]
            trk_rates += [rates[j, m0:m1].mean() for j in trk]
            line += (f"  swap{k}: retainers={len(ret)} already-there={already_there[-1]} "
                     f"relearn-latencies={sorted(int(v) for v in lats.values() if v > 0)}")
        print(line)
    print(f"\n{prefix}: swaps analysed = {len(n_retainers)}")
    print(f"  retainers per swap: {np.bincount(n_retainers).tolist()} (index = count)  mean={np.mean(n_retainers):.2f}")
    print(f"  neurons already holding the returning pattern at swap 2: "
          f"{already_there[1::2]}  (per seed, in order)")
    if single_slots:
        print(f"  swaps with exactly one retainer: {single_slots}; retainer in prior top tier: {single_in_top}/{single_slots}")
    if retainer_prev1 and tracker_prev1:
        p = mannwhitneyu(retainer_prev1, tracker_prev1, alternative='greater').pvalue
        print(f"  SWAP 1 ONLY, previous-phase aligned gap: retainers mean={np.mean(retainer_prev1):.3f} "
              f"(n={len(retainer_prev1)}) vs trackers mean={np.mean(tracker_prev1):.3f} "
              f"(n={len(tracker_prev1)}); MWU one-sided p={p:.4f}")
    # (swap-2 version deliberately not reported: the neuron that held the returning pattern has a
    #  very negative previous-frame gap and trivially counts as a tracker, which biases the test.)
    r = np.array(lat_relearn)
    if len(r):
        print(f"  re-learning latency to {latency_level}: n={len(r)} median={np.median(r):.0f}s mean={r.mean():.0f}s "
              f">=100s: {(r >= 100).sum()}/{len(r)} max={r.max():.0f}s")
    if ret_rates and trk_rates:
        print(f"  late-phase rates Hz: retainers {np.mean(ret_rates):.1f}  trackers {np.mean(trk_rates):.1f}")


if __name__ == '__main__':
    analyze(sys.argv[1])

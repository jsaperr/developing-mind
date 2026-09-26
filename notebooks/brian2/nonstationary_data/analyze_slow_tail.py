"""Slow second-swap tail: is the slow return caused by the neurons that already hold the returned-to
pattern (it's already covered -> weak pressure for the rest to also return), or by the returning
neurons competing among themselves?

Worked in ABSOLUTE block terms (block A mean w - block B mean w) so labels don't flip at the swap;
corr blocks must be [0,1,0]. At swap 2:
  holder   = already on A at end of phase 2 (a swap-1 retainer)
  returner = on B at end of phase 2, climbs back to +LEVEL during phase 3 (latency = time to cross)
  refuser  = on B at end of phase 2 and never gets back to +LEVEL (a swap-2 retainer)

The honest unit is the seed, not the returner: returners in one seed share that seed's holder
count, so the pooled per-returner correlation is pseudoreplicated. The per-seed test is primary.

Gate check: during a returner's climb, is it rate-close (high g_ij = 1/(1+|r_i-r_j|/gap_scale),
i.e. strong mutual inhibition) to the holders or to its co-returners?

Usage: analyze_slow_tail.py [glob prefix, default nonstat_n7_reliable_13_1p5]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent
LEVEL = 0.3   # same re-learn threshold as analyze_retainers.py


def g_ij(ri, rj, gap_scale):
    return 1.0 / (1.0 + np.abs(ri - rj) / gap_scale)


def block0_gap(d):
    w = np.array(d['weight_trace']); si = np.array(d['syn_i']); sj = np.array(d['syn_j'])
    return np.stack([w[(sj == j) & (si < 10)].mean(0) - w[(sj == j) & (si >= 10)].mean(0)
                     for j in range(d['n_post'])])


def corr(name, x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0:
        print(f"  {name}: not testable (n={ok.sum()})")
        return
    rho, p = spearmanr(x[ok], y[ok])
    print(f"  {name}: Spearman rho={rho:+.2f} p={p:.3f} (n={ok.sum()})")


def analyze(prefix):
    per_seed, rows = [], []
    for f in sorted(glob.glob(str(HERE / f"{prefix}_seed*.json"))):
        d = json.load(open(f))
        if d.get('status') != 'completed':
            continue
        assert d['phase_corr_blocks'] == [0, 1, 0]
        t = np.array(d['weight_trace_t']); n = d['n_post']; gs = d['gap_scale']
        r = np.array(d['r_trace']); rt = np.array(d['r_trace_t'])
        s2 = d['swap_times_s'][1]
        bg = block0_gap(d)
        end2 = bg[:, (t > s2 - 50) & (t <= s2)].mean(axis=1)
        holders = [j for j in range(n) if end2[j] > 0]
        lat = {}
        refusers = 0
        for j in (j for j in range(n) if end2[j] <= 0):
            m = (t >= s2) & (bg[j] >= LEVEL)
            if m.any():
                lat[j] = float(t[m][0] - s2)
            else:
                refusers += 1
        per_seed.append((d['seed'], len(holders), len(lat), refusers, sorted(int(v) for v in lat.values())))
        for j, L in lat.items():
            wm = (rt >= s2) & (rt <= s2 + L)
            co = [k for k in lat if k != j]
            rows.append((L, len(holders), len(lat),
                         np.mean([g_ij(r[j, wm], r[a, wm], gs).mean() for a in holders]) if holders else np.nan,
                         np.mean([g_ij(r[j, wm], r[k, wm], gs).mean() for k in co]) if co else np.nan))

    print(f"{prefix} -- swap 2 (return to A):")
    print(f"{'seed':>6} {'holders':>8} {'returners':>10} {'refusers':>9}  return latencies (s)")
    for sd, nh, nr, nf, lats in per_seed:
        print(f"{sd:>6} {nh:>8} {nr:>10} {nf:>9}  {lats}")

    by_h = {}
    for _, nh, _, _, lats in per_seed:
        if lats:
            by_h.setdefault(nh, []).append(max(lats))
    print("\nPer-seed (primary): slowest return latency by number of holders:",
          {k: sorted(v) for k, v in sorted(by_h.items())})
    ks = sorted(by_h)
    if len(ks) == 2:
        lo, hi = by_h[ks[0]], by_h[ks[1]]
        print(f"  MWU ({ks[1]} holders slower than {ks[0]}): p={mannwhitneyu(hi, lo, alternative='greater').pvalue:.3f}; "
              f"perfect separation: {max(lo) < min(hi)}")

    a = np.array(rows, dtype=float)
    print(f"\nPooled returners (pseudoreplicated, secondary): n={len(a)}, median {np.median(a[:, 0]):.0f}s, max {a[:, 0].max():.0f}s")
    corr("latency ~ # holders", a[:, 1], a[:, 0])
    corr("latency ~ # co-returners", a[:, 2], a[:, 0])
    print(f"  mean g to holders {np.nanmean(a[:, 3]):.3f} | to co-returners {np.nanmean(a[:, 4]):.3f}")
    corr("latency ~ g to holders", a[:, 3], a[:, 0])
    corr("latency ~ g to co-returners", a[:, 4], a[:, 0])


if __name__ == '__main__':
    analyze(sys.argv[1] if len(sys.argv) > 1 else "nonstat_n7_reliable_13_1p5")

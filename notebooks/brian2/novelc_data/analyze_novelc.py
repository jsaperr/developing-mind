"""Novel-C readout (A->B->C, C never seen; 3 disjoint blocks of 10). Written before looking at any
novel-C result, so the readout is fixed in advance.

Per neuron and time, the tuned gap for block b = mean w over block b - mean w over the other two
blocks. A neuron "favors" the block with the largest mean w at the end of a phase (last 50 s).

Questions:
  1. Capacity tax vs release (exploratory, no bar): at the end of phase 3, how many neurons still
     favor a stale block (A or B) that is never coming back, vs committed to C? Do phase-1 A-holders
     keep A through 2000 s of its absence?
  2. Commit latency to a novel block at swap 1 (to B) and swap 2 (to C). Threshold is
     plateau-relative (half of the best neuron's phase-1 plateau), the same yardstick as
     analyze_nonstationary.py, since absolute gaps differ from the 20-input rig.
  3. Swap-signal control. Prediction on record (arc 05, N=7 follow-up entry) before this ran:
     swap 2 is also novel, so its first-10 s dip should be about as large as swap 1's (unlike the
     A->B->A return, where it was 37% of swap 1's), and neurons still holding a stale block should
     NOT speed up at swap 2.
  4. w_total health through both swaps.

Usage: analyze_novelc.py [glob prefix, default novelc_n7_reliable_13_1p5]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
NAMES = "ABC"


def block_means(d):
    """(n_post, n_blocks, n_t) mean weight per block per neuron."""
    w = np.array(d['weight_trace']); si = np.array(d['syn_i']); sj = np.array(d['syn_j'])
    B, nb = d['block_size'], d['n_blocks']
    return np.stack([np.stack([w[(sj == j) & (si // B == b)].mean(0) for b in range(nb)])
                     for j in range(d['n_post'])])


def tuned_gap(bm, b):
    others = [k for k in range(bm.shape[1]) if k != b]
    return bm[:, b] - bm[:, others].mean(axis=1)


def analyze(prefix):
    files = sorted(glob.glob(str(HERE / f"{prefix}_seed*.json")))
    lat1, lat2, dips, stale_rate_change, commit_rate_change = [], [], [], [], []
    end3_counts, a_through, wt_range = [], [], []
    for f in files:
        d = json.load(open(f))
        if d.get('status') != 'completed':
            print("NOT COMPLETED:", Path(f).name)
            continue
        assert d['phase_corr_blocks'] == [0, 1, 2]
        t = np.array(d['weight_trace_t']); n = d['n_post']
        s1, s2 = d['swap_times_s']; T = d['total_s']
        bm = block_means(d)
        end = lambda s: bm[:, :, (t > s - 50) & (t <= s)].mean(axis=2)     # (n, n_blocks)
        fav1, fav2, fav3 = end(s1).argmax(1), end(s2).argmax(1), end(T).argmax(1)

        g = [tuned_gap(bm, b) for b in range(3)]
        plateau1 = g[0][:, (t > s1 - 50) & (t <= s1)].mean(axis=1).max()
        level = 0.5 * plateau1

        def latencies(block, s):
            out = []
            for j in range(n):
                m = (t >= s) & (g[block][j] >= level)
                out.append(float(t[m][0] - s) if m.any() else np.nan)
            return out
        l1 = latencies(1, s1); l2 = latencies(2, s2)
        lat1 += [x for x in l1 if np.isfinite(x)]
        lat2 += [x for x in l2 if np.isfinite(x)]

        counts = {NAMES[b]: int((fav3 == b).sum()) for b in range(3)}
        end3_counts.append(counts)
        a_holders_p2 = [j for j in range(n) if fav2[j] == 0]
        a_through.append((len(a_holders_p2), sum(1 for j in a_holders_p2 if fav3[j] == 0)))

        r = np.array(d['spike_rate_bins'])
        dip = lambda s: r[:, int(s):int(s) + 10].mean() - r[:, int(s) - 100:int(s)].mean()
        dips.append((dip(s1), dip(s2)))
        for j in range(n):
            dr = r[j, int(s2):int(s2) + 10].mean() - r[j, int(s2) - 100:int(s2)].mean()
            (stale_rate_change if fav2[j] == 0 else commit_rate_change).append(dr)

        wt = np.array(d['w_total_trace'])
        wt_range.append((wt[:, 50:].min(), wt[:, 50:].max()))

        print(f"seed {d['seed']}: favored block end of phase 1/2/3 = "
              f"{''.join(NAMES[b] for b in fav1)} / {''.join(NAMES[b] for b in fav2)} / "
              f"{''.join(NAMES[b] for b in fav3)}   commit latency (s) to B: "
              f"{[int(x) if np.isfinite(x) else None for x in l1]}  to C: "
              f"{[int(x) if np.isfinite(x) else None for x in l2]}  (level {level:.2f})")

    k = len(end3_counts)
    print(f"\n{prefix}: {k} completed seeds")
    print("1. end of phase 3 (world = C), neurons favoring each block, per seed:")
    for c in end3_counts:
        print(f"   {c}")
    stale = np.array([c['A'] + c['B'] for c in end3_counts])
    print(f"   stale (A or B) per seed: {stale.tolist()}  mean {stale.mean():.2f} of 7 "
          f"({100 * stale.mean() / 7:.0f}%)")
    print(f"   phase-2 A-holders still on A at end of phase 3 (held through 2000 s of absence): "
          f"{sum(b for _, b in a_through)}/{sum(a for a, _ in a_through)}")
    l1a, l2a = np.array(lat1), np.array(lat2)
    print(f"2. commit latency: to B median {np.median(l1a):.0f}s (n={len(l1a)}, max {l1a.max():.0f}) | "
          f"to C median {np.median(l2a):.0f}s (n={len(l2a)}, max {l2a.max():.0f})")
    dp = np.array(dips)
    print(f"3. first-10s dip: swap1 {dp[:, 0].mean():+.2f}  swap2 {dp[:, 1].mean():+.2f} Hz  "
          f"(ratio {dp[:, 1].mean() / dp[:, 0].mean():.2f}; A->B->A return ratio was 0.37)  "
          f"paired Wilcoxon p={wilcoxon(dp[:, 0], dp[:, 1]).pvalue:.3f}")
    print(f"   at swap 2, neurons still on A: {np.mean(stale_rate_change):+.2f} Hz (n={len(stale_rate_change)})"
          f"   others: {np.mean(commit_rate_change):+.2f} Hz (n={len(commit_rate_change)})")
    wr = np.array(wt_range)
    print(f"4. w_total after t=50s: [{wr[:, 0].min():.2f}, {wr[:, 1].max():.2f}] (target {d['target_total']})")


def post_hoc(prefix, ab_prefix="nonstat_n7_reliable_13_1p5"):
    """ADDED AFTER inspecting the trajectory figure (not part of the pre-fixed readout above).
    (a) do A-holders erode during phase 2, or hold flat? (b) which neurons never release A?
    (c) the per-neuron recognition test done per seed (not pseudoreplicated): mean first-10 s rate
    change at swap 2 of neurons holding A, when A returns (A->B->A, 20-input rig) vs when it does not
    (A->B->C, 30-input rig). Cross-rig comparison; the trackers' drop (-6.3 vs -6.0 Hz) is similar
    across rigs, which is the scale anchor. Also saves the per-neuron trajectory figure."""
    from scipy.stats import mannwhitneyu
    ab_dir = HERE.parent / "nonstationary_data"
    ero, resid, nc_seed = [], [], []
    for f in sorted(glob.glob(str(HERE / f"{prefix}_seed*.json"))):
        d = json.load(open(f)); t = np.array(d['weight_trace_t']); bm = block_means(d)
        gA = tuned_gap(bm, 0); r = np.array(d['spike_rate_bins'])
        fav = lambda s: bm[:, :, (t > s - 50) & (t <= s)].mean(axis=2).argmax(1)
        s1, s2 = (int(s) for s in d['swap_times_s']); T = int(d['total_s'])
        holders = [j for j in range(d['n_post']) if fav(s2)[j] == 0]
        for j in holders:
            ero.append(gA[j, (t > s1) & (t <= s1 + 100)].mean() - gA[j, (t > s2 - 100) & (t <= s2)].mean())
            if fav(T)[j] == 0:
                resid.append((d['seed'], j, round(float(gA[j, (t > s2 - 50) & (t <= s2)].mean()), 2),
                              round(float(gA[j, t > T - 50].mean()), 2)))
        nc_seed.append(np.mean([r[j, s2:s2 + 10].mean() - r[j, s2 - 100:s2].mean() for j in holders]))
    ero = np.array(ero)
    print(f"\nPOST HOC (a) A-holders' A tuned-gap loss over phase 2: median {np.median(ero):+.3f}, "
          f">0.1 in {np.sum(ero > 0.1)}/{len(ero)}, max {ero.max():.2f}")
    print(f"POST HOC (b) never release A (seed, neuron, A gap end-ph2 -> end-ph3): {resid}")

    ab_seed = []
    for f in sorted(glob.glob(str(ab_dir / f"{ab_prefix}_seed*.json"))):
        d = json.load(open(f)); t = np.array(d['weight_trace_t']); r = np.array(d['spike_rate_bins'])
        w = np.array(d['weight_trace']); si = np.array(d['syn_i']); sj = np.array(d['syn_j'])
        s2 = int(d['swap_times_s'][1]); m = (t > s2 - 50) & (t <= s2)
        hold = [j for j in range(d['n_post'])
                if w[(sj == j) & (si < 10)][:, m].mean() > w[(sj == j) & (si >= 10)][:, m].mean()]
        ab_seed.append(np.mean([r[j, s2:s2 + 10].mean() - r[j, s2 - 100:s2].mean() for j in hold]))
    print("POST HOC (c) per-seed mean rate change of A-holders at swap 2:")
    print(f"   A returns (A->B->A):  {np.round(ab_seed, 2).tolist()}")
    print(f"   A does not (A->B->C): {np.round(nc_seed, 2).tolist()}")
    print(f"   MWU p={mannwhitneyu(ab_seed, nc_seed).pvalue:.5f}; "
          f"perfect separation: {min(ab_seed) > max(nc_seed)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(str(HERE / f"{prefix}_seed*.json")))
    fig, axes = plt.subplots(3, len(files), figsize=(3.2 * len(files), 7.5), sharex=True, sharey=True,
                             squeeze=False)
    cols = plt.cm.tab10(np.arange(10))
    for c, f in enumerate(files):
        d = json.load(open(f)); t = np.array(d['weight_trace_t']); bm = block_means(d)
        for b in range(3):
            ax = axes[b, c]; g = tuned_gap(bm, b)
            for j in range(d['n_post']):
                ax.plot(t, g[j], lw=0.8, color=cols[j % 10])
            for s in d['swap_times_s']:
                ax.axvline(s, color='k', ls='--', lw=0.6)
            ax.axhline(0, color='grey', lw=0.4)
            if c == 0:
                ax.set_ylabel(f"tuned gap to {NAMES[b]}")
            if b == 0:
                ax.set_title(f"seed {d['seed']}", fontsize=9)
    fig.suptitle("novel-C: per-neuron tuned gap to each block (world A | B | C; dashed = swaps; colour = neuron)")
    fig.tight_layout()
    out = HERE / f"{prefix}.png"
    fig.savefig(out, dpi=80)
    print("saved", out.name)


if __name__ == '__main__':
    pfx = sys.argv[1] if len(sys.argv) > 1 else "novelc_n7_reliable_13_1p5"
    analyze(pfx)
    post_hoc(pfx)

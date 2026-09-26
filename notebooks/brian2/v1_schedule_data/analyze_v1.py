"""Readout for the v1 world A->B->C->A->C (seeds 33000-33007). Written, with its predictions,
BEFORE the batch was launched.

Swaps: 1 A->B (novel), 2 B->C (novel), 3 C->A (A is TWO contexts back), 4 A->C (C is ONE back).

Per neuron, the favored block at the end of a phase is the block with the largest mean weight over
its last 50 s. "Holding the incoming pattern at a swap" means favoring it over the 50 s before the
swap.

PREDICTIONS ON RECORD:
  V1-P1 (one-back retention, arc 05): at swap 3 almost nobody already holds A. Mean <= 1 neuron per
        seed, from the lock-in residue only. At swap 4, 2-4 neurons per seed already hold C. So
        "holders of the incoming pattern" is higher at swap 4 than at swap 3 in >= 6/8 seeds.
  V1-P2 (recognition follows holding, arc 05): holders of the incoming pattern speed up in the
        first 10 s at swap 4 by a mean of +4 to +8 Hz per seed (as at the A->B->A return). At swap
        3 there's no comparable population-level recognition, because there are no or few holders.
  V1-P3 (H same-rig lull test, arc 07): with phase-mean settled H readouts, A4 is more similar to
        A1 than to B2 or C3 in >= 6/8 seeds, and C5 is more similar to C3 than to A1, B2 or A4 in
        >= 6/8. Known risk, stated up front: the first context (A1) is weakly represented by every
        label-free readout (arc 07), which could sink the A4-A1 half without H being wrong about
        context.
Descriptive, no prediction: lock-in residue over time (neurons holding a pattern that's neither the
current context nor the one just before), commit latencies per swap, scalar dips, w_total health.

Usage: analyze_v1.py   (needs the conda env; saves v1_tuned_gaps.png)
"""
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / "interface_readout"))
from src.brian2_stdp.results_io import load_result
from analyze_s1_readout import corr, windows

PREFIX = "v1_n7_reliable_13_1p5"
NAMES = "ABC"
LABELS = ["A1", "B2", "C3", "A4", "C5"]


def prep(d):
    n = d['n_post']; si = np.array(d['syn_i']); sj = np.array(d['syn_j'])
    w = np.array(d['weight_trace'])
    wm = np.zeros((n, d['n_pre'], w.shape[1])); wm[sj, si] = w
    bm = np.stack([wm[:, b * d['block_size']:(b + 1) * d['block_size']].mean(axis=1)
                   for b in range(d['n_blocks'])], axis=1)          # (n, blocks, t)
    return wm, bm, np.array(d['spike_rate_bins'], dtype=float), np.array(d['weight_trace_t'])


def fav_at(bm, t, s):
    return bm[:, :, (t > s - 50) & (t <= s)].mean(axis=2).argmax(axis=1)


def main():
    files = sorted(glob.glob(str(HERE / f"{PREFIX}_seed*.json.gz")))
    runs = [d for d in (load_result(f) for f in files) if d.get('status') == 'completed']
    print(f"{PREFIX}: {len(runs)}/{len(files)} completed")
    blocks = runs[0]['phase_corr_blocks']
    n_sw = len(blocks) - 1
    holders = np.zeros((len(runs), n_sw), int); speed = np.full((len(runs), n_sw), np.nan)
    dips = np.zeros((len(runs), n_sw)); lat = {k: [] for k in range(n_sw)}
    residue = np.zeros((len(runs), len(blocks)), int); lull = []
    for i, d in enumerate(runs):
        wm, bm, r, t = prep(d)
        ends = [0.0] + d['swap_times_s'] + [d['total_s']]
        fav_end = [fav_at(bm, t, e) for e in ends[1:]]
        for ph in range(len(blocks)):
            ok = {blocks[ph]} | ({blocks[ph - 1]} if ph else set())
            residue[i, ph] = int(sum(1 for b in fav_end[ph] if b not in ok))
        for k, s in enumerate(d['swap_times_s']):
            s = int(s); inc = blocks[k + 1]
            h = [j for j in range(d['n_post']) if fav_at(bm, t, s)[j] == inc]
            holders[i, k] = len(h)
            if h:
                speed[i, k] = np.mean([r[j, s:s + 10].mean() - r[j, s - 100:s].mean() for j in h])
            dips[i, k] = r[:, s:s + 10].mean() - r[:, s - 100:s].mean()
            others = [b for b in range(d['n_blocks']) if b != inc]
            g = bm[:, inc] - bm[:, others].mean(axis=1)
            plateau = (bm[:, blocks[0]] - bm[:, [b for b in range(3) if b != blocks[0]]].mean(axis=1))[
                :, (t > ends[1] - 50) & (t <= ends[1])].mean(axis=1).max()
            for j in range(d['n_post']):
                if j in h:
                    continue
                m = (t >= s) & (g[j] >= 0.5 * plateau)
                if m.any():
                    lat[k].append(float(t[m][0] - s))
        ws = windows(d, wm, r, [0.0] + d['swap_times_s'], 50)
        Hm = [np.mean([x[4] for x in ws if x[0] == ph and x[1]], axis=0) for ph in range(len(blocks))]
        lull.append([[corr(Hm[a], Hm[b]) for b in range(len(blocks))] for a in range(len(blocks))])

    print("\nholders of the incoming pattern at each swap (per seed):")
    for k in range(n_sw):
        print(f"  swap {k + 1} ({NAMES[blocks[k]]}->{NAMES[blocks[k + 1]]}): {holders[:, k].tolist()}  "
              f"mean {holders[:, k].mean():.2f}")
    print(f"V1-P1: swap 4 holders > swap 3 holders in {int(np.sum(holders[:, 3] > holders[:, 2]))}/{len(runs)} seeds; "
          f"swap-3 mean {holders[:, 2].mean():.2f} (predicted <= 1)")
    print("\nV1-P2: mean first-10 s rate change of incoming-pattern holders, per seed (nan = no holders):")
    for k in range(n_sw):
        print(f"  swap {k + 1}: {np.round(speed[:, k], 2).tolist()}")
    print("\nscalar first-10 s population dip per swap (mean): " +
          "  ".join(f"swap{k + 1} {dips[:, k].mean():+.2f}" for k in range(n_sw)))
    print("commit latency of non-holders to the incoming pattern (median / max, s): " +
          "  ".join(f"swap{k + 1} {np.median(lat[k]):.0f}/{max(lat[k]):.0f} (n={len(lat[k])})"
                    for k in range(n_sw) if lat[k]))
    print(f"\nlock-in residue (neurons holding neither the current nor the previous context) at end of "
          f"each phase, per seed:\n  {residue.tolist()}\n  totals by phase: {residue.sum(axis=0).tolist()}")

    L = np.array(lull)
    print("\nV1-P3: mean H similarity between phase means (rows/cols " + ", ".join(LABELS) + "):")
    for a in range(len(blocks)):
        print(f"  {LABELS[a]}: " + "  ".join(f"{np.nanmean(L[:, a, b]):+.2f}" for b in range(len(blocks))))
    a4 = int(np.sum((L[:, 3, 0] > L[:, 3, 1]) & (L[:, 3, 0] > L[:, 3, 2])))
    c5 = int(np.sum((L[:, 4, 2] > L[:, 4, 0]) & (L[:, 4, 2] > L[:, 4, 1]) & (L[:, 4, 2] > L[:, 4, 3])))
    print(f"  A4 closest to A1 (vs B2, C3): {a4}/{len(runs)}   C5 closest to C3 (vs A1, B2, A4): {c5}/{len(runs)}")
    print(f"  per-seed sim(A4,A1): {np.round(L[:, 3, 0], 2).tolist()}")

    wt = [np.array(d['w_total_trace'])[:, 50:] for d in runs]
    print(f"\nw_total after t=50 s: [{min(x.min() for x in wt):.2f}, {max(x.max() for x in wt):.2f}] (target 10)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, len(runs), figsize=(3.2 * len(runs), 7.5), sharex=True, sharey=True, squeeze=False)
    cols = plt.cm.tab10(np.arange(10))
    for c, d in enumerate(runs):
        _, bm, _, t = prep(d)
        for b in range(3):
            g = bm[:, b] - bm[:, [k for k in range(3) if k != b]].mean(axis=1)
            ax = axes[b, c]
            for j in range(d['n_post']):
                ax.plot(t, g[j], lw=0.8, color=cols[j])
            for s in d['swap_times_s']:
                ax.axvline(s, color='k', ls='--', lw=0.6)
            ax.axhline(0, color='grey', lw=0.4)
            if c == 0:
                ax.set_ylabel(f"tuned gap to {NAMES[b]}")
            if b == 0:
                ax.set_title(f"seed {d['seed']}", fontsize=9)
    fig.suptitle("v1 A|B|C|A|C: per-neuron tuned gap to each block (dashed = swaps)")
    fig.tight_layout(); fig.savefig(HERE / "v1_tuned_gaps.png", dpi=80)
    print("saved v1_tuned_gaps.png")


if __name__ == '__main__':
    main()

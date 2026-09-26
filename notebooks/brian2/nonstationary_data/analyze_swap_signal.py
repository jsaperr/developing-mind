"""Does the substrate itself emit an observable signal when the world changes, and does that signal
distinguish a RETURN to a known pattern from a NEW one? (Candidate source of the phase/context
awareness episodic check (b) is blocked on -- see system_contract.md.)

Note on the phase-aligned gap: it flips sign AT a swap because the reference block changes, not
because weights move (checked below: weights are continuous across the swap). So "every neuron's
aligned gap goes negative at the swap" is re-referencing, not evidence the network noticed anything.
This script looks at firing rate instead, which the network could actually observe.

Checks (N=7, 13mV/1.5 A->B->A runs):
  1. Population rate change in the first 10s after a swap vs the same windows at non-swap times
     (t=500/1500/2500) as a null.
  2. Swap 1 (to novel B) vs swap 2 (return to familiar A), paired per seed.
  3. Decomposition at swap 2: neurons still holding A (retainers) vs neurons on B (trackers).
Confound for 2, not removed here: swap 2 is later and hits an already-split population. The
novel-C batch (../novelc_data, swap 2 goes to a never-seen pattern) is the control.

Usage: analyze_swap_signal.py [glob prefix, default nonstat_n7_reliable_13_1p5]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

HERE = Path(__file__).resolve().parent
WINDOWS = [(-100, 0), (0, 10), (10, 30), (30, 60), (60, 120), (120, 300)]


def block0_gap_at(d, t_lo, t_hi):
    """Absolute (block A - block B) mean weight per neuron over [t_lo, t_hi]."""
    w = np.array(d['weight_trace']); si = np.array(d['syn_i']); sj = np.array(d['syn_j'])
    t = np.array(d['weight_trace_t'])
    m = (t > t_lo) & (t <= t_hi)
    return np.array([w[(sj == j) & (si < 10)][:, m].mean() - w[(sj == j) & (si >= 10)][:, m].mean()
                     for j in range(d['n_post'])])


def analyze(prefix):
    files = sorted(glob.glob(str(HERE / f"{prefix}_seed*.json")))
    swap_w, null_w, dips, hold, trk, max_dw = [], [], [], [], [], []
    for f in files:
        d = json.load(open(f))
        if d.get('status') != 'completed':
            continue
        r = np.array(d['spike_rate_bins'])
        s1, s2 = (int(s) for s in d['swap_times_s'])
        win = lambda s: [r[:, max(0, s + a):s + b].mean() for a, b in WINDOWS]
        swap_w += [win(s1), win(s2)]
        null_w += [win(s) for s in (s1 // 2, (s1 + s2) // 2, (s2 + int(d['total_s'])) // 2)]
        dip = lambda s: r[:, s:s + 10].mean() - r[:, s - 100:s].mean()
        area = lambda s: (r[:, s:s + 120].mean(0) - r[:, s - 100:s].mean()).sum()
        dips.append((dip(s1), dip(s2), area(s1), area(s2)))

        g = block0_gap_at(d, s2 - 50, s2)
        for j in range(d['n_post']):
            dr = r[j, s2:s2 + 10].mean() - r[j, s2 - 100:s2].mean()
            (hold if g[j] > 0 else trk).append(dr)

        w = np.array(d['weight_trace']); t = np.array(d['weight_trace_t'])
        for s in (s1, s2):
            a, b = np.searchsorted(t, s - 2), np.searchsorted(t, s + 2)
            max_dw.append(np.abs(w[:, b] - w[:, a]).max())

    sw, nl, dips = np.array(swap_w), np.array(null_w), np.array(dips)
    print(f"{prefix}: {len(dips)} seeds")
    print("1. population mean rate (Hz), windows rel. to event:", [f"{a}..{b}s" for a, b in WINDOWS])
    print("   swap    :", np.round(sw.mean(0), 2).tolist())
    print("   non-swap:", np.round(nl.mean(0), 2).tolist())
    d_sw, d_nl = sw[:, 1] - sw[:, 0], nl[:, 1] - nl[:, 0]
    print(f"   first-10s change: swap {d_sw.mean():+.2f} (sd {d_sw.std():.2f}) vs non-swap "
          f"{d_nl.mean():+.2f} (sd {d_nl.std():.2f}); MWU p={mannwhitneyu(d_sw, d_nl).pvalue:.2g}")
    print(f"   max |dw| of any synapse across a 4s window at a swap: {max(max_dw):.3f}")
    print("2. novel (swap1) vs familiar return (swap2), paired per seed:")
    print(f"   first-10s dip: novel {dips[:, 0].mean():+.2f}  familiar {dips[:, 1].mean():+.2f}  "
          f"(familiar smaller in {np.sum(np.abs(dips[:, 1]) < np.abs(dips[:, 0]))}/{len(dips)}; "
          f"Wilcoxon p={wilcoxon(dips[:, 0], dips[:, 1]).pvalue:.4f})")
    print(f"   120s integrated deficit (Hz*s): novel {dips[:, 2].mean():.0f}  familiar {dips[:, 3].mean():.0f}")
    print("3. at the familiar return, first-10s rate change per neuron:")
    print(f"   retainers holding A (n={len(hold)}): {np.mean(hold):+.2f} Hz  [{min(hold):+.1f}, {max(hold):+.1f}]")
    print(f"   trackers on B       (n={len(trk)}): {np.mean(trk):+.2f} Hz  [{min(trk):+.1f}, {max(trk):+.1f}]")


if __name__ == '__main__':
    analyze(sys.argv[1] if len(sys.argv) > 1 else "nonstat_n7_reliable_13_1p5")

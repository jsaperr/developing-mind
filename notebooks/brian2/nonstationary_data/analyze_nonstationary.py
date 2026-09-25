"""Exploratory readout for the non-stationary-correlation batch. No pass/fail: prints per-seed
trajectories sampled on a coarse grid, plus descriptive landmarks, and saves small-multiple figures
of the full phase-aligned gap traces so the trajectories can be looked at directly.

Landmarks are descriptive, not thresholds: the phase-aligned gap of the best-tracking neuron at
the end of each phase, and the first time after each swap that it crosses 0 and crosses half of
its own phase-1 plateau (a plateau-relative yardstick, since absolute separation varies by seed).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.brian2_stdp.metrics import phase_aligned_per_neuron_gap

HERE = Path(__file__).resolve().parent
GRID_S = 100


def load(path):
    d = json.load(open(path))
    if d.get('status') != 'completed':
        return None
    w = np.array(d['weight_trace'])
    t = np.array(d['weight_trace_t'])
    gap, phase = phase_aligned_per_neuron_gap(
        w, t, np.array(d['syn_i']), np.array(d['syn_j']), d['n_post'], d['swap_times_s'],
        d['phase_corr_blocks'])
    return d, t, gap, phase


def first_cross(t, y, t_from, level):
    m = (t >= t_from) & (y >= level)
    return float(t[m][0] - t_from) if m.any() else None


def describe(name):
    files = sorted(glob.glob(str(HERE / f"nonstat_{name}_seed*.json")))
    print(f"\n===== {name}: {len(files)} result files =====")
    rows = []
    for f in files:
        got = load(f)
        if got is None:
            print(f"  {Path(f).name}: NOT COMPLETED")
            continue
        d, t, gap, phase = got
        swaps = d['swap_times_s']
        best = gap.max(axis=0)                      # best-tracking neuron's aligned gap over time
        ends = [0.0] + swaps + [d['total_s']]
        end_vals = [float(best[(t > e1 - 50) & (t <= e1)].mean()) for e1 in ends[1:]]
        plateau1 = end_vals[0]
        cross = []
        for s in swaps:
            cross.append((first_cross(t, best, s, 0.0), first_cross(t, best, s, 0.5 * plateau1)))
        wt = np.array(d['w_total_trace'])
        rows.append((d['seed'], end_vals, cross))
        grid = [int(np.searchsorted(t, g)) for g in range(GRID_S, int(d['total_s']) + 1, GRID_S)]
        print(f"  seed {d['seed']}: aligned-gap(best neuron) at end of phases 1/2/3 = "
              f"{[round(v, 2) for v in end_vals]}  | after swap1: cross0/half-plateau = {cross[0]}  "
              f"| after swap2 = {cross[1]}  | w_total range {wt.min():.2f}-{wt.max():.2f}")
        print("     best-neuron aligned gap every 100s: " +
              " ".join(f"{best[min(i, len(best) - 1)]:+.2f}" for i in grid))
        per_neuron_end = [np.round(gap[:, (t > e1 - 50) & (t <= e1)].mean(axis=1), 2).tolist() for e1 in ends[1:]]
        print(f"     per-neuron aligned gap at end of phases: {per_neuron_end}")
    return rows


def figures(name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(str(HERE / f"nonstat_{name}_seed*.json")))
    files = [f for f in files if json.load(open(f)).get('status') == 'completed']
    if not files:
        return
    n = len(files)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 6), sharey=True, squeeze=False)
    for ax, f in zip(axes.ravel(), files):
        d, t, gap, phase = load(f)
        for j in range(gap.shape[0]):
            ax.plot(t, gap[j], lw=0.9, label=f"neuron {j}")
        for s in d['swap_times_s']:
            ax.axvline(s, color='k', ls='--', lw=0.8)
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_title(f"seed {d['seed']}", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{name}: phase-aligned gap (corr - uncorr of the CURRENTLY correlated block)")
    fig.supxlabel("time (s); dashed = world swaps")
    fig.tight_layout()
    out = HERE / f"nonstationary_{name}.png"
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == '__main__':
    for nm in ["reliable_13_1p5", "strong_tight_gate"]:
        describe(nm)
        figures(nm)

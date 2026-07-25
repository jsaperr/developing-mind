"""Compares the coarse-dt (0.2ms) arm against the existing default-dt (0.1ms) runs at the same
setting (13mV/1.5, 600s), from the boundary sweep (seeds 24120-24127).

Exact seed-matched agreement is impossible by construction -- a different dt draws a different
`xi` noise realization -- so this is a distributional comparison: same classification split, and
overlapping late-window gap statistics. Also reports the wall-clock ratio, which is the whole
point of asking.
"""
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.brian2_stdp.metrics import classify_hierarchy, compute_competitive_metrics

HERE = Path(__file__).resolve().parent
BOUNDARY = HERE.parent / "boundary_sweep_data"
N_POST, N_CORR = 3, 10


def summarize(paths, label):
    rows = []
    for p in paths:
        d = json.load(open(p))
        if d.get("status") != "completed":
            continue
        comp = compute_competitive_metrics(np.array(d["weight_trace"]), np.array(d["syn_i"]),
                                            np.array(d["syn_j"]), N_POST, n_corr=N_CORR)
        hier = classify_hierarchy(comp["per_neuron_gap"], np.array(d["weight_trace_t"]))
        rows.append({
            "seed": d["seed"], "label": hier["label"],
            "late_std": hier["late_window_std"],
            "final_gaps": comp["per_neuron_gap"][:, -1],
            "wall": d.get("wall_elapsed"),
        })
    labels = [r["label"] for r in rows]
    stds = [r["late_std"] for r in rows]
    walls = [r["wall"] for r in rows if r["wall"]]
    print(f"--- {label} (n={len(rows)}) ---")
    print(f"  differentiate={labels.count('differentiate')}, converge={labels.count('converge')}, "
          f"disorder={labels.count('disorder')}")
    print(f"  late_window_std: mean={np.mean(stds):.4f}, min={np.min(stds):.4f}, max={np.max(stds):.4f}")
    if walls:
        print(f"  wall per run: mean={np.mean(walls):.1f}s")
    all_gaps = np.concatenate([r["final_gaps"] for r in rows])
    print(f"  final per-neuron gaps: mean={all_gaps.mean():.4f}, min={all_gaps.min():.4f}, "
          f"max={all_gaps.max():.4f}")
    return rows, np.mean(walls) if walls else None


def main():
    coarse = sorted(HERE.glob("dtcheck_dt0.2_seed*.json"))
    fine = sorted(BOUNDARY.glob("boundary_inhib13_gap1.50_seed*.json"))
    _, w_fine = summarize(fine, "dt=0.1ms (default, from boundary sweep)")
    _, w_coarse = summarize(coarse, "dt=0.2ms (coarse arm)")
    if w_fine and w_coarse:
        print(f"\nwall-clock speedup: {w_fine / w_coarse:.2f}x")


if __name__ == "__main__":
    main()

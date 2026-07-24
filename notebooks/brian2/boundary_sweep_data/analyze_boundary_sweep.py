"""Analysis for the boundary-mapping sweep. Two curves, reported separately per web's
instruction, not collapsed into one score:
  - reliability: fraction of seeds classified 'differentiate'.
  - reentry_rate: fraction of ALL seeds (not just differentiating ones) showing genuine tier
    reentry (detect_tier_reentry) -- an excluded neuron actually regaining top-tier membership,
    not noise-trading among neurons that never left.

Also reports disorder_rate (fraction classified 'disorder' by classify_hierarchy) alongside,
since disorder seeds by construction show constant top-tier instability -- worth knowing whether
reentry_rate at a given point is driven by genuine 5003-style one-time promotions or is mostly a
byproduct of seeds that never settled into anything stable in the first place.

Falsification question, stated before running (see run_boundary_sweep.py's docstring): does a
region exist with both reliability>=60% AND reentry_rate>0, or do the two curves trade off
cleanly with no overlap anywhere in the tested 10-13mV x 1.0-1.5 range?
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.brian2_stdp.metrics import classify_hierarchy, compute_competitive_metrics, detect_tier_reentry

OUT_DIR = Path(__file__).resolve().parent
N_POST = 3
N_CORR = 10
RELIABLE_THRESHOLD = 0.6


def load_results():
    by_combo = defaultdict(list)
    for path in sorted(OUT_DIR.glob("boundary_inhib*_seed*.json")):
        with open(path) as f:
            d = json.load(f)
        if d.get("status") != "completed":
            print(f"skipping {path.name}: status={d.get('status')}")
            continue
        by_combo[d["calibration_combo_name"]].append(d)
    return by_combo


def analyze_one_seed(d):
    weight_trace = np.array(d["weight_trace"])
    weight_trace_t = np.array(d["weight_trace_t"])
    syn_i = np.array(d["syn_i"])
    syn_j = np.array(d["syn_j"])

    comp = compute_competitive_metrics(weight_trace, syn_i, syn_j, N_POST, n_corr=N_CORR)
    hier = classify_hierarchy(comp["per_neuron_gap"], weight_trace_t)
    reentry = detect_tier_reentry(comp["per_neuron_gap"], weight_trace_t)
    return {
        "seed": d["seed"], "label": hier["label"], "late_window_std": hier["late_window_std"],
        "reentered": reentry["reentered"], "first_reentry_t": reentry["first_reentry_t"],
        "inhib_strength_mV": d["inhib_strength_mV"], "gap_scale": d["gap_scale"],
    }


def analyze_grid_point(combo_name, seed_results):
    per_seed = [analyze_one_seed(d) for d in seed_results]
    n_total = len(per_seed)
    n_differentiate = sum(1 for r in per_seed if r["label"] == "differentiate")
    n_converge = sum(1 for r in per_seed if r["label"] == "converge")
    n_disorder = sum(1 for r in per_seed if r["label"] == "disorder")
    n_reentered = sum(1 for r in per_seed if r["reentered"])

    reliability = n_differentiate / n_total
    reentry_rate = n_reentered / n_total
    disorder_rate = n_disorder / n_total

    inhib = per_seed[0]["inhib_strength_mV"]
    gap_scale = per_seed[0]["gap_scale"]
    return {
        "combo_name": combo_name, "inhib_strength_mV": inhib, "gap_scale": gap_scale,
        "n_seeds": n_total, "reliability": reliability, "reentry_rate": reentry_rate,
        "disorder_rate": disorder_rate, "n_differentiate": n_differentiate,
        "n_converge": n_converge, "n_disorder": n_disorder, "n_reentered": n_reentered,
        "per_seed": per_seed,
    }


def main():
    by_combo = load_results()
    if not by_combo:
        print("No completed boundary sweep results found yet.")
        return

    grid = [analyze_grid_point(name, seeds) for name, seeds in by_combo.items()]
    grid.sort(key=lambda r: (r["inhib_strength_mV"], r["gap_scale"]))

    print(f"{'inhib':>6} {'gap':>5} {'n':>3} {'reliability':>11} {'reentry_rate':>12} {'disorder_rate':>13}")
    for r in grid:
        print(f"{r['inhib_strength_mV']:>6.1f} {r['gap_scale']:>5.2f} {r['n_seeds']:>3d} "
              f"{r['reliability']:>11.2f} {r['reentry_rate']:>12.2f} {r['disorder_rate']:>13.2f}")

    overlap = [r for r in grid if r["reliability"] >= RELIABLE_THRESHOLD and r["reentry_rate"] > 0]
    print(f"\nOverlap region (reliability>={RELIABLE_THRESHOLD:.0%} AND reentry_rate>0): {len(overlap)} point(s)")
    for r in overlap:
        print(f"  inhib={r['inhib_strength_mV']}, gap_scale={r['gap_scale']}: "
              f"reliability={r['reliability']:.2f}, reentry_rate={r['reentry_rate']:.2f}, "
              f"disorder_rate={r['disorder_rate']:.2f}")

    out_path = OUT_DIR / "boundary_analysis.json"
    with open(out_path, "w") as f:
        json.dump(grid, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()

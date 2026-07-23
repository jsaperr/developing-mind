"""Analysis for the population-competition bistability sweep (run_sweep.py's output).

Tracks two things per grid point, per web's design -- not a single collapsed number:
  - Reliability: fraction of seeds that differentiate vs. converge (classify_differentiation,
    the same late-window cross-neuron-gap-std>0.03 criterion used for the original
    strong_tight_gate calibration/seed-expansion).
  - Richness (among differentiating seeds only): holder-identity swap count within the 600s
    window (count_identity_swaps) -- a PROXY for the identity-churn spectrum found at 5000s in
    the n=7 extension, not a full frozen/reorganization/never-locks-in classification (that
    needs the full duration; this only asks whether early dynamics already show real variation
    across seeds, or look uniformly sharp/rigid regardless of seed).

Stated up front, before looking at results (same falsification-first discipline as the rest of
this project): the interesting outcome is a region where differentiation is reliable (>80%) AND
richness (swap-count spread across differentiating seeds) stays real, not collapsed toward zero.
A region where reliability only rises as richness drops would be a genuine tradeoff finding, same
shape as savings-vs-content-fidelity in the Hopfield work -- report that plainly if that's what
the grid shows, don't fish for a "best of both" reading the data doesn't support.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.brian2_stdp.metrics import classify_differentiation, compute_competitive_metrics, count_identity_swaps

OUT_DIR = Path(__file__).resolve().parent
N_POST = 3
N_CORR = 10
RELIABLE_THRESHOLD = 0.8  # >=80% differentiate = "reliable", per web's stated target


def load_results():
    by_combo = defaultdict(list)
    for path in sorted(OUT_DIR.glob("sweep_inhib*_seed*.json")):
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
    label, late_std = classify_differentiation(comp["per_neuron_gap"], weight_trace_t)
    swaps = count_identity_swaps(comp["holder_identity"])
    return {
        "seed": d["seed"], "label": label, "late_window_std": late_std,
        "identity_swaps": swaps, "inhib_strength_mV": d["inhib_strength_mV"],
        "gap_scale": d["gap_scale"],
    }


def analyze_grid_point(combo_name, seed_results):
    per_seed = [analyze_one_seed(d) for d in seed_results]
    n_total = len(per_seed)
    differentiating = [r for r in per_seed if r["label"] == "differentiate"]
    n_diff = len(differentiating)
    reliability = n_diff / n_total if n_total > 0 else None

    swap_counts = [r["identity_swaps"] for r in differentiating]
    richness = {
        "n_differentiating": n_diff,
        "swap_counts": swap_counts,
        "mean_swaps": float(np.mean(swap_counts)) if swap_counts else None,
        "std_swaps": float(np.std(swap_counts)) if swap_counts else None,
        "min_swaps": min(swap_counts) if swap_counts else None,
        "max_swaps": max(swap_counts) if swap_counts else None,
    }

    inhib = per_seed[0]["inhib_strength_mV"]
    gap_scale = per_seed[0]["gap_scale"]
    return {
        "combo_name": combo_name, "inhib_strength_mV": inhib, "gap_scale": gap_scale,
        "n_seeds": n_total, "reliability": reliability, "richness": richness,
        "per_seed": per_seed,
    }


def main():
    by_combo = load_results()
    if not by_combo:
        print("No completed sweep results found yet.")
        return

    grid_report = [analyze_grid_point(name, seeds) for name, seeds in by_combo.items()]
    grid_report.sort(key=lambda r: (r["inhib_strength_mV"], r["gap_scale"]))

    print(f"{'inhib':>6} {'gap':>5} {'n':>3} {'reliability':>11} {'mean_swaps':>11} {'std_swaps':>10} {'swap_range':>14}")
    for r in grid_report:
        rel_str = f"{r['reliability']:.2f}" if r["reliability"] is not None else "n/a"
        rich = r["richness"]
        mean_str = f"{rich['mean_swaps']:.1f}" if rich["mean_swaps"] is not None else "n/a"
        std_str = f"{rich['std_swaps']:.1f}" if rich["std_swaps"] is not None else "n/a"
        range_str = f"[{rich['min_swaps']},{rich['max_swaps']}]" if rich["min_swaps"] is not None else "n/a"
        print(f"{r['inhib_strength_mV']:>6.1f} {r['gap_scale']:>5.1f} {r['n_seeds']:>3d} "
              f"{rel_str:>11} {mean_str:>11} {std_str:>10} {range_str:>14}")

    reliable_and_rich = [r for r in grid_report if r["reliability"] is not None
                          and r["reliability"] >= RELIABLE_THRESHOLD
                          and r["richness"]["std_swaps"] is not None
                          and r["richness"]["std_swaps"] > 0]
    print(f"\nReliable (>={RELIABLE_THRESHOLD:.0%}) AND richness present (std_swaps>0): "
          f"{len(reliable_and_rich)} grid point(s)")
    for r in reliable_and_rich:
        print(f"  inhib={r['inhib_strength_mV']}, gap_scale={r['gap_scale']}: "
              f"reliability={r['reliability']:.2f}, mean_swaps={r['richness']['mean_swaps']:.1f}, "
              f"std_swaps={r['richness']['std_swaps']:.1f}")

    out_path = OUT_DIR / "sweep_analysis.json"
    with open(out_path, "w") as f:
        json.dump(grid_report, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()

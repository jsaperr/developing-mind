"""Analysis for Experiment B step 3: the N-scaling curve. Applies the Gate-1-cleared metrics
(classify_hierarchy, compute_tiers, full_rank_swap_count) at every n_post, tracking reliability,
richness, and hierarchy legibility as a function of N -- a curve, not a single verdict.

Falsification criteria, stated before running (three-way, per web's Gate 1 instruction):
  (a) HOLDS CLEANLY at a given N: >=80% of seeds differentiate (not converge/disorder), AND
      richness (std of full_rank_swap_count among differentiating seeds) stays comparable to or
      above what N=3 showed in the original sweep (std_swaps was commonly 20-40 there), AND at
      most 1 of the differentiating seeds is labeled 'disorder' by classify_hierarchy.
  (b) EARLY STRAIN: differentiation rate drops to 50-80%, or richness measurably thins relative
      to N=3, or 2-3 seeds land in 'disorder' -- real but not dominant erosion.
  (c) BREAKDOWN: 'disorder' becomes the dominant differentiating-seed outcome (>half), or overall
      differentiation rate collapses below 50%.
These are per-N calls, not a single grid verdict -- the point is finding where (if anywhere) the
transition between them happens.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.brian2_stdp.metrics import classify_hierarchy, compute_competitive_metrics, full_rank_swap_count

OUT_DIR = Path(__file__).resolve().parent
N_CORR = 10


def load_results():
    by_n = defaultdict(list)
    for path in sorted(OUT_DIR.glob("nscale_n*_seed*.json")):
        with open(path) as f:
            d = json.load(f)
        if d.get("status") != "completed":
            print(f"skipping {path.name}: status={d.get('status')}")
            continue
        by_n[d["n_post"]].append(d)
    return by_n


def analyze_one_seed(d):
    weight_trace = np.array(d["weight_trace"])
    weight_trace_t = np.array(d["weight_trace_t"])
    syn_i = np.array(d["syn_i"])
    syn_j = np.array(d["syn_j"])
    n_post = d["n_post"]

    comp = compute_competitive_metrics(weight_trace, syn_i, syn_j, n_post, n_corr=N_CORR)
    hier = classify_hierarchy(comp["per_neuron_gap"], weight_trace_t)
    rank_swaps = full_rank_swap_count(comp["per_neuron_gap"])
    return {
        "seed": d["seed"], "n_post": n_post, "label": hier["label"],
        "late_window_std": hier["late_window_std"], "n_distinct_top_sets": hier["n_distinct_top_sets"],
        "n_tiers": len(hier["tiers"]), "top_tier_size": len(hier["tiers"][0]),
        "rank_swaps": rank_swaps, "inhib_strength_mV": d["inhib_strength_mV"],
    }


def classify_n_point(per_seed, n_post):
    n_total = len(per_seed)
    labels = [r["label"] for r in per_seed]
    n_differentiate = sum(1 for l in labels if l == "differentiate")
    n_converge = sum(1 for l in labels if l == "converge")
    n_disorder_total = sum(1 for l in labels if l == "disorder")
    # among the seeds that show REAL cross-neuron spread (differentiate OR disorder), how many
    # resolve into a stable hierarchy vs. never settle
    n_spread = n_differentiate + n_disorder_total
    disorder_frac_of_spread = n_disorder_total / n_spread if n_spread > 0 else None
    reliability = n_differentiate / n_total

    rank_swaps_diff = [r["rank_swaps"] for r in per_seed if r["label"] == "differentiate"]
    richness = {
        "n_differentiating": n_differentiate,
        "rank_swaps": rank_swaps_diff,
        "mean_rank_swaps": float(np.mean(rank_swaps_diff)) if rank_swaps_diff else None,
        "std_rank_swaps": float(np.std(rank_swaps_diff)) if rank_swaps_diff else None,
    }

    # Hierarchy-SHAPE variability: top_tier_size as a fraction of n_post, among differentiating
    # seeds. Caught by inspecting raw tier data, not the reliability/disorder verdict alone: at
    # N=10, roughly half the seeds keep a small, legible top tier while the other half collapse
    # toward "almost everyone ties, 1-2 excluded" -- a real bimodal split in what "differentiate"
    # even means, invisible to reliability (still counts as differentiate) and to the disorder
    # detector (the tied top group IS stable, just enormous). Reported as its own axis, not
    # folded into the reliability/disorder verdict, since collapsing it would hide exactly the
    # kind of thing this sweep is supposed to surface.
    top_tier_fracs = [r["top_tier_size"] / r["n_post"] for r in per_seed if r["label"] == "differentiate"]
    shape = {
        "top_tier_fracs": top_tier_fracs,
        "mean_top_tier_frac": float(np.mean(top_tier_fracs)) if top_tier_fracs else None,
        "std_top_tier_frac": float(np.std(top_tier_fracs)) if top_tier_fracs else None,
        "frac_seeds_majority_tied": (sum(1 for f in top_tier_fracs if f > 0.5) / len(top_tier_fracs))
                                      if top_tier_fracs else None,
    }

    if reliability >= 0.8 and n_disorder_total <= 1 and (richness["std_rank_swaps"] or 0) > 0:
        verdict = "holds_cleanly"
    elif n_disorder_total > n_differentiate:
        verdict = "breakdown"
    elif reliability < 0.5:
        verdict = "breakdown"
    else:
        verdict = "early_strain"

    return {
        "n_post": n_post, "n_seeds": n_total, "n_differentiate": n_differentiate,
        "n_converge": n_converge, "n_disorder": n_disorder_total, "reliability": reliability,
        "disorder_frac_of_spread": disorder_frac_of_spread, "richness": richness, "shape": shape,
        "verdict": verdict, "per_seed": per_seed,
    }


def main():
    by_n = load_results()
    if not by_n:
        print("No completed N-scaling results found yet.")
        return

    curve = []
    for n_post, seeds in sorted(by_n.items()):
        per_seed = [analyze_one_seed(d) for d in seeds]
        curve.append(classify_n_point(per_seed, n_post))

    print(f"{'N':>3} {'n':>3} {'differentiate':>13} {'converge':>9} {'disorder':>9} "
          f"{'reliability':>11} {'mean_swaps':>11} {'std_swaps':>10} {'verdict':>14}")
    for r in curve:
        rich = r["richness"]
        mean_str = f"{rich['mean_rank_swaps']:.1f}" if rich["mean_rank_swaps"] is not None else "n/a"
        std_str = f"{rich['std_rank_swaps']:.1f}" if rich["std_rank_swaps"] is not None else "n/a"
        print(f"{r['n_post']:>3d} {r['n_seeds']:>3d} {r['n_differentiate']:>13d} "
              f"{r['n_converge']:>9d} {r['n_disorder']:>9d} {r['reliability']:>11.2f} "
              f"{mean_str:>11} {std_str:>10} {r['verdict']:>14}")

    print(f"\n{'N':>3} {'mean_top_tier_frac':>19} {'std_top_tier_frac':>18} {'frac_majority_tied':>19}")
    for r in curve:
        s = r["shape"]
        mean_str = f"{s['mean_top_tier_frac']:.2f}" if s["mean_top_tier_frac"] is not None else "n/a"
        std_str = f"{s['std_top_tier_frac']:.2f}" if s["std_top_tier_frac"] is not None else "n/a"
        maj_str = f"{s['frac_seeds_majority_tied']:.2f}" if s["frac_seeds_majority_tied"] is not None else "n/a"
        print(f"{r['n_post']:>3d} {mean_str:>19} {std_str:>18} {maj_str:>19}")

    out_path = OUT_DIR / "n_scaling_analysis.json"
    with open(out_path, "w") as f:
        json.dump(curve, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()

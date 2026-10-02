"""Modal fidelity check: reproduce the arc-01 apre005 ensemble (seeds 2001-2008, 5000 s) remotely
and compare against the saved results in apre005_ensemble_data/.

Run from the repo root, in the developing-mind conda env:
    python -m modal run notebooks/brian2/modal_ensemble_check.py

The run body mirrors apre005_ensemble_data/run_single_seed.py exactly (frozen script, not edited;
it can't be imported on Modal because it hardcodes a parents[3] path).

Predictions (written before the run):
  P1. This single-neuron rig has no Brian2 membrane noise and the input is numpy-seeded, so on a
      matching toolchain final weights and group_mean_gap match the saved files bit for bit.
  P2. If P1 fails on float drift (different CPU), trajectories diverge slightly but statistics hold:
      post_rate ~18.7-18.9 Hz, min_group_mean_gap in [-0.022, -0.007], max_overlap_fraction in
      [0.68, 1.0] for all 8 seeds.
  P3. A miss at the statistics level means the Modal setup is not a faithful substitute.
"""
import json
from pathlib import Path

import modal

app = modal.App("developing-mind-ensemble-check")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    # versions match the local developing-mind conda env
    .pip_install("brian2==2.9.0", "cython==3.2.8", "numpy==2.0.1", "setuptools==82.0.1")
    .add_local_python_source("src")
)

TARGET_RATE = 20.0
APRE_VAL = 0.005
P_SHARE = 0.9
DURATION_S = 5000.0


@app.function(image=image, cpu=1.0, memory=2048, timeout=3600)
def run_long_seed(seed_val: int):
    import time

    import numpy as np
    from brian2 import SpikeMonitor, StateMonitor, ms, prefs, run, second, start_scope

    from src.brian2_stdp.metrics import compute_group_metrics
    from src.brian2_stdp.network import build_network
    from src.brian2_stdp.spikes import build_presynaptic_input

    start_scope()
    rng = np.random.default_rng(seed_val)
    idx, t = build_presynaptic_input(TARGET_RATE, P_SHARE, DURATION_S, rng)
    pre, post, syn = build_network(idx, t, APRE_VAL)
    post_spikes = SpikeMonitor(post)
    weight_trace = StateMonitor(syn, "w", record=True, dt=500 * ms)

    wall_start = time.time()
    run(DURATION_S * second)
    wall_elapsed = time.time() - wall_start

    metrics = compute_group_metrics(weight_trace.w[:], n_corr=10)
    final_w = np.array(syn.w[:])
    return {
        "seed": seed_val,
        "target": prefs.codegen.target,
        "wall_elapsed": wall_elapsed,
        "post_rate": post_spikes.count[0] / DURATION_S,
        "group_mean_gap": metrics["group_mean_gap"].tolist(),
        "overlap_fraction": metrics["overlap_fraction"].tolist(),
        "final_corr_w": final_w[:10].tolist(),
        "final_uncorr_w": final_w[10:].tolist(),
        "min_group_mean_gap": float(metrics["group_mean_gap"].min()),
        "max_overlap_fraction": float(metrics["overlap_fraction"].max()),
    }


@app.local_entrypoint()
def main():
    import numpy as np

    data_dir = Path(__file__).parent / "apre005_ensemble_data"
    seeds = list(range(2001, 2009))
    out_dir = Path(__file__).parent / "modal_ensemble_check_out"
    out_dir.mkdir(exist_ok=True)

    for r in run_long_seed.map(seeds):
        (out_dir / f"modal_seed{r['seed']}.json").write_text(json.dumps(r))
        ref = json.loads((data_dir / f"apre_ensemble_seed{r['seed']}.json").read_text())
        w_new = np.array(r["final_corr_w"] + r["final_uncorr_w"])
        w_ref = np.array(ref["final_corr_w"] + ref["final_uncorr_w"])
        g_new, g_ref = np.array(r["group_mean_gap"]), np.array(ref["group_mean_gap"])
        print(
            f"seed {r['seed']} [{r['target']}] wall {r['wall_elapsed']:.0f}s | "
            f"post_rate {r['post_rate']:.4f} vs {ref['post_rate']:.4f} | "
            f"min_gap {r['min_group_mean_gap']:.5f} vs {ref['min_group_mean_gap']:.5f} | "
            f"max_overlap {r['max_overlap_fraction']:.2f} vs {ref['max_overlap_fraction']:.2f} | "
            f"final_w max|diff| {np.abs(w_new - w_ref).max():.3g} | "
            f"gap-trace max|diff| {np.abs(g_new - g_ref).max():.3g} | "
            f"bit-identical {np.array_equal(w_new, w_ref) and np.array_equal(g_new, g_ref)}"
        )

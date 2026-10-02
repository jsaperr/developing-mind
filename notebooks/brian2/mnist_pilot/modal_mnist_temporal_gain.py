"""Temporal (causal) gain control on the reconciled substrate (exploration, 2026-10-02, Jasper: under an hour).
Per-image gain control uses image boundaries, an oracle. mnist_sim.build_input_temporal divides every pixel's rate by a
running estimate of total input (time constant tau_g, updated only while something is shown). So each image starts at the
previous image's gain and adapts. Input check: corr(spikes per image, ink) is 0.99 with none, 0.00 per-image, and 0.42 /
0.72 / 0.92 at tau_g 20 / 100 / 1000 ms.
Substrate: winner-take-all 20 mV (gate off) + leaky fair-share (0.05 mV, tau 100 s); N=40, budget 30, classes 0-3, 1 pass,
seeds 70100-70107. Per-image reference: mnist_leak_tau100 (74.8%, every digit, novelty 3.2x).
Arms: none, tau_g 5, 20, 100 ms. Files mnist_tg_<arm>_seed<S>.json.gz; scored by analyze_mnist_pilot.py <prefix> v3.

PREDICTIONS (before launch):
  TG-P1 tau_g 5 and 20 ms reach >= 65% with every digit >= 3 neurons in >= 6/8 (close to the per-image oracle).
  TG-P2 accuracy falls as tau_g grows; tau_g 100 ms < tau_g 20 ms.
  TG-P3 no gain control on this substrate: < 65% (the ink bias returns), and below every causal arm.
"""
import gzip
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from modal_common import IMAGE  # noqa: E402

image = (IMAGE.add_local_file(str(HERE.parent / "modal_common.py"), "/root/modal_common.py")
         .add_local_file(str(HERE / "mnist_sim.py"), "/root/mnist_sim.py")
         .add_local_file(str(HERE / "data" / "pilot_subset.npz"), "/root/pilot_subset.npz"))
app = modal.App("developing-mind-temporal-gain")
SEEDS = list(range(70100, 70108))
ARMS = {"none": None, "tg5": 0.005, "tg20": 0.02, "tg100": 0.1}


@app.function(image=image, cpu=1.0, memory=4096, timeout=2 * 3600, max_containers=40,
              retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_seed(arm: str, seed: int) -> dict:
    import numpy as np
    import mnist_sim as S
    subset = dict(np.load("/root/pilot_subset.npz"))
    fair = dict(theta_plus_mV=0.05, tau_theta_s=100.0, fair_share=True)
    tg = ARMS[arm]
    r = S.run_pilot(seed, subset, 30.0, n_post=40, normalize=False, adaptive=fair, wta=True, gain_tau_s=tg)
    r["arm"] = arm
    return r


@app.local_entrypoint()
def main():
    todo = [(a, s) for a in ARMS for s in SEEDS if not (HERE / f"mnist_tg_{a}_seed{s}.json.gz").exists()]
    print(f"{len(todo)} runs to do", flush=True)
    n = 0
    for r in run_seed.starmap(todo, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print(f"FAILED {r!r}", flush=True); continue
        with gzip.open(HERE / f"mnist_tg_{r['arm']}_seed{r['seed']}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(r, f)
        n += 1
        print(f"saved {r['arm']} {r['seed']}", flush=True)
    print(f"== {n}/{len(todo)} completed", flush=True)

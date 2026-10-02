"""MNIST pilot v3 (generalization plan step 4a): v2 found sharp specialization but no diversity (nearly all 40 neurons
specialized on one digit). v3 adds the missing rule, winner-take-all lateral inhibition, DERIVED rather than tuned: each
competitor spike pushes a neuron down by its full threshold distance (v_thresh - v_rest = 20 mV), ambiguity gate off.
Same 8 seeds as v1/v2 (paired). N=40, weight budget 30, adaptive threshold as v2 (Diehl & Cook's constants).
  wta_adapt       raw input + adaptive threshold + WTA (Diehl & Cook's recipe)
  wta_norm_adapt  the same with per-image gain control
Results: mnist_v3_<arm>_seed<S>.json.gz. Scored by `analyze_mnist_pilot.py <prefix> v3`.

Local smoke (200 training images, before launch):
- wta_adapt: about 10 spikes/image, 21 distinct winning neurons over 100 images, 37% of images with no response.
- wta_norm_adapt: about 11 spikes/image, 20 distinct winners, 1% silent.

v3 SCORING DEFINITIONS (fixed before any v3 data; v2 showed zero-response images break v1's definitions):
- A silent image (no spikes) is an ABSTENTION and counts as wrong in vote accuracy. Responsive-only accuracy is
  reported alongside.
- MP-P3 separation and MC-1 strength quintiles are computed over responsive images; the silent fraction is
  reported.
- MC-2: a silent test image counts as strange (nothing matched). The radius is the 5th percentile of best matches
  over responsive label-period images.

PREDICTIONS ON RECORD (written before launch):
  V3-P1 diversity: every class 0-3 gets >= 3 neurons assigned in >= 6/8 seeds, in both arms (v2: one class took
        31-40 of 40).
  V3-P2 the full pilot bar in wta_norm_adapt: specialization >= 60%, weight match >= 60%, vote accuracy >= 65% (silent
        = wrong).
  V3-P3 wta_adapt (raw input) is hurt by abstentions: > 20% of test images silent (faint digits can't beat the risen
        thresholds), so its accuracy with silent counted wrong is below wta_norm_adapt's.
  V3-P4 confidence: in wta_norm_adapt, the strongest-response quintile beats the weakest by >= 15 points (MC-1 bar),
        in >= 6/8 seeds.
  V3-P5 novelty: in wta_norm_adapt, held-out digits (4, 5) are strange >= 2x as often as known ones (MC-2 bar).
  V3-P6 the fingerprint separates classes (within > between, every pair, over responsive images) in >= 6/8 seeds,
        wta_norm_adapt.

Run from the repo root:  python -m modal run notebooks/brian2/mnist_pilot/modal_mnist_v3.py
"""
import gzip
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from modal_common import IMAGE  # noqa: E402

TARGET_TOTAL = 30.0
SEEDS = list(range(70100, 70108))
ARMS = ["wta_adapt", "wta_norm_adapt"]
ADAPT = dict(theta_plus_mV=0.05, tau_theta_s=1e4)
image = (IMAGE.add_local_file(str(HERE.parent / "modal_common.py"), "/root/modal_common.py")
         .add_local_file(str(HERE / "mnist_sim.py"), "/root/mnist_sim.py")
         .add_local_file(str(HERE / "data" / "pilot_subset.npz"), "/root/pilot_subset.npz"))
app = modal.App("developing-mind-mnist-v3")


@app.function(image=image, cpu=1.0, memory=4096, timeout=4 * 3600, max_containers=20,
              retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_seed(arm: str, seed: int) -> dict:
    import numpy as np
    import mnist_sim as S
    subset = dict(np.load("/root/pilot_subset.npz"))
    kw = dict(wta_adapt=dict(adaptive=ADAPT, wta=True), wta_norm_adapt=dict(normalize=True, adaptive=ADAPT, wta=True))[arm]
    r = S.run_pilot(seed, subset, TARGET_TOTAL, n_post=40, **kw)
    r["arm"] = arm
    return r


@app.local_entrypoint()
def main():
    todo = [(a, s) for a in ARMS for s in SEEDS if not (HERE / f"mnist_v3_{a}_seed{s}.json.gz").exists()]
    print(f"{len(todo)} runs to do", flush=True)
    n = 0
    for r in run_seed.starmap(todo, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print(f"FAILED {r!r}", flush=True); continue
        with gzip.open(HERE / f"mnist_v3_{r['arm']}_seed{r['seed']}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(r, f)
        n += 1
        print(f"{r['arm']} seed {r['seed']} saved (wall {r['wall_elapsed']:.0f}s)", flush=True)
    print(f"== {n}/{len(todo)} completed", flush=True)

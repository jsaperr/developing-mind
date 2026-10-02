"""MNIST pilot v2 (generalization plan step 4a): the two fixes for v1's ink confound, as separate arms, on the SAME 8
seeds as v1 (same image orders, same Brian2 noise, so every comparison with v1 is paired). N=40, weight budget 30
(v1's calibration; gain control keeps the mean drive the same).
  norm        per-image gain control: every image delivers the training set's mean total input rate
  adapt       Diehl & Cook's adaptive threshold (theta += 0.05 mV per spike, decay tau 1e4 s: their published
              constants, not tuned by us), raw inputs
  norm_adapt  both
Results: mnist_v2_<arm>_seed<S>.json.gz. Scored by analyze_mnist_pilot.py <prefix> with the SAME criteria as v1.

Local smoke (60 images, before launch): norm+adapt dropped from ~208 to ~27 spikes/image (mean theta 3.7 mV).
Recorded risk: with tau 1e4 s, theta may keep climbing until neurons fall nearly silent. Diehl & Cook compensated by
re-presenting low-response images at higher intensity; we haven't added that.

PREDICTIONS ON RECORD (written before launch):
  V2-P1 norm removes the ink confound but not the specialization failure: every class gets >= 3 neurons assigned in
        >= 6/8 seeds (v1: class 1 had 0 in 8/8), but specialization stays < 30% (competition is unchanged).
  V2-P2 adapt alone sparsifies but keeps the ink bias: spikes/image drop >= 5x vs v1, while class 1 still gets
        < 3 neurons in most seeds.
  V2-P3 norm_adapt is the best arm: specialization >= 40%, weight maps match the assigned class >= 50%, vote accuracy
        >= 65% on classes 0-3 (v1: 0%, 30%, 54%).
  V2-P4 with the ink confound gone, the strength signal is no longer inverted in norm_adapt: weakest-vs-strongest
        quintile gap >= 0 (v1: -51 points). (MC-1's full >= 15-point bar is reported but not expected.)
  V2-P5 novelty in norm_adapt: held-out strange ratio >= 2 (v1: 1.8).
  Failure mode to report, not hide: any arm where neurons go nearly silent (mean < 5 spikes/image in the test period).

Run from the repo root:  python -m modal run notebooks/brian2/mnist_pilot/modal_mnist_v2.py
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
ARMS = ["norm", "adapt", "norm_adapt"]
ADAPT = dict(theta_plus_mV=0.05, tau_theta_s=1e4)
image = (IMAGE.add_local_file(str(HERE.parent / "modal_common.py"), "/root/modal_common.py")
         .add_local_file(str(HERE / "mnist_sim.py"), "/root/mnist_sim.py")
         .add_local_file(str(HERE / "data" / "pilot_subset.npz"), "/root/pilot_subset.npz"))
app = modal.App("developing-mind-mnist-v2")


@app.function(image=image, cpu=1.0, memory=4096, timeout=4 * 3600, max_containers=20,
              retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_seed(arm: str, seed: int) -> dict:
    import numpy as np
    import mnist_sim as S
    subset = dict(np.load("/root/pilot_subset.npz"))
    kw = dict(norm=dict(normalize=True), adapt=dict(adaptive=ADAPT), norm_adapt=dict(normalize=True, adaptive=ADAPT))[arm]
    r = S.run_pilot(seed, subset, TARGET_TOTAL, n_post=40, **kw)
    r["arm"] = arm
    return r


@app.local_entrypoint()
def main():
    todo = [(a, s) for a in ARMS for s in SEEDS if not (HERE / f"mnist_v2_{a}_seed{s}.json.gz").exists()]
    print(f"{len(todo)} runs to do", flush=True)
    n = 0
    for r in run_seed.starmap(todo, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print(f"FAILED {r!r}", flush=True); continue
        with gzip.open(HERE / f"mnist_v2_{r['arm']}_seed{r['seed']}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(r, f)
        n += 1
        print(f"{r['arm']} seed {r['seed']} saved (wall {r['wall_elapsed']:.0f}s)", flush=True)
    print(f"== {n}/{len(todo)} completed", flush=True)

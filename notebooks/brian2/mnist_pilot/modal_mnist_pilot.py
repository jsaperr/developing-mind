"""MNIST pilot (generalization plan step 4a) on Modal: 8 seeds, N=40, target_total 30 (calibrate_mnist.py: the
smallest budget passing the pre-stated bar; 19.6 Hz, all neurons active). Core: mnist_sim.run_pilot (the same code
the calibration ran locally). Data: data/pilot_subset.npz from prep_mnist.py (git-ignored; shipped into the image).
Results are small (per-image spike counts + three weight snapshots), so they come back directly as
mnist_pilot_seed<S>.json.gz.

Caution recorded before launch: at target_total 30 every neuron fires for every image (~270 spikes per image across
40 neurons in calibration). Our competition is gentler than Diehl & Cook's (no adaptive threshold), so neurons may
all learn an "average digit" rather than specializing. If MP-P1 fails, the planned next step is the adaptive
threshold (a hand-built rule, logged as such), not a hand-tuned number.

PREDICTIONS ON RECORD (written before launch; scored by analyze_mnist_pilot.py). Labels are used only in scoring:
  MP-P1 specialization: >= 60% of neurons respond most to one class with a clear margin (their mean label-period
        response to their top class >= 1.5x their second class).
  MP-P2 weights look like digits: for >= 60% of neurons, the final weight map correlates more with the mean image of
        the neuron's assigned class than with any other known class's mean image.
  MP-P3 the fingerprint separates classes: on known-class test images, the mean within-class cosine of rectified
        fingerprints exceeds the between-class cosine for every class pair.
  MP-P4 classification (Diehl & Cook vote: each neuron assigned to its top label-period class; a test image goes to
        the class whose neurons respond most on average): >= 50% on the known classes (chance 25%). Exploratory.
  MC-1  calibration: across strength quintiles (unnormalized fingerprint size, quintiles of the run's own
        distribution), accuracy rises, and the weakest quintile is >= 15 points below the strongest.
  MC-2  novelty: memory (DormantGatedMemory, radius 0.8) runs over the training fingerprints at one check per image.
        A test image is "strange" when its best match to any stored entry (live or dormant) is below a self-set
        radius, the 5th percentile of label-period best matches. Held-out digits (4, 5) are strange >= 2x as often
        as known ones.
  Honest expectation: P1/P2 are genuinely uncertain (see the caution); P3 and MC-1 depend on them.

Run from the repo root:  python -m modal run notebooks/brian2/mnist_pilot/modal_mnist_pilot.py
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
image = (IMAGE.add_local_file(str(HERE / "mnist_sim.py"), "/root/mnist_sim.py")
         .add_local_file(str(HERE / "data" / "pilot_subset.npz"), "/root/pilot_subset.npz"))
app = modal.App("developing-mind-mnist-pilot")


@app.function(image=image, cpu=1.0, memory=4096, timeout=4 * 3600, max_containers=20,
              retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_seed(seed: int) -> dict:
    import numpy as np
    import mnist_sim as S
    subset = dict(np.load("/root/pilot_subset.npz"))
    return S.run_pilot(seed, subset, TARGET_TOTAL, n_post=40)


@app.local_entrypoint()
def main():
    todo = [s for s in SEEDS if not (HERE / f"mnist_pilot_seed{s}.json.gz").exists()]
    print(f"{len(todo)} seeds to run", flush=True)
    n = 0
    for r in run_seed.map(todo, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print(f"FAILED {r!r}", flush=True); continue
        with gzip.open(HERE / f"mnist_pilot_seed{r['seed']}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(r, f)
        n += 1
        print(f"seed {r['seed']} saved (wall {r['wall_elapsed']:.0f}s)", flush=True)
    print(f"== {n}/{len(todo)} completed", flush=True)

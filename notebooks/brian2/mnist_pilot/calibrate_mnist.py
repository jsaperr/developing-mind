"""Calibrates the weight budget (target_total) for rate-coded MNIST input BEFORE the pilot batch, the way
calibrate_novelc.py / calibrate_60.py did for the synthetic rigs. N=40, 196 inputs, 120 training images (60 s),
seed 70001, target_total in {10, 20, 30, 40}, everything else as the pilot.

Bar (stated before running, not tuned after): over the last 60 training images, the mean per-neuron rate during
presentations is in 3-20 Hz AND at least 50% of neurons spike at least once. Pick the SMALLEST passing
target_total. If none passes, report and stop (don't change gmax or the network).

Usage (repo root): python notebooks/brian2/mnist_pilot/calibrate_mnist.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2])); sys.path.insert(0, str(HERE))
import mnist_sim as S

z = dict(np.load(HERE / "data" / "pilot_subset.npz"))
small = dict(z)
for split in ("label", "test"):
    small[f"{split}_x"], small[f"{split}_y"] = z[f"{split}_x"][:4], z[f"{split}_y"][:4]
chosen = None
for tt in (10.0, 20.0, 30.0, 40.0):
    r = S.run_pilot(70001, small, tt, n_post=40, n_train=120)
    c = np.array(r["counts"])[:, 60:120]
    rate = c.sum(1) / (60 * S.PRESENT_S)
    active = float((c.sum(1) > 0).mean())
    ok = 3 <= rate.mean() <= 20 and active >= 0.5
    print(f"target_total {tt:4.0f}: mean rate {rate.mean():5.1f} Hz (max {rate.max():5.1f}), active neurons {active:.0%}, "
          f"spikes per image {c.sum(0).mean():.1f} -> {'PASS' if ok else 'miss'}  (wall {r['wall_elapsed']:.0f} s)", flush=True)
    if ok and chosen is None:
        chosen = tt
print("chosen target_total:", chosen)

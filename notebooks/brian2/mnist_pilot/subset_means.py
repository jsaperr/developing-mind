"""Mean 14x14 image per known class, from the pilot subset's TRAIN split (used to judge whether a neuron's weights
look like its class's digits)."""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def mean_images():
    z = np.load(HERE / "data" / "pilot_subset.npz")
    x, y = z["train_x"].reshape(len(z["train_y"]), -1).astype(float), z["train_y"]
    return {int(c): x[y == c].mean(0) for c in np.unique(y)}

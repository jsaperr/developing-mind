"""Builds the MNIST pilot subset (generalization plan step 4a) from the torchvision download in ./data
(git-ignored; MNIST isn't ours to redistribute). Deterministic (fixed rng), so anyone can regenerate it.

  14x14 images: 2x2 average pooling of the 28x28 originals, stored as uint8 0-255.
  known classes {0, 1, 2, 3}: train 300 per class and label 50 per class (both from the MNIST train split, disjoint),
                              test 100 per class (from the test split)
  held-out classes {4, 5}:    test 100 per class only (never seen in training; for the MC-2 novelty check)

Output: data/pilot_subset.npz with arrays {train,label,test}_{x,y}.
Usage (repo root): python notebooks/brian2/mnist_pilot/prep_mnist.py
"""
from pathlib import Path

import numpy as np
import torchvision

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
KNOWN, HELD = [0, 1, 2, 3], [4, 5]


def pool(x):
    x = x.astype(np.float32).reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def main():
    rng = np.random.default_rng(20261001)
    tr = torchvision.datasets.MNIST(str(DATA), train=True, download=False)
    te = torchvision.datasets.MNIST(str(DATA), train=False, download=False)
    trx, try_ = tr.data.numpy(), tr.targets.numpy()
    tex, tey = te.data.numpy(), te.targets.numpy()
    out = {k: [] for k in ("train_x", "train_y", "label_x", "label_y", "test_x", "test_y")}
    for c in KNOWN:
        idx = rng.permutation(np.where(try_ == c)[0])
        out["train_x"].append(trx[idx[:300]]); out["train_y"] += [c] * 300
        out["label_x"].append(trx[idx[300:350]]); out["label_y"] += [c] * 50
    for c in KNOWN + HELD:
        idx = rng.permutation(np.where(tey == c)[0])[:100]
        out["test_x"].append(tex[idx]); out["test_y"] += [c] * 100
    arrays = {}
    for split in ("train", "label", "test"):
        arrays[f"{split}_x"] = pool(np.concatenate(out[f"{split}_x"]))
        arrays[f"{split}_y"] = np.array(out[f"{split}_y"], dtype=np.int64)
    np.savez_compressed(DATA / "pilot_subset.npz", **arrays)
    for k, v in arrays.items():
        print(k, v.shape, v.dtype)
    m = arrays["train_x"].astype(float) / 255
    print(f"mean pixel intensity {m.mean():.3f}; mean summed intensity per image {m.reshape(len(m), -1).sum(1).mean():.1f}")


if __name__ == "__main__":
    main()

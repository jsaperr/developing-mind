"""Modal smoke test: can a seeded Brian2 Cython run execute remotely and reproduce?

Run from the repo root, in the developing-mind conda env:
    python -m modal run notebooks/brian2/modal_smoke.py

Runs the canonical single-synapse-group STDP rig for a short duration on Modal, seeds 0, 0, 1.
Expected: the two seed-0 results are identical, seed 1 differs, target is "cython".
"""
import modal

app = modal.App("developing-mind-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    # versions match the local developing-mind conda env; unpinned numpy pulls a release brian2 can't import
    .pip_install("brian2==2.9.0", "cython==3.2.8", "numpy==2.0.1", "setuptools==82.0.1")
    .add_local_python_source("src")
)


@app.function(image=image, cpu=1.0, memory=2048, timeout=600)
def run_seed(seed: int, duration_s: float = 20.0):
    import time

    import numpy as np
    from brian2 import defaultclock, ms, prefs, run, second, seed as b2_seed

    from src.brian2_stdp import network, spikes

    t0 = time.time()
    b2_seed(seed)
    defaultclock.dt = 0.2 * ms
    rng = np.random.default_rng(seed)
    idx, t = spikes.build_presynaptic_input(20, 0.5, duration_s, rng)
    pre, post, syn = network.build_network(idx, t, apre_val=0.005)
    run(duration_s * second)
    w = np.asarray(syn.w[:])
    return {
        "seed": seed,
        "target": prefs.codegen.target,
        "w_corr_mean": float(w[:10].mean()),
        "w_uncorr_mean": float(w[10:].mean()),
        "elapsed_s": round(time.time() - t0, 1),
    }


@app.local_entrypoint()
def main():
    for r in run_seed.starmap([(0,), (0,), (1,)]):
        print(r)

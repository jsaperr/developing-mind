"""MNIST pilot v4 + the unified substrate (generalization plan step 4a): v3's winner-take-all substrate with the
adaptive threshold replaced by a FAIR-SHARE threshold. Each spike raises its own neuron's theta by Diehl & Cook's 0.05 mV and
lowers every neuron's theta by 0.05/N, so the total is constant: no ratchet.
Why: on the continuously running click world, the published threshold (a one-way ratchet, decay 1e4 s) silenced the
network within ~300 s (N=7: 0 Hz, theta +35 mV). It's a train-then-freeze device, and the framework has no freeze. With
fair-share, a local check (400 s) kept every neuron active at a steady rate (N=7 ~9-10 Hz, N=40 ~2.7 Hz).
Same seeds as v1-v3 (paired); everything else as v3's wta_norm_adapt (gain control, WTA 20 mV, gate off, budget 30).
The same substrate runs on the click world in notebooks/brian2/unified_substrate/modal_unify.py.
Arm: wta_norm_fair. Results: mnist_v4_wta_norm_fair_seed<S>.json.gz. Scored by `analyze_mnist_pilot.py <prefix> v3`
(v3's silent-image definitions).

PREDICTIONS ON RECORD (written before launch):
  V4-P1 fair-share keeps v3's MNIST result: specialization >= 60%, every class 0-3 gets >= 3 neurons in >= 6/8 seeds, vote
        accuracy >= 60% (v3: 74%, 8/8, 65.1%).
  V4-P2 confidence and novelty survive: strongest-vs-weakest strength quintile >= +15 points in >= 6/8 seeds; held-out
        strange ratio >= 1.5 (v3: +30, 2.0x).

Run from the repo root:  python -m modal run notebooks/brian2/mnist_pilot/modal_mnist_v4.py
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
ARMS = ["wta_norm_fair"]
ADAPT = dict(theta_plus_mV=0.05, tau_theta_s=1e4)
FAIR = dict(theta_plus_mV=0.05, tau_theta_s=1e12, fair_share=True)   # no decay: the sum is conserved
image = (IMAGE.add_local_file(str(HERE.parent / "modal_common.py"), "/root/modal_common.py")
         .add_local_file(str(HERE / "mnist_sim.py"), "/root/mnist_sim.py")
         .add_local_file(str(HERE / "data" / "pilot_subset.npz"), "/root/pilot_subset.npz"))
app = modal.App("developing-mind-mnist-v4")


@app.function(image=image, cpu=1.0, memory=4096, timeout=4 * 3600, max_containers=20,
              retries=modal.Retries(max_retries=2, initial_delay=5.0))
def run_seed(arm: str, seed: int) -> dict:
    import numpy as np
    import mnist_sim as S
    subset = dict(np.load("/root/pilot_subset.npz"))
    kw = dict(wta_norm_fair=dict(normalize=True, adaptive=FAIR, wta=True))[arm]
    r = S.run_pilot(seed, subset, TARGET_TOTAL, n_post=40, **kw)
    r["arm"] = arm
    return r


@app.local_entrypoint()
def main():
    todo = [(a, s) for a in ARMS for s in SEEDS if not (HERE / f"mnist_v4_{a}_seed{s}.json.gz").exists()]
    print(f"{len(todo)} runs to do", flush=True)
    n = 0
    for r in run_seed.starmap(todo, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print(f"FAILED {r!r}", flush=True); continue
        with gzip.open(HERE / f"mnist_v4_{r['arm']}_seed{r['seed']}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(r, f)
        n += 1
        print(f"{r['arm']} seed {r['seed']} saved (wall {r['wall_elapsed']:.0f}s)", flush=True)
    print(f"== {n}/{len(todo)} completed", flush=True)

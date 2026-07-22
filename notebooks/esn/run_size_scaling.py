import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.esn.reservoir import build_reservoir
from src.esn.memory_capacity import memory_capacity

BEST_SR = 1.1  # from stage 1
DURATION = 3000  # shorter than stage 1's 5000 -- large-reservoir runs are expensive, trend is what matters here
MAX_LAG = 30
SIZES = [300, 1000, 3000]  # dropped 10000: eigendecomposition (~141s) plus run cost made it by
                            # far the dominant cost (~90% of total wall-clock) for a single data
                            # point: tried scipy.sparse.linalg's iterative top-eigenvalue solver
                            # to speed that up, but it doesn't help on an unstructured dense
                            # matrix (still needs many O(n^2) matvecs to converge) -- the
                            # 300->1000->3000 trend already answers whether size scaling helps
SEEDS_PER_SIZE = {300: 3, 1000: 3, 3000: 3}

# Real bug caught before trusting the first run of this sweep: fixed ridge_alpha=1e-6 (fine at
# stage 1's n=300 scale, where training samples >> features) causes catastrophic overfitting
# once n_units approaches or exceeds the ~2000 available training samples at duration=3000 --
# confirmed directly (train R^2=0.9999, test R^2=-4.08 at n_units=3000 with the old alpha).
# alpha=0.1 gives test R^2=0.89 at n=3000 in a direct sweep; used uniformly across all sizes
# tested here for comparability, not tuned per-size.
RIDGE_ALPHA = 0.1

results = {}
for n_units in SIZES:
    n_seeds = SEEDS_PER_SIZE[n_units]
    totals = []
    wall_times = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(3000 * seed + n_units)
        t0 = time.time()
        W_in, W = build_reservoir(n_units, BEST_SR, rng=rng)
        mc_per_lag, total_mc = memory_capacity(W_in, W, leak_rate=0.3, duration=DURATION, max_lag=MAX_LAG,
                                                ridge_alpha=RIDGE_ALPHA, rng=rng)
        wall_times.append(time.time() - t0)
        totals.append(total_mc)
    results[n_units] = {
        'total_mc_mean': float(np.mean(totals)),
        'total_mc_std': float(np.std(totals)) if len(totals) > 1 else None,
        'n_seeds': n_seeds,
        'wall_time_mean': float(np.mean(wall_times)),
    }
    print(f"n_units={n_units}: total_MC={results[n_units]['total_mc_mean']:.2f} "
          f"(std={results[n_units]['total_mc_std']}, n_seeds={n_seeds}) "
          f"wall_time={results[n_units]['wall_time_mean']:.1f}s")

out_path = Path(__file__).parent / 'size_scaling_results.json'
with open(out_path, 'w') as f:
    json.dump({'sizes': SIZES, 'results': results, 'best_sr': BEST_SR, 'duration': DURATION, 'max_lag': MAX_LAG}, f, indent=2)
print(f"saved -> {out_path}")

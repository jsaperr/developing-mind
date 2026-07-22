import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.esn.reservoir import build_reservoir
from src.esn.phase_task import classification_capacity

N_UNITS = 300
BEST_SR = 1.1  # from stage 1
N_PHASES = 3
PHASE_LEN = 400  # matches the 400-step phases stage 2 would use
DURATION = 8000
LAGS = [1, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1000]
N_SEEDS = 5
CHANCE = 1 / N_PHASES
HORIZON_THRESHOLD = CHANCE + 0.1  # "meaningfully above chance" -- stated before running

per_seed_acc = []
for seed in range(N_SEEDS):
    rng = np.random.default_rng(4000 + seed)
    W_in, W = build_reservoir(N_UNITS, BEST_SR, input_dim=N_PHASES, rng=rng)
    acc = classification_capacity(W_in, W, leak_rate=0.3, n_phases=N_PHASES, phase_len=PHASE_LEN,
                                   duration=DURATION, lags=LAGS, rng=rng)
    per_seed_acc.append(acc)
    print(f"seed={seed}: {[round(acc[k], 3) for k in LAGS]}")

mean_acc = {k: float(np.mean([a[k] for a in per_seed_acc])) for k in LAGS}
std_acc = {k: float(np.std([a[k] for a in per_seed_acc])) for k in LAGS}

horizon = None
for k in LAGS:
    if mean_acc[k] < HORIZON_THRESHOLD:
        horizon = k
        break

print(f"\nmean accuracy per lag: {[round(mean_acc[k], 3) for k in LAGS]}")
print(f"chance={CHANCE:.3f}, horizon_threshold={HORIZON_THRESHOLD:.3f}")
print(f"effective classification horizon: {horizon if horizon else f'>{LAGS[-1]} (never dropped below threshold)'}")

out_path = Path(__file__).parent / 'classification_capacity_results.json'
with open(out_path, 'w') as f:
    json.dump({
        'n_units': N_UNITS, 'spectral_radius': BEST_SR, 'n_phases': N_PHASES,
        'phase_len': PHASE_LEN, 'duration': DURATION, 'lags': LAGS, 'n_seeds': N_SEEDS,
        'chance': CHANCE, 'horizon_threshold': HORIZON_THRESHOLD,
        'mean_acc': mean_acc, 'std_acc': std_acc, 'effective_horizon': horizon,
    }, f, indent=2)
print(f"saved -> {out_path}")

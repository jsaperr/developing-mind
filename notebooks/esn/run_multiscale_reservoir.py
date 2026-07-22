import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.esn.reservoir import build_reservoir, build_multiscale_leak_rates
from src.esn.memory_capacity import memory_capacity
from src.esn.phase_task import classification_capacity

N_UNITS = 300  # same total size as baseline -- this tests whether MIXING timescales helps,
                # not just adding more units
BEST_SR = 1.1
FAST_RATE = 0.3   # matches stage 1's single leak rate
SLOW_RATE = 0.02  # matches decay_fast's actual rate, a deliberate thematic echo of the
                   # two-layer memory's own fast/slow split
N_PHASES = 3
PHASE_LEN = 400
MC_DURATION = 5000
MC_MAX_LAG = 30
CLS_DURATION = 8000
CLS_LAGS = [1, 50, 100, 200, 300, 400, 500, 700, 1000]
N_SEEDS = 5

# Stated before running: "meaningfully extends" means >=20% increase in total linear MC, OR
# >=20% increase in the classification effective horizon (last lag with mean accuracy above
# chance+0.1), relative to the single-timescale baseline at the same total unit count.
MEANINGFUL_THRESHOLD = 0.20
CHANCE = 1 / N_PHASES
HORIZON_THRESHOLD = CHANCE + 0.1


def run_condition(leak_rate_arg, label):
    mc_totals = []
    cls_accs = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(5000 + seed)
        W_in_mc, W_mc = build_reservoir(N_UNITS, BEST_SR, rng=rng)
        leak = leak_rate_arg(N_UNITS, rng) if callable(leak_rate_arg) else leak_rate_arg
        _, total_mc = memory_capacity(W_in_mc, W_mc, leak_rate=leak, duration=MC_DURATION, max_lag=MC_MAX_LAG, rng=rng)
        mc_totals.append(total_mc)

        rng2 = np.random.default_rng(6000 + seed)
        W_in_cls, W_cls = build_reservoir(N_UNITS, BEST_SR, input_dim=N_PHASES, rng=rng2)
        leak2 = leak_rate_arg(N_UNITS, rng2) if callable(leak_rate_arg) else leak_rate_arg
        acc = classification_capacity(W_in_cls, W_cls, leak_rate=leak2, n_phases=N_PHASES,
                                       phase_len=PHASE_LEN, duration=CLS_DURATION, lags=CLS_LAGS, rng=rng2)
        cls_accs.append(acc)

    mean_acc = {k: float(np.mean([a[k] for a in cls_accs])) for k in CLS_LAGS}
    horizon = None
    for k in CLS_LAGS:
        if mean_acc[k] < HORIZON_THRESHOLD:
            horizon = k
            break
    result = {
        'label': label,
        'total_mc_mean': float(np.mean(mc_totals)),
        'total_mc_std': float(np.std(mc_totals)),
        'cls_mean_acc': mean_acc,
        'cls_effective_horizon': horizon,
    }
    print(f"{label}: total_MC={result['total_mc_mean']:.2f}+/-{result['total_mc_std']:.2f}, "
          f"cls_horizon={horizon}, cls_acc={[round(mean_acc[k],3) for k in CLS_LAGS]}")
    return result


baseline = run_condition(FAST_RATE, 'single-timescale baseline (leak=0.3)')
multiscale = run_condition(lambda n, rng: build_multiscale_leak_rates(n, FAST_RATE, SLOW_RATE, fast_fraction=0.5, rng=rng),
                            'multi-timescale (50/50 fast=0.3 / slow=0.02)')

mc_ratio = multiscale['total_mc_mean'] / baseline['total_mc_mean'] - 1
horizon_ratio = None
if baseline['cls_effective_horizon'] and multiscale['cls_effective_horizon']:
    horizon_ratio = multiscale['cls_effective_horizon'] / baseline['cls_effective_horizon'] - 1
elif multiscale['cls_effective_horizon'] is None and baseline['cls_effective_horizon'] is not None:
    horizon_ratio = float('inf')  # multiscale never dropped below threshold within tested lags

meaningful = (mc_ratio >= MEANINGFUL_THRESHOLD) or (horizon_ratio is not None and horizon_ratio >= MEANINGFUL_THRESHOLD)

print(f"\nMC ratio (multiscale/baseline - 1): {mc_ratio:+.1%}")
print(f"Classification horizon ratio: {horizon_ratio}")
print(f"Meaningfully extends (>= {MEANINGFUL_THRESHOLD:.0%} on either metric)? {meaningful}")

out_path = Path(__file__).parent / 'multiscale_results.json'
with open(out_path, 'w') as f:
    json.dump({
        'n_units': N_UNITS, 'spectral_radius': BEST_SR, 'fast_rate': FAST_RATE, 'slow_rate': SLOW_RATE,
        'meaningful_threshold': MEANINGFUL_THRESHOLD, 'baseline': baseline, 'multiscale': multiscale,
        'mc_ratio': mc_ratio, 'horizon_ratio': horizon_ratio, 'meaningful': meaningful,
    }, f, indent=2)
print(f"saved -> {out_path}")

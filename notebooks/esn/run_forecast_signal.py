import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.esn.reservoir import build_reservoir, build_multiscale_leak_rates
from src.esn.forecast_task import forecast_signal

# Stage 2a per web's design: does the reservoir carry a genuine FORWARD-looking forecast signal
# ("will stale pattern p return within W steps"), not just the BACKWARD-looking memory tested by
# follow-ups 2/3 ("what happened k steps ago")? Standalone testbed only -- no episodic.py touch.

N_UNITS = 300
BEST_SR = 1.1  # from stage 1
FAST_RATE = 0.3
SLOW_RATE = 0.02  # multi-timescale reservoir = follow-up 3's winning config, used as-is here

N_CORE = 4
MEAN_INTERVAL = 400  # matches follow-up 2's phase_len, so results are comparable in scale
GAMMA_SHAPE = 4      # CV = 1/sqrt(4) = 0.5 -- real irregularity, not a disguised fixed period
VISIT_LEN = 40
DURATION = 20000     # duration/mean_interval ~ 50 visits/pattern -> ~200 core visits total,
                      # enough for a stable AUC estimate per pattern after washout+split
W_FORECAST = 200     # half of mean_interval: a non-trivial forecast horizon (not near-certain,
                      # not near-impossible)
WASHOUT = 1000        # ~2.5 mean intervals before any pattern's forecast is evaluated
N_SEEDS = 5
RIDGE_ALPHA = 1e-2   # n_units=300 vs ~10k+ train samples per pattern -- far from the
                      # feature/sample overfitting regime hit in run_size_scaling.py, no need
                      # for that script's heavier alpha=0.1

# Falsification criteria, stated before running (the explicit lesson from follow-up 3's broken
# threshold-crossing check): CONFIRMS a real forecast signal if mean reservoir AUC clears the
# staleness-only baseline by >= 0.05 absolute AND mean reservoir AUC > 0.65 in its own right
# (not just "barely beats an already-weak baseline"). REJECTS if the margin isn't met, or if
# reservoir AUC collapses toward 0.5 (chance) -- which would suggest follow-ups 2/3's success was
# schedule-regularity leakage in a different guise, per web's explicit hypothesis to test here.
MIN_MARGIN = 0.05
MIN_RESERVOIR_AUC = 0.65

per_seed_reservoir_auc = []
per_seed_baseline_auc = []
per_seed_reservoir_acc = []
per_seed_baseline_acc = []
per_pattern_detail = []

for seed in range(N_SEEDS):
    rng = np.random.default_rng(7000 + seed)
    W_in, W = build_reservoir(N_UNITS, BEST_SR, input_dim=N_CORE + 1, rng=rng)
    leak = build_multiscale_leak_rates(N_UNITS, FAST_RATE, SLOW_RATE, fast_fraction=0.5, rng=rng)
    results = forecast_signal(W_in, W, leak_rate=leak, n_core=N_CORE, mean_interval=MEAN_INTERVAL,
                               gamma_shape=GAMMA_SHAPE, visit_len=VISIT_LEN, duration=DURATION,
                               w_forecast=W_FORECAST, washout=WASHOUT, ridge_alpha=RIDGE_ALPHA, rng=rng)

    res_aucs = [r['reservoir_auc'] for r in results.values() if r['reservoir_auc'] is not None]
    base_aucs = [r['baseline_auc'] for r in results.values() if r['baseline_auc'] is not None]
    res_accs = [r['reservoir_acc'] for r in results.values()]
    base_accs = [r['baseline_acc'] for r in results.values()]

    per_seed_reservoir_auc.append(float(np.mean(res_aucs)))
    per_seed_baseline_auc.append(float(np.mean(base_aucs)))
    per_seed_reservoir_acc.append(float(np.mean(res_accs)))
    per_seed_baseline_acc.append(float(np.mean(base_accs)))
    per_pattern_detail.append({str(p): r for p, r in results.items()})

    print(f"seed={seed}: n_patterns_evaluated={len(results)}, "
          f"reservoir_auc={per_seed_reservoir_auc[-1]:.3f}, baseline_auc={per_seed_baseline_auc[-1]:.3f}, "
          f"reservoir_acc={per_seed_reservoir_acc[-1]:.3f}, baseline_acc={per_seed_baseline_acc[-1]:.3f}")

mean_res_auc = float(np.mean(per_seed_reservoir_auc))
mean_base_auc = float(np.mean(per_seed_baseline_auc))
std_res_auc = float(np.std(per_seed_reservoir_auc))
std_base_auc = float(np.std(per_seed_baseline_auc))
margin = mean_res_auc - mean_base_auc

confirmed = (margin >= MIN_MARGIN) and (mean_res_auc > MIN_RESERVOIR_AUC)

print(f"\nmean reservoir AUC: {mean_res_auc:.3f} +/- {std_res_auc:.3f}")
print(f"mean baseline AUC:  {mean_base_auc:.3f} +/- {std_base_auc:.3f}")
print(f"margin (reservoir - baseline): {margin:+.3f}")
print(f"falsification result -- confirmed real forecast signal (margin>={MIN_MARGIN} AND reservoir_auc>{MIN_RESERVOIR_AUC})? {confirmed}")

out_path = Path(__file__).parent / 'forecast_signal_results.json'
with open(out_path, 'w') as f:
    json.dump({
        'n_units': N_UNITS, 'spectral_radius': BEST_SR, 'fast_rate': FAST_RATE, 'slow_rate': SLOW_RATE,
        'n_core': N_CORE, 'mean_interval': MEAN_INTERVAL, 'gamma_shape': GAMMA_SHAPE,
        'visit_len': VISIT_LEN, 'duration': DURATION, 'w_forecast': W_FORECAST, 'washout': WASHOUT,
        'n_seeds': N_SEEDS, 'ridge_alpha': RIDGE_ALPHA,
        'min_margin': MIN_MARGIN, 'min_reservoir_auc': MIN_RESERVOIR_AUC,
        'per_seed_reservoir_auc': per_seed_reservoir_auc, 'per_seed_baseline_auc': per_seed_baseline_auc,
        'per_seed_reservoir_acc': per_seed_reservoir_acc, 'per_seed_baseline_acc': per_seed_baseline_acc,
        'mean_reservoir_auc': mean_res_auc, 'mean_baseline_auc': mean_base_auc,
        'std_reservoir_auc': std_res_auc, 'std_baseline_auc': std_base_auc,
        'margin': margin, 'confirmed': confirmed,
        'per_pattern_detail': per_pattern_detail,
    }, f, indent=2)
print(f"saved -> {out_path}")

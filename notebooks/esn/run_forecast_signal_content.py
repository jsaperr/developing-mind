import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.esn.reservoir import build_reservoir, build_multiscale_leak_rates
from src.esn.forecast_task import forecast_signal_content

# Stage 2b per web's design: stage 2a fed the reservoir pure timing (one-hot identity), nothing
# about content, and lost to a staleness-only baseline. Does feeding actual pattern CONTENT
# (with a deliberate associative-priming mechanism -- content drifts toward a pattern before its
# scheduled return, see forecast_task.py's generate_visitation_schedule_with_content) let the
# reservoir carry a genuine content-driven context signal that identity alone structurally
# couldn't represent? Same schedule timing, same reservoir config, same forecast target, same
# staleness-only baseline as stage 2a, so this isolates content-vs-identity as the one variable
# under test. Standalone testbed only -- no episodic.py touch.

N_UNITS = 300
BEST_SR = 1.1
FAST_RATE = 0.3
SLOW_RATE = 0.02  # same multi-timescale config as stage 2a, unchanged

N_CORE = 4
PATTERN_DIM = 32          # unit-norm content vectors, same convention as src/hopfield/two_layer.py
MEAN_INTERVAL = 400        # identical to stage 2a
GAMMA_SHAPE = 4
VISIT_LEN = 40
DURATION = 20000
W_FORECAST = 200
WASHOUT = 1000
PRIMING_WINDOW = 150       # shorter than W_FORECAST -- a genuine leading indicator, not a
                            # relabeling of the same window being forecast
PRIMING_PROB = 0.5         # moderate, not deterministic -- content drift precedes return only
                            # some of the time, same as real associative priming would be
PRIMING_NOISE_STD = 0.5    # corrupted echo, not an exact preview of the pattern
N_SEEDS = 5
RIDGE_ALPHA = 1e-2

# Falsification bar, stated before running -- identical to stage 2a's for direct comparability
# (a different bar here would look like moving the goalposts): confirms a real content-driven
# forecast signal if mean reservoir AUC clears the staleness-only baseline by >= 0.05 absolute
# AND mean reservoir AUC > 0.65 in its own right. Rejects otherwise. Per Jasper/web's explicit
# instruction: if this also fails, stop chasing check (b) via reservoirs and log it as closed
# debt -- two structurally different, root-caused negative results is a complete answer, not a
# reason to try a third mechanism.
MIN_MARGIN = 0.05
MIN_RESERVOIR_AUC = 0.65

per_seed_reservoir_auc = []
per_seed_baseline_auc = []
per_seed_reservoir_acc = []
per_seed_baseline_acc = []
per_pattern_detail = []

for seed in range(N_SEEDS):
    rng = np.random.default_rng(8000 + seed)
    W_in, W = build_reservoir(N_UNITS, BEST_SR, input_dim=PATTERN_DIM, rng=rng)
    leak = build_multiscale_leak_rates(N_UNITS, FAST_RATE, SLOW_RATE, fast_fraction=0.5, rng=rng)
    results = forecast_signal_content(W_in, W, leak_rate=leak, n_core=N_CORE, pattern_dim=PATTERN_DIM,
                                       mean_interval=MEAN_INTERVAL, gamma_shape=GAMMA_SHAPE, visit_len=VISIT_LEN,
                                       duration=DURATION, w_forecast=W_FORECAST, priming_window=PRIMING_WINDOW,
                                       priming_prob=PRIMING_PROB, priming_noise_std=PRIMING_NOISE_STD,
                                       washout=WASHOUT, ridge_alpha=RIDGE_ALPHA, rng=rng)

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
print(f"falsification result -- confirmed real content-driven forecast signal (margin>={MIN_MARGIN} AND reservoir_auc>{MIN_RESERVOIR_AUC})? {confirmed}")

out_path = Path(__file__).parent / 'forecast_signal_content_results.json'
with open(out_path, 'w') as f:
    json.dump({
        'n_units': N_UNITS, 'spectral_radius': BEST_SR, 'fast_rate': FAST_RATE, 'slow_rate': SLOW_RATE,
        'n_core': N_CORE, 'pattern_dim': PATTERN_DIM, 'mean_interval': MEAN_INTERVAL, 'gamma_shape': GAMMA_SHAPE,
        'visit_len': VISIT_LEN, 'duration': DURATION, 'w_forecast': W_FORECAST, 'washout': WASHOUT,
        'priming_window': PRIMING_WINDOW, 'priming_prob': PRIMING_PROB, 'priming_noise_std': PRIMING_NOISE_STD,
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

"""Calibration check for a 3-disjoint-block novel-C rig BEFORE running any batch.

30 presynaptic inputs = 3 blocks of 10; one block correlated per phase. Hold target_total=10
(w_init=1/3) so the homeostatic scaling pins total synaptic drive to the SAME value as the
20-input runs -> firing rate should stay in the ~3-20Hz band with gmax unchanged. This run
confirms that empirically (phase-1 = block A correlated, 400s) rather than assuming it.

Falsifiable calibration bar (stated before running, not tuned after): per-neuron steady rate in
3-20Hz and w_total within ~[9.5,10.5]. If it misses, report and stop -- do not chase gmax.
"""
import sys
from pathlib import Path
import numpy as np
from brian2 import SpikeMonitor, StateMonitor, defaultclock, mV, ms, run, second, start_scope

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.brian2_stdp.network import build_competitive_population_network, scale_inhib_for_n
from src.brian2_stdp.spikes import generate_correlated_group, generate_uncorrelated_group

TARGET_RATE = 20.0; P_SHARE = 0.9; N_POST = 3; DT_MS = 0.2
N_BLOCKS = 3; BLOCK = 10; N_PRE = N_BLOCKS * BLOCK
TARGET_TOTAL = 10.0; W_INIT = TARGET_TOTAL / N_PRE
INHIB_REF = 13.0; GAP = 1.5
CALIB_S = 400.0


def multiblock_input(corr_block, duration_s, rng):
    """One phase: block `corr_block` (indices [corr_block*BLOCK,+BLOCK)) is correlated (p_share),
    all other blocks independent Poisson. Rate-matched across all blocks."""
    idxs, ts = [], []
    for b in range(N_BLOCKS):
        if b == corr_block:
            gi, gt = generate_correlated_group(BLOCK, TARGET_RATE, P_SHARE, duration_s, 2.0, rng)
        else:
            gi, gt = generate_uncorrelated_group(BLOCK, TARGET_RATE, duration_s, rng)
        idxs.append(np.asarray(gi) + b * BLOCK)
        ts.append(np.asarray(gt))
    idx = np.concatenate(idxs).astype(int); t = np.concatenate(ts)
    order = np.argsort(t)
    return idx[order], t[order] * second


start_scope()
defaultclock.dt = DT_MS * ms
rng = np.random.default_rng(90000)
idx, t = multiblock_input(0, CALIB_S, rng)
per_conn = scale_inhib_for_n(N_POST, reference_inhib_mV=INHIB_REF, reference_n_post=3)
pre, post, syn, inhib = build_competitive_population_network(
    N_POST, idx, t, 0.005, per_conn * mV, GAP, n_pre=N_PRE, w_init=W_INIT, target_total=TARGET_TOTAL)
spikes = SpikeMonitor(post)
wt = StateMonitor(post, 'w_total', record=True, dt=1 * second)
run(CALIB_S * second)

rate = np.array(spikes.count[:]) / CALIB_S
wtot = wt.w_total[:]
band = wtot[:, int(50):]  # after initial transient
print(f"N_PRE={N_PRE} w_init={W_INIT:.4f} target_total={TARGET_TOTAL}  inhib/conn={per_conn:.2f}mV")
print(f"per-neuron rate (Hz): {np.round(rate,1).tolist()}   [bar: 3-20 Hz]")
print(f"w_total steady band per neuron (t>50s): "
      f"{[f'[{band[j].min():.2f},{band[j].max():.2f}]' for j in range(N_POST)]}   [bar: ~9.5-10.5]")
ok_rate = np.all((rate >= 3) & (rate <= 20))
ok_w = np.all((band >= 9.3) & (band <= 10.7))
print(f"CALIBRATION {'PASS' if (ok_rate and ok_w) else 'CHECK'}: rate_ok={ok_rate} w_ok={ok_w}")

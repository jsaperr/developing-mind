"""Rest: what does the substrate do when nothing is happening? Framework v5, section I: "the resting
state isn't nothing". Every earlier run fed the network a context the whole time.

World: A(1000) B(1000) C(1000) REST(3000) B(1000) s. The same 30-input disjoint rig, N=7, 13mV/1.5, v1's
frozen runner. REST = no synchrony at all: every one of the 30 wires clicks independently at the same
20 Hz, so the average rate never changes. Only the timing structure disappears. Before rest the
substrate holds C (current) and B (one back); A has been released. B returns after rest: was it kept
through the silence, or did rest act like a change?

The input builder is local (the src set builder requires a non-empty synced set): phases with a set
use spikes.build_set_phase_input; the rest phase uses spikes.generate_uncorrelated_group for all 30.

PREDICTIONS ON RECORD (written before launch; scored by analyze_rest.py, which replays memory too):
 Substrate
  RE-P1 tuning survives rest: no context's mean holder count changes by more than 0.5 between rest +300 s
        and rest end, and B still has >= 2 holders per seed just before it returns (it had ~3.4).
        Nothing recruits neurons during rest, because there's no new context to capture them.
  RE-P2 but selectivity blurs: the mean preferred-block weight share falls by >= 10% of its excess over
        uniform (1/3) during rest. Random timing gives STDP no consistent signal, and its depression
        bias slowly erodes the learned structure.
  RE-P3 rest is a smaller event than a change: the 60 s weight displacement peak in the first 300 s of
        rest is below the peak after C's arrival, in >= 6/8 seeds.
 Memory (default DormantGatedMemory, rectified readout)
  RE-P4 ghosts: during rest the fingerprint is made of whatever the above-average neurons happen to be
        tuned to, so it resembles C or B. At W=10, memory names B or C in >= 30% of rest checks, and
        doesn't store a separate "rest" memory in most seeds (<= 3/8).
  RE-P5 ghost recognitions rehearse memory: at W=10 (horizon ~1500 s < 3000 s of rest), B's memory is
        still live (not dormant) when B returns in >= 5/8 seeds.

Usage: run_rest_seed.py <seed> <out.json.gz>
"""
import sys
from pathlib import Path

import numpy as np
from brian2 import second

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / "v1_schedule_data"))
import run_v1_seed as v1
from src.brian2_stdp.results_io import load_result, save_result
from src.brian2_stdp.spikes import build_set_phase_input, generate_uncorrelated_group

A, B, C = list(range(0, 10)), list(range(10, 20)), list(range(20, 30))
SETS = [A, B, C, []]                       # context 3 = rest (no synced group)
SCHEDULE = [0, 1, 2, 3, 1]
DURATIONS = [1000.0, 1000.0, 1000.0, 3000.0, 1000.0]


def build_with_rest(rate, p_share, durs, ids, rng, n_blocks=None, block_size=None):
    all_idx, all_t, start = [], [], 0.0
    for dur, cid in zip(durs, ids):
        guard = 0.001 if start > 0 else 0.0
        if SETS[cid]:
            i, t = build_set_phase_input(rate, p_share, [dur], [SETS[cid]], 30, rng)
            t = np.asarray(t / second)
        else:
            i, t = generate_uncorrelated_group(30, rate, dur, rng)
        all_idx.append(np.asarray(i, int)); all_t.append(np.asarray(t) + start + guard)
        start += dur
    idx = np.concatenate(all_idx); t = np.concatenate(all_t)
    order = np.argsort(t)
    return idx[order], t[order] * second


if __name__ == '__main__':
    seed, out = int(sys.argv[1]), sys.argv[2]
    v1.build_multiblock_phase_input = build_with_rest
    v1.PHASE_CORR_BLOCKS = SCHEDULE
    v1.PHASE_DURATIONS_S = DURATIONS
    v1.run_v1_seed(seed, out)
    d = load_result(out)
    d['world'] = 'rest'
    d['context_sets'] = SETS
    save_result(out, d)
    print(f"rest seed {seed} -> {out} ({d['status']})")

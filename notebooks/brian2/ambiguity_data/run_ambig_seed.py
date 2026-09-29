"""Ambiguity (the Necker-cube world): after the substrate has learned A then B (so it holds both, one back),
present BOTH at once. Wires 1-10 are synced to one hidden rhythm and wires 11-20 to a second, independent
rhythm, both at full strength (p_share 0.9); wires 21-30 are independent. Then plain A again.

World: A(1000) B(1000) AB(2000) A(1000) s. The disjoint 30-input rig, N=7, 13mV/1.5, v1's frozen runner.
The builder is local (a phase can carry several synced groups, each with its own master), rates are
matched, and src is untouched.

Framework link: v5 treats basin boundaries ("moments of unresolved ambiguity") as where curiosity lives.
Binocular rivalry / Necker flips need neural fatigue (adaptation). This substrate has none, so a lack of
flipping is itself informative.

PREDICTIONS ON RECORD (written before launch; scored by analyze_ambig.py):
 Substrate
  AM-P1 coexistence, no capture: from the end of B to the end of AB, neither A's nor B's mean holder count
        changes by more than 0.5 (both interpretations already own neurons; nothing needs recruiting).
  AM-P2 no rivalry: during AB (after +300 s), individual neurons switch preferred context (A <-> B) at most
        once per seed on average. There's no fatigue mechanism to drive alternation.
  AM-P3 AB's onset is a small event: the 60 s weight-displacement peak in AB's first 300 s is below the peak
        after B's arrival, in >= 6/8 seeds.
 Memory (default DormantGatedMemory, rectified readout, W=10)
  AM-P4 the fingerprint is a blend: during settled AB, its median cosine to A and to B are both between
        0.5 and 0.8 (neither matches within the radius).
  AM-P5 memory flags it, then invents a category: NOVEL in >= 40% of settled AB checks, and a new memory
        whose content is nearest the A+B blend is stored in >= 5/8 seeds.
  AM-P6 the blend doesn't hijack A: when plain A returns, A is named in > half of its settled checks in 8/8.

Usage: run_ambig_seed.py <seed> <out.json.gz>
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
from src.brian2_stdp.spikes import generate_correlated_group, generate_uncorrelated_group

A, B, C = list(range(0, 10)), list(range(10, 20)), list(range(20, 30))
GROUPS = {0: [A], 1: [B], 2: [C], 3: [A, B]}          # context 3 = AB, two independent synced groups
CONTEXT_SETS = [A, B, C, A + B]
SCHEDULE = [0, 1, 3, 0]
DURATIONS = [1000.0, 1000.0, 2000.0, 1000.0]


def build_multi(rate, p_share, durs, ids, rng, n_blocks=None, block_size=None):
    all_idx, all_t, start = [], [], 0.0
    for dur, cid in zip(durs, ids):
        guard = 0.001 if start > 0 else 0.0
        used = []
        for g in GROUPS[cid]:
            gi, gt = generate_correlated_group(len(g), rate, p_share, dur, 2.0, rng)
            all_idx.append(np.asarray(g)[np.asarray(gi, int)]); all_t.append(np.asarray(gt) + start + guard)
            used += g
        other = np.setdiff1d(np.arange(30), used)
        ui, ut = generate_uncorrelated_group(len(other), rate, dur, rng)
        all_idx.append(other[np.asarray(ui, int)]); all_t.append(np.asarray(ut) + start + guard)
        start += dur
    idx = np.concatenate(all_idx).astype(int); t = np.concatenate(all_t)
    order = np.argsort(t)
    return idx[order], t[order] * second


if __name__ == '__main__':
    seed, out = int(sys.argv[1]), sys.argv[2]
    v1.build_multiblock_phase_input = build_multi
    v1.PHASE_CORR_BLOCKS = SCHEDULE
    v1.PHASE_DURATIONS_S = DURATIONS
    v1.run_v1_seed(seed, out)
    d = load_result(out)
    d['world'] = 'ambiguity_AB'
    d['context_sets'] = CONTEXT_SETS
    save_result(out, d)
    print(f"ambiguity seed {seed} -> {out} ({d['status']})")

"""Generalization plan, step 1 (short-phase arm, disjoint): the v1b schedule A->B->C->A->B with 300 s
phases instead of 1000 s, on the same 30-input 3-block rig (disjoint blocks), N=7, 13mV/1.5. Reuses
v1's frozen runner with only the schedule durations overridden.

Why: step 0 found that after checkb's short (300 s) AND partly overlapping fillers, the substrate
dropped the just-departed filler and kept older core residue, instead of one-back. This arm keeps
the phases short but makes the contexts disjoint, so it separates "short phase" from "overlap".

PREDICTIONS ON RECORD (written before launch; scored by analyze_short_phase.py):
  SP-P1 one-back holds with short disjoint phases: at C's arrival (swap 2) the new context recruits
        from A's holders (A drops to <= 0.5 holders per seed by +150 s) and B keeps >= 60% of its
        holders; at the two-back returns (swaps 3, 4) the incoming context has <= 1 holder per seed
        just before the swap. If this holds, step 0's drop was caused by overlap, not phase length.
  SP-P2 every phase is learned within its 300 s: the phase's own context holds >= 4 of 7 neurons by
        its end, for every phase (seed mean).

Usage: run_short_seed.py <seed> <out.json.gz>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / "v1_schedule_data"))
import run_v1_seed as v1
from src.brian2_stdp.results_io import load_result, save_result

SCHEDULE = [0, 1, 2, 0, 1]
DURATIONS = [300.0] * 5

if __name__ == '__main__':
    seed, out = int(sys.argv[1]), sys.argv[2]
    v1.PHASE_CORR_BLOCKS = SCHEDULE
    v1.PHASE_DURATIONS_S = DURATIONS
    v1.run_v1_seed(seed, out)
    d = load_result(out)
    d['world'] = 'short300_disjoint'
    save_result(out, d)
    print(f"short-phase seed {seed} -> {out} ({d['status']})")

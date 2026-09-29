"""Generalization plan, step 1 (3000 s arm, disjoint): the v1b schedule A->B->C->A->B with 3000 s phases
instead of 1000 s, on the same 30-input rig (disjoint blocks), N=7, 13mV/1.5. Reuses v1's frozen runner
with only the schedule durations overridden (exactly as short_phase_data/run_short_seed.py does for 300 s).

Why: 300 s and 1000 s disjoint phases both give one-back. The other direction is untested: do retainers
that hold a context for a long time harden so much that the next change can't release them?

PREDICTIONS ON RECORD (written before launch; scored by analyze_long3000.py with step 0's holder metric):
  LP-P1 one-back holds at 3000 s: at C's arrival (swap 2) B keeps >= 60% of its holders at +150 s, and A
        (held through all 3000 s of B) is down to <= 1 holder per seed by +300 s; at the two-back returns
        (swaps 3, 4) the incoming context has <= 1 holder per seed just before the swap.
  LP-P2 re-assignment stays change-triggered: between +300 s after a swap and the next swap (2700 s of
        stable world), no context's mean holder count changes by more than 0.5.

Usage: run_long3000_seed.py <seed> <out.json.gz>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / "v1_schedule_data"))
import run_v1_seed as v1
from src.brian2_stdp.results_io import load_result, save_result

SCHEDULE = [0, 1, 2, 0, 1]
DURATIONS = [3000.0] * 5

if __name__ == '__main__':
    seed, out = int(sys.argv[1]), sys.argv[2]
    v1.PHASE_CORR_BLOCKS = SCHEDULE
    v1.PHASE_DURATIONS_S = DURATIONS
    v1.run_v1_seed(seed, out)
    d = load_result(out)
    d['world'] = 'long3000_disjoint'
    save_result(out, d)
    print(f"long-phase seed {seed} -> {out} ({d['status']})")

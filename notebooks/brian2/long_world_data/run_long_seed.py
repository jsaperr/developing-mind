"""Long world: 20 phases x 1000 s (20,000 s), five contexts with graded overlap, each visited 4
times, on the same 30-input rig as the set worlds, N=7, 13mV/1.5. Built for the persistent
character store (dormant entries) on real input: many returns after memory has forgotten a
context (misattribution under real exposure), and a context's readout at its 4th visit vs its 1st
(drift). It also extends the one-back question to 20 phases (does the lock-in residue
accumulate? An open Brian2 item).

Contexts (input sets; prototype cosines in brackets):
  A 0-9, B 10-19, C 20-29 (disjoint, -0.5); D = 0-6 + 20-22 (70% overlap with A, 0.55);
  E = 5-14 (50% with A and with B, 0.25).
Schedule (fixed, rng 2026, no immediate repeats, 4 visits each; gaps between visits 2-10 phases):
  A E A B D B C A C D B E D C E C A E D B

Reuses v1's frozen runner with the set-based input builder, exactly as run_set_seed.py does.

PREDICTIONS ON RECORD (substrate side, written before launch):
  LW-P1 one-back holds for 20 phases: holders of the incoming context at returns more than one back
        average <= 1 per seed, and don't trend upward from the first half of the run to the second.
  LW-P2 no drift: a context's settled rectified readout at its 4th visit has cosine >= 0.9 with its
        1st-visit readout (seed mean, every context).

Usage: run_long_seed.py <seed> <out.json.gz>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / "v1_schedule_data"))
import run_v1_seed as v1
from src.brian2_stdp.results_io import load_result, save_result
from src.brian2_stdp.spikes import build_set_phase_input

SETS = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30)),
        list(range(0, 7)) + [20, 21, 22], list(range(5, 15))]
SCHEDULE = [0, 4, 0, 1, 3, 1, 2, 0, 2, 3, 1, 4, 3, 2, 4, 2, 0, 4, 3, 1]
DURATIONS = [1000.0] * len(SCHEDULE)

if __name__ == '__main__':
    seed, out = int(sys.argv[1]), sys.argv[2]
    v1.build_multiblock_phase_input = (lambda rate, p, durs, ids, rng, n_blocks=None, block_size=None:
                                       build_set_phase_input(rate, p, durs, [SETS[i] for i in ids], 30, rng))
    v1.PHASE_CORR_BLOCKS = SCHEDULE
    v1.PHASE_DURATIONS_S = DURATIONS
    v1.run_v1_seed(seed, out)
    d = load_result(out)
    d['world'] = 'long20'
    d['context_sets'] = SETS
    save_result(out, d)
    print(f"long world seed {seed} -> {out} ({d['status']})")

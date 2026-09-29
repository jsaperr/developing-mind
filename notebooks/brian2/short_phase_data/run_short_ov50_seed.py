"""Generalization step 1, overlap arm: the ov50 world (v1b schedule A B C A B, pairwise 50% overlapping
input sets, as set_worlds_data/run_set_seed.py 'ov50') with 300 s phases instead of 1000 s. N=7,
13mV/1.5, same 30-input rig, frozen v1 runner with the set-input builder (exactly as run_set_seed.py).

Why: the 300 s disjoint arm (run_short_seed.py) stayed one-back, so step 0's deviation (checkb: the
just-departed filler dropped, older core residue kept) came with input overlap. Mechanism guess: a
new context that shares inputs with the current one recruits the current context's neurons too, since
they already respond to the shared inputs. So the new context takes most of the population and few
retainers are left.

PREDICTIONS ON RECORD (written before launch; scored by analyze_short_ov50.py, which also scores the
existing ov50 1000 s runs with the same metric):
  SO-P1 overlap alone breaks the even split: 150 s after each of swaps 1 and 2, the incoming context
        holds >= 5 of 7 neurons (seed mean) in BOTH ov50 at 300 s and ov50 at 1000 s (disjoint: ~3.1-3.4).
  SO-P2 so the just-departed context keeps < 60% of its holders at the next change (disjoint: 86-96%),
        in both.

Usage: run_short_ov50_seed.py <seed> <out.json.gz>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / "v1_schedule_data"))
import run_v1_seed as v1
from src.brian2_stdp.results_io import load_result, save_result
from src.brian2_stdp.spikes import build_set_phase_input

SETS = [list(range(0, 10)), list(range(5, 15)), list(range(0, 5)) + list(range(10, 15))]

if __name__ == '__main__':
    seed, out = int(sys.argv[1]), sys.argv[2]
    v1.build_multiblock_phase_input = (lambda rate, p, durs, ids, rng, n_blocks=None, block_size=None:
                                       build_set_phase_input(rate, p, durs, [SETS[i] for i in ids], 30, rng))
    v1.PHASE_CORR_BLOCKS = [0, 1, 2, 0, 1]
    v1.PHASE_DURATIONS_S = [300.0] * 5
    v1.run_v1_seed(seed, out)
    d = load_result(out)
    d['world'] = 'short300_ov50'
    d['context_sets'] = SETS
    save_result(out, d)
    print(f"short ov50 seed {seed} -> {out} ({d['status']})")

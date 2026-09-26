"""Worlds whose contexts are arbitrary input SETS (spikes.build_set_phase_input), on the same
30-input rig as novel-C / v1, at N=7, 13mV/1.5.

  ov50    v1b schedule A->B->C->A->B, pairwise 50% overlapping contexts. Prototype cosine 0.25,
          so the contexts are distinct but close; memory's novelty threshold is 0.5.
  ov70    same schedule, pairwise 70% overlap. Prototype cosine 0.55, ABOVE memory's novelty
          threshold, so memory can't tell the contexts apart by novelty alone.
  checkb  the integrated version of episodic check (b) (experiments.md, 2026-07-19): core
          contexts A, B, C (disjoint blocks) with dominant phases, and brief, never-returning
          filler contexts between them: A(1000) F1(300) B(1000) F2(300) C(1000) F3(300) A(1000) s.
          The fillers are fixed random 10-subsets (rng 2026), each with prototype cosine <= 0.25 to
          every core.
(0% overlap = the existing v1b data, disjoint blocks, prototype cosine -0.5.)

Reuses v1's frozen runner. Its input builder is swapped for the set-based one via module attribute
(the runner looks the builder up at call time). Per-input statistics per phase are unchanged: 10
correlated inputs, 20 independent, same rates. So the novel-C calibration applies, and target_total
is still 10 with w_init 1/3. The saved result gains 'context_sets' (context id -> input indices);
'phase_corr_blocks' holds context ids.

Usage: run_set_seed.py <ov50|ov70|checkb> <seed> <out.json.gz>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / "v1_schedule_data"))
import run_v1_seed as v1
from src.brian2_stdp.results_io import load_result, save_result
from src.brian2_stdp.spikes import build_set_phase_input

A, B, C = list(range(0, 10)), list(range(10, 20)), list(range(20, 30))
WORLDS = {
    "ov50": dict(sets=[list(range(0, 10)), list(range(5, 15)), list(range(0, 5)) + list(range(10, 15))],
                 schedule=[0, 1, 2, 0, 1], durations=[1000.0] * 5),
    "ov70": dict(sets=[list(range(0, 10)), list(range(3, 13)), [0, 1, 2, 3, 4, 5, 6, 10, 11, 12]],
                 schedule=[0, 1, 2, 0, 1], durations=[1000.0] * 5),
    "checkb": dict(sets=[A, B, C,
                         [0, 2, 3, 9, 10, 12, 15, 17, 18, 29],
                         [3, 6, 7, 16, 17, 21, 22, 24, 26, 29],
                         [0, 2, 3, 4, 7, 11, 12, 18, 19, 24]],
                   schedule=[0, 3, 1, 4, 2, 5, 0], durations=[1000.0, 300.0, 1000.0, 300.0, 1000.0, 300.0, 1000.0]),
}

if __name__ == '__main__':
    world, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    cfg = WORLDS[world]
    sets = cfg['sets']
    v1.build_multiblock_phase_input = (lambda rate, p, durs, ids, rng, n_blocks=None, block_size=None:
                                       build_set_phase_input(rate, p, durs, [sets[i] for i in ids], 30, rng))
    v1.PHASE_CORR_BLOCKS = cfg['schedule']
    v1.PHASE_DURATIONS_S = cfg['durations']
    v1.run_v1_seed(seed, out)
    d = load_result(out)
    d['world'] = world
    d['context_sets'] = sets
    save_result(out, d)
    print(f"set world {world} seed {seed} -> {out} ({d['status']})")

"""FORK v2 (2026-09-29): identical to run_fork_seed.py except that Brian2's own random generator is seeded
(brian2.seed(seed)) before the run. v1 assumed the network had no randomness of its own; it does (sigma_v, the
per-timestep membrane noise that lets neurons differentiate), and nothing seeded it, so v1's "forks" differed
from second 1 (FK-P1 failed). A seeding check confirmed that brian2.seed makes runs bit-identical.

PREDICTIONS FOR v2 (written after seeing v1's results, which are therefore not blind to them; v1 showed that
independent neural noise alone makes wiring as different as independent seeds, while function (fingerprints)
stays at ~0.95-0.96 and history effects are as predicted):
  F2-P1 true fork: every weight identical across branches before 2000 s.
  F2-P2 small differences compound: the twins (now differing ONLY in input clicks after 2000 s) start at
        distance 0 and diverge to >= 50% of the independent-seed distance within 1000 s, reaching >= 80% by
        5000 s.
  F2-P3 function stays shared: twin fingerprint cosine over the settled test >= 0.9.
  F2-P4 history effects as in v1: at the test's start the detour holds <= 1 A-neuron and the twins >= 2.5, and
        by 5000 s the detour-twin distance is <= 1.5x the twin distance.
  F2-P5 memory keeps the detour: C character (w_char >= 1.5) at the end, W=10: detour 8/8, twins 0/8.

Original v1 docstring follows.

The fork (framework v5, section VII): checkpoint a developing system, duplicate it, and give the copies
different lives. The doc predicts small differences compound ("so they respond differently to subsequent
input, so they diverge further").

Implementation: the network uses no randomness of its own (no rand() anywhere; all randomness is in the
seeded input). Runs that get bit-identical input up to time T are therefore bit-identical up to T; that's a
true fork, and it's verified in the analysis (weights must match exactly before the fork point). Same
30-input disjoint rig, N=7, 13mV/1.5, v1's frozen runner; the input builder is local.

Per seed, three branches from one shared past:
  shared   A(1000) B(1000)                  same input in every branch (rng = seed)
  twin1    B(1000), fresh random clicks      rng = (seed, 1)   \\ same world, different noise:
  twin2    B(1000), fresh random clicks      rng = (seed, 2)   /  what does chance alone do?
  detour   C(1000)                           rng = (seed, 3)      a genuinely different experience
  test     A(1000) B(1000)                   IDENTICAL input in every branch (rng = (seed, 99))

PREDICTIONS ON RECORD (written before launch; scored by analyze_fork.py):
  FK-P1 it's a true fork: every weight is identical across the three branches before 2000 s.
  FK-P2 chance alone keeps twins close: the twin1-twin2 weight distance stays below 30% of the distance
        between independent seeds (neuron-matched), through the branch and the test.
  FK-P3 the detour responds differently to identical input: at the test's start the detour holds <= 1 A-neuron
        per seed, the twins >= 2.5 (A was one back for the twins, two back for the detour).
  FK-P4 the substrate forgets the detour: at the end (after the shared test), the detour-twin1 distance is
        <= 1.5x the twin1-twin2 distance (back to the noise floor).
  FK-P5 memory doesn't: with the default memory (W=50), the detour's memory still carries character for C at
        the end (live or dormant, w_char >= 1.5) in 8/8 seeds, and the twins' in 0/8. The forks end up
        differing in memory, not in the network.

Usage: run_fork_seed.py <seed> <twin1|twin2|detour> <out.json.gz>
"""
import sys
from pathlib import Path

import numpy as np
from brian2 import second

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / "v1_schedule_data"))
import brian2
import run_v1_seed as v1
from src.brian2_stdp.results_io import load_result, save_result
from src.brian2_stdp.spikes import build_set_phase_input

SETS = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]
BRANCH = {"twin1": (1, 1), "twin2": (1, 2), "detour": (2, 3)}      # (context, rng code)


def make_builder(seed, branch):
    ctx, code = BRANCH[branch]
    rngs = [None, None, np.random.default_rng([seed, code]), np.random.default_rng([seed, 99]), None]

    def build(rate, p_share, durs, ids, rng, n_blocks=None, block_size=None):
        test_rng = rngs[3]
        all_idx, all_t, start = [], [], 0.0
        for k, (dur, cid) in enumerate(zip(durs, ids)):
            r = rng if k < 2 else (rngs[2] if k == 2 else test_rng)
            guard = 0.001 if start > 0 else 0.0
            i, t = build_set_phase_input(rate, p_share, [dur], [SETS[cid]], 30, r)
            all_idx.append(np.asarray(i, int)); all_t.append(np.asarray(t / second) + start + guard)
            start += dur
        idx = np.concatenate(all_idx); t = np.concatenate(all_t)
        order = np.argsort(t, kind="stable")
        return idx[order], t[order] * second
    return build, [0, 1, ctx, 0, 1]


if __name__ == '__main__':
    seed, branch, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    build, schedule = make_builder(seed, branch)
    v1.build_multiblock_phase_input = build
    v1.PHASE_CORR_BLOCKS = schedule
    v1.PHASE_DURATIONS_S = [1000.0] * 5
    brian2.seed(seed)
    v1.run_v1_seed(seed, out)
    d = load_result(out)
    d['world'] = 'forkv2_' + branch
    d['branch'] = branch
    d['context_sets'] = SETS
    save_result(out, d)
    print(f"fork v2 seed {seed} {branch} -> {out} ({d['status']})")

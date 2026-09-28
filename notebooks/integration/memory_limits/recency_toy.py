"""Why does recency (C > B in final w_char) fail in the integrated check (b), even on clean input?
Memory-only toy, no substrate: clean prototype queries plus jitter, the check (b) schedule
A F1 B F2 C F3 A (fillers 0.3 x a core phase, random never-returning 10-subsets of 30 inputs,
cores disjoint blocks), memory = promoted GatedEpisodicMemory with the features off (the "clean"
arm of readout_set_worlds.py). The phase length L (in memory steps) is swept; W=50 s is L=20,
W=10 s is L=100, the July fixed-X original was L=400.

Background. In the July fixed-X original, recency came out as final w_char C > B by a small margin
(6.24 vs 5.97), and w_char's only way to fall is decay_char = 0.0005/step (time constant 2000
steps). The eviction horizon is ~150-170 steps since the last win. A W=50 check (b) run is ~100
steps long.

Arms:
  evict     the real memory (staleness threshold 150), as in the set-world clean arm
  no-evict  positive control: eviction off (fixed-X-like), so every core survives
  +relax    after the schedule, 300 extra steps of the two-layer update with no retrieval (w_fast
            relaxes, residual consolidation finishes, no eviction): removes consolidation lag

PREDICTIONS ON RECORD (written before this script was first run):
  RC-P1 lag is part of it: at L=20, no-evict, C > B is 0/8 at the end of the schedule, and the
        C - B gap moves toward C after +relax.
  RC-P2 lag isn't all of it: at L=20, even after +relax, C > B stays <= 4/8. With a 2000-step
        decay constant, a ~50-step age difference can't separate B from C.
  RC-P3 positive control: no-evict, L=400: C > B >= 6/8 (the July shape).
  RC-P4 the structural conflict: with eviction, there is no L in {20, 50, 100, 200, 400} where
        all three cores survive in >= 6/8 AND C > B in >= 5/8. Old enough for w_char decay to
        separate B from C means older than the eviction horizon, and pure phases give an
        off-phase core no wins to reset it.

Usage: recency_toy.py   (any env with torch; writes recency_toy_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.hopfield.episodic_consolidating import GatedEpisodicMemory
from src.integration.interface import unit

N_IN = 30
SIGMA = 0.032  # per-dim jitter giving consecutive-query cosine ~0.97, the substrate's settled value
LENGTHS = [20, 50, 100, 200, 400]
SEEDS = range(8)
RELAX = 300


def schedule(L, rng):
    core = [unit(np.isin(np.arange(N_IN), range(b * 10, b * 10 + 10)).astype(float)) for b in range(3)]
    fill = [unit(np.isin(np.arange(N_IN), rng.choice(N_IN, 10, replace=False)).astype(float)) for _ in range(3)]
    Lf = max(1, int(round(0.3 * L)))
    seq = [(core[0], L, 'A'), (fill[0], Lf, 'F'), (core[1], L, 'B'), (fill[1], Lf, 'F'),
           (core[2], L, 'C'), (fill[2], Lf, 'F'), (core[0], L, 'A2')]
    q, lab = [], []
    for p, n, name in seq:
        for _ in range(n):
            q.append(unit(p + SIGMA * rng.standard_normal(N_IN))); lab.append(name)
    return np.array(q), lab


def run(q, lab, gs, evict, relax):
    m = GatedEpisodicMemory(dim=N_IN, gap_scale=gs, eta=0.0, match_floor=None,
                            staleness_threshold=150 if evict else 10 ** 9)
    born, gaps = {}, []
    for i, x in enumerate(torch.tensor(q, dtype=torch.float32)):
        if len(m.mem.patterns) >= 2:
            s = np.sort((torch.stack(m.mem.patterns) @ x).numpy())
            gaps.append(float(s[-1] - s[-2]))
        o = m.step(x)
        if o['created'] is not None:
            born[o['created']] = lab[i]
    mem = m.mem
    if relax:
        zero = torch.zeros(len(mem.patterns))
        wf, wc = torch.tensor(mem.w_fast), torch.tensor(mem.w_char)
        for _ in range(RELAX):
            wf, wc = mem._update_fn(wf, wc, zero)
        mem.w_fast, mem.w_char = wf.tolist(), wc.tolist()
    wc = {}
    for i, w in zip(mem.ids, mem.w_char):
        if born.get(i) in ('A', 'B', 'C'):
            wc[born[i]] = max(wc.get(born[i], 0.0), w)
    return wc, gaps


def main():
    out = {}
    for L in LENGTHS:
        # gap_scale grounded as in every earlier world: 0.6 x median top1-top2 gap under plain memory
        g_all = []
        for sd in SEEDS:
            q, lab = schedule(L, np.random.default_rng(1000 + sd))
            g_all += run(q, lab, 0.1534, True, False)[1]
        gs = 0.6 * float(np.median(g_all))
        out[L] = {"gap_scale": gs}
        for arm, evict, relax in (("evict", True, False), ("evict+relax", True, True),
                                  ("no-evict", False, False), ("no-evict+relax", False, True)):
            alive, cb, ah, gapcb = 0, 0, 0, []
            for sd in SEEDS:
                q, lab = schedule(L, np.random.default_rng(1000 + sd))
                wc = run(q, lab, gs, evict, relax)[0]
                if all(k in wc for k in 'ABC'):
                    alive += 1; cb += wc['C'] > wc['B']; ah += wc['A'] > max(wc['B'], wc['C'])
                    gapcb.append(wc['C'] - wc['B'])
            row = dict(cores_alive=alive, C_over_B=cb, A_highest=ah,
                       mean_C_minus_B=float(np.mean(gapcb)) if gapcb else None)
            out[L][arm] = row
            print(f"L={L:4d} {arm:15s} cores alive {alive}/8  A highest {ah}/8  C>B {cb}/8  "
                  f"mean w_char C-B {row['mean_C_minus_B'] if gapcb else float('nan'):+.3f}")
    json.dump(out, open(HERE / "recency_toy_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

"""Two cheap follow-ups to recency_toy.py (same schedule, same clean queries, same memory), one per
option for where recency should live. Memory only, no substrate.

Option 1: recency is timing, not character. Re-score the same runs with the variables that
already carry timing: staleness (steps since last win; C more recent = lower) and w_fast (the
fast layer, ~50-step decay). Only seeds where both B's and C's original entries are alive at the
end can be scored.

Option 2: character outlives episodes. A toy separate character store (notebook-only, src
untouched): every content the memory creates gets a character record (pattern, w_char). While
its episode is alive, the record mirrors the episode's w_char (the existing rule). When the episode
is evicted, the record persists and only decays (decay_char, toward 1, no consolidation). When a
new episode is created whose content matches a record (cosine >= 0.8, the recommended radius), it
links to that record and starts with the record's w_char instead of 1. Ordering is read from the
records, so an evicted core still has a character.

PREDICTIONS ON RECORD (written before this script was first run):
  RK-P1 option 1: wherever B and C are both alive at the end (L=20, 50 with eviction; all L
        without eviction), C is more recent than B by staleness in 8/8, and by w_fast in >= 7/8.
  RK-P2 option 2, long phases (L >= 100, eviction on): all three core records exist 8/8, A highest
        >= 6/8, C > B >= 6/8. The records get the time that the episodes didn't.
  RK-P3 option 2, short phases (L = 20, 50): C > B stays <= 4/8. Character recency exists only on
        character's own timescale; a separate store doesn't change that.
  RK-P4 option 2 doesn't break primacy: A highest >= 6/8 at every L.

Usage: recency_carriers.py   (any env with torch; writes recency_carriers_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recency_toy as RT
from src.hopfield.episodic_consolidating import GatedEpisodicMemory
from src.hopfield.two_layer import CONSTANTS

LINK = 0.8


def run_option1(q, lab, gs, evict):
    m = GatedEpisodicMemory(dim=RT.N_IN, gap_scale=gs, eta=0.0, match_floor=None,
                            staleness_threshold=150 if evict else 10 ** 9)
    born = {}
    for i, x in enumerate(torch.tensor(q, dtype=torch.float32)):
        o = m.step(x)
        if o['created'] is not None:
            born[o['created']] = lab[i]
    st, wf = {}, {}
    for k, i in enumerate(m.mem.ids):
        if born.get(i) in ('B', 'C'):
            st[born[i]] = m.mem.staleness[k]; wf[born[i]] = m.mem.w_fast[k]
    return st, wf


def run_option2(q, lab, gs):
    m = GatedEpisodicMemory(dim=RT.N_IN, gap_scale=gs, eta=0.0, match_floor=None)
    mem = m.mem
    records = []          # dicts: pattern, w_char, label (first creator's label, scoring only), alive
    link = {}             # episode id -> record index
    for i, x in enumerate(torch.tensor(q, dtype=torch.float32)):
        o = m.step(x)
        if o['created'] is not None:
            p = mem.patterns[mem.ids.index(o['created'])]
            sims = [float(r['pattern'] @ p) for r in records]
            if sims and max(sims) >= LINK:
                ri = int(np.argmax(sims))
                mem.w_char[mem.ids.index(o['created'])] = records[ri]['w_char']
            else:
                records.append(dict(pattern=p.clone(), w_char=1.0, label=lab[i])); ri = len(records) - 1
            link[o['created']] = ri
        alive = {link[e]: k for k, e in enumerate(mem.ids) if e in link}
        for ri, r in enumerate(records):
            if ri in alive:
                r['w_char'] = mem.w_char[alive[ri]]
            else:
                r['w_char'] += CONSTANTS.decay_char * (1 - r['w_char'])
    return {r['label']: r['w_char'] for r in records if r['label'] in ('A', 'B', 'C')}


def main():
    out = {}
    for L in RT.LENGTHS:
        g_all = []
        for sd in RT.SEEDS:
            q, lab = RT.schedule(L, np.random.default_rng(1000 + sd))
            g_all += RT.run(q, lab, 0.1534, True, False)[1]
        gs = 0.6 * float(np.median(g_all))
        out[L] = {}
        for evict in (True, False):
            n, st_ok, wf_ok = 0, 0, 0
            for sd in RT.SEEDS:
                q, lab = RT.schedule(L, np.random.default_rng(1000 + sd))
                st, wf = run_option1(q, lab, gs, evict)
                if 'B' in st and 'C' in st:
                    n += 1; st_ok += st['C'] < st['B']; wf_ok += wf['C'] > wf['B']
            key = "opt1 evict" if evict else "opt1 no-evict"
            out[L][key] = dict(scorable=n, C_recent_staleness=st_ok, C_over_B_wfast=wf_ok)
            print(f"L={L:4d} {key:14s} scorable {n}/8  C more recent by staleness {st_ok}/{n}  by w_fast {wf_ok}/{n}")
        n, ah, cb, gap = 0, 0, 0, []
        for sd in RT.SEEDS:
            q, lab = RT.schedule(L, np.random.default_rng(1000 + sd))
            wc = run_option2(q, lab, gs)
            if all(k in wc for k in 'ABC'):
                n += 1; ah += wc['A'] > max(wc['B'], wc['C']); cb += wc['C'] > wc['B']; gap.append(wc['C'] - wc['B'])
        out[L]["opt2 store"] = dict(records=n, A_highest=ah, C_over_B=cb, mean_C_minus_B=float(np.mean(gap)))
        print(f"L={L:4d} opt2 store     core records {n}/8  A highest {ah}/8  C>B {cb}/8  mean C-B {np.mean(gap):+.3f}")
    json.dump(out, open(HERE / "recency_carriers_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

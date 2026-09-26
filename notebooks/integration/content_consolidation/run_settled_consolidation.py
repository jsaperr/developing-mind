"""Settled-only content consolidation: refresh on every step, move content only when the query is steady.

Motivation (strong_tight_gate v1c, W=10 s): with the absolute floor in place, 8 "absorption" events
still occurred. The queries genuinely matched the entries (cos 0.80-0.86), so the floor correctly
let them through. The hazard is the TRANSITIONAL query itself: right after a change it reads
"not-departing", which truly overlaps several contexts. 6/8 flipped entries had been CREATED from
such a query at an earlier change, and 2/8 were well-formed entries pulled across by one.

Separation under test: on transitional steps, the staleness REFRESH (rehearsal) is what makes
long-range memory work (causally tested), while moving CONTENT is what causes absorption. So keep
retrieval and refresh on every step, but consolidate content only when the stability gate is open
(query steady). That's the same signal that already gates creation, so no new parameter.

Worlds: 13mV/1.5 A->B->A, A->B->C, v1, v1b, v1c; strong_tight_gate v1b, v1c. Both clocks.
Best = stability gate TAU 0.9 + gated consolidation eta 0.1 + absolute floor THETA.

PREDICTIONS ON RECORD (written before this script was first run):
  SC-P1: absorption = 0 in every world and clock with settled-only consolidation (best had 8 at
         strong_tight_gate v1c W=10 s).
  SC-P2: no key recognition result drops by more than 1 seed. v1b B two-back genuine at W=50 s (8/8
         at both operating points); v1b B alive at W=10 s (13mV 6/8, stg 8/8); v1c B genuine at
         W=50 s (8/8 at both); v1 C one-back genuine at W=50 s (8/8).
Descriptive: accuracy and two-back reports after novel swaps.

Usage: run_settled_consolidation.py   (conda env; writes settled_consolidation_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "coupling_v0"))
import run_coupling_v0 as T
import run_content_consolidation as C
from run_creation_gates import stable_allow

B2 = ROOT / "notebooks" / "brian2"
WORLDS = [
    ("13mV A->B->A", B2 / "nonstationary_data", "nonstat_n7_reliable_13_1p5_seed*.json", [(2, 0, "A one-back")]),
    ("13mV A->B->C", B2 / "novelc_data", "novelc_n7_reliable_13_1p5_seed*.json", []),
    ("13mV v1", B2 / "v1_schedule_data", "v1_n7_reliable_13_1p5_seed*.json.gz", [(3, 0, "A two-back"), (4, 2, "C one-back")]),
    ("13mV v1b", B2 / "v1b_schedule_data", "v1b_n7_reliable_13_1p5_seed*.json.gz", [(4, 1, "B two-back")]),
    ("13mV v1c", B2 / "v1c_schedule_data", "v1c_n7_reliable_13_1p5_seed*.json.gz", [(5, 1, "B rel-to-ret 2000s")]),
    ("stg v1b", B2 / "stg_replication_data", "stg_v1b_n7_seed*.json.gz", [(4, 1, "B two-back")]),
    ("stg v1c", B2 / "stg_replication_data", "stg_v1c_n7_seed*.json.gz", [(5, 1, "B rel-to-ret 2000s")]),
]
TAU, ETA = 0.9, 0.1


def main():
    out = {}
    for name, ddir, pat, returns in WORLDS:
        runs = T.load_world(ddir, pat)
        for W in T.CLOCKS:
            rng = np.random.default_rng(0); torch.manual_seed(0)
            streams = [T.build_streams(d, wm, r, n, bs, W, rng) for d, wm, r, n, bs in runs]
            g1 = np.concatenate([T.run_memory(s, "substrate", T.OLD_GAP_SCALE, T.THETA)['gaps'] for s in streams])
            gs = 0.6 * float(np.median(g1))
            key = f"{name} W{W}"
            out[key] = {}
            line = f"{key:22s}"
            for vname, settled_only in (("best", False), ("settled-only", True)):
                res = []
                for s in streams:
                    allow = stable_allow(s['q']['substrate'], TAU)
                    res.append(C.run_cc(s, 'substrate', gs, ETA, True, allow=allow, match_floor=T.THETA,
                                        consolidate_when=allow if settled_only else None))
                sc = [T.score(s, x) for s, x in zip(streams, res)]
                row = dict(acc=float(np.mean([x['acc'] for x in sc])), absorbed=int(sum(x['absorbed'] for x in res)),
                           two_back=int(sum(sw['other'] for x in sc for sw in x['swaps'] if not sw['returning'])))
                rec = []
                for rp, op, label in returns:
                    gen = int(sum(C.genuine(s, x, rp, op) > 0.5 for s, x in zip(streams, res)))
                    alive = int(sum(any(x['born_phase'][i] == op for i in x['alive_ids'][int(np.where(s['ph'] == rp)[0][0]) - 1])
                                    for s, x in zip(streams, res)))
                    row[label] = dict(genuine=gen, alive=alive)
                    rec.append(f"{label} alive {alive} gen {gen}")
                out[key][vname] = row
                line += f" | {vname}: acc {row['acc']:.3f} absorbed {row['absorbed']:2d} two-back {row['two_back']:3d} " + "; ".join(rec)
            print(line)
    json.dump(out, open(HERE / "settled_consolidation_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

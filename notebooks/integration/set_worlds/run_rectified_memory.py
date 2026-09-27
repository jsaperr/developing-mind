"""Memory replay with the RECTIFIED readout H+ vs the current H, using the adopted memory (promoted src
GatedEpisodicMemory: commit only when steady and not re-learning, rehearse always, TRANSITIONAL).

Motivation (compare_readouts.py): on settled windows, H+ = unit(sum_j max(r_j - mean r, 0) w_j)
restores readout quality under 50% overlap (0.46 -> 0.96), removes previous-context leakage
(H: about -0.7, H+: about 0), and also fixes the cold start (phase 1 0.2-0.3 -> 0.96). But memory
depends on what the readout says DURING transitions (rehearsal, capture), so this is the real test.

Worlds with returns: 13mV A->B->A, v1, v1b, v1c; strong_tight_gate v1b, v1c; ov50. Both clocks.
gap_scale is re-grounded per readout, world and clock by the original procedure (0.6 x the
top1-top2 gap median under plain memory). The steady flag is computed on whichever readout is in
use; the changing flag is readout-independent (weights).

PREDICTIONS ON RECORD (written before this script was first run):
  HP-P1 the first context gets recognized: A->B->A A-return genuine at W=10 s >= 7/8 (H: 6/8); v1
        two-back A at W=50 s >= 6/8 (H: 4/8).
  HP-P2 no key non-first recognition drops by more than 1 seed vs H: two-back B genuine at W=50 s
        (13mV v1b, stg v1b, ov50); v1c B at W=50 s; v1 C one-back at W=50 s.
  HP-P3 overlap rehearsal recovers: ov50 B alive at its two-back return, W=10 s, >= 4/8 (H: 1/8).
  HP-P4 absorption stays 0 everywhere with the adopted rule.

Usage: run_rectified_memory.py   (conda env; writes rectified_memory_summary.json)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from src.brian2_stdp.results_io import load_result
from src.integration import interface as I
import readout_set_worlds as RS

B2 = ROOT / "notebooks" / "brian2"
WORLDS = [
    ("13mV A->B->A", B2 / "nonstationary_data" / "nonstat_n7_reliable_13_1p5_seed*.json", [(2, 0, "A return (first ctx)")]),
    ("13mV v1", B2 / "v1_schedule_data" / "v1_n7_reliable_13_1p5_seed*.json.gz", [(3, 0, "A two-back (first)"), (4, 2, "C one-back")]),
    ("13mV v1b", B2 / "v1b_schedule_data" / "v1b_n7_reliable_13_1p5_seed*.json.gz", [(3, 0, "A two-back (first)"), (4, 1, "B two-back")]),
    ("13mV v1c", B2 / "v1c_schedule_data" / "v1c_n7_reliable_13_1p5_seed*.json.gz", [(5, 1, "B rel-to-ret 2000s")]),
    ("stg v1b", B2 / "stg_replication_data" / "stg_v1b_n7_seed*.json.gz", [(3, 0, "A two-back (first)"), (4, 1, "B two-back")]),
    ("stg v1c", B2 / "stg_replication_data" / "stg_v1c_n7_seed*.json.gz", [(5, 1, "B rel-to-ret 2000s")]),
    ("ov50", B2 / "set_worlds_data" / "ov50_n7_seed*.json.gz", [(3, 0, "A two-back (first)"), (4, 1, "B two-back")]),
]
READOUTS = {
    "H": lambda r, w: I.h_readout(r, w),
    "H+": lambda r, w: I.unit(np.clip(r - r.mean(), 0, None) @ w),
}


def load(pat):
    runs = []
    for f in sorted(glob.glob(str(pat))):
        d = load_result(f)
        if d.get('status') != 'completed':
            continue
        n_pre = int(max(d['syn_i'])) + 1
        d.setdefault('context_sets', [list(range(b * 10, b * 10 + 10)) for b in range(n_pre // 10)])
        w = np.array(d['weight_trace']); wm = np.zeros((d['n_post'], n_pre, w.shape[1]))
        wm[np.array(d['syn_j']), np.array(d['syn_i'])] = w
        runs.append((d, wm, np.array(d['spike_rate_bins'], dtype=float), I.changing_per_second(wm)))
    return runs


def build(d, wm, r, chg, W, readout, rng):
    n_pre = wm.shape[1]
    ps = [0.0] + list(d['swap_times_s']); ids = d['phase_corr_blocks']
    protos = np.array([I.unit(np.isin(np.arange(n_pre), s).astype(float)) for s in d['context_sets']])
    starts, ph, q = [], [], []
    for a in range(0, int(d['total_s']) - W + 1, W):
        p = int(np.searchsorted(ps, a, side='right') - 1)
        if p + 1 < len(ps) and a + W > ps[p + 1]:
            continue
        starts.append(a); ph.append(p); q.append(READOUTS[readout](r[:, a:a + W].mean(1), wm[:, :, a:a + W].mean(2)))
    ph = np.array(ph); q = np.array(q)
    true = np.array([ids[p] for p in ph]); prev = np.array([ids[p - 1] if p else -1 for p in ph])
    settled = np.array([a - ps[p] >= RS.SETTLE for a, p in zip(starts, ph)])
    return dict(starts=starts, ph=ph, true=true, prev=prev, settled=settled, protos=protos, q=q, clean=q,
                steady=I.steady_flags(q, 0.9), changing=I.window_any(chg, starts, W))


def main():
    out = {}
    for name, pat, returns in WORLDS:
        runs = load(pat)
        for W in (10, 50):
            key = f"{name} W{W}"; out[key] = {}
            line = f"{key:18s}"
            for ro in READOUTS:
                rng = np.random.default_rng(0); torch.manual_seed(0)
                streams = [build(d, wm, r, chg, W, ro, rng) for d, wm, r, chg in runs]
                gs = RS.ground_gap_scale(streams)
                res = [RS.run_arm(s, 'adopted', gs) for s in streams]
                sc = [RS.score(s, x) for s, x in zip(streams, res)]
                row = dict(acc_committed=float(np.nanmean([x['acc_committed'] for x in sc])),
                           absorbed=int(sum(x['absorbed'] for x in res)))
                txt = []
                for rp, op, label in returns:
                    g = int(sum(RS.genuine(s, x, rp, op) > 0.5 for s, x in zip(streams, res)))
                    a = int(sum(RS.alive_at(s, x, rp, op) for s, x in zip(streams, res)))
                    row[label] = dict(genuine=g, alive=a); txt.append(f"{label}: alive {a} gen {g}")
                out[key][ro] = row
                line += f"\n     {ro:2s}: acc {row['acc_committed']:.3f} absorbed {row['absorbed']} | " + " | ".join(txt)
            print(line)
    json.dump(out, open(HERE / "rectified_memory_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

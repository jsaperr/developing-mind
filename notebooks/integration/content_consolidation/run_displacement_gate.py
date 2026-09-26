"""Displacement gate: memory doesn't COMMIT (create entries or move content) while the substrate is
still re-learning. Retrieval and the staleness refresh (rehearsal) continue on every step.

The signal (analyze_change_signal.py): substrate weight displacement over the last 60 s,
D(t) = sum |w(t) - w(t-60)|. It's elevated after 72/72 changes (onset ~12 s, median ~210-230 s)
and never in settled windows. Here the flag is CAUSAL and LABEL-FREE: flag(t) = D(t) > median of
D over [t-900, t-60] + 3 x 1.4826 x MAD over that span. With fewer than 100 s of history (the start
of the run) the flag is on, since the substrate is still forming. A memory step (a W-second window)
is flagged if any second in it is flagged. The parameters (60 s, 900 s, 3 x robust sd) are fixed a
priori to match the signal analysis. They weren't tuned here.

Variants (forked memory; src/ untouched):
  best         stability gate on creation (TAU 0.9) + gated consolidation (eta 0.1) + absolute floor
  best + disp  as best, AND creation and consolidation both blocked on flagged steps  <- primary
  disp only    displacement gate replaces the stability gate on creation (and gates consolidation)

Worlds: 13mV A->B->A, A->B->C, v1, v1b, v1c; strong_tight_gate v1b, v1c. Both clocks.

Scoring: overall accuracy, plus accuracy and two-back ("other") reports on UNFLAGGED steps only.
While the flag is up, memory can't have an entry for a brand-new context yet, so it reports
something stale by construction. The durable outcomes are what matter: absorption, persistent
capture, recognition.

PREDICTIONS ON RECORD (written before this script was first run):
  DG-P1: absorption = 0 in every world and clock for best + disp (best: up to 8 at stg v1c W=10).
  DG-P2: persistent capture removed. Two-back reports on UNFLAGGED steps after novel swaps are
         <= 5 per world at W=10 s for best + disp.
  DG-P3: key recognitions don't drop by more than 1 seed vs best: v1b B two-back genuine W=50 s
         (8/8, both points), v1b B alive W=10 s (13mV 6/8, stg 8/8), v1c B genuine W=50 s (8/8,
         both), v1 C one-back genuine W=50 s (8/8).
  DG-P4 (the expected cost): OVERALL accuracy falls below best in most worlds at W=10 s, while
         accuracy on unflagged steps is >= best's in most worlds.
Descriptive: disp only; first-context recognition; fraction of steps flagged.

Usage: run_displacement_gate.py   (conda env; writes displacement_gate_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "coupling_v0"))
import run_coupling_v0 as T
import run_content_consolidation as C
from run_creation_gates import stable_allow
from run_settled_consolidation import WORLDS

TAU, ETA = 0.9, 0.1
L, TRAIL, K, MIN_HIST = 60, 900, 3.0, 100


def change_flag(wm):
    """Per-second causal flag from 60 s weight displacement; wm is (n_post, n_pre, n_t)."""
    n_t = wm.shape[2]
    D = np.full(n_t, np.nan)
    D[L:] = np.abs(wm[:, :, L:] - wm[:, :, :-L]).sum(axis=(0, 1))
    flag = np.ones(n_t, bool)
    for t in range(L, n_t):
        hist = D[max(L, t - TRAIL):t - L]
        hist = hist[np.isfinite(hist)]
        if len(hist) < MIN_HIST:
            continue
        med = np.median(hist)
        flag[t] = D[t] > med + K * 1.4826 * np.median(np.abs(hist - med))
    return flag


def step_flags(d, flag, W):
    """Flag per memory step, with the same window tiling and swap-straddle skipping as S1 windows()."""
    ps = [0.0] + list(d['swap_times_s']); Tt = int(d['total_s']); out = []
    for a in range(0, Tt - W + 1, W):
        b = a + W
        ph = int(np.searchsorted(ps, a, side='right') - 1)
        if ph + 1 < len(ps) and b > ps[ph + 1]:
            continue
        out.append(bool(flag[a:min(b, len(flag))].any()))
    return np.array(out)


def main():
    out = {}
    for name, ddir, pat, returns in WORLDS:
        runs = T.load_world(ddir, pat)
        flags_1s = [change_flag(wm) for d, wm, r, n, bs in runs]
        for W in T.CLOCKS:
            rng = np.random.default_rng(0); torch.manual_seed(0)
            streams = [T.build_streams(d, wm, r, n, bs, W, rng) for d, wm, r, n, bs in runs]
            sflags = [step_flags(d, f, W) for (d, *_), f in zip(runs, flags_1s)]
            assert all(len(sf) == len(s['ph']) for sf, s in zip(sflags, streams)), "step alignment"
            g1 = np.concatenate([T.run_memory(s, "substrate", T.OLD_GAP_SCALE, T.THETA)['gaps'] for s in streams])
            gs = 0.6 * float(np.median(g1))
            key = f"{name} W{W}"
            out[key] = {"flagged_fraction": float(np.mean([sf.mean() for sf in sflags]))}
            print(f"\n{key}  (flagged {out[key]['flagged_fraction']:.2f} of steps)")
            for vname in ("best", "best + disp", "disp only"):
                res = []
                for s, sf in zip(streams, sflags):
                    st = stable_allow(s['q']['substrate'], TAU)
                    calm = (lambda f: (lambda i: not f[i]))(sf)
                    if vname == "best":
                        allow, cw = st, None
                    elif vname == "best + disp":
                        allow, cw = (lambda a, c: (lambda i: a(i) and c(i)))(st, calm), calm
                    else:
                        allow, cw = calm, calm
                    res.append(C.run_cc(s, 'substrate', gs, ETA, True, allow=allow, match_floor=T.THETA,
                                        consolidate_when=cw))
                sc = [T.score(s, x) for s, x in zip(streams, res)]
                acc_unfl, other_unfl = [], 0
                for s, x, sf in zip(streams, res, sflags):
                    m = ~sf
                    acc_unfl.append(float(np.mean(x['reported'][m] == s['true'][m])) if m.any() else np.nan)
                    # two-back reports on unflagged steps, within novel phases after their swap
                    novel_ph = {sw['swap'] for sw in T.score(s, x)['swaps'] if not sw['returning']}
                    for i in np.where(m)[0]:
                        p = s['ph'][i]
                        if p in novel_ph and x['reported'][i] not in (s['true'][i], s['prev'][i]):
                            other_unfl += 1
                row = dict(acc=float(np.mean([x['acc'] for x in sc])), acc_unflagged=float(np.nanmean(acc_unfl)),
                           absorbed=int(sum(x['absorbed'] for x in res)), two_back_unflagged=int(other_unfl),
                           two_back_all=int(sum(sw['other'] for x in sc for sw in x['swaps'] if not sw['returning'])),
                           created=float(np.mean([x['created'] for x in res])))
                rec = []
                for rp, op, label in returns:
                    gen = int(sum(C.genuine(s, x, rp, op) > 0.5 for s, x in zip(streams, res)))
                    alive = int(sum(any(x['born_phase'][i] == op for i in x['alive_ids'][int(np.where(s['ph'] == rp)[0][0]) - 1])
                                    for s, x in zip(streams, res)))
                    row[label] = dict(genuine=gen, alive=alive); rec.append(f"{label}: alive {alive} gen {gen}")
                out[key][vname] = row
                print(f"  {vname:12s} acc {row['acc']:.3f} (unflagged {row['acc_unflagged']:.3f})  absorbed {row['absorbed']}  "
                      f"two-back unflagged {row['two_back_unflagged']:3d} (all {row['two_back_all']:3d})  created {row['created']:.1f}  "
                      + " | ".join(rec))
    json.dump(out, open(HERE / "displacement_gate_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

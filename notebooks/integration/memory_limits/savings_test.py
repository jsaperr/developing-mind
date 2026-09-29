"""Savings: does a forgotten context that comes back to its dormant character get re-learned faster
than one starting fresh? The framework (v5, III.3) asks about "forgetting as raised threshold rather
than deletion". Replay only: the long world at W=10 (where contexts are forgotten between visits),
rectified readout, src DormantGatedMemory (radius 0.8, anchoring, NOVEL report).

Arms: dormant ON (default) vs OFF (prune_eps huge, so an evicted entry is pruned at once and never
reawakens; everything else identical). Compared at the SAME phases: every phase in which the ON arm
reawakened a dormant entry for the phase's own context.

Two kinds of "faster", kept apart:
  recognition time  steps from phase start until the phase's context is named in 3 consecutive
                    reports. Recognition needs an entry, and creation is gated by the substrate
                    settling, not by strength. Strength only breaks ties and can't speed up a clear
                    content match.
  strength regain   the new entry's w_fast 20 steps after creation. Inherited w_char raises the
                    w_fast learning multiplier (1 + k (w_char - 1), capped at 3).

PREDICTIONS ON RECORD (written before this script was first run):
  SV-P1 recognition time is NOT faster: the median paired difference (OFF - ON) is within +/-1 step.
  SV-P2 strength is regained faster: w_fast at +20 steps is >= 1.5x higher with dormant ON (median
        paired ratio), in >= 80% of paired phases.
  SV-P3 no accuracy cost: named-report accuracy within 0.005 of OFF.

Usage: savings_test.py   (conda env; writes savings_test_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchored_consolidation as AC
from src.hopfield.episodic_consolidating import TRANSITIONAL
from src.hopfield.episodic_dormant import NOVEL, DormantGatedMemory

R, RS, LW = AC.R, AC.RS, AC.LW


def run(s, gs, dormant):
    m = DormantGatedMemory(dim=30, gap_scale=gs, prune_eps=0.05 if dormant else 1e9)
    tag, per_phase = {}, {}
    reports, created_at = [], {}
    wf_track = {}
    starts = {int(p): int(np.where(s['ph'] == p)[0][0]) for p in np.unique(s['ph'])}
    for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
        o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
        p, true = int(s['ph'][i]), int(s['true'][i])
        if o['created'] is not None:
            tag[o['created']] = int(np.argmax(s['protos'] @ x.numpy()))
            if tag[o['created']] == true and p not in created_at:
                created_at[p] = (i, o['created'], o['reawakened'] is not None)
        if p in created_at:
            j, e, _ = created_at[p]
            if i == j + 20 and e in m.mem.ids:
                wf_track[p] = m.mem.w_fast[m.mem.ids.index(e)]
        rep = o['report']
        named = None if rep is TRANSITIONAL or rep == NOVEL else tag.get(rep)
        reports.append(named)
    recog = {}
    for p, st in starts.items():
        idx = [i for i in range(st, len(reports)) if s['ph'][i] == p]
        true = int(s['true'][idx[0]])
        run_len = 0
        for k, i in enumerate(idx):
            run_len = run_len + 1 if reports[i] == true else 0
            if run_len == 3:
                recog[p] = k - 2; break
    named = [(r, int(t)) for r, t in zip(reports, s['true']) if r is not None]
    acc = float(np.mean([r == t for r, t in named]))
    return dict(created_at=created_at, wf20=wf_track, recog=recog, acc=acc)


def main():
    runs = R.load(LW.PAT)
    rng = np.random.default_rng(0); torch.manual_seed(0)
    streams = [R.build(d, wm, r, chg, 10, "H+", rng) for d, wm, r, chg in runs]
    gs = RS.ground_gap_scale(streams)
    d_rec, ratios, accs = [], [], []
    for s in streams:
        on, off = run(s, gs, True), run(s, gs, False)
        accs.append((on['acc'], off['acc']))
        for p, (i, e, reawakened) in on['created_at'].items():
            if not reawakened:
                continue
            if p in on['recog'] and p in off['recog']:
                d_rec.append(off['recog'][p] - on['recog'][p])
            if p in on['wf20'] and p in off['wf20']:
                ratios.append(on['wf20'][p] / off['wf20'][p])
    out = dict(paired_phases_recognition=len(d_rec), recog_diff_median=float(np.median(d_rec)) if d_rec else None,
               recog_diff_mean=float(np.mean(d_rec)) if d_rec else None,
               paired_phases_wfast=len(ratios), wfast_ratio_median=float(np.median(ratios)) if ratios else None,
               wfast_ratio_ge_1p5=float(np.mean(np.array(ratios) >= 1.5)) if ratios else None,
               acc_on=float(np.mean([a for a, _ in accs])), acc_off=float(np.mean([b for _, b in accs])))
    print(f"reawakened return phases: recognition paired n={out['paired_phases_recognition']}, "
          f"OFF-ON steps median {out['recog_diff_median']}, mean {out['recog_diff_mean']:.2f}")
    print(f"  w_fast at +20 steps: paired n={out['paired_phases_wfast']}, ON/OFF median ratio "
          f"{out['wfast_ratio_median']:.2f}, share >= 1.5x {out['wfast_ratio_ge_1p5']:.0%}")
    print(f"  named-report accuracy ON {out['acc_on']:.4f}  OFF {out['acc_off']:.4f}")
    json.dump(out, open(HERE / "savings_test_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

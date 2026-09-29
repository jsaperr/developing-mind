"""A NOVEL report: when memory's best match is below its own novelty radius, say "nothing fits"
instead of naming that match. Replay only, existing substrate runs, rectified readout, radius 0.8,
anchored consolidation (anchored_consolidation.AnchoredGated), both clocks, all 8 worlds.
Retrieval, creation, consolidation and eviction are unchanged; only the REPORT changes.

Motivation (post-hoc diagnosis, experiments_integration.md 2026-09-28): every correct settled report
had a winner matching the query at >= 0.8, while 73-79% of wrong ones matched below 0.8. Memory
names its closest old context while it knows the query is new (creation waits for steady).

Reports: TRANSITIONAL while the substrate re-learns (as adopted); else NOVEL if the winner's cosine
to the query < 0.8; else the winner's content tag (nearest prototype, re-tagged when content moves,
as in readout_set_worlds.run_arm). Scoring on all steps outside TRANSITIONAL; labels only here.

PREDICTIONS ON RECORD (written before this script was first run):
  NV-P1 accuracy of named reports rises by >= 0.05 in ov70 W=10 and >= 0.015 in the long world W=10,
        and falls in no cell.
  NV-P2 NOVEL eats no good reports: of the reports that were correct without NOVEL, <= 0.5% become
        NOVEL, in every cell.
  NV-P3 NOVEL is rare where nothing is new: <= 1% of settled steps in the disjoint worlds (v1b, v1c,
        stg), both clocks.
  NV-P4 NOVEL fires where it should: in ov70 W=10 and the long world W=10, >= 60% of the settled
        steps that were wrong without NOVEL are reported NOVEL instead.

Usage: novel_report.py   (conda env; writes novel_report_summary.json)
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

R, RS, SR, LW = AC.R, AC.RS, AC.SR, AC.LW
NOVEL = "NOVEL"
RADIUS = AC.RADIUS


def run(s, gs):
    m = AC.AnchoredGated(dim=s['q'].shape[1], gap_scale=gs)
    tag, named, novel_rep, settled = {}, [], [], []
    for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
        o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
        if o['created'] is not None:
            tag[o['created']] = int(np.argmax(s['protos'] @ x.numpy()))
        wid = o['winner']
        if wid in m.mem.ids:
            p = m.mem.patterns[m.mem.ids.index(wid)]
            if o['consolidation_rate'] > 0:
                tag[wid] = int(np.argmax(s['protos'] @ p.numpy()))
            sim = float(p @ x)
        else:
            sim = 1.0
        if o['report'] is TRANSITIONAL:
            named.append(None); novel_rep.append(None)
        else:
            named.append(tag[wid]); novel_rep.append(NOVEL if sim < RADIUS else tag[wid])
        settled.append(bool(s['settled'][i]))
    return named, novel_rep, settled


def score(s, named, novel_rep, settled):
    true = s['true']
    idx = [i for i in range(len(named)) if named[i] is not None]
    acc_old = np.mean([named[i] == true[i] for i in idx])
    nm = [i for i in idx if novel_rep[i] != NOVEL]
    acc_new = np.mean([novel_rep[i] == true[i] for i in nm]) if nm else np.nan
    correct = [i for i in idx if named[i] == true[i]]
    wrong_settled = [i for i in idx if named[i] != true[i] and settled[i]]
    st = [i for i in idx if settled[i]]
    return dict(acc_old=acc_old, acc_new=acc_new,
                good_lost=np.mean([novel_rep[i] == NOVEL for i in correct]) if correct else 0.0,
                novel_settled=np.mean([novel_rep[i] == NOVEL for i in st]) if st else 0.0,
                wrong_caught=(np.mean([novel_rep[i] == NOVEL for i in wrong_settled]) if wrong_settled else np.nan),
                n_wrong_settled=len(wrong_settled))


def main():
    out = {}
    worlds = [(n, p) for n, p, _ in SR.WORLDS] + [("long world", LW.PAT)]
    for name, pat in worlds:
        runs = R.load(pat)
        for W in (10, 50):
            rng = np.random.default_rng(0); torch.manual_seed(0)
            streams = [R.build(d, wm, r, chg, W, "H+", rng) for d, wm, r, chg in runs]
            gs = RS.ground_gap_scale(streams)
            sc = [score(s, *run(s, gs)) for s in streams]
            n_wrong = sum(x['n_wrong_settled'] for x in sc)
            caught = [x['wrong_caught'] for x in sc if x['n_wrong_settled']]
            row = dict(acc_named_before=float(np.mean([x['acc_old'] for x in sc])),
                       acc_named_after=float(np.nanmean([x['acc_new'] for x in sc])),
                       good_reports_lost=float(np.mean([x['good_lost'] for x in sc])),
                       novel_frac_settled=float(np.mean([x['novel_settled'] for x in sc])),
                       wrong_settled=int(n_wrong),
                       wrong_caught=float(np.mean(caught)) if caught else None)
            out[f"{name} W{W}"] = row
            print(f"{name:10s} W={W:2d}  accuracy of named reports {row['acc_named_before']:.3f} -> {row['acc_named_after']:.3f} | "
                  f"good reports lost {row['good_reports_lost']:.2%} | NOVEL on settled steps {row['novel_frac_settled']:.2%} | "
                  f"settled wrong {n_wrong}, caught as NOVEL "
                  + (f"{row['wrong_caught']:.0%}" if caught else "-"))
    json.dump(out, open(HERE / "novel_report_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

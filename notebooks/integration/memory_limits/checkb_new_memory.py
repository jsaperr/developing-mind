"""The integrated check (b) (set_worlds_data/checkb: A F1 B F2 C F3 A; cores disjoint, fillers never
return), re-scored with the new default memory (src DormantGatedMemory: radius 0.8, anchoring,
dormant entries, NOVEL report) next to the previous one (GatedEpisodicMemory, radius 0.5).
Replay only, rectified readout, both clocks.

Scores (labels only here):
  A recognized  A's return phase is NAMED correctly in > half of its settled steps (report level,
                so a reawakened entry counts; the older "born in A's first phase" metric would not)
  character     per core, the largest w_char among entries (live, and dormant for the new memory)
                whose content is nearest that core; all 3 present / A highest / C > B
  accuracy      named-report accuracy; NOVEL share of settled steps

PREDICTIONS ON RECORD (written before this script was first run):
  CN-P1 all three cores still carry character at the end in 8/8 with the new memory at BOTH clocks
        (the old memory at W=10 kept all three live in only 2/8: forgetting deleted character).
  CN-P2 A recognized: new >= old at both clocks (old: W=10 5/8, W=50 8/8 by the genuine metric).
  CN-P3 primacy holds (A highest >= 6/8, new memory, both clocks); recency doesn't appear (C > B
        <= 4/8): these runs are ~100-520 memory steps, far below w_char's ~2000-step timescale.
  CN-P4 accuracy of named reports: new >= old - 0.005 at both clocks.

Usage: checkb_new_memory.py   (conda env; writes checkb_new_memory_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchored_consolidation as AC
from src.hopfield.episodic_consolidating import TRANSITIONAL, GatedEpisodicMemory
from src.hopfield.episodic_dormant import NOVEL, DormantGatedMemory

R, RS = AC.R, AC.RS


def run(s, gs, new):
    m = DormantGatedMemory(dim=30, gap_scale=gs) if new else GatedEpisodicMemory(dim=30, gap_scale=gs)
    tag, reports = {}, []
    for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
        o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
        if o['created'] is not None:
            tag[o['created']] = int(np.argmax(s['protos'] @ x.numpy()))
        rep = o['report']
        reports.append(None if rep is TRANSITIONAL else ("N" if rep == NOVEL else tag.get(rep)))
    items = m.character() if new else [(p, w, 'live') for p, w in zip(m.mem.patterns, m.mem.w_char)]
    char = {}
    for p, w, _ in items:
        c = int(np.argmax(s['protos'] @ p.numpy()))
        char[c] = max(char.get(c, 0.0), float(w))
    ret = np.where((s['ph'] == 6) & s['settled'])[0]
    a_rec = np.mean([reports[i] == 0 for i in ret]) > 0.5
    named = [(reports[i], int(s['true'][i])) for i in range(len(reports)) if reports[i] not in (None, "N")]
    st = [i for i in range(len(reports)) if s['settled'][i] and reports[i] is not None]
    return dict(a_rec=bool(a_rec), char=char, acc=float(np.mean([r == t for r, t in named])),
                novel=float(np.mean([reports[i] == "N" for i in st])))


def main():
    runs = R.load(R.B2 / "set_worlds_data" / "checkb_n7_seed*.json.gz")
    out = {}
    for W in (10, 50):
        rng = np.random.default_rng(0); torch.manual_seed(0)
        streams = [R.build(d, wm, r, chg, W, "H+", rng) for d, wm, r, chg in runs]
        gs = RS.ground_gap_scale(streams)
        for new in (False, True):
            res = [run(s, gs, new) for s in streams]
            full = [x for x in res if all(c in x['char'] for c in (0, 1, 2))]
            row = dict(A_recognized=int(sum(x['a_rec'] for x in res)), cores_with_character=len(full),
                       A_highest=int(sum(x['char'][0] > max(x['char'][1], x['char'][2]) for x in full)),
                       C_over_B=int(sum(x['char'][2] > x['char'][1] for x in full)),
                       acc=float(np.mean([x['acc'] for x in res])), novel_settled=float(np.mean([x['novel'] for x in res])))
            key = f"W{W} {'new (dormant, r0.8)' if new else 'old (gated, r0.5)'}"
            out[key] = row
            print(f"{key:26s} A recognized {row['A_recognized']}/8 | all 3 cores carry character {row['cores_with_character']}/8 "
                  f"| A highest {row['A_highest']}/{len(full)} C>B {row['C_over_B']}/{len(full)} | acc {row['acc']:.3f} "
                  f"| NOVEL {row['novel_settled']:.1%}")
    json.dump(out, open(HERE / "checkb_new_memory_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

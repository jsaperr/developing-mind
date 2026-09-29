"""POST-HOC diagnostics (2026-09-28), run after anchored_consolidation.py and long_world_readout.py, not
pre-registered. (1) da_check: are D-phase queries/entries A-like? (2) ov70_merge_check: are merged entries born
blended? (3) ov70_report_check: do wrong reports come from below-radius matches? Outputs recorded in
experiments_integration.md (2026-09-28 entries)."""
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import long_world_readout as L, numpy as np, torch
from src.hopfield.episodic_consolidating import GatedEpisodicMemory
runs = L.R.load(L.PAT)
rng = np.random.default_rng(0); torch.manual_seed(0)
streams = [L.R.build(d, wm, r, chg, 10, "H+", rng) for d, wm, r, chg in runs]
gs = L.RS.ground_gap_scale(streams)
# every episode created during a D phase (no store): what does its content look like?
n, alike_A, rep_A, settledD = 0, 0, 0, 0
for s in streams:
    m = GatedEpisodicMemory(dim=30, gap_scale=gs, theta=0.8, match_floor=0.8)
    tag = {}
    for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
        o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
        if o['created'] is not None:
            tag[o['created']] = int(np.argmax(s['protos'] @ x.numpy()))
            if s['true'][i] == 3:
                n += 1; alike_A += tag[o['created']] == 0
        if s['true'][i] == 3 and s['settled'][i]:
            settledD += 1; rep_A += (o['report'] is not None and tag.get(o['winner']) == 0)
print(f"episodes created in D phases: {n}, whose content is closest to A: {alike_A}")
print(f"settled D steps reported as A (no store): {rep_A}/{settledD}")
q = np.concatenate([s['q'][(s['true'] == 3) & s['settled']] for s in streams])
print("settled D queries: cosine to D proto median %.2f, to A proto median %.2f, frac closer to A %.2f" % (
    np.median(q @ streams[0]['protos'][3]), np.median(q @ streams[0]['protos'][0]),
    np.mean(q @ streams[0]['protos'][0] > q @ streams[0]['protos'][3])))

"""POST-HOC diagnostics (2026-09-28), run after anchored_consolidation.py and long_world_readout.py, not
pre-registered. (1) da_check: are D-phase queries/entries A-like? (2) ov70_merge_check: are merged entries born
blended? (3) ov70_report_check: do wrong reports come from below-radius matches? Outputs recorded in
experiments_integration.md (2026-09-28 entries)."""
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchored_consolidation as AC, numpy as np, torch
R, RS = AC.R, AC.RS
runs = R.load(R.B2 / "set_worlds_data" / "ov70_n7_seed*.json.gz")
rng = np.random.default_rng(0); torch.manual_seed(0)
streams = [R.build(d, wm, r, chg, 10, "H+", rng) for d, wm, r, chg in runs]
gs = RS.ground_gap_scale(streams)
blend, total, steps_after_change = 0, 0, []
merged_birth = []
for s in streams:
    m = AC.AnchoredGated(dim=30, gap_scale=gs)
    wins = {}
    ps = [0] + list(np.where(np.diff(s['ph']) != 0)[0] + 1)
    for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
        o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
        if o['created'] is not None:
            c = s['protos'] @ x.numpy(); top = np.sort(c)[::-1]
            total += 1; blend += top[0] < 0.9
            steps_after_change.append(i - max(p for p in ps if p <= i))
        if s['settled'][i] and o['report'] is not None:
            wins.setdefault(o['winner'], {}).setdefault(int(s['true'][i]), 0)
            wins[o['winner']][int(s['true'][i])] += 1
    for e, c in wins.items():
        if sum(v >= 10 for v in c.values()) >= 2:
            p0 = m.mem.birth[e].numpy() if e in m.mem.birth else None
            if p0 is not None:
                merged_birth.append(np.round(np.sort(s['protos'] @ p0)[::-1][:2], 2).tolist())
print(f"entries created: {total}; birth content a blend (best prototype cosine < 0.9): {blend}")
print("steps after a phase change at creation (W=10 steps):", np.bincount(steps_after_change)[:40].tolist())
print("merged entries' birth content, top-2 prototype cosines:", merged_birth)

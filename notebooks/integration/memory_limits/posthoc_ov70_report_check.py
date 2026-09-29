"""POST-HOC diagnostics (2026-09-28), run after anchored_consolidation.py and long_world_readout.py, not
pre-registered. (1) da_check: are D-phase queries/entries A-like? (2) ov70_merge_check: are merged entries born
blended? (3) ov70_report_check: do wrong reports come from below-radius matches? Outputs recorded in
experiments_integration.md (2026-09-28 entries)."""
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchored_consolidation as AC, numpy as np, torch
R, RS = AC.R, AC.RS
for world in ("ov70", "long"):
    pat = R.B2 / "set_worlds_data" / "ov70_n7_seed*.json.gz" if world == "ov70" else AC.LW.PAT
    runs = R.load(pat)
    rng = np.random.default_rng(0); torch.manual_seed(0)
    streams = [R.build(d, wm, r, chg, 10, "H+", rng) for d, wm, r, chg in runs]
    gs = RS.ground_gap_scale(streams)
    wrong_below = wrong_above = right_below = right = 0
    for s in streams:
        m = AC.AnchoredGated(dim=30, gap_scale=gs)
        born = {}
        for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
            o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
            if o['created'] is not None:
                born[o['created']] = int(s['true'][i])
            if not s['settled'][i] or o['report'] is None:
                continue
            k = m.mem.ids.index(o['winner']) if o['winner'] in m.mem.ids else None
            sim = float(m.mem.patterns[k] @ x) if k is not None else 1.0
            ok = born.get(o['winner']) == int(s['true'][i])
            if ok:
                right += 1; right_below += sim < 0.8
            else:
                wrong_below += sim < 0.8; wrong_above += sim >= 0.8
    print(f"{world} W=10 anchored, settled committed reports: correct {right} (of which winner sim < 0.8: {right_below}); "
          f"wrong with winner sim < 0.8: {wrong_below}; wrong with sim >= 0.8: {wrong_above}")

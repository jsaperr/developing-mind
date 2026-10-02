"""Scores U-P1..P4 (predictions in modal_unify.py, written before launch) on the compact files unify_n<N>_seed<S>.json.gz
(loaded with modal_common.load_compact, which reproduces these analyses exactly). Written before any result was read.
Metrics are step 2's (analyze_nscale.py), applied to the unified substrate. The old-substrate N=7/40 step 2 numbers are
printed for comparison.

Usage: analyze_unify.py   (conda env; writes unify_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(ROOT / "notebooks" / "brian2" / "set_worlds_data"))
sys.path.insert(0, str(ROOT / "notebooks" / "integration" / "set_worlds"))
import analyze_release_timing as RT
import modal_common as MC
import readout_set_worlds as RS
from src.hopfield.episodic_consolidating import TRANSITIONAL
from src.hopfield.episodic_dormant import NOVEL, DormantGatedMemory

R, I = RT.R, RT.I
SW = [1000, 2000, 3000, 4000]


def hold(runs, c, t):
    return float(np.mean([RT.holders(d, wm, c, t) for d, wm, r, ch in runs]))


def main():
    out = {}
    old = json.load(open(ROOT / "notebooks" / "brian2" / "n_scaling_v1b_data" / "nscale_summary.json"))
    for n in (7, 40):
        runs = [MC.load_compact(f) for f in sorted(HERE.glob(f"unify_n{n}_seed*.json.gz"))]
        runs = [x for x in runs if x[0].get("status") == "completed"]
        if not runs:
            print(f"N={n}: none"); continue
        k = len(runs)
        late = [r[:, 4500:5000].mean(1) for d, wm, r, ch in runs]
        rate, active = float(np.mean([x.mean() for x in late])), float(np.mean([(x > 0.1).mean() for x in late]))
        b_keep = hold(runs, 1, 2150) / max(hold(runs, 1, 2000), 1e-9)
        a3, b4 = hold(runs, 0, 3000) / n, hold(runs, 1, 4000) / n
        fired, fp = [], []
        for d, wm, r, ch in runs:
            ch = np.asarray(ch, bool)
            for i, s in enumerate(SW):
                fired.append(bool(ch[s:s + 300].any()))
                end = SW[i + 1] if i + 1 < len(SW) else int(d['total_s'])
                fp.append(float(ch[s + 300:end].mean()))
        rng = np.random.default_rng(0); torch.manual_seed(0)
        s10 = [R.build(d, wm, r, ch, 10, "H+", rng) for d, wm, r, ch in runs]
        qt = float(np.median([s['q'][i] @ s['protos'][s['true'][i]] for s in s10 for i in range(len(s['q'])) if s['settled'][i]]))
        rng = np.random.default_rng(0); torch.manual_seed(0)
        s50 = [R.build(d, wm, r, ch, 50, "H+", rng) for d, wm, r, ch in runs]
        gs = RS.ground_gap_scale(s50)
        got = 0
        for s in s50:
            m = DormantGatedMemory(dim=30, gap_scale=gs); tag, rb = {}, []
            for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
                o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
                if o['created'] is not None:
                    tag[o['created']] = int(np.argmax(s['protos'] @ x.numpy()))
                rep = o['report']; lab = None if (rep is TRANSITIONAL or rep == NOVEL) else tag.get(rep)
                if s['settled'][i] and s['ph'][i] == 4:
                    rb.append(lab == 1)
            got += np.mean(rb) > 0.5
        row = dict(seeds=k, late_rate=rate, late_active=active, B_keep=b_keep, A_before3=a3, B_before4=b4,
                   change_fired=float(np.mean(fired)), change_settled_on=float(np.mean(fp)), readout=qt, B_named_W50=int(got),
                   theta_range_mV=[float(np.min([np.min(d['theta_final_mV']) for d, *_ in runs])), float(np.max([np.max(d['theta_final_mV']) for d, *_ in runs]))])
        out[str(n)] = row
        o = old.get(str(n), {})
        print(f"N={n} ({k} seeds) | U-P1 late rate {rate:.2f} Hz, active {active:.0%} | U-P2 B keeps {b_keep:.0%} (old {o.get('B_keep', float('nan')):.0%}), "
              f"before two-back A {a3:.0%} B {b4:.0%} of N (old {o.get('A_before_swap3', float('nan')):.0%}/{o.get('B_before_swap4', float('nan')):.0%}) | "
              f"U-P3 change fired {row['change_fired']:.0%}, settled-on {row['change_settled_on']:.1%} | U-P4 readout {qt:.3f}, "
              f"B named W50 {got}/{k} | theta range {row['theta_range_mV'][0]:.1f}..{row['theta_range_mV'][1]:.1f} mV")
    json.dump(out, open(HERE / "unify_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

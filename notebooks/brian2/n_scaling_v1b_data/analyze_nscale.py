"""Scores NS-P1..P5 (predictions in modal_nscale.py, written before launch) on nscale_v1b_n<N>_seed<S>.json.gz.
Written before any result was read.

Holders: step 0's metric (each neuron's 50 s mean weight vector, nearest centered block prototype).
Change signal: src/integration/interface.changing_per_second at its fixed defaults (the tripwire: no re-tuning).
Readout and memory: the default rectified readout and src DormantGatedMemory, via the integration replay code.

Usage: analyze_nscale.py   (conda env; writes nscale_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "notebooks" / "brian2" / "set_worlds_data"))
sys.path.insert(0, str(ROOT / "notebooks" / "integration" / "set_worlds"))
import analyze_release_timing as RT
import readout_set_worlds as RS
from src.hopfield.episodic_consolidating import TRANSITIONAL
from src.hopfield.episodic_dormant import NOVEL, DormantGatedMemory

R, I = RT.R, RT.I
NS = [5, 7, 10, 15, 40]
SW = [1000, 2000, 3000, 4000]


def hold(runs, c, t):
    return float(np.mean([RT.holders(d, wm, c, t) for d, wm, r, ch in runs]))


def main():
    out = {}
    for n in NS:
        runs = R.load(HERE / f"nscale_v1b_n{n}_seed*.json.gz")
        if not runs:
            print(f"N={n}: no completed runs"); continue
        k = len(runs)
        row = dict(seeds=k, wall_s=float(np.mean([d['wall_elapsed'] for d, *_ in runs])))
        # NS-P1 / P2 (holders)
        b_keep = hold(runs, 1, 2150) / max(hold(runs, 1, 2000), 1e-9)
        a_before3, b_before4 = hold(runs, 0, 3000) / n, hold(runs, 1, 4000) / n
        a_retain = hold(runs, 0, 2000) / n
        row.update(B_keep=b_keep, A_before_swap3=a_before3, B_before_swap4=b_before4, A_retainer_frac=a_retain,
                   holders={str(t): [hold(runs, c, t) for c in range(3)] for t in (1000, 2000, 2150, 3000, 3150, 4000, 4150, 5000)})
        # NS-P3 (change signal at fixed defaults)
        fired, fp = [], []
        for d, wm, r, ch in runs:
            ch = np.asarray(ch, bool)
            for i, s in enumerate(SW):
                fired.append(bool(ch[s:s + 300].any()))
                end = SW[i + 1] if i + 1 < len(SW) else int(d['total_s'])
                fp.append(float(ch[s + 300:end].mean()))
        row.update(change_fired=float(np.mean(fired)), change_settled_on=float(np.mean(fp)))
        # NS-P4 (readout) and NS-P5 (memory)
        rng = np.random.default_rng(0); torch.manual_seed(0)
        s10 = [R.build(d, wm, r, ch, 10, "H+", rng) for d, wm, r, ch in runs]
        q_true = [s['q'][i] @ s['protos'][s['true'][i]] for s in s10 for i in range(len(s['q'])) if s['settled'][i]]
        between = []
        for s in s10:
            means = {}
            for p in np.unique(s['ph']):
                m = s['q'][(s['ph'] == p) & s['settled']]
                if len(m):
                    means.setdefault(int(s['true'][s['ph'] == p][0]), []).append(I.unit(m.mean(0)))
            ctx = {c: I.unit(np.mean(v, 0)) for c, v in means.items()}
            cs = sorted(ctx); between += [float(ctx[a] @ ctx[b]) for i, a in enumerate(cs) for b in cs[i + 1:]]
        row.update(readout_settled_median=float(np.median(q_true)), between_max=float(max(between)))
        for W in (50, 10):
            rng = np.random.default_rng(0); torch.manual_seed(0)
            streams = s10 if W == 10 else [R.build(d, wm, r, ch, W, "H+", rng) for d, wm, r, ch in runs]
            gs = RS.ground_gap_scale(streams)
            got_b, got_a = 0, 0
            for s in streams:
                m = DormantGatedMemory(dim=30, gap_scale=gs)
                tag, rb, ra = {}, [], []
                for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
                    o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
                    if o['created'] is not None:
                        tag[o['created']] = int(np.argmax(s['protos'] @ x.numpy()))
                    rep = o['report']
                    lab = None if (rep is TRANSITIONAL or rep == NOVEL) else tag.get(rep)
                    if s['settled'][i] and s['ph'][i] == 4:
                        rb.append(lab == 1)
                    if s['settled'][i] and s['ph'][i] == 3:
                        ra.append(lab == 0)
                got_b += np.mean(rb) > 0.5; got_a += np.mean(ra) > 0.5
            row[f"W{W}"] = dict(B_two_back_named=int(got_b), A_two_back_named=int(got_a))
        out[str(n)] = row
        print(f"N={n:2d} ({k} seeds, wall {row['wall_s']:.0f} s) | B keeps {b_keep:.0%} at C | before two-back: A {a_before3:.0%}, "
              f"B {b_before4:.0%} of N | A retainers at end of B {a_retain:.0%} | change fired {row['change_fired']:.0%}, "
              f"on when settled {row['change_settled_on']:.1%} | readout {row['readout_settled_median']:.3f}, between max "
              f"{row['between_max']:.2f} | B named W50 {row['W50']['B_two_back_named']}/{k} W10 {row['W10']['B_two_back_named']}/{k}"
              f" | A named W50 {row['W50']['A_two_back_named']}/{k}")
    print("\nNS-P1 B keep >= 60% and incoming <= 15% | NS-P2 retainers 35-65% | NS-P3 fired >= 90%, settled-on < 5% | "
          "NS-P4 readout >= 0.9, between < 0.8 | NS-P5 B named (W50) >= 7/8 -- at every N")
    json.dump(out, open(HERE / "nscale_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

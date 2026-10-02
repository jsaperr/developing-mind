"""Scores F-P1..P3 (predictions in modal_factorial.py, written before launch) on fact_<cell>_n<N>_seed<S>.json.gz with
analyze_unify.py's metrics, unchanged; the unified cell (unify_n<N>_seed<S>) is printed alongside. Written before results.
MNIST arms (F-P4, F-P5): analyze_mnist_pilot.py mnist_fact_<arm> v3.

Usage: analyze_factorial.py   (conda env; writes factorial_summary.json and factorial_output.txt)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_unify as AU

MC, R, RS, RT = AU.MC, AU.R, AU.RS, AU.RT


def score(prefix, n):
    runs = [MC.load_compact(f) for f in sorted(HERE.glob(f"{prefix}_n{n}_seed*.json.gz"))]
    runs = [x for x in runs if x[0].get("status") == "completed"]
    if not runs:
        return None
    k = len(runs)
    late = [r[:, 4500:5000].mean(1) for d, wm, r, ch in runs]
    hold = lambda c, t: float(np.mean([RT.holders(d, wm, c, t) for d, wm, r, ch in runs]))
    fired, fp = [], []
    for d, wm, r, ch in runs:
        ch = np.asarray(ch, bool)
        for i, s in enumerate(AU.SW):
            fired.append(bool(ch[s:s + 300].any()))
            end = AU.SW[i + 1] if i + 1 < len(AU.SW) else int(d['total_s'])
            fp.append(float(ch[s + 300:end].mean()))
    rng = np.random.default_rng(0); torch.manual_seed(0)
    s10 = [R.build(d, wm, r, ch, 10, "H+", rng) for d, wm, r, ch in runs]
    qt = float(np.median([s['q'][i] @ s['protos'][s['true'][i]] for s in s10 for i in range(len(s['q'])) if s['settled'][i]]))
    rng = np.random.default_rng(0); torch.manual_seed(0)
    s50 = [R.build(d, wm, r, ch, 50, "H+", rng) for d, wm, r, ch in runs]
    gs = RS.ground_gap_scale(s50)
    got = 0
    for s in s50:
        m = AU.DormantGatedMemory(dim=30, gap_scale=gs); tag, rb = {}, []
        for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
            o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
            if o['created'] is not None:
                tag[o['created']] = int(np.argmax(s['protos'] @ x.numpy()))
            rep = o['report']; lab = None if (rep is AU.TRANSITIONAL or rep == AU.NOVEL) else tag.get(rep)
            if s['settled'][i] and s['ph'][i] == 4:
                rb.append(lab == 1)
        got += np.mean(rb) > 0.5
    # active-neuron count: how many neurons carry the settled response (share of N above 0.1 Hz, and above 1 Hz)
    act = float(np.mean([(x > 0.1).mean() for x in late])); act1 = float(np.mean([(x > 1.0).mean() for x in late]))
    return dict(seeds=k, late_rate=float(np.mean([x.mean() for x in late])), active=act, active_1hz=act1,
                B_keep=hold(1, 2150) / max(hold(1, 2000), 1e-9), A_before3=hold(0, 3000) / n, B_before4=hold(1, 4000) / n,
                change_fired=float(np.mean(fired)), change_settled_on=float(np.mean(fp)), readout=qt, B_named_W50=int(got),
                passes=bool(qt >= 0.9 and got >= 7))


def main():
    out, lines = {}, []
    for prefix in ("fact_ctrl", "fact_fair", "fact_wta", "unify"):
        for n in (7, 40):
            row = score(prefix, n)
            if row is None:
                lines.append(f"{prefix} N={n}: none"); continue
            out[f"{prefix}_n{n}"] = row
            lines.append(f"{prefix:9s} N={n:2d} ({row['seeds']} seeds) | rate {row['late_rate']:.2f} Hz, active {row['active']:.0%} (>1 Hz {row['active_1hz']:.0%}) | "
                         f"B keeps {row['B_keep']:.0%}, incoming before two-back A {row['A_before3']:.0%} B {row['B_before4']:.0%} | "
                         f"change fired {row['change_fired']:.0%}, settled-on {row['change_settled_on']:.1%} | readout {row['readout']:.3f}, "
                         f"B named W50 {row['B_named_W50']}/{row['seeds']} | {'PASS' if row['passes'] else 'fail'}")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "factorial_output.txt").write_text(txt + "\n", encoding="utf-8")
    json.dump(out, open(HERE / "factorial_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

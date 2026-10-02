"""Scores MC-P1..P5 (predictions in modal_many.py, written before launch) on many6_n<N>_seed<S>.json.gz.
Written before any result was read.

World: A B C D E F | E C F B F A (12 x 1000 s). A return's depth k = the incoming context's position in the
recency list just before it returns (1 = the just-departed context).
Holders: step 0's metric with six block prototypes. Change signal: interface.changing_per_second at fixed defaults.
Memory: the default src DormantGatedMemory. A return counts as REMEMBERED when the context is named in more than half
of its settled checks AND the entry that first names it is old (born before the return, or reawakened from dormant).

Usage: analyze_many.py   (conda env; writes many_summary.json)
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
NS = [7, 40]
L = "ABCDEF"


def returns(sched):
    """[(phase index, context, depth)] for every phase whose context appeared before."""
    out, recency = [], []
    for p, c in enumerate(sched):
        if c in recency:
            out.append((p, c, recency.index(c)))
        if c in recency:
            recency.remove(c)
        recency.insert(0, c)
    return out


def holders60(d, wm, c, t):
    """Step 0's holder metric (RT.holders), sized to the rig's own input count. RT.holders hard-codes 30 inputs;
    this fix was made after the first scoring attempt crashed on it, before any result was read. Same metric."""
    n_pre = wm.shape[1]
    protos = np.array([I.unit(np.isin(np.arange(n_pre), s).astype(float)) for s in d['context_sets']])
    wbar = wm[:, :, t - 50:t].mean(axis=2)
    return sum(int(np.argmax(protos @ I.unit(wbar[j]))) == c for j in range(d['n_post']))


def hold(runs, c, t):
    return float(np.mean([holders60(d, wm, c, t) for d, wm, r, ch in runs]))


def main():
    out = {}
    for n in NS:
        runs = R.load(HERE / f"many6_n{n}_seed*.json.gz")
        if not runs:
            print(f"N={n}: no completed runs"); continue
        d0 = runs[0][0]; sched = d0['phase_corr_blocks']; sw = [int(x) for x in d0['swap_times_s']]; k = len(runs)
        row = dict(seeds=k)
        keeps = [hold(runs, sched[i], s + 150) / max(hold(runs, sched[i], s), 1e-9) for i, s in enumerate(sw)]
        row['keep_min'] = float(min(keeps)); row['keep_each'] = [round(x, 2) for x in keeps]
        rets = returns(sched)
        inc = [(p, c, dep, hold(runs, c, sw[p - 1]) / n) for p, c, dep in rets]
        row['incoming'] = [(L[c], dep, round(v, 3)) for p, c, dep, v in inc]
        fired, fp = [], []
        for d, wm, r, ch in runs:
            ch = np.asarray(ch, bool)
            for i, s in enumerate(sw):
                fired.append(bool(ch[s:s + 300].any()))
                end = sw[i + 1] if i + 1 < len(sw) else int(d['total_s'])
                fp.append(float(ch[s + 300:end].mean()))
        row.update(change_fired=float(np.mean(fired)), change_settled_on=float(np.mean(fp)))
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
            per_ret = {p: [0, 0, 0] for p, c, dep in rets}           # remembered, named-only, dormant-reawakened
            for s in streams:
                m = DormantGatedMemory(dim=60, gap_scale=gs)
                tag, born, reawoken, first = {}, {}, set(), {}
                named = {p: [] for p, c, dep in rets}
                start = {p: int(np.where(s['ph'] == p)[0][0]) for p, c, dep in rets}
                for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
                    o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
                    if o['created'] is not None:
                        tag[o['created']] = int(np.argmax(s['protos'] @ x.numpy())); born[o['created']] = i
                        if o['reawakened'] is not None:
                            reawoken.add(o['created'])
                    rep = o['report']
                    lab = None if (rep is TRANSITIONAL or rep == NOVEL) else tag.get(rep)
                    p = int(s['ph'][i])
                    if p in named:
                        c = sched[p]
                        if s['settled'][i]:
                            named[p].append(lab == c)
                        if lab == c and p not in first:
                            first[p] = rep
                for p, c, dep in rets:
                    half = np.mean(named[p]) > 0.5 if named[p] else False
                    w = first.get(p)
                    old = w is not None and (born.get(w, -1) < start[p] or w in reawoken)
                    per_ret[p][0] += bool(half and old); per_ret[p][1] += bool(half)
                    per_ret[p][2] += bool(w in reawoken) if w is not None else 0
            row[f"W{W}"] = {f"{L[c]} (phase {p + 1}, {dep} back)": dict(remembered=v[0], named=v[1], via_dormant=v[2])
                            for (p, c, dep), v in zip(rets, per_ret.values())}
        out[str(n)] = row
        print(f"\n===== N={n} ({k} seeds) =====")
        print(f"MC-P1 just-departed keeps at +150 s, every change: min {row['keep_min']:.0%}  {row['keep_each']}")
        print("MC-P2 incoming holders just before each return (share of N): "
              + "  ".join(f"{a}@{dep}back {v:.0%}" for a, dep, v in row['incoming']))
        print(f"MC-P3 change fired {row['change_fired']:.0%}, on when settled {row['change_settled_on']:.1%}")
        print(f"MC-P4 readout {row['readout_settled_median']:.3f}, between max {row['between_max']:.2f}")
        for W in (50, 10):
            print(f"MC-P5 W={W}: " + " | ".join(f"{key}: remembered {v['remembered']}/{k} (named {v['named']}, via dormant {v['via_dormant']})"
                                                for key, v in row[f"W{W}"].items()))
    json.dump(out, open(HERE / "many_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

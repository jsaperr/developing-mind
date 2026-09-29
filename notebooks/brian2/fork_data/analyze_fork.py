"""Scores FK-P1..P5 (predictions in run_fork_seed.py, written before launch) on the fork runs (seeds
48000-48007; branches twin1, twin2, detour; shared A B up to 2000 s, branch 2000-3000 s, identical test A B
3000-5000 s).

Distances between weight states (7 neurons x 30 inputs) are neuron-matched: rows are paired by the Hungarian
algorithm to minimize total distance, since different seeds can learn the same thing on different neurons.
Memory is the default src DormantGatedMemory on the rectified readout.

ADDED 2026-09-29, before any fork result existed (committed while the batch ran): FK-P2b, a functional
comparison. Raw weights can differ while what the network represents stays the same (principles.md: stability
is a population readout that tolerates synapse churn). During the test all branches get IDENTICAL clicks, so
compare the twins' rectified fingerprints window by window (10 s):
  FK-P2b twins represent the same thing: the median cosine between twin1's and twin2's fingerprints over the
         settled test windows is >= 0.9, whatever the raw weight distance does. The same statistic between
         independent seeds (same world schedule, different history noise) is reported as the reference.

Usage: analyze_fork.py   (conda env; writes fork_summary.json)
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "notebooks" / "brian2" / "set_worlds_data"))
sys.path.insert(0, str(ROOT / "notebooks" / "integration" / "set_worlds"))
import analyze_release_timing as RT
import readout_set_worlds as RS
from src.hopfield.episodic_consolidating import TRANSITIONAL
from src.hopfield.episodic_dormant import NOVEL, DormantGatedMemory

R, I = RT.R, RT.I
P3 = np.array([I.unit(np.isin(np.arange(30), s).astype(float)) for s in ([*range(0, 10)], [*range(10, 20)], [*range(20, 30)])])
BR = ("twin1", "twin2", "detour")
CHECK = [1900, 2100, 2500, 3000, 3500, 4000, 4500, 5000]


def W(wm, t):
    return wm[:, :, t - 20:t].mean(axis=2)


def dist(a, b):
    c = ((a[:, None, :] - b[None, :, :]) ** 2).sum(2)
    r, k = linear_sum_assignment(c)
    return float(np.sqrt(c[r, k].sum()))


def holders(wm, c, t):
    wbar = wm[:, :, t - 50:t].mean(axis=2)
    return sum(int(np.argmax(P3 @ I.unit(wbar[j]))) == c for j in range(wm.shape[0]))


def main():
    PREFIX = sys.argv[1] if len(sys.argv) > 1 else "fork"
    runs = {b: R.load(HERE / f"{PREFIX}_{b}_n7_seed*.json.gz") for b in BR}
    seeds = [d['seed'] for d, *_ in runs['twin1']]
    assert all([d['seed'] for d, *_ in runs[b]] == seeds for b in BR), "branch/seed mismatch"
    print(f"fork seeds: {len(seeds)}")
    out = {}

    mx = max(float(np.abs(runs[b][k][1][:, :, :2000] - runs['twin1'][k][1][:, :, :2000]).max())
             for b in BR for k in range(len(seeds)))
    first_diff = []
    for k in range(len(seeds)):
        dd = np.abs(runs['detour'][k][1] - runs['twin1'][k][1]).max(axis=(0, 1))
        first_diff.append(int(np.argmax(dd > 0)))
    print(f"FK-P1: max |weight difference| between branches before 2000 s: {mx:.3g} (== 0?); first differing second: {sorted(set(first_diff))}")

    print("neuron-matched weight distance (mean over seeds):")
    print("   t (s)   twin1-twin2   detour-twin1   independent seeds")
    rows = []
    for t in CHECK:
        tw = np.mean([dist(W(runs['twin1'][k][1], t), W(runs['twin2'][k][1], t)) for k in range(len(seeds))])
        de = np.mean([dist(W(runs['detour'][k][1], t), W(runs['twin1'][k][1], t)) for k in range(len(seeds))])
        ind = np.mean([dist(W(runs['twin1'][a][1], t), W(runs['twin1'][b][1], t)) for a, b in combinations(range(len(seeds)), 2)])
        rows.append((t, tw, de, ind))
        print(f"  {t:6d}   {tw:11.3f}   {de:12.3f}   {ind:17.3f}")
    ratio = max(tw / ind for t, tw, de, ind in rows if t >= 2100)
    print(f"FK-P2: largest twin / independent ratio after the fork: {ratio:.0%} (<= 30%?)")

    a_tw = np.mean([holders(runs[b][k][1], 0, 3000) for b in ("twin1", "twin2") for k in range(len(seeds))])
    a_de = np.mean([holders(runs['detour'][k][1], 0, 3000) for k in range(len(seeds))])
    print(f"FK-P3: A-neurons at the test's start: twins {a_tw:.2f} (>= 2.5?), detour {a_de:.2f} (<= 1?)")

    end = rows[-1]
    per_seed = [dist(W(runs['detour'][k][1], 5000), W(runs['twin1'][k][1], 5000)) /
                max(dist(W(runs['twin1'][k][1], 5000), W(runs['twin2'][k][1], 5000)), 1e-9) for k in range(len(seeds))]
    print(f"FK-P4: at 5000 s detour-twin1 {end[2]:.3f} vs twin1-twin2 {end[1]:.3f} -> ratio {end[2] / end[1]:.2f} (<= 1.5?); "
          f"per seed ratios {[round(x, 2) for x in per_seed]}")
    def fingerprints(run):
        d, wm, r, ch = run
        return {a: I.readout(r[:, a:a + 10].mean(1), wm[:, :, a:a + 10].mean(2), kind="rectified")
                for a in list(range(3300, 4000, 10)) + list(range(4300, 5000, 10))}
    fp = {b: [fingerprints(run) for run in runs[b]] for b in BR}
    tw_cos = [np.median([fp['twin1'][k][a] @ fp['twin2'][k][a] for a in fp['twin1'][k]]) for k in range(len(seeds))]
    de_cos = [np.median([fp['detour'][k][a] @ fp['twin1'][k][a] for a in fp['twin1'][k]]) for k in range(len(seeds))]
    ind_cos = [np.median([fp['twin1'][a][t] @ fp['twin1'][b][t] for t in fp['twin1'][a]]) for a, b in combinations(range(len(seeds)), 2)]
    print(f"FK-P2b: fingerprint cosine over the settled test (identical clicks): twin1-twin2 median {np.median(tw_cos):.3f} "
          f"(>= 0.9?; per seed {[round(float(x), 2) for x in tw_cos]}), detour-twin1 {np.median(de_cos):.3f}, independent seeds {np.median(ind_cos):.3f}")
    out['functional'] = dict(twin_cos=[float(x) for x in tw_cos], detour_cos=[float(x) for x in de_cos], independent_cos=float(np.median(ind_cos)))
    out['substrate'] = dict(max_prefix_diff=mx, distances=rows, A_at_test=[a_tw, a_de], end_ratio_per_seed=per_seed)

    for Wc in (50, 10):
        res = {}
        for b in BR:
            rng = np.random.default_rng(0); torch.manual_seed(0)
            streams = [R.build(d, wm, r, ch, Wc, "H+", rng) for d, wm, r, ch in runs[b]]
            gs = RS.ground_gap_scale(streams) if b == "twin1" else res['gs']
            res['gs'] = gs
            c_char, first_a, named_a = 0, [], []
            for s in streams:
                m = DormantGatedMemory(dim=30, gap_scale=gs)
                tag, t_first = {}, None
                for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
                    o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
                    if o['created'] is not None:
                        tag[o['created']] = int(np.argmax(P3 @ x.numpy()))
                    rep = o['report']
                    k = None if (rep is TRANSITIONAL or rep == NOVEL) else tag[rep]
                    if s['ph'][i] == 3:
                        if k == 0 and t_first is None:
                            t_first = s['starts'][i] - 3000
                        if s['settled'][i]:
                            named_a.append(k == 0)
                first_a.append(t_first if t_first is not None else np.nan)
                c_char += any(int(np.argmax(P3 @ p.numpy())) == 2 and w >= 1.5 for p, w, _ in m.character())
            res[b] = dict(C_character=c_char, first_A_named_s=float(np.nanmean(first_a)), A_named_settled=float(np.mean(named_a)))
        print(f"W={Wc}: " + " | ".join(f"{b}: C character {res[b]['C_character']}/8, test-A first named after "
                                        f"{res[b]['first_A_named_s']:.0f} s, named A {res[b]['A_named_settled']:.0%} of settled" for b in BR))
        out[f"W{Wc}"] = {b: res[b] for b in BR}
    print("FK-P5 (W=50): detour C character 8/8 and twins 0/8?")
    json.dump(out, open(HERE / f"{PREFIX}_summary.json", "w"), indent=1, default=float)


if __name__ == '__main__':
    main()

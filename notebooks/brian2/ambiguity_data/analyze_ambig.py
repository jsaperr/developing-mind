"""Scores AM-P1..P6 (predictions in run_ambig_seed.py, written before launch) on the ambiguity world
A(1000) B(1000) AB(2000) A(1000). Substrate: holders (step 0's metric, contexts A/B/C), per-neuron switching,
weight displacement. Memory: the default src DormantGatedMemory replayed on the rectified readout.

Usage: analyze_ambig.py   (conda env; writes ambig_summary.json)
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
P3 = np.array([I.unit(np.isin(np.arange(30), s).astype(float)) for s in ([*range(0, 10)], [*range(10, 20)], [*range(20, 30)])])
PAB = I.unit(np.isin(np.arange(30), [*range(0, 20)]).astype(float))
P4 = np.vstack([P3, PAB])
LAB = ["A", "B", "C", "AB"]


def prefs(wm, t):
    wbar = wm[:, :, t - 50:t].mean(axis=2)
    return [int(np.argmax(P3 @ I.unit(wbar[j]))) for j in range(wm.shape[0])]


def displacement(wm):
    w = wm.reshape(-1, wm.shape[2]); out = np.zeros(wm.shape[2])
    out[60:] = np.linalg.norm(w[:, 60:] - w[:, :-60], axis=0)
    return out


def main():
    runs = R.load(HERE / "ambig_n7_seed*.json.gz")
    print(f"ambiguity seeds: {len(runs)}")
    out = {}
    marks = [(2000, "end of B (AB starts)"), (2300, "AB +300"), (3000, "AB +1000"), (4000, "end of AB"), (4150, "A back +150"), (5000, "end")]
    H = {}
    print("mean holders per context (A B C):")
    for t, lab in marks:
        v = [float(np.mean([prefs(wm, t).count(c) for d, wm, r, ch in runs])) for c in range(3)]
        H[t] = v; print(f"  {lab:22s} " + "  ".join(f"{x:4.2f}" for x in v))
    dA, dB = abs(H[4000][0] - H[2000][0]), abs(H[4000][1] - H[2000][1])
    print(f"AM-P1: change end of B -> end of AB: A {dA:.2f}, B {dB:.2f} (both <= 0.5?)")

    sw = []
    for d, wm, r, ch in runs:
        seq = [prefs(wm, t) for t in range(2300, 4001, 10)]
        n = 0
        for j in range(wm.shape[0]):
            s = [x[j] for x in seq if x[j] in (0, 1)]
            n += sum(1 for a, b in zip(s, s[1:]) if a != b)
        sw.append(n)
    print(f"AM-P2: A<->B preference switches per seed during AB (after +300 s): mean {np.mean(sw):.2f}, per seed {sw} (mean <= 1?)")

    pk = np.array([(displacement(wm)[1000:1300].max(), displacement(wm)[2000:2300].max()) for d, wm, r, ch in runs])
    print(f"AM-P3: displacement peak after B's arrival {pk[:, 0].mean():.3f}, at AB onset {pk[:, 1].mean():.3f}; "
          f"AB smaller in {(pk[:, 1] < pk[:, 0]).sum()}/8 (>= 6?)")
    out['substrate'] = dict(holders={str(k): v for k, v in H.items()}, switches=sw, disp=pk.mean(0).tolist())

    for W in (10, 50):
        rng = np.random.default_rng(0); torch.manual_seed(0)
        streams = [R.build(d, wm, r, ch, W, "H+", rng) for d, wm, r, ch in runs]
        gs = RS.ground_gap_scale(streams)
        cosA, cosB, counts, blend_mem, a_back, flips = [], [], {k: 0 for k in ["A", "B", "C", "AB", "N", "T"]}, 0, 0, []
        for s in streams:
            m = DormantGatedMemory(dim=30, gap_scale=gs)
            tag, made_blend, ret, prev, nflip = {}, False, [], None, 0
            for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
                o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
                if o['created'] is not None:
                    tag[o['created']] = int(np.argmax(P4 @ x.numpy()))
                    if s['ph'][i] == 2 and tag[o['created']] == 3:
                        made_blend = True
                rep = o['report']
                k = "T" if rep is TRANSITIONAL else "N" if rep == NOVEL else LAB[tag[rep]]
                if s['ph'][i] == 2 and s['settled'][i]:
                    counts[k] += 1
                    cosA.append(float(P3[0] @ x.numpy())); cosB.append(float(P3[1] @ x.numpy()))
                    if k in ("A", "B"):
                        if prev is not None and prev != k:
                            nflip += 1
                        prev = k
                if s['ph'][i] == 3 and s['settled'][i]:
                    ret.append(k == "A")
            blend_mem += made_blend; a_back += np.mean(ret) > 0.5; flips.append(nflip)
        tot = sum(counts.values()); share = {k: v / tot for k, v in counts.items()}
        print(f"W={W}: settled AB fingerprint cosine to A median {np.median(cosA):.2f}, to B median {np.median(cosB):.2f} | "
              "memory during settled AB: " + "  ".join(f"{k} {share[k]:.0%}" for k in share)
              + f" | blend memory stored {blend_mem}/8 | A<->B flips in named reports per seed {flips} | A named on return {a_back}/8")
        out[f"W{W}"] = dict(cosA=float(np.median(cosA)), cosB=float(np.median(cosB)), reports=share,
                            blend_memory=blend_mem, flips=flips, A_back=a_back)
    print("AM-P4 (W=10): both medians in 0.5-0.8?  AM-P5 (W=10): NOVEL >= 40% and blend memory >= 5/8?  AM-P6: A back 8/8?")
    json.dump(out, open(HERE / "ambig_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

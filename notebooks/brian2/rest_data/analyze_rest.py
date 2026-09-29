"""Scores RE-P1..P5 (predictions in run_rest_seed.py, written before launch) on the rest world
A(1000) B(1000) C(1000) REST(3000) B(1000). Substrate: holders (step 0's metric), selectivity, weight
displacement. Memory: the default src DormantGatedMemory replayed on the rectified readout (W=10 and 50).

Usage: analyze_rest.py   (conda env; writes rest_summary.json)
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

R = RT.R
REST0, REST1 = 3000, 6000
LAB = "ABCR"


def protos3():
    return np.array([RT.I.unit(np.isin(np.arange(30), s).astype(float)) for s in ([*range(0, 10)], [*range(10, 20)], [*range(20, 30)])])


def holders(d, wm, c, t):
    P = protos3(); wbar = wm[:, :, t - 50:t].mean(axis=2)
    return sum(int(np.argmax(P @ RT.I.unit(wbar[j]))) == c for j in range(d['n_post']))


def selectivity(wm, t):
    wbar = wm[:, :, t - 50:t].mean(axis=2)
    blocks = np.stack([wbar[:, 0:10].sum(1), wbar[:, 10:20].sum(1), wbar[:, 20:30].sum(1)], 1)
    return float((blocks.max(1) / blocks.sum(1)).mean())


def displacement(wm):
    w = wm.reshape(-1, wm.shape[2])
    out = np.zeros(wm.shape[2])
    out[60:] = np.linalg.norm(w[:, 60:] - w[:, :-60], axis=0)
    return out


def main():
    runs = R.load(HERE / "rest_n7_seed*.json.gz")
    print(f"rest seeds: {len(runs)}")
    out = {}
    times = [(2950 + 50, "end of C (rest starts)"), (3300, "rest +300"), (4500, "rest +1500"), (6000, "end of rest"), (6150, "B back +150"), (7000, "end")]
    print("mean holders per context (A B C):")
    H = {}
    for t, lab in times:
        v = [float(np.mean([holders(d, wm, c, t) for d, wm, r, chg in runs])) for c in range(3)]
        H[t] = v
        print(f"  {lab:24s} " + "  ".join(f"{x:4.2f}" for x in v))
    drift = max(abs(H[6000][c] - H[3300][c]) for c in range(3))
    print(f"RE-P1: largest holder change during rest {drift:.2f} (<= 0.5?); B holders at end of rest {H[6000][1]:.2f} (>= 2?)")

    s0 = np.mean([selectivity(wm, 3300) for d, wm, r, c in runs]); s1 = np.mean([selectivity(wm, 6000) for d, wm, r, c in runs])
    sC = np.mean([selectivity(wm, 3000) for d, wm, r, c in runs])
    frac = (s0 - s1) / (s0 - 1 / 3)
    print(f"RE-P2: preferred-block share: end of C {sC:.3f}, rest +300 {s0:.3f}, end of rest {s1:.3f} -> fall = {frac:.0%} of excess over 1/3 (>= 10%?)")

    peaks = []
    for d, wm, r, c in runs:
        disp = displacement(wm)
        peaks.append((disp[2000:2300].max(), disp[3000:3300].max(), disp[3300:6000].max()))
    peaks = np.array(peaks)
    n_smaller = int((peaks[:, 1] < peaks[:, 0]).sum())
    print(f"RE-P3: 60 s displacement peak after C's arrival {peaks[:, 0].mean():.3f}, at rest onset {peaks[:, 1].mean():.3f}, "
          f"rest +300..end {peaks[:, 2].mean():.3f}; rest onset smaller in {n_smaller}/8 (>= 6?)")
    flagged = [float(RT.I.changing_per_second(wm)[3000:3300].mean()) for d, wm, r, c in runs]
    print(f"      change flag on during the first 300 s of rest: {np.mean(flagged):.0%} of seconds (mean over seeds)")
    out['substrate'] = dict(holders={str(k): v for k, v in H.items()}, rest_drift=drift, selectivity=[sC, s0, s1],
                            selectivity_fall=frac, disp_peaks=peaks.mean(0).tolist(), rest_onset_smaller=n_smaller)

    for W in (10, 50):
        rng = np.random.default_rng(0); torch.manual_seed(0)
        streams = [R.build(d, wm, r, chg, W, "H+", rng) for d, wm, r, chg in runs]
        gs = RS.ground_gap_scale(streams)
        rest_counts, created_rest, b_live, b_named, ghost_cos = {k: 0 for k in "ABCRNT"}, 0, 0, 0, []
        for s in streams:
            m = DormantGatedMemory(dim=30, gap_scale=gs)
            P = np.vstack([protos3(), np.zeros(30)])
            tag, made_rest, b_alive_at_return = {}, False, None
            ret = [i for i in range(len(s['ph'])) if s['ph'][i] == 4 and s['settled'][i]]
            for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
                if s['ph'][i] == 4 and b_alive_at_return is None:
                    b_alive_at_return = any(tag.get(e) == 1 for e in m.mem.ids)
                o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
                if o['created'] is not None:
                    tag[o['created']] = int(np.argmax(P @ x.numpy()))
                    if s['ph'][i] == 3:
                        made_rest = True
                if s['ph'][i] == 3:
                    rep = o['report']
                    k = "T" if rep is TRANSITIONAL else "N" if rep == NOVEL else LAB[tag[rep]]
                    rest_counts[k] += 1
                    ghost_cos.append(float(np.max(protos3() @ x.numpy())))
            reps = []
            created_rest += made_rest; b_live += bool(b_alive_at_return)
        tot = sum(rest_counts.values())
        share = {k: v / tot for k, v in rest_counts.items()}
        print(f"W={W}: memory during rest: " + "  ".join(f"{k} {share[k]:.0%}" for k in "ABCRNT")
              + f" | stored a memory during rest in {created_rest}/8 | B's memory live when B returns {b_live}/8"
              + f" | rest fingerprint's best cosine to A/B/C: median {np.median(ghost_cos):.2f}")
        out[f"W{W}"] = dict(rest_reports=share, created_during_rest=created_rest, B_live_at_return=b_live,
                            rest_best_cos_median=float(np.median(ghost_cos)))
    print("RE-P4 (W=10): names B or C in >= 30% of rest checks? stores a rest memory in <= 3/8?  RE-P5 (W=10): B live >= 5/8?")
    json.dump(out, open(HERE / "rest_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

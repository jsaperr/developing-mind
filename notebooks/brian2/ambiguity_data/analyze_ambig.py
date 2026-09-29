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
    out['substrate'] = dict(holders={str(k): v for k, v in H.items()}, switches=[int(x) for x in sw], disp=pk.mean(0).tolist())

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
                            blend_memory=int(blend_mem), flips=[int(f) for f in flips], A_back=int(a_back))
    print("AM-P4 (W=10): both medians in 0.5-0.8?  AM-P5 (W=10): NOVEL >= 40% and blend memory >= 5/8?  AM-P6: A back 8/8?")
    json.dump(out, open(HERE / "ambig_summary.json", "w"), indent=1)


def post_hoc():
    """POST-HOC (added after seeing the results, not predicted). Memory flickered between naming A and B
    instead of seeing a blend. Mechanism guess: the rectified readout H+ only counts neurons firing ABOVE the
    population mean. With both groups driven about equally, whichever group is slightly ahead in a window takes
    the whole fingerprint, so the readout itself makes a winner-take-all decision from noise.
    Checks: (1) in settled AB windows, how often the above-average set contains only one group's neurons;
    (2) the unrectified contrast readout H on the same windows, which should give a blend (NOVEL / a blend
    memory) instead of a flicker, if the mechanism is right; (3) dominance durations of the flicker."""
    runs = R.load(HERE / "ambig_n7_seed*.json.gz")
    W = 10
    one_group, n_win = 0, 0
    for d, wm, r, ch in runs:
        st = [a for a in range(2300, 4000 - W + 1, W)]
        for a in st:
            rates = r[:, a:a + W].mean(1); pr = prefs(wm, a + W)
            above = [j for j in range(len(rates)) if rates[j] > rates.mean()]
            grp = {pr[j] for j in above if pr[j] in (0, 1)}
            one_group += len(grp) == 1; n_win += 1
    print(f"POST-HOC (1): settled AB 10 s windows where the above-average neurons all hold ONE of A/B: {one_group / n_win:.0%}")
    for ro in ("H+", "H"):
        rng = np.random.default_rng(0); torch.manual_seed(0)
        streams = [R.build(d, wm, r, ch, W, ro, rng) for d, wm, r, ch in runs]
        gs = RS.ground_gap_scale(streams)
        cA, cB, cnt, blend, runs_len = [], [], {k: 0 for k in ["A", "B", "C", "AB", "N", "T"]}, 0, []
        for s in streams:
            m = DormantGatedMemory(dim=30, gap_scale=gs)
            tag, made, cur, ln = {}, False, None, 0
            for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
                o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
                if o['created'] is not None:
                    tag[o['created']] = int(np.argmax(P4 @ x.numpy()))
                    made |= s['ph'][i] == 2 and tag[o['created']] == 3
                rep = o['report']
                k = "T" if rep is TRANSITIONAL else "N" if rep == NOVEL else LAB[tag[rep]]
                if s['ph'][i] == 2 and s['settled'][i]:
                    cnt[k] += 1; cA.append(float(P3[0] @ x.numpy())); cB.append(float(P3[1] @ x.numpy()))
                    if k in ("A", "B"):
                        if k == cur: ln += 1
                        else:
                            if cur is not None: runs_len.append(ln)
                            cur, ln = k, 1
            blend += made
        tot = sum(cnt.values())
        print(f"POST-HOC (2) readout {ro:2s}: settled AB cosine to A median {np.median(cA):.2f}, to B {np.median(cB):.2f} | "
              "reports " + "  ".join(f"{k} {cnt[k] / tot:.0%}" for k in cnt) + f" | blend memory stored {blend}/8"
              + (f" | (3) dominance run length: mean {np.mean(runs_len):.2f} checks, P(run=1) {np.mean(np.array(runs_len) == 1):.0%}" if runs_len else ""))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'posthoc':
        post_hoc()
    elif len(sys.argv) == 1:
        main()


def post_hoc_timescale():
    """POST-HOC (4): is the activity dominance structured (rivalry-like episodes) or white noise? Per second in
    settled AB (2300-4000 s): D(t) = mean rate of A-holders - mean rate of B-holders. Report its
    autocorrelation time (lag where autocorrelation first drops below 1/e) and the mean dominance episode
    length (run of same sign of D smoothed over 5 s). White noise gives ~1 s; structured rivalry gives far
    longer episodes. Control: the same statistic for two random halves of neurons during pure A (4300-5000 s)."""
    runs = R.load(HERE / "ambig_n7_seed*.json.gz")
    def stats(D):
        D = D - D.mean(); ac = np.correlate(D, D, 'full')[len(D) - 1:]; ac = ac / ac[0]
        tau = int(np.argmax(ac < 1 / np.e)) if (ac < 1 / np.e).any() else len(ac)
        sm = np.convolve(D, np.ones(5) / 5, 'same'); sg = np.sign(sm); sg = sg[sg != 0]
        runs_ = np.diff(np.flatnonzero(np.r_[True, sg[1:] != sg[:-1], True]))
        return tau, runs_.mean()
    taus, eps, ctl_t, ctl_e = [], [], [], []
    for d, wm, r, ch in runs:
        pr = prefs(wm, 2300)
        a = [j for j in range(len(pr)) if pr[j] == 0]; b = [j for j in range(len(pr)) if pr[j] == 1]
        if not a or not b:
            continue
        D = r[a, 2300:4000].mean(0) - r[b, 2300:4000].mean(0)
        t_, e_ = stats(D); taus.append(t_); eps.append(e_)
        half = len(pr) // 2
        Dc = r[:half, 4300:5000].mean(0) - r[half:, 4300:5000].mean(0)
        t_, e_ = stats(Dc); ctl_t.append(t_); ctl_e.append(e_)
    print(f"POST-HOC (4): A-vs-B activity difference in AB: autocorrelation time median {np.median(taus)} s, "
          f"mean dominance episode {np.mean(eps):.1f} s | control (random halves, pure A): {np.median(ctl_t)} s, {np.mean(ctl_e):.1f} s")


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'timescale':
    post_hoc_timescale()


def post_hoc_magnitude():
    """POST-HOC (5): the readout normalizes, so any small A-minus-B activity difference becomes a full-strength
    fingerprint. Does the UNNORMALIZED H+ magnitude, sum_j max(r_j - mean r, 0) * w_j before unit(), carry the
    ambiguity instead? Compare settled 10 s windows in pure A (phase 1 and the final A) against settled AB."""
    runs = R.load(HERE / "ambig_n7_seed*.json.gz")
    mag = {"pure (A, B phases)": [], "ambiguous (AB)": []}
    for d, wm, r, ch in runs:
        for lo, hi, key in ((300, 1000, "pure (A, B phases)"), (1300, 2000, "pure (A, B phases)"), (2300, 4000, "ambiguous (AB)")):
            for a in range(lo, hi - 10 + 1, 10):
                rates = r[:, a:a + 10].mean(1); w = wm[:, :, a:a + 10].mean(2)
                v = np.clip(rates - rates.mean(), 0, None) @ w
                mag[key].append(float(np.linalg.norm(v - v.mean())))
    p, q = np.array(mag["pure (A, B phases)"]), np.array(mag["ambiguous (AB)"])
    thr = np.percentile(p, 5)
    print(f"POST-HOC (5): unnormalized H+ magnitude, median: pure {np.median(p):.2f}, ambiguous {np.median(q):.2f} "
          f"(ratio {np.median(q) / np.median(p):.2f}); AB windows below the 5th percentile of pure windows: {np.mean(q < thr):.0%}")


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'magnitude':
    post_hoc_magnitude()

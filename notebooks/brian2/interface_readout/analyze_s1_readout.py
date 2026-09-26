"""S1 (system_contract.md): does a LABEL-FREE readout of the substrate carry context?

Every STDP result so far is measured with experimenter labels (which block is correlated). A
downstream memory layer only sees rates, weights and spikes. Analysis only, on saved runs:
  A->B->A  N=7 13mV/1.5  ../nonstationary_data/nonstat_n7_reliable_13_1p5_* (20 inputs, 8 seeds)
  A->B->C  N=7 13mV/1.5  ../novelc_data/novelc_n7_reliable_13_1p5_*       (30 inputs, 8 seeds)
Block labels are used ONLY to score the readouts (which phase is which context), never to build them.

Readouts, each computed per non-overlapping window of W seconds (W in 10, 50, 200 = the clock
question, Q2):
  R  rate vector: each neuron's mean rate in the window. dim N=7. (Contract reading I.)
  W  weight state: every synapse's mean weight in the window, rows ordered by (post, pre) index.
     dim N*n_pre = 140 / 210. (Contract reading II, population weights as one pattern.)
  H  hybrid: sum_j (r_j - mean r) * w_j, where w_j is neuron j's weight vector over presynaptic
     inputs. That's the input-space direction the currently more active neurons are tuned to.
     dim n_pre = 20 / 30. Readings I and II combined, which is what a linear decoder reading
     activity through learned tuning computes.
Similarity: Pearson correlation between two readout vectors (centered cosine; raw cosine is near
1 for any two all-positive vectors). "Settled" = windows starting at least 300 s into a phase
(the swap dip recovers over about 300 s).

Tests:
  T1 within-context stability: mean similarity of consecutive settled windows in the same phase.
     Inflated by temporal proximity (weights move slowly), so read it as churn tolerance, not
     context evidence.
  T2 adjacent-context distinctness: mean similarity between settled windows of phase k and k+1.
  T3 LULL TEST (A->B->A; the key one): using each phase's mean settled readout, is
     sim(A3, A1) > sim(A3, B2)? Same context 2000 s apart against a different context that's
     adjacent in time. Passing means context beats temporal adjacency. Counted per seed.
  T4 one-back contamination (A->B->C, descriptive): sim(C3, B2) vs sim(C3, A1).
  T5 scale: T2's between-context similarity against the regime the Hopfield layer was validated
     in (random 64-d unit vectors, cosine about 0 +/- 0.125, episodic_layer_v1.ipynb).
  Also: the cross-neuron rate spread per phase, to diagnose a degenerate (uniform) rate vector.

PREDICTIONS ON RECORD, written before this script was first run:
  S1-P1 (the other session's): R discriminates contexts poorly and W well (T2). The reason given:
        homeostatic scaling holds every neuron near the same rate, and context lives in weights.
  S1-P2 (derived from the logged retainer/refuser counts, not from any readout): W FAILS the lull
        test, with sim(A3,A1) <= sim(A3,B2) in at least 4/8 seeds. One-back retention means
        holders (A in both) and refusers (B in both) make phase 3 match phase 2 on about 4-6 of 7
        neurons, against about 4-5 of 7 for phase 1.
  S1-P3: R is degenerate in phase 1 (every neuron is on A, so rates are near uniform), so R's lull
        test is uninformative whatever it says. Checked via the rate spread.
  S1-P4: H PASSES the lull test in at least 6/8 seeds. Weighting by current rate emphasizes the
        neurons tuned to the current input and down-weights retainers of the old one.
Everything else is descriptive.

Usage: analyze_s1_readout.py        (prints tables; saves s1_similarity_matrices.png)
"""
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = [
    ("A->B->A", HERE.parent / "nonstationary_data", "nonstat_n7_reliable_13_1p5"),
    ("A->B->C", HERE.parent / "novelc_data", "novelc_n7_reliable_13_1p5"),
]
WINDOWS_S = [10, 50, 200]
SETTLE_S = 300
READOUTS = ["R", "W", "H"]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else np.nan


def load_run(path):
    d = json.load(open(path))
    if d.get('status') != 'completed':
        return None
    n = d['n_post']; si = np.array(d['syn_i']); sj = np.array(d['syn_j'])
    n_pre = int(si.max()) + 1
    w = np.array(d['weight_trace'])                   # (n_syn, n_t), 1 s samples
    wm = np.zeros((n, n_pre, w.shape[1]))
    wm[sj, si] = w                                     # (post, pre, t) -- index order, label-free
    r = np.array(d['spike_rate_bins'])                 # (n, n_bins) Hz, 1 s bins
    phase_starts = [0.0] + list(d['swap_times_s'])
    return d, wm, r, phase_starts


def windows(d, wm, r, phase_starts, W):
    """List of (phase_index, settled, R, Wvec, H) per non-overlapping window."""
    T = int(d['total_s']); out = []
    for a in range(0, T - W + 1, W):
        b = a + W
        ph = int(np.searchsorted(phase_starts, a, side='right') - 1)
        if ph + 1 < len(phase_starts) and b > phase_starts[ph + 1]:
            continue                                   # window straddles a swap
        rv = r[:, a:b].mean(axis=1)
        wmat = wm[:, :, a:b].mean(axis=2)              # (n, n_pre)
        out.append((ph, a - phase_starts[ph] >= SETTLE_S, rv, wmat.ravel(),
                    (rv - rv.mean()) @ wmat))
    return out


def phase_mean(ws, ph, k):
    v = [x[k] for x in ws if x[0] == ph and x[1]]
    return np.mean(v, axis=0) if v else None


def analyze():
    rng = np.random.default_rng(0)
    rand64 = [corr(*[v / np.linalg.norm(v) for v in rng.standard_normal((2, 64))]) for _ in range(2000)]
    print(f"T5 reference: Pearson between random 64-d unit vectors: mean {np.mean(rand64):+.3f} "
          f"sd {np.std(rand64):.3f}  (the regime the Hopfield layer was validated in)\n")

    fig_data = {}
    for name, ddir, prefix in DATASETS:
        runs = [x for x in (load_run(f) for f in sorted(glob.glob(str(ddir / f"{prefix}_seed*.json")))) if x]
        print(f"===== {name} ({len(runs)} seeds) =====")
        spread = {}
        for d, wm, r, ps in runs:
            for ph in range(len(ps)):
                a = int(ps[ph]) + SETTLE_S; b = int(ps[ph + 1]) if ph + 1 < len(ps) else int(d['total_s'])
                spread.setdefault(ph, []).append(r[:, a:b].mean(axis=1).std())
        print("cross-neuron rate spread (sd of settled per-neuron mean rate, Hz) by phase: " +
              "  ".join(f"phase{ph + 1} {np.mean(v):.2f}" for ph, v in spread.items()))

        for W in WINDOWS_S:
            print(f"\n  window {W} s")
            for k, ro in enumerate(READOUTS, start=2):
                t1, t2, lull, t4 = [], [], [], []
                for d, wm, r, ps in runs:
                    ws = windows(d, wm, r, ps, W)
                    settled = [x for x in ws if x[1]]
                    t1 += [corr(settled[i][k], settled[i + 1][k]) for i in range(len(settled) - 1)
                           if settled[i][0] == settled[i + 1][0]]
                    for ph in range(len(ps) - 1):
                        A = [x[k] for x in settled if x[0] == ph]; B = [x[k] for x in settled if x[0] == ph + 1]
                        t2 += [corr(a, b) for a in A for b in B]
                    m = [phase_mean(ws, ph, k) for ph in range(len(ps))]
                    if name == "A->B->A":
                        lull.append((corr(m[2], m[0]), corr(m[2], m[1])))
                    else:
                        t4.append((corr(m[2], m[1]), corr(m[2], m[0])))
                    if W == 50 and d['seed'] in (31000, 32000):
                        fig_data[(name, ro)] = ws
                line = (f"    {ro}: T1 within {np.nanmean(t1):+.3f} | T2 adjacent-between {np.nanmean(t2):+.3f}"
                        f" (sd {np.nanstd(t2):.3f})")
                if lull:
                    L = np.array(lull)
                    passes = int(np.sum(L[:, 0] > L[:, 1]))
                    line += (f" | T3 lull: sim(A3,A1) {np.nanmean(L[:, 0]):+.3f} vs sim(A3,B2) "
                             f"{np.nanmean(L[:, 1]):+.3f}, pass {passes}/{len(L)}")
                if t4:
                    T = np.array(t4)
                    line += (f" | T4: sim(C3,B2) {np.nanmean(T[:, 0]):+.3f} vs sim(C3,A1) {np.nanmean(T[:, 1]):+.3f},"
                             f" C3 closer to B2 in {int(np.sum(T[:, 0] > T[:, 1]))}/{len(T)}")
                print(line)
        print()
    return fig_data


def figure(fig_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row, name in enumerate(["A->B->A", "A->B->C"]):
        for col, ro in enumerate(READOUTS):
            ws = fig_data[(name, ro)]; k = 2 + col
            M = np.array([[corr(a[k], b[k]) for b in ws] for a in ws])
            ax = axes[row, col]
            im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
            edges = [i for i in range(1, len(ws)) if ws[i][0] != ws[i - 1][0]]
            for e in edges:
                ax.axhline(e - 0.5, color='k', lw=0.6); ax.axvline(e - 0.5, color='k', lw=0.6)
            ax.set_title(f"{name}  readout {ro}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.6, label="Pearson similarity")
    fig.suptitle("S1: window-by-window similarity, 50 s windows (seed 31000 top, 32000 bottom); lines = world swaps")
    out = HERE / "s1_similarity_matrices.png"
    fig.savefig(out, dpi=80)
    print("saved", out.name)


if __name__ == '__main__':
    figure(analyze())

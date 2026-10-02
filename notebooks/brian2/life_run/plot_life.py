"""Figure: one life (seed 54000) on a single time axis (2026-10-02, descriptive).
Rows: (1) the schedule (home contexts coloured, one-offs grey, rests white) with era boundaries; (2) character (max w_char
over live + dormant entries per home context; default memory, W=10, as in analyze_life.py); (3) memory store size (live,
dormant); (4) the two change signals per phase (weight signal = re-learning, activity signal = any switch); (5) how many
neurons hold each home context (50 s assignments).
Usage: plot_life.py [seed]   (writes life_seed<S>.png)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "notebooks" / "integration" / "set_worlds"), str(ROOT / "notebooks" / "integration" / "familiar_switch")]
import analyze_familiar_switch as FS
import analyze_life as AL
import readout_set_worlds as RS
from src.hopfield.episodic_dormant import DormantGatedMemory

HOME = list("ABCDEFG")
COL = dict(A="#1f77b4", B="#ff7f0e", C="#2ca02c", D="#d62728", E="#9467bd", F="#8c564b", G="#e377c2")


def main(seed=54000):
    runs = AL.load()
    d = next(r for r in runs if r["seed"] == seed)
    S = [AL.stream(r, 10) for r in runs]
    gs = RS.ground_gap_scale(S)
    s = S[[r["seed"] for r in runs].index(seed)]
    m = DormantGatedMemory(dim=60, gap_scale=gs)
    tk, ch, live, dorm = [], {h: [] for h in HOME}, [], []
    for i, x in enumerate(torch.tensor(s["q"], dtype=torch.float32)):
        m.step(x, steady=bool(s["steady"][i]), changing=bool(s["changing"][i]))
        if i % 20 == 0:
            tk.append(i * 10 / 3600)
            c = {}
            for pat, w, st in m.character():
                k = s["pn"][int(np.argmax(s["protos"] @ np.asarray(pat, float)))]
                c[k] = max(c.get(k, 1.0), float(w))
            for h in HOME:
                ch[h].append(c.get(h, 1.0))
            live.append(len(m.mem.ids)); dorm.append(len(m.dormant))
    names, ps = d["phase_names"], d["phase_s"]
    fw = np.asarray(d["change_flags"], bool)
    fa = FS.rule(FS.activity_series(np.array(d["rates10"], float)))
    A = np.array(d["assign_50s"]); pn = d["proto_names"]
    fig, ax = plt.subplots(5, 1, figsize=(15, 11), sharex=True, gridspec_kw=dict(height_ratios=[0.6, 2, 1, 0.8, 1.6]))
    for p, n in enumerate(names):
        c = COL.get(n, "#bbbbbb" if n.startswith("N") else "white")
        ax[0].axvspan(p * ps / 3600, (p + 1) * ps / 3600, color=c, lw=0)
    ax[0].set_yticks([]); ax[0].set_title(f"A life in eras (seed {seed}): 200 phases x 500 s; colours = home contexts, grey = one-offs, white = rest", loc="left")
    for e in (60, 140):
        for a in ax:
            a.axvline(e * ps / 3600, color="k", lw=0.8, ls="--")
    for h in HOME:
        ax[1].plot(tk, ch[h], color=COL[h], label=h, lw=1.5)
    ax[1].set_ylabel("character\n(max w_char)"); ax[1].legend(ncol=7, loc="upper left", fontsize=9)
    ax[2].plot(tk, live, color="k", label="live entries"); ax[2].plot(tk, dorm, color="grey", label="dormant entries")
    ax[2].set_ylabel("memory store"); ax[2].legend(loc="upper left", fontsize=9)
    for p, n in enumerate(names[1:], start=1):
        if n == "REST":
            continue
        a0 = p * ps
        x = a0 / 3600
        ax[3].plot([x], [1], "|", color="tab:red" if fw[int(a0):int(a0) + 300].any() else "lightgrey", ms=10)
        ax[3].plot([x], [0], "|", color="tab:blue" if fa[int(a0) // 10:int(a0) // 10 + 30].any() else "lightgrey", ms=10)
    ax[3].set_yticks([0, 1]); ax[3].set_yticklabels(["activity\n(any switch)", "weights\n(re-learning)"]); ax[3].set_ylim(-0.6, 1.6)
    t50 = np.arange(len(A)) * 50 / 3600
    bottom = np.zeros(len(A))
    for h in HOME:
        if h in pn:
            v = (A == pn.index(h)).sum(1)
            ax[4].fill_between(t50, bottom, bottom + v, color=COL[h], lw=0, step="post"); bottom += v
    ax[4].fill_between(t50, bottom, 40, color="#dddddd", lw=0, step="post")
    ax[4].set_ylabel("neurons holding\neach context"); ax[4].set_ylim(0, 40); ax[4].set_xlabel("simulated hours")
    ax[0].text(30 * ps / 3600, 0.5, "era 1: A, B (C)", ha="center", va="center", fontsize=10, transform=ax[0].get_xaxis_transform())
    ax[0].text(100 * ps / 3600, 0.5, "era 2: C, D (G, A, B)", ha="center", va="center", fontsize=10, transform=ax[0].get_xaxis_transform())
    ax[0].text(170 * ps / 3600, 0.5, "era 3: E, F (A-D)", ha="center", va="center", fontsize=10, transform=ax[0].get_xaxis_transform())
    fig.tight_layout()
    out = HERE / f"life_seed{seed}.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 54000)

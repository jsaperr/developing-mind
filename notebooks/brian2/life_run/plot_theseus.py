"""Figure (2026-10-02, descriptive): wiring vs function across a life. For each home context and each of its visits: cosine of
the full weight vector, and of the functional readout (W=10 rectified fingerprint), to the context's FIRST visit (settled
means). Raw weight trace, one seed. Usage: plot_theseus.py <raw npz> [compact]  (writes theseus_seed<S>.png)"""
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COL = dict(A="#1f77b4", B="#ff7f0e", C="#2ca02c", D="#d62728", E="#9467bd", F="#8c564b")


def unit(v):
    v = np.asarray(v, float) - np.mean(v); n = np.linalg.norm(v); return v / n if n else v


def main():
    comp = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "life_seed54000.json.gz")
    d = json.loads(gzip.decompress(open(comp, "rb").read()))
    tr = np.load(sys.argv[1])["trace"]
    names, ps = d["phase_names"], int(d["phase_s"]); q = np.array(d["fingerprints"]["W10"]["rect"])
    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5))
    for h, c in COL.items():
        vis = [p for p, n in enumerate(names) if n == h]
        w0 = unit(tr[vis[0] * ps + 300:(vis[0] + 1) * ps].mean(0)); q0 = unit(q[(vis[0] * ps + 300) // 10:(vis[0] + 1) * ps // 10].mean(0))
        t = [(p * ps + ps / 2) / 3600 for p in vis]
        wc = [unit(tr[p * ps + 300:(p + 1) * ps].mean(0)) @ w0 for p in vis]
        qc = [unit(q[(p * ps + 300) // 10:(p + 1) * ps // 10].mean(0)) @ q0 for p in vis]
        ax.plot(t, qc, "-o", color=c, ms=4, lw=1.5, label=f"{h} function")
        ax.plot(t, wc, "--", color=c, lw=1, alpha=0.8)
        ax.plot(t, wc, "x", color=c, ms=4, alpha=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("simulated hours"); ax.set_ylabel("cosine to the context's first visit")
    ax.set_title(f"Ship of Theseus (seed {d['seed']}): function (solid, dots) stays ~0.97-0.99 while the wiring (dashed, x) drifts to "
                 "~unrelated", loc="left", fontsize=10)
    ax.legend(ncol=6, fontsize=8, loc="center right")
    ax.set_ylim(-0.2, 1.05)
    fig.tight_layout(); out = HERE / f"theseus_seed{d['seed']}.png"; fig.savefig(out, dpi=110); print("wrote", out)


if __name__ == "__main__":
    main()

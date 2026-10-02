"""Post-hoc (2026-10-02): synaptic turnover vs function over the life (raw 1 s weight trace, one seed).
(1) Strong-synapse persistence: of the top 10% synapses at t0 (every 2000 s), the share still top 10% at t0 + lag.
(2) For each home context: cosine of the FULL weight vector (2400 synapses) and of the context's functional readout (the W=10
rectified fingerprint) between its first and its last visit (settled means).
Usage: analyze_turnover.py <raw npz> [compact json.gz]   (writes turnover_output.txt here)
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAGS = [60, 600, 3600, 3 * 3600, 10 * 3600]


def unit(v):
    v = np.asarray(v, float) - np.mean(v); n = np.linalg.norm(v); return v / n if n else v


def main():
    raw = sys.argv[1]
    comp = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "life_seed54000.json.gz")
    d = json.loads(gzip.decompress(open(comp, "rb").read()))
    tr = np.load(raw)["trace"]                      # (100000, 2400) float32
    k = tr.shape[1] // 10
    top = lambda t: set(np.argpartition(tr[t], -k)[-k:].tolist())
    lines = [f"Turnover, seed {d['seed']}"]
    pers = {L: [] for L in LAGS}
    for t0 in range(1000, tr.shape[0], 2000):
        a = top(t0)
        for L in LAGS:
            if t0 + L < tr.shape[0]:
                pers[L].append(len(a & top(t0 + L)) / k)
    lines.append("top-10% synapses still top-10% after: " + " | ".join(f"{L / 3600:g} h: {np.mean(v):.0%}" for L, v in pers.items()) + " (chance 10%)")
    names, ps = d["phase_names"], int(d["phase_s"]); q = np.array(d["fingerprints"]["W10"]["rect"])
    for h in "ABCDEF":
        vis = [p for p, n in enumerate(names) if n == h]
        if len(vis) < 2:
            continue
        f, l = vis[0], vis[-1]
        wf = tr[f * ps + 300:(f + 1) * ps].mean(0); wl = tr[l * ps + 300:(l + 1) * ps].mean(0)
        qf = unit(q[(f * ps + 300) // 10:(f + 1) * ps // 10].mean(0)); ql = unit(q[(l * ps + 300) // 10:(l + 1) * ps // 10].mean(0))
        lines.append(f"{h}: first vs last visit ({(l - f) * ps / 3600:.1f} h apart): wiring cosine {unit(wf) @ unit(wl):.3f}, function cosine {qf @ ql:.3f}")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "turnover_output.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

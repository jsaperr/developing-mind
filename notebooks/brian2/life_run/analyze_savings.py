"""Post-hoc (2026-10-02): substrate-level savings over the life. For each switch into a home context, the time until >= 25% of
neurons are assigned to it (50 s weight assignments), comparing FIRST visits with returns of contexts the network had let go
(<= 5% of neurons holding them at the switch). Faster re-learning would mean a latent trace in the weights.
Also the latent trace itself: at those switches, the mean weight onto the incoming context's 10 inputs, relative to the
uniform level (budget / 60). All 7 compact files. Usage: analyze_savings.py (writes savings_output.txt here)
"""
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HOME = set("ABCDEF")      # G overlaps A and C, excluded


def main():
    first, ret = [], []
    for f in sorted(HERE.glob("life_seed*.json.gz")):
        d = json.loads(gzip.decompress(f.read_bytes())); names, ps, pn = d["phase_names"], int(d["phase_s"]), d["proto_names"]
        A = np.array(d["assign_50s"]); seen = set()
        for p, n in enumerate(names):
            if n not in HOME or p == 0:
                seen.add(n); continue
            c = pn.index(n); j0 = p * ps // 50
            held0 = np.mean(A[j0 - 1] == c)
            share = (A[j0:j0 + ps // 50] == c).mean(1)
            hit = np.where(share >= 0.25)[0]
            t = (hit[0] + 1) * 50 if len(hit) else None
            if n not in seen:
                first.append(t)
            elif held0 <= 0.05:
                ret.append(t)
            seen.add(n)
    f_ok = [t for t in first if t is not None]; r_ok = [t for t in ret if t is not None]
    lines = [f"Substrate savings (7 seeds): time to >= 25% of neurons on the incoming context",
             f"first visits: n={len(first)}, reached {len(f_ok) / len(first):.0%}, median {np.median(f_ok):.0f} s",
             f"returns after being let go (<= 5% held): n={len(ret)}, reached {len(r_ok) / len(ret):.0%}, median {np.median(r_ok):.0f} s"]
    txt = "\n".join(lines); print(txt); (HERE / "savings_output.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


def fine(raw, comp=None):
    """1 s resolution from the raw weight trace (one seed): time until >= 25% of neurons' weights are nearest the incoming
    context's prototype; first visits vs returns with <= 5% held."""
    import sys
    sys.path.insert(0, str(HERE.parents[2]))
    from src.integration.interface import unit
    d = json.loads(gzip.decompress(open(comp or str(HERE / "life_seed54000.json.gz"), "rb").read()))
    tr = np.load(raw)["trace"]; z = np.load(raw); si, sj = z["syn_i"], z["syn_j"]
    names, ps, pn = d["phase_names"], int(d["phase_s"]), d["proto_names"]
    P = np.array([unit(np.isin(np.arange(60), s).astype(float)) for s in [d["phase_sets"][names.index(k)] for k in pn]])
    def assign(t):
        m = np.zeros((40, 60)); m[sj, si] = tr[t]
        m = m - m.mean(1, keepdims=True); m /= np.linalg.norm(m, axis=1, keepdims=True)
        return np.argmax(m @ P.T, 1)
    first, ret, seen = [], [], set()
    for p, n in enumerate(names):
        if n not in HOME or p == 0:
            seen.add(n); continue
        c = pn.index(n); a0 = p * ps
        held0 = np.mean(assign(a0 - 1) == c)
        t_hit = next((t for t in range(1, ps) if np.mean(assign(a0 + t) == c) >= 0.25), None)
        (first if n not in seen else ret if held0 <= 0.05 else []).append(t_hit)
        seen.add(n)
    out = [f"1 s resolution, seed {d['seed']}: first visits n={len(first)}, median {np.median([t for t in first if t]):.0f} s "
           f"(range {min(t for t in first if t)}-{max(t for t in first if t)}); returns after being let go n={len(ret)}, median "
           f"{np.median([t for t in ret if t]):.0f} s (IQR {np.percentile([t for t in ret if t], 25):.0f}-{np.percentile([t for t in ret if t], 75):.0f})"]
    print(out[0])
    with open(HERE / "savings_output.txt", "a", encoding="utf-8") as fh:
        fh.write(out[0] + "\n")

"""Post-hoc (2026-10-02): recognition latency from raw spikes. After each switch into a home context, score every 1 s window:
each context's score = mean rate of the neurons assigned to it by the 50 s weight assignment FROZEN at the switch (so only
already-existing wiring counts). Latency = first second at which the incoming context has the top score and keeps it for >= 8
of the next 10 s. Split by the incoming context's recency depth (1 = one back) and by how many neurons already held it.
Usage: analyze_recognition_latency.py <raw npz> [compact]   (writes recognition_latency_output.txt here)
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HOME = set("ABCDEFG")


def main():
    comp = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "life_seed54000.json.gz")
    d = json.loads(gzip.decompress(open(comp, "rb").read()))
    z = np.load(sys.argv[1]); st = z["spike_t"].astype(np.float64); si = z["spike_i"].astype(int)
    names, ps, pn = d["phase_names"], int(d["phase_s"]), d["proto_names"]; A = np.array(d["assign_50s"])
    R = np.zeros((40, int(d["total_s"])), np.float32)
    np.add.at(R, (si, np.minimum(st.astype(int), R.shape[1] - 1)), 1)
    rows, rec = [], []
    for p, n in enumerate(names):
        if p > 0 and n in HOME and names[p - 1] != "REST":
            a0 = p * ps; asg = A[a0 // 50 - 1]; c = pn.index(n)
            held = np.mean(asg == c)
            dep = rec.index(n) if n in rec else -1
            lat = None
            if held > 0:
                ctxs = sorted(set(asg.tolist()))
                sc = np.array([R[asg == k, a0:a0 + 300].mean(0) for k in ctxs])      # (n_ctx, 300)
                top = np.array(ctxs)[np.argmax(sc, 0)] == c
                for t in range(0, 290):
                    if top[t] and top[t:t + 10].sum() >= 8:
                        lat = t + 1; break
            rows.append((dep, held, lat))
        if n != "REST":
            if n in rec:
                rec.remove(n)
            rec.insert(0, n)
    lines = [f"Recognition latency, seed {d['seed']}, {len(rows)} home switches (not from rest)"]
    for lab, sel in (("one back", lambda r: r[0] == 1), ("two back", lambda r: r[0] == 2), ("3+ back", lambda r: r[0] >= 3),
                     ("first visit", lambda r: r[0] == -1)):
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        lat = [r[2] for r in rs if r[2] is not None]
        lines.append(f"{lab:11s} n={len(rs):3d} | held by {np.mean([r[1] for r in rs]):.0%} of neurons at the switch | recognized from "
                     f"existing wiring within 300 s: {len(lat) / len(rs):.0%}, median latency {np.median(lat) if lat else float('nan'):.0f} s")
    for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 1.01)):
        rs = [r for r in rows if lo <= r[1] < hi]
        lat = [r[2] for r in rs if r[2] is not None]
        if rs:
            lines.append(f"held {lo:.0%}-{min(hi, 1):.0%}: n={len(rs)}, recognized {len(lat) / len(rs):.0%}, median {np.median(lat) if lat else float('nan'):.0f} s")
    txt = "\n".join(lines); print(txt)
    (HERE / "recognition_latency_output.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

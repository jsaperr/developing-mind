"""Post-hoc (2026-10-02): do context assemblies co-fire during REST, when the input has no structure? Raw life data (one
seed, every spike; spike times stored as float32, so ~4-8 ms resolution late in the life: 25 ms bins).
For each rest phase (and, as a reference, the settled part of the home phase before it): bin each neuron's spikes in 25 ms,
z-score per neuron, correlation matrix. Pairs are "co-tuned" if both neurons' 50 s weight assignment (at the start of the
rest) is the same context, else "cross". Assemblies are ranked by recency (1 = the context just before the rest).
Usage: analyze_rest_replay.py <raw npz> [compact json.gz]   (writes rest_replay_output.txt here)
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BIN = 0.025


def corr_matrix(st, si, t0, t1, n=40):
    m = (st >= t0) & (st < t1)
    nb = int((t1 - t0) / BIN)
    C = np.zeros((n, nb), np.float32)
    np.add.at(C, (si[m].astype(int), np.minimum(((st[m] - t0) / BIN).astype(int), nb - 1)), 1)
    C -= C.mean(1, keepdims=True); sd = C.std(1, keepdims=True); sd[sd == 0] = 1; C /= sd
    return (C @ C.T) / nb, C.mean(0) * 0 + C.sum(0)


def main():
    raw = sys.argv[1]
    comp = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "life_seed54000.json.gz")
    d = json.loads(gzip.decompress(open(comp, "rb").read()))
    z = np.load(raw); st = z["spike_t"].astype(np.float64); si = z["spike_i"]
    names, ps, pn = d["phase_names"], d["phase_s"], d["proto_names"]; A = np.array(d["assign_50s"])
    iu = np.triu_indices(40, 1)
    rows = []
    for p, n in enumerate(names):
        if n != "REST" or p == 0:
            continue
        a0 = p * ps
        asg = A[int(a0 // 50) - 1]
        same = asg[iu[0]] == asg[iu[1]]
        # recency rank of each neuron's context among structured phases before this rest
        rec = []
        for q in range(p - 1, -1, -1):
            if names[q] != "REST" and pn.index(names[q]) not in rec:
                rec.append(pn.index(names[q]))
        out = {}
        for label, (t0, t1) in (("rest", (a0 + 50, a0 + ps)), ("before (active)", (a0 - ps + 300, a0))):
            Cm, _ = corr_matrix(st, si, t0, t1)
            c = Cm[iu]
            out[label] = (c[same].mean() if same.any() else np.nan, c[~same].mean())
        # per-assembly co-firing in rest by recency rank
        Cm, _ = corr_matrix(st, si, a0 + 50, a0 + ps)
        by_rank = {}
        for ctx in set(asg.tolist()):
            members = np.where(asg == ctx)[0]
            if len(members) < 2:
                continue
            sub = Cm[np.ix_(members, members)][np.triu_indices(len(members), 1)].mean()
            r = rec.index(ctx) + 1 if ctx in rec else 99
            by_rank.setdefault(min(r, 4), []).append(sub)
        rows.append((p, names[p - 1], out, by_rank))
    lines = [f"Rest replay, seed {d['seed']}: {len(rows)} rests; mean pairwise correlation of 25 ms spike counts"]
    rs = np.array([[r[2]["rest"][0], r[2]["rest"][1], r[2]["before (active)"][0], r[2]["before (active)"][1]] for r in rows])
    lines.append(f"co-tuned vs cross pairs: REST {np.nanmean(rs[:, 0]):+.4f} vs {np.nanmean(rs[:, 1]):+.4f} "
                 f"(co-tuned higher in {np.mean(rs[:, 0] > rs[:, 1]):.0%} of rests) | ACTIVE before {np.nanmean(rs[:, 2]):+.4f} vs {np.nanmean(rs[:, 3]):+.4f}")
    agg = {}
    for r in rows:
        for k, v in r[3].items():
            agg.setdefault(k, []).extend(v)
    lines.append("within-assembly co-firing in rest by recency (1 = context just before the rest; 4 = 4th or older/one-off): "
                 + " | ".join(f"{k}: {np.mean(v):+.4f} (n={len(v)})" for k, v in sorted(agg.items())))
    txt = "\n".join(lines)
    print(txt)
    (HERE / "rest_replay_output.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

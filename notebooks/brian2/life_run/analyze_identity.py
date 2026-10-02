"""Post-hoc, exploratory (no predictions; 2026-10-02): neuron identity over the life run. When a home context returns after
an absence, are its holders (neurons whose 50 s weights are nearest its prototype at the end of the phase) the same neurons
that held it at the end of its previous visit? Chance = previous holders / N. Also per-neuron loyalty (share of its home-
context time on its favourite), distinct home contexts held, and assignment switches over the life.
Usage: analyze_identity.py  (conda env; reads life_seed*.json.gz here, writes identity_output.txt)
"""
import gzip, json, glob, numpy as np
from collections import Counter
HOME = set("ABCDEFG")
from pathlib import Path
HERE = Path(__file__).resolve().parent
fs = sorted(glob.glob(str(HERE / "life_seed*.json.gz")))
ov, chance, gaps, loyal, ncontexts, switches, rec_ov = [], [], [], [], [], [], {}
for f in fs:
    d = json.loads(gzip.decompress(open(f, 'rb').read())); names = d['phase_names']; pn = d['proto_names']; A = np.array(d['assign_50s'])
    last = {}
    for p, n in enumerate(names):
        if n not in HOME: continue
        c = pn.index(n); end = (p + 1) * 10 - 1                      # last 50 s assignment of the phase
        holders = set(np.where(A[end] == c)[0])
        if n in last and holders:
            prev_holders, prev_p = last[n]
            if prev_holders and p - prev_p > 1:
                o = len(holders & prev_holders) / len(holders)
                # chance: same number drawn at random from 40
                ch = len(prev_holders) / 40
                ov.append(o); chance.append(ch); gaps.append((p - prev_p) * 500)
        last[n] = (holders, p)
    # per-neuron life: share of time on its favourite home context, distinct home contexts held, switches
    for j in range(40):
        seq = [pn[a] for a in A[:, j]]
        h = [s for s in seq if s in HOME]
        if h:
            loyal.append(Counter(h).most_common(1)[0][1] / len(h)); ncontexts.append(len(set(h)))
        switches.append(sum(1 for a, b in zip(seq, seq[1:]) if a != b))
ov, chance, gaps = map(np.array, (ov, chance, gaps))
print(f"returns after absence: {len(ov)}; same neurons re-recruited: {ov.mean():.0%} of the returning context's holders were its previous holders (chance {chance.mean():.0%})")
for lo, hi in ((0, 2000), (2000, 10000), (10000, 1e9)):
    m = (gaps >= lo) & (gaps < hi)
    if m.any(): print(f"  gap {lo/1000:.0f}-{hi/1000 if hi < 1e9 else 'inf'} ks: overlap {ov[m].mean():.0%} vs chance {chance[m].mean():.0%} (n={m.sum()})")
print(f"neurons: loyalty to favourite home context {np.mean(loyal):.0%} (quartiles {np.percentile(loyal,25):.0%}/{np.percentile(loyal,50):.0%}/{np.percentile(loyal,75):.0%}); distinct home contexts held over a life: {np.mean(ncontexts):.1f}; assignment switches per life {np.mean(switches):.0f}")

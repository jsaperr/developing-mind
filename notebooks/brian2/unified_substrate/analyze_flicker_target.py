"""Post-hoc (2026-10-02): where does the rectified readout point during flicker dips (settled 10 s windows with cosine < 0.5
to the current context) on the leaky fair-share cells? Current / previous context / the other context (cosine >= 0.5) /
none. Output: flicker_target_output.txt."""
import sys, glob, numpy as np
R = r"C:\Users\urawi\OneDrive\Desktop\a wide variety of vscode projects\developing-mind"
H = R + r"\notebooks\brian2\unified_substrate"
sys.path[:0] = [R, R + r"\notebooks\brian2"]
import modal_common as MC
from src.integration import interface as I
for pre in ("leak_tau100", "leak_tau50"):
    for n in (7, 40):
        where = {"current": 0, "previous context": 0, "the other context": 0, "none (all < 0.5)": 0}; dips = 0; dur = []
        for f in sorted(glob.glob(H + f"\{pre}_n{n}_seed*.json.gz")):
            d, wm, r, ch = MC.load_compact(f)
            P = np.array([I.unit(np.isin(np.arange(30), s).astype(float)) for s in d['context_sets']]); sch = d['phase_corr_blocks']
            for p in range(1, 5):
                cur, prev = sch[p], sch[p - 1]; other = ({0, 1, 2} - {cur, prev}).pop()
                run = 0
                for a in range(p * 1000 + 300, p * 1000 + 1000, 10):
                    q = I.readout(r[:, a:a + 10].mean(1), wm[:, :, a:a + 10].mean(2)); c = P @ q
                    if c[cur] < 0.5:
                        dips += 1; run += 1
                        k = int(np.argmax(c))
                        where["current" if k == cur else "previous context" if (k == prev and c[k] >= 0.5) else "the other context" if (k == other and c[k] >= 0.5) else "none (all < 0.5)"] += 1
                    elif run:
                        dur.append(run * 10); run = 0
        print(f"{pre} N={n}: {dips} dip windows -> " + ", ".join(f"{k} {v / max(dips, 1):.0%}" for k, v in where.items()) + f" | median dip length {np.median(dur) if dur else 0:.0f} s")

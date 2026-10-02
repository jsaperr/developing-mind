"""Post-hoc diagnostic (2026-10-02): first-visit vs return fingerprint similarity per digit set, within-first-visit drift,
and between-set similarity (W=50 settled means). Explains the {0,1} vs {2,3} asymmetry in split_output.txt."""
import sys, numpy as np
R = r"C:\Users\urawi\OneDrive\Desktop\a wide variety of vscode projects\developing-mind"
sys.path[:0] = [R + r"\notebooks\brian2\split_mnist"]
import analyze_split_mnist as A
from src.integration.interface import unit
runs = A.load()
for W in (50,):
    rows = []
    for d in runs:
        s = A.stream(d, W); q, ph, st = s["q"], s["ph"], s["settled"]
        m = lambda p, sl=slice(None): unit(q[(ph == p) & st].mean(0))
        early = unit(q[(ph == 0) & st][:len(q[(ph == 0) & st]) // 3].mean(0)); late = unit(q[(ph == 0) & st][-len(q[(ph == 0) & st]) // 3:].mean(0))
        rows.append((m(0) @ m(3), m(1) @ m(4), early @ late, m(0) @ m(1), m(1) @ m(2), m(3) @ m(4)))
    r = np.array(rows)
    print(f"W={W}: {{0,1}} first vs return {r[:,0].mean():.3f} | {{2,3}} first vs return {r[:,1].mean():.3f} | {{0,1}} early vs late within first visit {r[:,2].mean():.3f}")
    print(f"between sets: phase0-phase1 {r[:,3].mean():.3f}, phase1-phase2 {r[:,4].mean():.3f}, phase3-phase4 {r[:,5].mean():.3f}")

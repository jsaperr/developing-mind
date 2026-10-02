"""Flicker and window map for the leaky fair-share cells (L-P5, L-P6 in modal_leak.py's follow-up). For each click cell:
the share of settled 10 s windows (>= 300 s into a phase, all 5 phases) whose rectified readout has cosine < 0.5 (and
< 0) to the current context's prototype, and the median. MNIST accuracy and class coverage come from
analyze_mnist_pilot.py mnist_leak_tau<x> v3.

Usage: analyze_leak_flicker.py   (conda env; writes leak_flicker_output.txt)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE), str(HERE.parent)]
import modal_common as MC
from src.integration import interface as I


def flicker(prefix, n):
    v = []
    for f in sorted(HERE.glob(f"{prefix}_n{n}_seed*.json.gz")):
        d, wm, r, ch = MC.load_compact(f)
        protos = np.array([I.unit(np.isin(np.arange(30), s).astype(float)) for s in d['context_sets']])
        sched = d['phase_corr_blocks']
        for p in range(5):
            v += [I.readout(r[:, a:a + 10].mean(1), wm[:, :, a:a + 10].mean(2)) @ protos[sched[p]]
                  for a in range(p * 1000 + 300, p * 1000 + 1000, 10)]
    return np.array(v)


def main():
    lines = []
    for prefix in ("fact_ctrl", "leak_tau10", "leak_tau30", "leak_tau50", "leak_tau100", "leak_tau1000"):
        for n in (7, 40):
            v = flicker(prefix, n)
            if not len(v):
                lines.append(f"{prefix} N={n}: none"); continue
            lines.append(f"{prefix:12s} N={n:2d} | settled median {np.median(v):.2f} | flicker (< 0.5) {np.mean(v < 0.5):.1%} | < 0 {np.mean(v < 0):.1%}")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "leak_flicker_output.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

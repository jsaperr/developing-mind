"""Scores GM-P1..P4 (predictions in modal_gain_map.py, before launch): click cells with analyze_factorial.score (readout,
memory) plus flicker (share of settled 10 s windows with rectified readout < 0.5 to the current context), alongside the
leak_tau100 reference (tp 0.05). MNIST: analyze_mnist_pilot.py mnist_gmap_<tag> v3 (run separately).

Usage: analyze_gain_map.py   (conda env; writes gain_map_output.txt)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_factorial as AF
import analyze_leak_flicker as LF

CELLS = [("leak_tau100", "tp 0.05 / tau 100 (gain 5, reference)"), ("gmap_tp0p2_tau25", "tp 0.2 / tau 25 (gain 5)"),
         ("gmap_tp0p0125_tau400", "tp 0.0125 / tau 400 (gain 5)"), ("gmap_tp0p01_tau100", "tp 0.01 / tau 100 (gain 1)"),
         ("gmap_tp0p0025_tau400", "tp 0.0025 / tau 400 (gain 1)"),
         ("gmap_tp0p5_tau10", "tp 0.5 / tau 10 (gain 5)"), ("gmap_tp1_tau5", "tp 1.0 / tau 5 (gain 5)"), ("gmap_tp0p4_tau25", "tp 0.4 / tau 25 (gain 10)")]


def main():
    lines = []
    for prefix, label in CELLS:
        row = AF.score(prefix, 40)
        if row is None:
            lines.append(f"{label}: none"); continue
        v = LF.flicker(prefix, 40)
        lines.append(f"{label:38s} | readout {row['readout']:.3f}, B named W50 {row['B_named_W50']}/{row['seeds']} | "
                     f"flicker {np.mean(v < 0.5):.1%} (< 0: {np.mean(v < 0):.1%}) | {'PASS' if row['passes'] else 'fail'}")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "gain_map_output.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

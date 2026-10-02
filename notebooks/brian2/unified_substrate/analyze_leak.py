"""Scores L-P1, L-P2 (predictions in modal_leak.py, written before launch) with analyze_factorial.py's metrics, plus the
habituation trace that found the mechanism: the readout cosine to the current context's prototype and the rate CV across
neurons at 10, 30, 60, 120, 300, 600 and 900 s into phases 2-5 (seed means). The factorial cells are printed alongside.
MNIST (L-P3, L-P4): analyze_mnist_pilot.py mnist_leak_tau<x> v3.

Usage: analyze_leak.py   (conda env; writes leak_output.txt and leak_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_factorial as AF

I = AF.AU.I
TS = (10, 30, 60, 120, 300, 600, 900)


def habituation(prefix, n):
    rows = []
    for f in sorted(HERE.glob(f"{prefix}_n{n}_seed*.json.gz")):
        d, wm, r, ch = AF.MC.load_compact(f)
        protos = np.array([I.unit(np.isin(np.arange(30), s).astype(float)) for s in d['context_sets']])
        sched = d['phase_corr_blocks']; out = []
        for t0 in TS:
            cs, cv = [], []
            for p in range(1, 5):
                a = p * 1000 + t0 - 10
                cs.append(I.readout(r[:, a:a + 10].mean(1), wm[:, :, a:a + 10].mean(2)) @ protos[sched[p]])
                rr = r[:, a:a + 10].mean(1); cv.append(rr.std() / max(rr.mean(), 1e-9))
            out.append((np.mean(cs), np.mean(cv)))
        rows.append(out)
    return np.mean(rows, 0) if rows else None


def main():
    out, lines = {}, []
    for prefix in ("fact_ctrl", "unify", "dose_tp0p0005", "leak_tau10", "leak_tau100", "leak_tau1000"):
        for n in (7, 40):
            row = AF.score(prefix, n)
            if row is None:
                lines.append(f"{prefix} N={n}: none"); continue
            h = habituation(prefix, n)
            row["habituation"] = dict(t_s=list(TS), readout=h[:, 0].tolist(), rate_cv=h[:, 1].tolist())
            out[f"{prefix}_n{n}"] = row
            lines.append(f"{prefix:13s} N={n:2d} | readout {row['readout']:.3f}, B named W50 {row['B_named_W50']}/{row['seeds']}, "
                         f"change fired {row['change_fired']:.0%}, incoming before two-back A {row['A_before3']:.0%} B {row['B_before4']:.0%} | "
                         f"{'PASS' if row['passes'] else 'fail'} | readout in phase " + " ".join(f"{t}s {c:+.2f}" for t, c in zip(TS, h[:, 0]))
                         + " | rate CV " + " ".join(f"{v:.2f}" for v in h[:, 1]))
    txt = "\n".join(lines)
    print(txt)
    (HERE / "leak_output.txt").write_text(txt + "\n", encoding="utf-8")
    json.dump(out, open(HERE / "leak_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

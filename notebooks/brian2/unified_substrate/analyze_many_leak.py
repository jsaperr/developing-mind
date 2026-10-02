"""Scores ML-P1..P3 (predictions in modal_many_leak.py, written before launch): step 3's analyze_many.py, unchanged, run on
the compact files in many_leak/tau<x>/ (modal_common.load_compact reproduces the full-file analyses exactly), plus the
flicker measure (share of settled 10 s windows whose rectified readout has cosine < 0.5 to the current context).
Written before results.

Usage: analyze_many_leak.py   (conda env; writes many_leak/tau<x>/many_summary.json and many_leak_output.txt)
"""
import contextlib
import glob
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE.parent), str(ROOT / "notebooks" / "brian2" / "many_contexts_data")]
import analyze_many as AM
import modal_common as MC
from src.integration import interface as I

AM.R.load = lambda pat: [x for x in (MC.load_compact(f) for f in sorted(glob.glob(str(pat)))) if x[0].get("status") == "completed"]


def flicker(d_dir, n):
    v = []
    for d, wm, r, ch in AM.R.load(d_dir / f"many6_n{n}_seed*.json.gz"):
        protos = np.array([I.unit(np.isin(np.arange(60), s).astype(float)) for s in d['context_sets']])
        for p, c in enumerate(d['phase_corr_blocks']):
            v += [I.readout(r[:, a:a + 10].mean(1), wm[:, :, a:a + 10].mean(2)) @ protos[c]
                  for a in range(p * 1000 + 300, p * 1000 + 1000, 10)]
    return np.array(v)


def main():
    out = []
    for tau in (50, 100):
        AM.HERE = HERE / "many_leak" / f"tau{tau}"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            AM.main()
        out.append(f"######## tau {tau} s ########" + buf.getvalue())
        for n in (7, 40):
            v = flicker(AM.HERE, n)
            if len(v):
                out.append(f"flicker N={n}: settled median {np.median(v):.3f}, < 0.5 in {np.mean(v < 0.5):.1%}, < 0 in {np.mean(v < 0):.1%}")
    txt = "\n".join(out)
    print(txt)
    (HERE / "many_leak_output.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Self-report from label-free signals (exploration toward the framework's metacognition layer; analysis only, nothing in
src; 2026-10-02). Can the system say what situation it is in, using only its own signals, on the 100,000 s life?

Signals per 10 s step (all already exist; no new hand-set number, cutoffs self-set from the system's own history):
  W  weight change flag (re-learning; interface.changing_per_second's rule, from the life run)
  Ac activity change flag (the familiar-switch rule on 10 s rates)
  M  memory's report (default DormantGatedMemory, W=10): a named entry, NOVEL, or TRANSITIONAL
  St response strength = || max(r - mean r, 0) || over the 40 neurons (how decisively some neurons out-fire the rest);
     "weak" = below the 5th percentile of the system's own strength over the preceding 10,000 s (a running, self-set cutoff)
Self-report (first rule that applies):
  RE-LEARNING     W on
  SWITCH (fam.)   Ac on and W off
  RESTING         St weak
  NOVEL           M == NOVEL
  KNOWN           otherwise (memory names something, or is transitional)
Truth (labels used only for scoring), per step: rest phase -> resting; one-off phase -> novel; home phase: first 300 s after a
change -> "just changed" (either RE-LEARNING or SWITCH counts), later -> known.
Written before computing. Expectations (stated now, not formal predictions): known and just-changed are read well; resting
is read from strength; one-offs read as RE-LEARNING early (new wiring) and then NOVEL or KNOWN-of-a-wrong-entry later.

v2 (post-hoc, after v1's results; run with argument v2): rest cutoff = Otsu threshold of the system's own strength
distribution over the preceding 10,000 s (the split that best separates its two modes; no assumed fraction); "just changed"
scored over the first 120 s; KNOWN split into KNOWN-NEW (names an entry born in the current phase) and KNOWN-OLD.

Usage: analyze_self_report.py [v2]   (conda env; reads the life run; writes self_report_output.txt)
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks" / "brian2" / "life_run"), str(ROOT / "notebooks" / "integration" / "set_worlds"),
                str(ROOT / "notebooks" / "integration" / "familiar_switch")]
import analyze_familiar_switch as FS
import analyze_life as AL
import readout_set_worlds as RS
from src.hopfield.episodic_dormant import NOVEL

V2 = len(sys.argv) > 1 and sys.argv[1] == "v2"
STATES = ["RE-LEARNING", "SWITCH", "RESTING", "NOVEL", "KNOWN-NEW", "KNOWN-OLD"] if V2 else ["RE-LEARNING", "SWITCH", "RESTING", "NOVEL", "KNOWN"]


def otsu(x):
    h, e = np.histogram(x, 64); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); mu = np.cumsum(h * c)
    tot = w[-1]; between = (mu[-1] * w - mu * tot) ** 2 / np.maximum(w * (tot - w), 1e-9)
    return c[int(np.argmax(between[:-1]))]
TRUTH = ["just changed", "resting", "novel", "known"]


def main():
    runs = AL.load()
    S = [AL.stream(d, 10) for d in runs]
    gs = RS.ground_gap_scale(S)
    conf = {t: Counter() for t in TRUTH}
    for d, s in zip(runs, S):
        mm = AL.run_memory(s, gs)
        names, ps = d["phase_names"], int(d["phase_s"])
        r = np.array(d["rates10"], float)
        fa = FS.rule(FS.activity_series(r))
        fw = np.asarray(d["change_flags"], bool)
        st = np.linalg.norm(np.clip(r - r.mean(0), 0, None), axis=0)
        # memory's raw report per step: re-run to capture NOVEL (run_memory maps NOVEL to None)
        from src.hopfield.episodic_dormant import DormantGatedMemory
        import torch
        m = DormantGatedMemory(dim=60, gap_scale=gs); novel = []; born = {}; newname = []
        for i, x in enumerate(torch.tensor(s["q"], dtype=torch.float32)):
            o = m.step(x, steady=bool(s["steady"][i]), changing=bool(s["changing"][i]))
            novel.append(o["report"] == NOVEL)
            if o["created"] is not None:
                born[o["created"]] = int(s["ph"][i])
            newname.append(born.get(o["report"], -1) == int(s["ph"][i]) if isinstance(o["report"], int) else False)
        for b in range(len(s["q"])):
            p = b * 10 // ps
            hist = st[max(0, b - 1000):b]
            weak = len(hist) >= 100 and st[b] < (otsu(hist) if V2 else np.percentile(hist, 5))
            if fw[b * 10:b * 10 + 10].any():
                rep = "RE-LEARNING"
            elif fa[b]:
                rep = "SWITCH"
            elif weak:
                rep = "RESTING"
            elif novel[b]:
                rep = "NOVEL"
            else:
                rep = ("KNOWN-NEW" if newname[b] else "KNOWN-OLD") if V2 else "KNOWN"
            n = names[p]; into = b * 10 - p * ps
            truth = "resting" if n == "REST" else "novel" if n.startswith("N") else ("just changed" if (p > 0 and into < (120 if V2 else 300)) else "known")
            conf[truth][rep] += 1
    lines = [f"Self-report on the life run ({len(runs)} seeds; rows = truth, columns = what the system reports, % of row)",
             f"{'':14s}" + "".join(f"{c:>13s}" for c in STATES) + "      n"]
    for t in TRUTH:
        tot = sum(conf[t].values())
        lines.append(f"{t:14s}" + "".join(f"{conf[t][c] / tot:13.0%}" for c in STATES) + f"  {tot:6d}")
    good = {"just changed": ("RE-LEARNING", "SWITCH"), "resting": ("RESTING",), "novel": ("NOVEL", "RE-LEARNING", "KNOWN-NEW"),
            "known": ("KNOWN", "KNOWN-OLD", "KNOWN-NEW")}
    lines.append("correct-ish share: " + ", ".join(f"{t} {sum(conf[t][c] for c in good[t]) / sum(conf[t].values()):.0%}" for t in TRUTH))
    txt = "\n".join(lines)
    print(txt)
    (HERE / ("self_report_v2_output.txt" if V2 else "self_report_output.txt")).write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Scores SM-P1..P4 and MC-3 (predictions in modal_split_mnist.py) on split_seed<S>.json.gz. Written before any 4b data.

Memory stream per clock W (10, 50 s): the compact rectified fingerprints; steady = interface.steady_flags(q, 0.9); changing =
interface.window_any(change flags, starts, W); gap_scale grounded by readout_set_worlds.ground_gap_scale; memory = the
default DormantGatedMemory (dim 196). Entries are content-tagged at birth by the nearest digit-set prototype. Prototype =
unit mean of that seed's settled fingerprints during the set's FIRST visit (phases 1-3) at the same clock. (Changed before
any 4b data: a mini-run test showed the mean-image prototypes are 0.78-0.84 alike and fingerprints match them only ~0.72,
so tags were noisy.) The mean-image readout is still reported. Labels are used only here, for scoring. Settled = >= 300 s
into a phase.
  SM-P1/P2: a return (phase 4 = {0,1}, phase 5 = {2,3}) is REMEMBERED when its set is named in more than half of its settled
            checks AND the entry that first names it is old (born before the phase or reawakened), as in analyze_many.py.
            Also reported: the wrong-set share of settled named checks (absorption proxy), and the NOVEL share.
  SM-P3:    share of neurons whose 50 s weights are nearest {0,1} just before its return (t = 3 * phase_s - 50).
  SM-P4:    weight change flags in [switch, switch + 300 s); the activity signal (familiar-switch rule on 10 s rates).
  MC-3:     phases 2, 3 (new sets): a NOVEL check before the first correct naming; phase 4 (return): no NOVEL before it.

Usage: analyze_split_mnist.py   (conda env; writes split_output.txt)
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks" / "integration" / "set_worlds"), str(ROOT / "notebooks" / "integration" / "familiar_switch"),
                str(ROOT / "notebooks" / "brian2" / "life_run")]
import analyze_familiar_switch as FS
import readout_set_worlds as RS
from src.hopfield.episodic_consolidating import TRANSITIONAL
from src.hopfield.episodic_dormant import NOVEL, DormantGatedMemory
from src.integration import interface as I

SETTLE = 300
CTX = [0, 1, 2, 0, 1]          # phase -> digit-set index ({0,1}, {2,3}, {4,5})


def load():
    return [d for d in (json.loads(gzip.decompress(f.read_bytes())) for f in sorted(HERE.glob("split_seed*.json.gz")))
            if d.get("status") == "completed"]


def stream(d, W):
    q = np.array(d["fingerprints"][f"W{W}"]["rect"], float)
    starts = np.arange(len(q)) * W
    ph = np.minimum((starts // d["phase_s"]).astype(int), len(CTX) - 1)
    settled = (starts - ph * d["phase_s"]) >= SETTLE
    fp_protos = np.array([I.unit(q[(ph == p) & settled].mean(0)) for p in range(3)])      # first visits of {0,1}, {2,3}, {4,5}
    return dict(q=q, starts=starts, ph=ph, true=np.array([CTX[p] for p in ph]), protos=fp_protos,
                img_protos=np.array(d["protos"], float), settled=settled, steady=I.steady_flags(q, 0.9),
                changing=I.window_any(np.asarray(d["change_flags"], bool), starts, W))


def memory(s, gs):
    m = DormantGatedMemory(dim=s["q"].shape[1], gap_scale=gs)
    tag, born, reawoken, labels, first_name, first_novel = {}, {}, set(), [], {}, {}
    for i, x in enumerate(torch.tensor(s["q"], dtype=torch.float32)):
        o = m.step(x, steady=bool(s["steady"][i]), changing=bool(s["changing"][i]))
        if o["created"] is not None:
            tag[o["created"]] = int(np.argmax(s["protos"] @ x.numpy())); born[o["created"]] = i
            if o["reawakened"] is not None:
                reawoken.add(o["created"])
        rep = o["report"]; p = int(s["ph"][i])
        lab = None if (rep is TRANSITIONAL or rep == NOVEL) else tag.get(rep)
        labels.append("NOVEL" if rep == NOVEL else lab)
        if rep == NOVEL and p not in first_novel:
            first_novel[p] = i
        if lab is not None and lab == s["true"][i] and p not in first_name:
            first_name[p] = (i, rep)
    return dict(labels=labels, born=born, reawoken=reawoken, first_name=first_name, first_novel=first_novel)


def main():
    runs = load()
    if not runs:
        print("no completed runs"); return
    k = len(runs); lines = [f"Split-MNIST: {k} seeds"]
    for W in (50, 10):
        S = [stream(d, W) for d in runs]
        gs = RS.ground_gap_scale(S)
        M = [memory(s, gs) for s in S]
        rem = {3: 0, 4: 0}; wrong, novel, mc3 = [], [], 0
        for s, mm in zip(S, M):
            for p in (3, 4):
                idx = np.where((s["ph"] == p) & s["settled"])[0]
                labs = [mm["labels"][i] for i in idx]
                half = np.mean([l == CTX[p] for l in labs]) > 0.5 if labs else False
                fn = mm["first_name"].get(p); start = int(np.where(s["ph"] == p)[0][0])
                old = fn is not None and (mm["born"].get(fn[1], -1) < start or fn[1] in mm["reawoken"])
                rem[p] += bool(half and old)
            named = [(l, s["true"][i]) for i, l in enumerate(mm["labels"]) if s["settled"][i] and isinstance(l, int)]
            wrong.append(np.mean([l != t for l, t in named]) if named else np.nan)
            novel.append(np.mean([mm["labels"][i] == "NOVEL" for i in range(len(s["q"])) if s["settled"][i]]))
            ok_new = all(p in mm["first_novel"] and (p not in mm["first_name"] or mm["first_novel"][p] < mm["first_name"][p][0]) for p in (1, 2))
            ok_ret = 3 in mm["first_name"] and not (3 in mm["first_novel"] and mm["first_novel"][3] < mm["first_name"][3][0])
            mc3 += ok_new and ok_ret
        lines.append(f"W={W} (gap_scale {gs:.3f}) | {{0,1}} return remembered {rem[3]}/{k}, {{2,3}} return {rem[4]}/{k} | "
                     f"wrong-set share of settled named checks {np.nanmean(wrong):.1%} | NOVEL share of settled checks {np.mean(novel):.1%} | MC-3 {mc3}/{k}")
    held, wfire, afire, rd, rd_ret = [], [], [], [], []
    for d in runs:
        A = np.array(d["assign_50s"]); ps = int(d["phase_s"])
        held.append(np.mean(A[(3 * ps - 50) // 50] == 0))
        f = np.asarray(d["change_flags"], bool); fa = FS.rule(FS.activity_series(np.array(d["rates10"], float)))
        wfire.append(sum(f[p * ps:p * ps + 300].any() for p in range(1, 5)))
        afire.append(sum(fa[p * ps // 10:p * ps // 10 + 30].any() for p in range(1, 5)))
    s10 = [stream(d, 10) for d in runs]
    for s in s10:
        rd += [s["q"][i] @ s["img_protos"][s["true"][i]] for i in range(len(s["q"])) if s["settled"][i]]
        ret = [s["q"][i] @ s["protos"][s["true"][i]] for i in range(len(s["q"])) if s["settled"][i] and s["ph"][i] >= 3]
        rd_ret.append(np.median(ret) if ret else np.nan)
    lines.append(f"SM-P3 neurons nearest {{0,1}} just before its return: {np.mean(held):.0%} (per seed {', '.join(f'{h:.0%}' for h in held)})")
    lines.append(f"SM-P4 weight change signal fired after >= 3 of 4 switches in {sum(w >= 3 for w in wfire)}/{k} seeds (mean {np.mean(wfire):.1f}/4); "
                 f"activity signal all 4 in {sum(a == 4 for a in afire)}/{k}")
    lines.append(f"readout (W=10, settled): cosine to the true set's mean image, median {np.median(rd):.3f}; returns vs the set's "
                 f"first-visit fingerprint, median {np.nanmean(rd_ret):.3f}")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "split_output.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Scores LL-P1..P6 (predictions in modal_life.py, written before launch) on the compact files life_seed<S>.json.gz.
Written while the runs were in flight, before any result existed.

Memory stream (per seed): the compact rectified fingerprints at W = 10 s (primary) and 50 s (reported alongside),
steady = interface.steady_flags(q, 0.9), changing = interface.window_any(change flags, window starts, W), gap_scale
grounded per clock by the original procedure (readout_set_worlds.ground_gap_scale). Memory = the default src
DormantGatedMemory. Entries are content-tagged: the nearest prototype over EVERY structured context of the life (home A-G
and the one-offs), re-tagged never (tag at birth, as in every earlier scorer). Settled = >= 300 s into a phase.

Operational definitions, fixed now:
  LL-P1 character of a home context = max w_char over character() entries (live + dormant) whose pattern's nearest
        prototype is that context; 1.0 (baseline) if there is none. Scored at the end of the life. Pass in a seed when the
        era means order E,F > C,D > A,B (mean of the pair). Strict order (min of the later pair > max of the earlier) is
        reported too.
  LL-P2 every phase in eras 2-3 (phase index >= 60) whose context is A or B. Remembered = named in more than half of its
        settled checks AND the entry that first names it is old (born before the phase or reawakened from dormant); the
        many-contexts definition.
  LL-P3 rest phases: share of ALL checks in rest phases where memory names a home context (A-G); per rest whose previous
        phase was a home context, does the most-named home context equal it (no home named counts as a miss).
  LL-P4 len(memory.dormant) right after the last step of phase 99 (midpoint) and at the end; pass when the seed-mean ratio
        end / midpoint <= 1.3. Live counts reported too.
  LL-P5 drift: cosine between the mean settled W=10 fingerprint of A's first visit and of A's last visit, seed mean.
        Residue: from the 50 s assignments, the share of neurons whose nearest prototype is a one-off context that is
        not the current phase's, in the last assignment of each era (t = 29950, 69950, 99950 s). Pass when every era is
        <= 10% and no era exceeds the one before by more than one neuron (1/40).
  LL-P6 changes INTO a structured context = every phase start after the first whose phase is not rest. Fired = any flag in
        [start, start + 300). Settled-on = the flagged share of seconds from start + 300 to the phase end, over
        structured phases only. Rest phases are reported separately.
Also reported: settled readout cosine to the true prototype (median), and the number of one-offs that got a memory entry.

Usage: analyze_life.py   (conda env; writes life_output.txt and life_summary.json)
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "notebooks" / "integration" / "set_worlds"))
import readout_set_worlds as RS
from src.hopfield.episodic_consolidating import TRANSITIONAL
from src.hopfield.episodic_dormant import NOVEL, DormantGatedMemory
from src.integration import interface as I

HOME = list("ABCDEFG")
SETTLE = 300
ERA_ENDS = [60, 140, 200]


def load():
    out = []
    for f in sorted(HERE.glob("life_seed*.json.gz")):
        d = json.loads(gzip.decompress(f.read_bytes()))
        if d.get("status") == "completed":
            out.append(d)
    return out


def stream(d, W):
    ps = float(d["phase_s"]); names = d["phase_names"]; pn = d["proto_names"]
    protos = np.array([I.unit(np.isin(np.arange(d["n_pre"]), s).astype(float)) for s in
                       [d["phase_sets"][names.index(n)] for n in pn]])
    q = np.array(d["fingerprints"][f"W{W}"]["rect"], float)
    starts = np.arange(len(q)) * W
    ph = (starts // ps).astype(int)
    true = np.array([pn.index(names[p]) if names[p] != "REST" else -1 for p in ph])
    return dict(q=q, starts=starts, ph=ph, true=true, protos=protos, pn=pn, names=names,
                settled=(starts - ph * ps) >= SETTLE, steady=I.steady_flags(q, 0.9),
                changing=I.window_any(np.asarray(d["change_flags"], bool), starts, W))


def run_memory(s, gs):
    m = DormantGatedMemory(dim=s["q"].shape[1], gap_scale=gs)
    tag, born, reawoken, labels, mid = {}, {}, set(), [], None
    first_name = {}
    last_mid = int(np.where(s["ph"] == 99)[0][-1])
    for i, x in enumerate(torch.tensor(s["q"], dtype=torch.float32)):
        o = m.step(x, steady=bool(s["steady"][i]), changing=bool(s["changing"][i]))
        if o["created"] is not None:
            tag[o["created"]] = int(np.argmax(s["protos"] @ x.numpy())); born[o["created"]] = i
            if o["reawakened"] is not None:
                reawoken.add(o["created"])
        rep = o["report"]
        lab = None if (rep is TRANSITIONAL or rep == NOVEL) else tag.get(rep)
        labels.append(lab)
        p = int(s["ph"][i])
        if lab is not None and lab == s["true"][i] and p not in first_name:
            first_name[p] = rep
        if i == last_mid:
            mid = dict(dormant=len(m.dormant), live=len(m.mem.ids))
    end = dict(dormant=len(m.dormant), live=len(m.mem.ids))
    char = {}
    for pat, w, st in m.character():
        c = s["pn"][int(np.argmax(s["protos"] @ np.asarray(pat, float)))]
        char[c] = max(char.get(c, 1.0), float(w))
    return dict(labels=labels, born=born, reawoken=reawoken, first_name=first_name, mid=mid, end=end, char=char,
                tagged=set(s["pn"][t] for t in tag.values()))


def score(runs, W, lines):
    S = [stream(d, W) for d in runs]
    gs = RS.ground_gap_scale(S)
    M = [run_memory(s, gs) for s in S]
    k = len(runs); names = runs[0]["phase_names"]; pn = S[0]["pn"]
    row = dict(gap_scale=gs)
    # LL-P1
    ch = [{h: mm["char"].get(h, 1.0) for h in HOME} for mm in M]
    era = lambda c, a, b: (c[a] + c[b]) / 2
    p1 = [era(c, "E", "F") > era(c, "C", "D") > era(c, "A", "B") for c in ch]
    p1s = [min(c["E"], c["F"]) > max(c["C"], c["D"]) and min(c["C"], c["D"]) > max(c["A"], c["B"]) for c in ch]
    row["P1"] = dict(pass_seeds=int(sum(p1)), strict_seeds=int(sum(p1s)), char_mean={h: float(np.mean([c[h] for c in ch])) for h in HOME})
    # LL-P2
    rets = [p for p in range(60, len(names)) if names[p] in ("A", "B")]
    rem = tot = 0
    for s, mm in zip(S, M):
        for p in rets:
            c = pn.index(names[p]); idx = np.where((s["ph"] == p) & s["settled"])[0]
            half = np.mean([mm["labels"][i] == c for i in idx]) > 0.5 if len(idx) else False
            w = mm["first_name"].get(p); start = int(np.where(s["ph"] == p)[0][0])
            old = w is not None and (mm["born"].get(w, -1) < start or w in mm["reawoken"])
            rem += bool(half and old); tot += 1
    row["P2"] = dict(remembered=rem, returns=tot, share=rem / max(tot, 1))
    # LL-P3
    home_idx = {pn.index(h) for h in HOME if h in pn}
    named = checks = hit = rests = 0
    for s, mm in zip(S, M):
        for p in [p for p, n in enumerate(names) if n == "REST"]:
            idx = np.where(s["ph"] == p)[0]
            labs = [mm["labels"][i] for i in idx]
            named += sum(l in home_idx for l in labs); checks += len(labs)
            if p > 0 and names[p - 1] in HOME:
                rests += 1
                hl = [l for l in labs if l in home_idx]
                hit += bool(hl) and max(set(hl), key=hl.count) == pn.index(names[p - 1])
    row["P3"] = dict(home_named_share=named / max(checks, 1), prior_most_named=hit / max(rests, 1), rests=rests)
    # LL-P4
    dm = np.mean([mm["mid"]["dormant"] for mm in M]); de = np.mean([mm["end"]["dormant"] for mm in M])
    row["P4"] = dict(dormant_mid=float(dm), dormant_end=float(de), ratio=float(de / max(dm, 1e-9)),
                     live_mid=float(np.mean([mm["mid"]["live"] for mm in M])), live_end=float(np.mean([mm["end"]["live"] for mm in M])),
                     per_seed=[(mm["mid"]["dormant"], mm["end"]["dormant"]) for mm in M])
    # readout + one-offs
    qt = [s["q"][i] @ s["protos"][s["true"][i]] for s in S for i in range(len(s["q"])) if s["settled"][i] and s["true"][i] >= 0]
    novel = [n for n in pn if n.startswith("N")]
    row["readout_settled_median"] = float(np.median(qt))
    row["oneoffs_with_entry"] = float(np.mean([sum(n in mm["tagged"] for n in novel) for mm in M])); row["oneoffs"] = len(novel)
    c1 = row["P1"]["char_mean"]
    lines += [f"--- W={W} s (gap_scale {gs:.4f}, {k} seeds) ---",
              f"LL-P1 era order E,F > C,D > A,B in {row['P1']['pass_seeds']}/{k} (strict {row['P1']['strict_seeds']}/{k}); "
              f"mean character " + " ".join(f"{h} {c1[h]:.2f}" for h in HOME),
              f"LL-P2 A/B returns in eras 2-3 remembered by an old memory: {rem}/{tot} = {row['P2']['share']:.0%}",
              f"LL-P3 rest: a home context named in {row['P3']['home_named_share']:.0%} of checks; most-named = the one before "
              f"in {row['P3']['prior_most_named']:.0%} of {rests} rests",
              f"LL-P4 dormant entries mid {dm:.1f} -> end {de:.1f} (x{row['P4']['ratio']:.2f}); live {row['P4']['live_mid']:.1f} -> "
              f"{row['P4']['live_end']:.1f}; per seed {row['P4']['per_seed']}",
              f"readout settled median {row['readout_settled_median']:.3f}; one-offs that got an entry {row['oneoffs_with_entry']:.1f}/{len(novel)}"]
    return row


def main():
    runs = load()
    if not runs:
        print("no completed runs"); return
    k = len(runs); names = runs[0]["phase_names"]; ps = float(runs[0]["phase_s"])
    lines = [f"Life in eras: {k} seeds, {len(names)} phases x {ps:.0f} s"]
    out = {}
    # LL-P5 drift
    S10 = [stream(d, 10) for d in runs]
    a_vis = [p for p, n in enumerate(names) if n == "A"]
    cos = []
    for s in S10:
        f = [I.unit(s["q"][(s["ph"] == p) & s["settled"]].mean(0)) for p in (a_vis[0], a_vis[-1])]
        cos.append(float(f[0] @ f[1]))
    # LL-P5 residue
    pn = runs[0]["proto_names"]
    res = []
    for d in runs:
        A = np.array(d["assign_50s"]); row = []
        for e in ERA_ENDS:
            t = e * ps - 50; j = int(t // 50); cur = names[int(t // ps)]
            past = {pn.index(n) for n in names[:int(t // ps)] if n.startswith("N") and n != cur}
            row.append(float(np.mean([a in past for a in A[j]])))
        res.append(row)
    res = np.array(res).mean(0)
    p5 = bool(np.mean(cos) >= 0.9 and (res <= 0.10).all() and all(res[i + 1] <= res[i] + 1 / runs[0]["n_post"] for i in range(2)))
    out["P5"] = dict(A_first_last_cos=cos, cos_mean=float(np.mean(cos)), residue_by_era=res.tolist(), pass_=p5)
    # LL-P6
    fired, on, on_rest = [], [], []
    for d in runs:
        f = np.asarray(d["change_flags"], bool)
        for p, n in enumerate(names):
            a, b = int(p * ps), int((p + 1) * ps)
            if n == "REST":
                on_rest.append(float(f[a:b].mean())); continue
            if p > 0:
                fired.append(bool(f[a:a + 300].any()))
            on.append(float(f[a + 300:b].mean()))
    out["P6"] = dict(fired=float(np.mean(fired)), settled_on=float(np.mean(on)), rest_on=float(np.mean(on_rest)))
    lines += [f"LL-P5 A first vs last visit cosine {np.mean(cos):.3f} (per seed {', '.join(f'{c:.2f}' for c in cos)}); "
              f"neurons on past one-offs at era ends {', '.join(f'{x:.1%}' for x in res)} -> {'PASS' if p5 else 'FAIL'}",
              f"LL-P6 change signal fired after {out['P6']['fired']:.0%} of changes into a structured context; on {out['P6']['settled_on']:.1%} "
              f"of settled seconds; on {out['P6']['rest_on']:.0%} of rest seconds"]
    for W in (10, 50):
        out[f"W{W}"] = score(runs, W, lines)
    txt = "\n".join(lines)
    print(txt)
    (HERE / "life_output.txt").write_text(txt + "\n", encoding="utf-8")
    json.dump(out, open(HERE / "life_summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

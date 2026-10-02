"""Familiar-switch detection: does ACTIVITY change mark the switches the weight-based change signal misses?

Motivation (life run, notebooks/brian2/life_run, 2026-10-02): the change signal (interface.changing_per_second, D(t) = sum
|w(t) - w(t-60)|) is a RE-LEARNING detector. It fires on 96-100% of changes into a context the population doesn't hold, but
on only 36% of one-back returns (27% when >= 30% of neurons already hold the incoming context): nothing re-wires, so nothing
fires. The system has no "a familiar switch just happened" event. Arc 07 found that rates alone say "now vs just before"
(system_contract.md, S1), so firing activity should change at EVERY switch, wired or not.

Data: the life run's compact files (8 seeds, 200 x 500 s phases, N=40), already on disk; analysis only, no new sim.

Activity-change signal (no new hand-set number; the weight signal's rule and constants, applied to rates):
  r(b) = per-neuron rate in 10 s block b (the data's resolution). Window mean m(b) = mean of r over the last 60 s (blocks
  b-5..b, L = 60 s as in the weight signal). Dr(b) = sum_j |m_j(b) - m_j(b - 6)|, the change between the last 60 s and the 60
  s before. Flag when Dr(b) > median + 3 * 1.4826 * MAD of Dr over [b - 90, b - 6] blocks (trail 900 s), True while
  history < 100 s (10 blocks), exactly changing_per_second's rule.
Secondary (readout-based): the same rule on J(b) = 1 - cos(q(b), q(b - 6)), q = the W=10 rectified fingerprint.

Changes scored: every home -> home phase change (both phases home contexts A-G, no rest or one-off on either side), as in the
life run's breakdown. Depth = the incoming context's position in the recency list of structured contexts (1 = one back).
Fired = any flag in [change, change + 300 s). Settled = >= 300 s into a phase.

PREDICTIONS ON RECORD (written before the activity signal was computed on any data):
  FS-P1 the activity signal fires within 300 s after >= 90% of home -> home changes, AND >= 90% of one-back returns (where
        the weight signal fired 36%).
  FS-P2 it stays quiet when settled: on for < 5% of settled 10 s blocks in structured phases.
  FS-P3 the two signals together separate two kinds of switch. "Familiar switch" = activity fired, weights didn't; "re-learning"
        = both fired. Familiar share >= 50% of one-back returns and <= 20% of returns at depth >= 3.
  FS-P4 memory agrees with the split: the default memory (W=10, as in analyze_life.py) names the incoming context within the
        first 60 s (6 steps) of the phase in >= 80% of familiar switches, and in a lower share of re-learning switches.
  FS-P5 activity is the faster signal: when both fire, the activity flag's first block comes no later than the weight flag's
        first second in >= 75% of changes (compared at 10 s resolution).
Reported, not predicted: the readout-jump version (J), rest onsets and one-off changes.

Usage: analyze_familiar_switch.py   (conda env; reads ../../brian2/life_run/life_seed*.json.gz; writes familiar_switch_output.txt)
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIFE = ROOT / "notebooks" / "brian2" / "life_run"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(LIFE)); sys.path.insert(0, str(ROOT / "notebooks" / "integration" / "set_worlds"))
import analyze_life as AL
import readout_set_worlds as RS

HOME = set("ABCDEFG")
L_B, TRAIL_B, MINH_B, K = 6, 90, 10, 3.0          # 60 s, 900 s, 100 s in 10 s blocks; k as in the weight signal


def rule(D):
    """changing_per_second's flag rule on a precomputed series (NaN where undefined)."""
    flag = np.ones(len(D), bool)
    for t in range(len(D)):
        hist = D[max(0, t - TRAIL_B):max(0, t - L_B)]
        hist = hist[np.isfinite(hist)]
        if len(hist) < MINH_B or not np.isfinite(D[t]):
            continue
        med = np.median(hist)
        flag[t] = D[t] > med + K * 1.4826 * np.median(np.abs(hist - med))
    return flag


def activity_series(r):
    """r: (n_post, n_blocks) rates. -> Dr per block."""
    c = np.cumsum(np.c_[np.zeros(r.shape[0]), r], axis=1)
    m = np.full(r.shape, np.nan)
    m[:, 5:] = (c[:, 6:] - c[:, :-6]) / 6.0                 # mean over blocks b-5..b
    D = np.full(r.shape[1], np.nan)
    D[11:] = np.abs(m[:, 11:] - m[:, 5:-6]).sum(0)
    return D


def jump_series(q):
    J = np.full(len(q), np.nan)
    J[L_B:] = 1 - np.sum(q[L_B:] * q[:-L_B], axis=1)
    return J


def main():
    runs = AL.load()
    k = len(runs); names = runs[0]["phase_names"]; ps = int(runs[0]["phase_s"]); bp = ps // 10
    S = [AL.stream(d, 10) for d in runs]
    gs = RS.ground_gap_scale(S)
    M = [AL.run_memory(s, gs) for s in S]
    rows = []
    act_settled, jmp_settled = [], []
    other = {"rest onset": [], "into one-off": [], "one-off -> home": [], "rest -> home": []}
    for d, s, mm in zip(runs, S, M):
        r = np.array(d["rates10"], float)
        fa = rule(activity_series(r)); fj = rule(jump_series(s["q"]))
        fw = np.asarray(d["change_flags"], bool)
        A = np.array(d["assign_50s"]); pn = d["proto_names"]
        rec = []
        for p, n in enumerate(names):
            a = p * bp
            if n != "REST":
                act_settled += list(fa[a + 30:a + bp]); jmp_settled += list(fj[a + 30:a + bp])
            if p > 0:
                prev = names[p - 1]
                if n in HOME and prev in HOME:
                    dep = rec.index(n) if n in rec else -1
                    wfire = fw[p * ps:p * ps + 300]; afire = fa[a:a + 30]
                    w1 = int(np.argmax(wfire)) // 10 if wfire.any() else None
                    a1 = int(np.argmax(afire)) if afire.any() else None
                    c = pn.index(n)
                    early = any(mm["labels"][i] == c for i in range(a, a + 6))
                    rows.append(dict(dep=dep, act=bool(afire.any()), jmp=bool(fj[a:a + 30].any()), w=bool(wfire.any()),
                                     a1=a1, w1=w1, hold=float(np.mean(A[p * bp // 5 - 1] == c)), early=early))
                elif n == "REST":
                    other["rest onset"].append(bool(fa[a:a + 30].any()))
                elif n.startswith("N"):
                    other["into one-off"].append(bool(fa[a:a + 30].any()))
                elif prev.startswith("N"):
                    other["one-off -> home"].append(bool(fa[a:a + 30].any()))
                elif prev == "REST":
                    other["rest -> home"].append(bool(fa[a:a + 30].any()))
            if n != "REST":
                if n in rec:
                    rec.remove(n)
                rec.insert(0, n)
    R = lambda f, sel=lambda x: True: float(np.mean([f(x) for x in rows if sel(x)])) if any(sel(x) for x in rows) else float("nan")
    one = lambda x: x["dep"] == 1
    deep = lambda x: x["dep"] >= 3
    fam = lambda x: x["act"] and not x["w"]
    rel = lambda x: x["act"] and x["w"]
    both = [x for x in rows if x["act"] and x["w"]]
    p5 = float(np.mean([x["a1"] <= x["w1"] for x in both])) if both else float("nan")
    out = dict(seeds=k, gap_scale=gs, n_changes=len(rows),
               P1=dict(all=R(lambda x: x["act"]), one_back=R(lambda x: x["act"], one), weights_all=R(lambda x: x["w"]),
                       weights_one_back=R(lambda x: x["w"], one)),
               P2=dict(settled_on=float(np.mean(act_settled))),
               P3=dict(familiar_one_back=R(fam, one), familiar_deep=R(fam, deep), familiar_all=R(fam)),
               P4=dict(early_named_familiar=R(lambda x: x["early"], fam), early_named_relearning=R(lambda x: x["early"], rel)),
               P5=dict(activity_first=p5, n_both=len(both)),
               jump=dict(all=R(lambda x: x["jmp"]), one_back=R(lambda x: x["jmp"], one), settled_on=float(np.mean(jmp_settled))),
               other={kk: (float(np.mean(v)), len(v)) for kk, v in other.items()},
               by_depth={str(dd): dict(n=sum(x["dep"] == dd for x in rows), act=R(lambda x: x["act"], lambda x, dd=dd: x["dep"] == dd),
                                        w=R(lambda x: x["w"], lambda x, dd=dd: x["dep"] == dd),
                                        familiar=R(fam, lambda x, dd=dd: x["dep"] == dd))
                         for dd in sorted({x["dep"] for x in rows})})
    o = out
    lines = [f"Familiar-switch detection on the life run: {k} seeds, {len(rows)} home -> home changes",
             f"FS-P1 activity signal fired after {o['P1']['all']:.0%} of changes, {o['P1']['one_back']:.0%} of one-back returns "
             f"(weight signal: {o['P1']['weights_all']:.0%}, {o['P1']['weights_one_back']:.0%})",
             f"FS-P2 activity signal on for {o['P2']['settled_on']:.1%} of settled blocks",
             f"FS-P3 familiar switch (activity yes, weights no): one-back {o['P3']['familiar_one_back']:.0%}, depth >= 3 "
             f"{o['P3']['familiar_deep']:.0%}, all {o['P3']['familiar_all']:.0%}",
             f"FS-P4 memory names the incoming context within 60 s: familiar {o['P4']['early_named_familiar']:.0%}, "
             f"re-learning {o['P4']['early_named_relearning']:.0%}",
             f"FS-P5 activity flag no later than the weight flag in {o['P5']['activity_first']:.0%} of {o['P5']['n_both']} changes where both fired",
             f"readout-jump version: fired {o['jump']['all']:.0%}, one-back {o['jump']['one_back']:.0%}, settled on {o['jump']['settled_on']:.1%}",
             "activity signal on other changes: " + ", ".join(f"{kk} {v[0]:.0%} of {v[1]}" for kk, v in o['other'].items()),
             "by depth: " + " | ".join(f"{dd}: n {v['n']} act {v['act']:.0%} w {v['w']:.0%} fam {v['familiar']:.0%}" for dd, v in o['by_depth'].items())]
    txt = "\n".join(lines)
    print(txt)
    (HERE / "familiar_switch_output.txt").write_text(txt + "\n", encoding="utf-8")
    json.dump(out, open(HERE / "familiar_switch_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

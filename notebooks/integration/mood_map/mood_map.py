"""The mood map: label-free internal states from signals the system already has, and the question under it:
can the system tell a real recognition from a ghost (rest) or a coin-flip (ambiguity) by itself?

Replay only (existing recordings): v1b (13mV/1.5), the 20-phase long world, checkb, rest, ambiguity. 8 seeds
each, rectified readout, default src DormantGatedMemory, W=10 (and W=50).

Signals per memory check (no labels):
  changing   the substrate's re-learning flag (interface.changing_per_second, the commit rule's signal)
  match      the winning memory's cosine to the fingerprint (step()['similarity']) >= 0.8, the radius
  strength   the fingerprint's size BEFORE normalizing: || centered( sum_j max(r_j - mean r, 0) w_j ) ||.
             Found in the ambiguity run: it halves when the call is close, and normalization throws it away.
  cutoff     SELF-SET and causal (the tripwire rule; no hand-picked value): the 5th percentile of strength over
             the run's own confident recognitions so far. A confident recognition is a check that matches, isn't
             changing, is steady, and (after a 30-check burn-in) is itself at or above the current cutoff.
             Only confident moments calibrate confidence. Nothing is flagged during burn-in.

States (in this order):
  returning  changing, and a memory matches
  lost       changing, and nothing matches
  strange    not changing, nothing matches      (novel but stable: where memory stores new things)
  uncertain  not changing, matches, strength below the cutoff   (weak recognition: ghost or coin-flip?)
  home       not changing, matches, strength at or above the cutoff

Scoring uses labels only here. A recognition is any check whose winner matches (>= 0.8) and isn't changing.
  real       in a settled phase of a real context, winner born in (tagged as) the true context
  ghost      during the REST phase of the rest world
  coin-flip  during the settled AB phase of the ambiguity world
  error      in a settled real phase, winner tagged as the wrong context

PREDICTIONS ON RECORD (written before this script was first run), W=10:
  MM-P1 the cutoff flags >= 50% of rest ghost recognitions as uncertain.
  MM-P2 it flags >= 50% of ambiguity coin-flip recognitions as uncertain.
  MM-P3 it flags <= 10% of real recognitions (pooled over v1b, long world, checkb and the real phases of rest
        and ambiguity).
  MM-P4 occupancy: settled v1b phases are >= 85% "home"; the rest phase and settled AB are each <= 50% "home".

Usage: mood_map.py   (conda env; writes mood_map_summary.json and mood_map_timelines.png)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "notebooks" / "integration" / "set_worlds"))
import readout_set_worlds as RS
import run_rectified_memory as R
from src.hopfield.episodic_consolidating import TRANSITIONAL
from src.hopfield.episodic_dormant import DormantGatedMemory
from src.integration import interface as I

B2 = R.B2
WORLDS = {
    "v1b": B2 / "v1b_schedule_data" / "v1b_n7_reliable_13_1p5_seed*.json.gz",
    "long world": B2 / "long_world_data" / "long20_n7_seed*.json.gz",
    "checkb": B2 / "set_worlds_data" / "checkb_n7_seed*.json.gz",
    "rest": B2 / "rest_data" / "rest_n7_seed*.json.gz",
    "ambiguity": B2 / "ambiguity_data" / "ambig_n7_seed*.json.gz",
}
STATES = ["home", "uncertain", "strange", "returning", "lost"]
BURN = 30


def strength(r, wm, a, W):
    rates = r[:, a:a + W].mean(1); w = wm[:, :, a:a + W].mean(2)
    v = np.clip(rates - rates.mean(), 0, None) @ w
    return float(np.linalg.norm(v - v.mean()))


def special_phase(world, d, p):
    """label for scoring: 'ghost' (rest phase), 'coin' (AB phase), 'real', or None (filler / unknown)."""
    cid = d['phase_corr_blocks'][p]
    sets = d['context_sets']
    if world == "rest" and len(sets[cid]) == 0:
        return "ghost"
    if world == "ambiguity" and len(sets[cid]) == 20:
        return "coin"
    if world == "checkb" and cid >= 3:
        return None                                  # never-returning fillers: not scored
    return "real"


def run_world(world, pat, W):
    runs = R.load(pat)
    rng = np.random.default_rng(0); torch.manual_seed(0)
    streams = [R.build(d, wm, r, ch, W, "H+", rng) for d, wm, r, ch in runs]
    gs = RS.ground_gap_scale(streams)
    rows = []
    for (d, wm, r, ch), s in zip(runs, streams):
        m = DormantGatedMemory(dim=30, gap_scale=gs)
        tag, pool, states = {}, [], []
        for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
            o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
            if o['created'] is not None:
                tag[o['created']] = int(np.argmax(s['protos'] @ x.numpy()))
            st_ = strength(r, wm, s['starts'][i], W)
            match = o['similarity'] >= 0.8
            cutoff = np.percentile(pool, 5) if len(pool) >= BURN else -np.inf
            if s['changing'][i]:
                state = "returning" if match else "lost"
            elif not match:
                state = "strange"
            else:
                state = "home" if st_ >= cutoff else "uncertain"
                if s['steady'][i] and (len(pool) < BURN or st_ >= cutoff):
                    pool.append(st_)
            p = int(s['ph'][i])
            kind = special_phase(world, d, p)
            recog = match and not s['changing'][i]
            label = None
            if recog and kind == "ghost":
                label = "ghost"
            elif recog and kind == "coin" and s['settled'][i]:
                label = "coin"
            elif recog and kind == "real" and s['settled'][i]:
                label = "real" if tag.get(o['winner']) == int(s['true'][i]) else "error"
            states.append(state)
            rows.append(dict(state=state, label=label, kind=kind, settled=bool(s['settled'][i]),
                             flagged=state == "uncertain", seed=d['seed'], t=s['starts'][i]))
    return rows, streams, runs[0][0]


def main():
    out, timelines = {}, {}
    for W in (10, 50):
        allrows = {}
        for world, pat in WORLDS.items():
            rows, streams, d0 = run_world(world, pat, W)
            allrows[world] = rows
            if W == 10:
                seed0 = rows[0]['seed']
                timelines[world] = ([r_['t'] for r_ in rows if r_['seed'] == seed0], [r_['state'] for r_ in rows if r_['seed'] == seed0],
                                    streams[0], d0)
        pooled = [r_ for w in allrows for r_ in allrows[w]]

        def rate(lab):
            x = [r_['flagged'] for r_ in pooled if r_['label'] == lab]
            return (float(np.mean(x)) if x else float('nan')), len(x)

        res = {lab: rate(lab) for lab in ("ghost", "coin", "real", "error")}
        occ = {}
        for world, rows in allrows.items():
            for kind in ("real", "ghost", "coin"):
                sel = [r_['state'] for r_ in rows if r_['kind'] == kind and (r_['settled'] or kind == "ghost")]
                if sel:
                    occ[f"{world} / {kind}"] = {s_: sel.count(s_) / len(sel) for s_ in STATES}
        print(f"\n===== W={W} =====")
        print("share of recognitions flagged 'uncertain' by the self-set cutoff:")
        for lab, (v, n) in res.items():
            print(f"  {lab:6s} {v:6.1%}  (n={n})")
        print("state occupancy (settled phases; the whole rest phase):")
        for k, v in occ.items():
            print(f"  {k:24s} " + "  ".join(f"{s_} {v[s_]:4.0%}" for s_ in STATES))
        out[f"W{W}"] = dict(flagged={k: v[0] for k, v in res.items()}, n={k: v[1] for k, v in res.items()}, occupancy=occ)
    print("\nMM-P1 ghost >= 50%?  MM-P2 coin >= 50%?  MM-P3 real <= 10%?  MM-P4 v1b/real home >= 85%, rest/ghost & ambiguity/coin home <= 50%? (W=10)")
    json.dump(out, open(HERE / "mood_map_summary.json", "w"), indent=1)
    plot(timelines)


def plot(timelines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    col = {"home": "#2E7D5B", "uncertain": "#E0A526", "strange": "#7B5EA7", "returning": "#4A90C2", "lost": "#C8553D"}
    fig, axes = plt.subplots(len(timelines), 1, figsize=(12, 1.35 * len(timelines) + 0.8), squeeze=False)
    for ax, (world, (ts, sts, s, d)) in zip(axes[:, 0], timelines.items()):
        for t_, st_ in zip(ts, sts):
            ax.axvspan(t_, t_ + 10, ymin=0.0, ymax=0.62, color=col[st_], lw=0)
        swaps = [0] + list(d['swap_times_s']) + [d['total_s']]
        for k in range(len(swaps) - 1):
            cid = d['phase_corr_blocks'][k]; n = len(d['context_sets'][cid])
            name = "rest" if n == 0 else "AB" if n == 20 else ("ABC"[cid] if cid < 3 else f"F{cid - 2}")
            ax.text((swaps[k] + swaps[k + 1]) / 2, 0.82, name, ha="center", va="center", fontsize=8, transform=ax.get_xaxis_transform())
            ax.axvline(swaps[k], ymin=0.66, ymax=1.0, color="#888", lw=0.6)
        ax.set_xlim(0, d['total_s']); ax.set_yticks([]); ax.set_ylabel(world, rotation=0, ha="right", va="center", fontsize=9)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[-1, 0].set_xlabel("simulated time (s)")
    fig.legend(handles=[Patch(color=col[s_], label=s_) for s_ in STATES], loc="upper center", ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(HERE / "mood_map_timelines.png", dpi=150)
    print("figure:", HERE / "mood_map_timelines.png")


if __name__ == '__main__':
    main()

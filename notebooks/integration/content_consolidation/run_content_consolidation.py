"""Content consolidation (forked memory) on the coupling-toy worlds.

Uses episodic_content_fork.ConsolidatingEpisodicMemory, a subclass fork. src/hopfield is not
touched. Streams, arms, grounding (v0's gap_scale per world/clock), the THETA=0.5 creation rule
and content-based scoring are all identical to coupling_v0/run_coupling_v0.py.

SANITY FIRST: with eta=0 the fork must reproduce v0 exactly (same winner ids, same reported
contexts, same created counts) on every seed of every world at both clocks. The script aborts if
it doesn't.

Variants (substrate arm unless noted):
  eta=0                 v0 baseline, through the fork
  gated eta=0.1         PRIMARY. The winner's content moves toward the query at rate 0.1*(1-g).
                        About a 10-step time constant when unambiguous.
  gated eta=0.03, 0.3   sensitivity
  ungated eta=0.1       rate 0.1 regardless of ambiguity (the principle test)
  clean arm, gated 0.1  for CC-P4
Entries are re-tagged by content after every consolidation (their nearest context prototype).

Metrics:
  genuine recognition  at a return, the entry born in the returning context's ORIGINAL phase wins
                       more than half of the return phase (counted per seed)
  two-back             steps reporting neither the current nor the previous context, after novel swaps
  absorption           an ESTABLISHED entry (won at least 10 settled steps while tagged with the
                       true context X) is later tagged with some other context Y. An established
                       memory pulled into a different context.

PREDICTIONS ON RECORD (written before this script was first run). v0 baselines are in
experiments_integration.md.
  CC-P1: gated eta=0.1 raises genuine recognition on the substrate arm:
         A->B->A return (W=10 s) 1/8 -> >= 5/8; v1 C return (W=50 s) 2/8 -> >= 5/8;
         v1 A two-back return (W=50 s) 2/8 -> >= 4/8 (weaker: first context).
  CC-P2: gated eta=0.1 cuts two-back reports after novel swaps (W=10 s):
         A->B->C 126 -> <= 40, v1 115 -> <= 40. The captured "not-old" entry should drift to the
         new context once queries settle.
  CC-P3: ungated eta=0.1 produces MORE absorption events than gated eta=0.1 (summed over all worlds
         and clocks), and lower mean accuracy.
  CC-P4: the clock problem is untouched. Clean arm, v1, W=10 s: A two-back genuine recognition
         stays 0/8, because the entry is evicted before A returns and content can't fix that.

Usage: run_content_consolidation.py   (conda env; writes content_consolidation_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "coupling_v0"))
import run_coupling_v0 as T
from episodic_content_fork import ConsolidatingEpisodicMemory

RETURNS = {"ABA": [(2, 0)], "ABCAC": [(3, 0), (4, 2)]}   # (return phase index, original phase index)
VARIANTS = [("eta=0", 0.0, True), ("gated 0.1", 0.1, True), ("gated 0.03", 0.03, True),
            ("gated 0.3", 0.3, True), ("ungated 0.1", 0.1, False)]
ESTABLISHED = 10


def run_cc(stream, arm, gap_scale, eta, gate, allow=None, match_floor=None, no_refresh=None):
    """allow: optional creation gate allow(step) -> bool (None = always allow; empty memory always may).
    match_floor: optional absolute-match condition for consolidation (None = off).
    no_refresh: optional no_refresh(step) -> bool. On such steps a win does NOT reset the winner's
    staleness (it ages like everyone else). This is a causal-test lesion, not a design proposal."""
    Q = torch.tensor(stream['q'][arm], dtype=torch.float32)
    protos = stream['protos']                                   # numpy float64, exactly as v0 tags
    mem = ConsolidatingEpisodicMemory(dim=Q.shape[1], eta=eta, gate_content=gate, gap_scale=gap_scale,
                                      match_floor=match_floor)
    tag, born_phase, reported, winner_ids, alive_ids = {}, {}, [], [], []
    wins = {}; established = {}; absorbed = set()
    created = 0
    for s in range(len(Q)):
        q = Q[s]
        novel = not mem.patterns or float((torch.stack(mem.patterns) @ q).max()) < T.THETA
        if novel and (not mem.patterns or allow is None or allow(s)):
            mem.add_pattern(q.clone(), s)
            tag[mem.ids[-1]] = int(np.argmax(protos @ q.numpy()))
            born_phase[mem.ids[-1]] = int(stream['ph'][s])
            created += 1
        if len(mem.patterns) == 1:                       # same handling as v0's step_memory
            w_fast, w_char = mem._update_fn(torch.tensor(mem.w_fast), torch.tensor(mem.w_char), torch.tensor([1.0]))
            mem.w_fast, mem.w_char = w_fast.tolist(), w_char.tolist()
            mem.staleness[0] = 0
            wi, g = 0, 0.0
        else:
            prev_stale = list(mem.staleness)
            wi, _, g = mem.retrieve_and_update(q)
            if no_refresh is not None and no_refresh(s):
                mem.staleness[wi] = prev_stale[wi] + 1      # lesion: this win doesn't refresh
        wid = mem.ids[wi]
        reported.append(tag[wid]); winner_ids.append(wid)   # report BEFORE this step's consolidation
        if mem.consolidate(wi, q, g) > 0:
            tag[wid] = int(np.argmax(protos @ mem.patterns[wi].numpy()))
        if stream['settled'][s] and reported[-1] == stream['true'][s]:
            c = wins.setdefault(wid, {}); c[reported[-1]] = c.get(reported[-1], 0) + 1
            if wid not in established and c[reported[-1]] >= ESTABLISHED:
                established[wid] = reported[-1]
        if wid in established and tag[wid] != established[wid]:
            absorbed.add(wid)
        mem.prune_step(s)
        alive_ids.append(tuple(mem.ids))
    return dict(reported=np.array(reported), winner_ids=winner_ids, born_phase=born_phase,
                created=created, absorbed=len(absorbed), evicted=len(mem.eviction_log), alive_ids=alive_ids)


def genuine(stream, res, ret_phase, orig_phase):
    m = stream['ph'] == ret_phase
    w = np.array(res['winner_ids'])[m]
    return float(np.mean([res['born_phase'][x] == orig_phase for x in w])) if m.any() else np.nan


def main():
    v0 = json.load(open(HERE.parent / "coupling_v0" / "coupling_v0_summary.json"))
    out = {}
    for world, ddir, pat in T.WORLDS:
        runs = T.load_world(ddir, pat)
        for W in T.CLOCKS:
            key = f"{world}_W{W}"
            rng = np.random.default_rng(0); torch.manual_seed(0)      # identical streams to v0
            streams = [T.build_streams(d, wm, r, n, bs, W, rng) for d, wm, r, n, bs in runs]
            gs = v0[key]['gap_scale']
            # sanity: eta=0 fork == v0
            for s in streams:
                a = T.run_memory(s, 'substrate', gs, T.THETA); b = run_cc(s, 'substrate', gs, 0.0, True)
                assert a['winner_ids'] == b['winner_ids'] and np.array_equal(a['reported'], b['reported']) \
                    and a['created'] == b['created'], f"fork diverges from v0 at eta=0 ({key})"
            out[key] = {}
            print(f"\n===== {key} (gap_scale {gs:.4f}; fork == v0 at eta=0: OK) =====")
            arms = [("substrate", n, e, gt) for n, e, gt in VARIANTS] + [("clean", "gated 0.1", 0.1, True)]
            for arm, name, eta, gate in arms:
                res = [run_cc(s, arm, gs, eta, gate) for s in streams]
                sc = [T.score(s, x) for s, x in zip(streams, res)]
                two_back = int(sum(sw['other'] for x in sc for sw in x['swaps'] if not sw['returning']))
                gen = {f"ret{rp}": [genuine(s, x, rp, op) for s, x in zip(streams, res)]
                       for rp, op in RETURNS.get(world, [])}
                row = dict(acc=float(np.mean([x['acc'] for x in sc])), two_back=two_back,
                           absorbed=int(sum(x['absorbed'] for x in res)),
                           created=float(np.mean([x['created'] for x in res])),
                           genuine={k: int(sum(v > 0.5 for v in vals)) for k, vals in gen.items()})
                out[key][f"{arm} {name}"] = row
                g_txt = "  ".join(f"genuine {k}: {v}/8" for k, v in row['genuine'].items())
                print(f"  {arm:9s} {name:12s} acc {row['acc']:.3f}  two-back {two_back:4d}  "
                      f"absorbed {row['absorbed']:3d}  created {row['created']:5.1f}  {g_txt}")
    json.dump(out, open(HERE / "content_consolidation_summary.json", "w"), indent=1)


if __name__ == '__main__':
    main()

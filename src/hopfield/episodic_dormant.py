"""Episodic memory whose forgetting hits episodes, not character: dormant entries plus anchored
consolidation, at a single similarity radius. Promoted from the notebook toys in
notebooks/integration/memory_limits/ (character_store_real, dormant_long_toy, long_world_readout,
anchored_consolidation). It's a SEPARATE module: episodic.py and episodic_consolidating.py are
unchanged. Built 2026-09-28 on Jasper's go (dormant entries + anchoring); see
experiments_integration.md and framework_drift.md item 1 (framework v5, III.3: "make forgetting hit
the episodic layer and not the gradient layer").

What it adds to GatedEpisodicMemory (the adopted commit rule is inherited unchanged: create only
when novel AND steady AND not re-learning; consolidate only when not re-learning; rehearse always;
TRANSITIONAL while re-learning):

  1. One similarity radius (default 0.8) for every "is this the same thing?" question: the
     novelty threshold, the consolidation floor, the anchor, and relinking.
  2. Anchored consolidation. Each entry keeps its birth pattern. Content moves toward a query only if
     the query is within the radius of the birth pattern, and never moves outside that radius.
     Content can sharpen but can't change identity (stops the slide seen at 70% overlap on the
     fast clock).
  3. Dormant entries. Eviction demotes an entry instead of deleting it: it keeps its pattern and
     w_char, stops taking part in retrieval, and its w_char decays toward 1 (decay_char). It's pruned
     once w_char - 1 < prune_eps, when it no longer carries any character. When a new entry is
     created whose content matches a dormant one (cosine >= radius), the dormant entry reawakens:
     the new entry starts with its w_char instead of 1 (character outlives the episode, and
     relearning a forgotten context starts from its old strength).
  4. NOVEL report (novel_report=True, the default): when the winner matches the query below the
     radius, report NOVEL instead of naming it (memory knows the query is new but can't create
     yet). Pre-registered test notebooks/integration/memory_limits/novel_report.py: it passed all
     four predictions. It catches 81-90% of wrong reports where they cluster, loses <= 0.19% of
     correct ones, and fires on <= 0.36% of settled steps where nothing is new. Pass
     novel_report=False for the plain report.

Step order matches the toys exactly (tested): snapshot live entries -> the inherited step (create,
retrieve/update, consolidate, prune) -> relink a newly created entry to a matching dormant one ->
demote this step's evictions (at their start-of-step pattern and w_char) -> decay and prune dormant.
"""
import torch

from .episodic_consolidating import (ADOPTED, TRANSITIONAL, ConsolidatingEpisodicMemory,
                                     GatedEpisodicMemory)
from .two_layer import CONSTANTS

NOVEL = "NOVEL"
RADIUS = 0.8
PRUNE_EPS = 0.05


class AnchoredConsolidatingMemory(ConsolidatingEpisodicMemory):
    """ConsolidatingEpisodicMemory whose content can't leave the radius around its birth pattern."""

    def __init__(self, dim, anchor=RADIUS, **kwargs):
        super().__init__(dim, **kwargs)
        self.anchor = anchor
        self.birth = {}                      # entry id -> birth pattern

    def add_pattern(self, vec, step):
        idx = super().add_pattern(vec, step)
        self.birth[self.ids[idx]] = vec.clone()
        return idx

    def consolidate(self, winner, query, g):
        p0 = self.birth[self.ids[winner]]
        if float(p0 @ query) < self.anchor:
            return 0.0
        old = self.patterns[winner].clone()
        rate = super().consolidate(winner, query, g)
        if rate > 0 and float(self.patterns[winner] @ p0) < self.anchor:
            self.patterns[winner] = old
            return 0.0
        return rate


class DormantGatedMemory(GatedEpisodicMemory):
    """The adopted commit rule, plus anchoring and dormant entries at one similarity radius.
    Call step(query, steady, changing) once per memory step."""

    def __init__(self, dim, gap_scale, radius=RADIUS, eta=ADOPTED['eta'], gate_content=ADOPTED['gate_content'],
                 prune_eps=PRUNE_EPS, novel_report=True, **episodic_kwargs):
        super().__init__(dim, gap_scale, theta=radius, eta=eta, gate_content=gate_content, match_floor=radius,
                         **episodic_kwargs)
        self.mem = AnchoredConsolidatingMemory(dim, anchor=radius, eta=eta, gate_content=gate_content,
                                               match_floor=radius, gap_scale=gap_scale, **episodic_kwargs)
        self.radius = radius
        self.prune_eps = prune_eps
        self.novel_report = novel_report
        self.dormant = []                    # dicts: id (of the demoted entry), pattern, w_char

    def step(self, query, steady=True, changing=False):
        """Returns the parent's dict (winner, created, report, g, consolidation_rate) plus
        reawakened (id of the dormant entry a new entry inherited from, or None), demoted (ids evicted
        into dormancy this step), and similarity (winner's cosine to the query)."""
        mem = self.mem
        snap = {e: (mem.patterns[k].clone(), mem.w_char[k]) for k, e in enumerate(mem.ids)}
        out = super().step(query, steady=steady, changing=changing)

        out['reawakened'] = None
        if out['created'] is not None and self.dormant:
            k = mem.ids.index(out['created'])
            sims = [float(d['pattern'] @ mem.patterns[k]) for d in self.dormant]
            best = max(range(len(sims)), key=sims.__getitem__)
            if sims[best] >= self.radius:
                d = self.dormant.pop(best)
                mem.w_char[k] = d['w_char']
                out['reawakened'] = d['id']

        alive = set(mem.ids)
        out['demoted'] = [e for e in snap if e not in alive]
        for e in out['demoted']:
            self.dormant.append(dict(id=e, pattern=snap[e][0], w_char=snap[e][1]))
            mem.birth.pop(e, None)
        for d in self.dormant:
            d['w_char'] += CONSTANTS.decay_char * (1 - d['w_char'])
        self.dormant = [d for d in self.dormant if d['w_char'] - 1 >= self.prune_eps]

        if out['winner'] in alive:
            out['similarity'] = float(mem.patterns[mem.ids.index(out['winner'])] @ query)
        else:
            out['similarity'] = 1.0
        if self.novel_report and out['report'] is not TRANSITIONAL and out['similarity'] < self.radius:
            out['report'] = NOVEL
        return out

    def character(self):
        """Every entry that still carries character: (pattern, w_char, state) for live and dormant."""
        live = [(p, w, 'live') for p, w in zip(self.mem.patterns, self.mem.w_char)]
        return live + [(d['pattern'], d['w_char'], 'dormant') for d in self.dormant]

"""Episodic memory with content consolidation and the adopted commit rule, promoted from the coupling
toy's fork (notebooks/integration/content_consolidation/episodic_content_fork.py). It's a SEPARATE
module: src/hopfield/episodic.py (the validated original) is unchanged and still the baseline.
Adopted by Jasper 2026-09-26; see experiments_integration.md and system_contract.md (Q4).

What it adds to EpisodicMemory (everything else is inherited unchanged: gated retrieval,
w_fast/w_char update, staleness, gated eviction):

  1. Content consolidation. The winner's stored pattern moves toward the query,
     p <- unit(p + alpha (q - p)), with alpha = eta (1 - g), where g is the same top1-top2
     ambiguity gate used for strength. It only happens if the winner actually matches
     (cosine >= match_floor), because the relative gate can't see novelty (principles.md,
     caveat under "strength breaks ties").
  2. The commit rule. Memory CREATES an entry only when the query is novel (best cosine < theta)
     AND steady AND the substrate isn't re-learning. It CONSOLIDATES content only when the
     substrate isn't re-learning. Retrieval and the staleness refresh happen on EVERY step:
     rehearsal during transitions is what keeps memories alive past their horizon (causally
     tested).
  3. The transitional output. While the substrate is re-learning, step() reports TRANSITIONAL
     instead of a (stale) context. The system knows it's in that window, since the flag is its own.
  4. The single-pattern case. src/hopfield/two_layer.retrieve_gated indexes the second-best
     similarity, so it can't run with one stored pattern. Here the lone pattern wins with weight
     1 and g=0, as the toy did. retrieve_gated itself is left alone.

Signals (steady / changing) come from src/integration/interface.py, and the caller supplies them
per step. gap_scale must be grounded by the caller from its own query statistics (the toy used
0.6 x the median top1-top2 gap, the original grounding procedure). It isn't defaulted.
"""
import torch

from .episodic import EpisodicMemory

TRANSITIONAL = None
ADOPTED = dict(theta=0.5, eta=0.1, gate_content=True, match_floor=0.5)


def _unit(v):
    v = v - v.mean()
    n = torch.linalg.norm(v)
    return v / n if n > 0 else v


class ConsolidatingEpisodicMemory(EpisodicMemory):
    """EpisodicMemory plus content consolidation (see module docstring). With eta=0 it behaves
    identically to EpisodicMemory."""

    def __init__(self, dim, eta=0.0, gate_content=True, match_floor=None, **kwargs):
        super().__init__(dim, **kwargs)
        self.eta = eta
        self.gate_content = gate_content
        self.match_floor = match_floor

    def consolidate(self, winner, query, g):
        """Move the winner's content toward the query; returns the rate used (0 = no move)."""
        if self.eta == 0.0:
            return 0.0
        if self.match_floor is not None and float(self.patterns[winner] @ query) < self.match_floor:
            return 0.0
        alpha = self.eta * (1.0 - g) if self.gate_content else self.eta
        if alpha > 0:
            p = self.patterns[winner]
            self.patterns[winner] = _unit(p + alpha * (query - p))
        return alpha


class GatedEpisodicMemory:
    """The adopted configuration. Call step(query, steady, changing) once per memory step."""

    def __init__(self, dim, gap_scale, theta=ADOPTED['theta'], eta=ADOPTED['eta'],
                 gate_content=ADOPTED['gate_content'], match_floor=ADOPTED['match_floor'], **episodic_kwargs):
        self.mem = ConsolidatingEpisodicMemory(dim, eta=eta, gate_content=gate_content, match_floor=match_floor,
                                               gap_scale=gap_scale, **episodic_kwargs)
        self.theta = theta
        self.t = 0

    def step(self, query, steady=True, changing=False):
        """query: 1-D tensor (unit, centered). steady: the query hasn't jumped. changing: the substrate
        is still re-learning. Returns dict(winner=id, created=id or None, report=id or
        TRANSITIONAL, g=ambiguity gate, consolidation_rate)."""
        mem, s = self.mem, self.t
        created = None
        novel = not mem.patterns or float((torch.stack(mem.patterns) @ query).max()) < self.theta
        if novel and (not mem.patterns or (steady and not changing)):
            mem.add_pattern(query.clone(), s)
            created = mem.ids[-1]
        if len(mem.patterns) == 1:
            w_fast, w_char = mem._update_fn(torch.tensor(mem.w_fast), torch.tensor(mem.w_char), torch.tensor([1.0]))
            mem.w_fast, mem.w_char = w_fast.tolist(), w_char.tolist()
            mem.staleness[0] = 0
            wi, g = 0, 0.0
        else:
            wi, _, g = mem.retrieve_and_update(query)
        winner = mem.ids[wi]
        rate = mem.consolidate(wi, query, g) if not changing else 0.0
        mem.prune_step(s)
        self.t += 1
        return dict(winner=winner, created=created, report=TRANSITIONAL if changing else winner, g=g,
                    consolidation_rate=rate)

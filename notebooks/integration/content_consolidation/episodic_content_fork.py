"""FORK of src/hopfield/episodic.py: episodic memory whose entry CONTENT consolidates.

This is a fork, deliberately kept apart from the validated module. It subclasses EpisodicMemory,
leaves every inherited behaviour (retrieval, w_fast/w_char update, staleness, gated eviction)
exactly as it is, and adds one thing: after a step, the winning entry's stored pattern moves
toward the query. src/hopfield/ is not modified. With eta=0 this class is behaviourally
identical to EpisodicMemory (checked at runtime by run_content_consolidation.py against the
v0 toy).

Why: coupling toy v0 found that in the validated memory, consolidation acts on STRENGTH (w_char)
only, and an entry's CONTENT is frozen at creation. When the substrate feeds it, entries are often
snapshots of transitional or early-learning queries, so a settled context returning doesn't match
them and memory creates a duplicate (experiments_integration.md).

Update, applied to the winner only:
    p <- unit(p + alpha * (q - p)),   alpha = eta * (1 - g)   if gate_content  else  eta
where g = 1/(1 + gap/gap_scale) is the SAME ambiguity gate already used for strength. So content
moves at full rate only when retrieval is unambiguous and not at all when it's ambiguous. This is
"strength breaks ties, never overrides matches" (principles.md) applied to content. The
gate_content=False variant is the test of whether that matters. unit() centers and normalizes,
matching how the toy's queries are built. When memory holds a single pattern there's no competitor
to be ambiguous with, so g=0.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.hopfield.episodic import EpisodicMemory


def _unit(v):
    v = v - v.mean()
    n = torch.linalg.norm(v)
    return v / n if n > 0 else v


class ConsolidatingEpisodicMemory(EpisodicMemory):
    """match_floor (added after the absorption finding, default None = off, so earlier runs are
    unchanged): consolidate only if the winner's CURRENT content actually matches the query,
    cosine >= match_floor. The top1-top2 gate g is relative and can't see novelty. This is the
    absolute condition the principles.md caveat calls for."""

    def __init__(self, dim, eta=0.0, gate_content=True, match_floor=None, **kwargs):
        super().__init__(dim, **kwargs)
        self.eta = eta
        self.gate_content = gate_content
        self.match_floor = match_floor

    def consolidate(self, winner, query, g):
        """Move the winner's stored content toward the query. No-op when eta == 0."""
        if self.eta == 0.0:
            return 0.0
        if self.match_floor is not None and float(self.patterns[winner] @ query) < self.match_floor:
            return 0.0
        alpha = self.eta * (1.0 - g) if self.gate_content else self.eta
        if alpha > 0:
            p = self.patterns[winner]
            self.patterns[winner] = _unit(p + alpha * (query - p))
        return alpha

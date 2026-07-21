"""Episodic grow-and-prune memory layer.

Canonical version: notebooks/hopfield/episodic_layer_v2.ipynb (the coverage-gap fix -- eviction
gated on distance past the eviction threshold, not distance to a peer), validated by
episodic_layer_v3_falsification.ipynb. v4/v5 found and attempted (unsuccessfully) to patch a
separate, still-open primacy/recency issue on top of this same mechanism -- they don't change
prune_step/retrieve_and_update, so they aren't reflected here; see experiments.md.

Unlike the original notebook, this reuses two_layer.make_update_fn/retrieve_gated directly
(batched over the whole pattern population each step) instead of a separately-maintained scalar
reimplementation of the same formula -- the duplication a reviewer flagged as a regression risk.
"""
import torch

from .two_layer import EPISODIC_OPERATING_POINT, make_update_fn, retrieve_gated


class EpisodicMemory:
    def __init__(self, dim, staleness_threshold=150, gap_scale_evict=20.0, strength_bonus=10.0,
                 k=EPISODIC_OPERATING_POINT["k"], w_fast_max=EPISODIC_OPERATING_POINT["w_fast_max"],
                 gap_scale=EPISODIC_OPERATING_POINT["gap_scale"]):
        self.dim = dim
        self.staleness_threshold = staleness_threshold
        self.gap_scale_evict = gap_scale_evict
        self.strength_bonus = strength_bonus
        self.gap_scale = gap_scale
        self._update_fn = make_update_fn(k, w_fast_max)

        self.patterns = []       # list of 1D tensors
        self.w_fast = []         # list of floats
        self.w_char = []         # list of floats
        self.staleness = []      # list of ints, steps since last retrieval win
        self.birth_step = []     # for logging
        self.next_id = 0
        self.ids = []            # stable ids, since list indices shift on eviction
        self.eviction_log = []   # dicts with step, id, staleness, w_char, g_evict, n_eligible

    def add_pattern(self, vec, step):
        self.patterns.append(vec)
        self.w_fast.append(1.0)
        self.w_char.append(1.0)
        self.staleness.append(0)
        self.birth_step.append(step)
        self.ids.append(self.next_id)
        self.next_id += 1
        return len(self.patterns) - 1

    def retrieve_and_update(self, query):
        X = torch.stack(self.patterns)
        w_fast_t = torch.tensor(self.w_fast)
        w_char_t = torch.tensor(self.w_char)

        retrieved, weights, g = retrieve_gated(query, X, w_fast_t, w_char_t, self.gap_scale)
        winner = weights.argmax().item()

        w_fast_t, w_char_t = self._update_fn(w_fast_t, w_char_t, weights)
        self.w_fast = w_fast_t.tolist()
        self.w_char = w_char_t.tolist()
        for i in range(len(self.patterns)):
            self.staleness[i] = 0 if i == winner else self.staleness[i] + 1

        return winner, weights, g

    def prune_step(self, step, force_ungated=False):
        """Evict at most one entry per call, gated per-candidate by distance past threshold
        (staleness_over), not by requiring a peer to compare against. Fixes the v1 coverage
        gap where a lone eligible candidate fell through to unconditional eviction."""
        eligible = [i for i in range(len(self.patterns)) if self.staleness[i] > self.staleness_threshold]
        if len(eligible) == 0:
            return None

        best_idx = None
        best_score = None
        best_g = None
        for i in eligible:
            staleness_over = max(self.staleness[i] - self.staleness_threshold, 0)
            g_evict = 1.0 if force_ungated else 1 / (1 + staleness_over / self.gap_scale_evict)
            score = staleness_over - g_evict * self.strength_bonus * (self.w_char[i] - 1)
            if best_score is None or score > best_score:
                best_idx, best_score, best_g = i, score, g_evict

        if best_score <= 0:
            # strength's gated protection outweighs staleness_over -- nobody evicted this step
            return None

        evict_idx = best_idx
        info = {
            "step": step, "id": self.ids[evict_idx], "staleness": self.staleness[evict_idx],
            "w_char": self.w_char[evict_idx], "w_fast": self.w_fast[evict_idx],
            "g_evict": best_g, "n_eligible": len(eligible),
        }
        self.eviction_log.append(info)

        del self.patterns[evict_idx]
        del self.w_fast[evict_idx]
        del self.w_char[evict_idx]
        del self.staleness[evict_idx]
        del self.birth_step[evict_idx]
        del self.ids[evict_idx]
        return info

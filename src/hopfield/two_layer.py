"""Canonical two-layer (fast/slow) Hopfield update rule and ambiguity-gated retrieval.

Source of truth for the mechanism validated across notebooks/hopfield/two_layer_consolidation.ipynb
(saturating w_char headroom fix) -> two_layer_rate_modulated_savings_v2.ipynb (matching w_fast
headroom + capped rate-modulation multiplier) -> two_layer_ambiguity_gated.ipynb (ambiguity gate
on retrieval bias). See experiments.md and principles.md ("strength breaks ties, never overrides
matches") for the reasoning behind each piece.

k, w_fast_max, and gap_scale are deliberately NOT defaulted here -- experiments.md is explicit
that no single combination was adopted project-wide. episodic.py reuses a specific "known-good"
point (EPISODIC_OPERATING_POINT) rather than re-tuning; other callers should choose explicitly.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TwoLayerConstants:
    decay_fast: float = 0.02
    increment_fast: float = 0.1
    decay_char: float = 0.0005
    consolidation_rate: float = 0.01
    char_weight: float = 1.0
    w_char_max: float = 10.0
    max_multiplier: float = 3.0
    beta: float = 4.0


CONSTANTS = TwoLayerConstants()

# Reused as-is by the episodic layer (episodic_layer_v2.ipynb onward) -- "a known-good operating
# point, not re-tuned", not a claim that this is the right default for every future use.
EPISODIC_OPERATING_POINT = dict(k=0.5, w_fast_max=10.0, gap_scale=0.1534)


def make_update_fn(k, w_fast_max, c: TwoLayerConstants = CONSTANTS):
    """Returns update_two_layer(w_fast, w_char, retrieval_weights) -> (w_fast, w_char).

    w_fast/w_char/retrieval_weights are torch tensors, one entry per pattern -- callers with a
    single pattern (e.g. episodic memory updating its whole population each step) just pass
    whole-population tensors, not per-pattern scalars.
    """
    def update_two_layer(w_fast, w_char, retrieval_weights):
        multiplier = (1 + k * (w_char - 1)).clamp(max=c.max_multiplier)
        increment_fast_effective = c.increment_fast * multiplier
        fast_headroom = (w_fast_max - w_fast).clamp(min=0) / w_fast_max
        w_fast = w_fast + c.decay_fast * (1 - w_fast) + increment_fast_effective * retrieval_weights * fast_headroom

        char_headroom = (c.w_char_max - w_char).clamp(min=0) / c.w_char_max
        w_char = w_char + c.decay_char * (1 - w_char) + c.consolidation_rate * (w_fast - 1).clamp(min=0) * char_headroom
        return w_fast, w_char
    return update_two_layer


def retrieve_gated(query, X, w_fast, w_char, gap_scale, c: TwoLayerConstants = CONSTANTS):
    """Ambiguity-gated retrieval: strength (w_fast/w_char) only breaks ties, gated by how
    ambiguous the top1-top2 content-similarity gap already is."""
    similarities = X @ query
    sorted_sims, _ = torch.sort(similarities, descending=True)
    gap = (sorted_sims[0] - sorted_sims[1]).item()
    g = 1 / (1 + gap / gap_scale)

    biased = c.beta * similarities + g * (torch.log(w_fast) + c.char_weight * torch.log(w_char))
    weights = F.softmax(biased, dim=0)
    retrieved = X.T @ weights
    return retrieved, weights, g

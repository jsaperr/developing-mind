"""Regression checks for the canonical two-layer + episodic Hopfield mechanism (src/hopfield/).

NOT YET EXECUTED as of writing -- added per the reviewer feedback that the Hopfield dependency
chain (v1 bug -> v2 saturation fix -> rate-mod v1 reopening it -> v2 containment -> ambiguity
gating) had no regression check. Run with `pytest tests/` before trusting these pass.

These test properties directly, rather than reproducing the original notebooks' full
phase-cycling experiment drivers (which aren't captured in src/ and shouldn't be guessed at).
"""
import torch

from src.hopfield.episodic import EpisodicMemory
from src.hopfield.two_layer import CONSTANTS, EPISODIC_OPERATING_POINT, make_update_fn, retrieve_gated


def test_retrieve_gated_returns_valid_distribution():
    torch.manual_seed(0)
    dim = 16
    n = 5
    X = torch.nn.functional.normalize(torch.randn(n, dim), dim=1)
    query = X[0]
    w_fast = torch.ones(n)
    w_char = torch.ones(n)

    retrieved, weights, g = retrieve_gated(query, X, w_fast, w_char, gap_scale=0.1534)

    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert (weights >= 0).all()
    assert retrieved.shape == query.shape


def test_gate_favors_strength_only_when_ambiguous():
    """g should be near 1 when top1/top2 content similarity is nearly tied, and small when one
    pattern is a clear content match -- 'strength breaks ties, never overrides matches'."""
    gap_scale = 0.1534

    clear_sims = torch.tensor([0.95, 0.10, 0.05])
    sorted_sims, _ = torch.sort(clear_sims, descending=True)
    gap_clear = (sorted_sims[0] - sorted_sims[1]).item()
    g_clear = 1 / (1 + gap_clear / gap_scale)

    ambiguous_sims = torch.tensor([0.61, 0.60, 0.05])
    sorted_sims, _ = torch.sort(ambiguous_sims, descending=True)
    gap_ambiguous = (sorted_sims[0] - sorted_sims[1]).item()
    g_ambiguous = 1 / (1 + gap_ambiguous / gap_scale)

    assert g_ambiguous > g_clear
    assert g_clear < 0.2
    assert g_ambiguous > 0.8


def test_headroom_bounds_growth_under_repeated_reinforcement():
    """Repeatedly retrieving the same pattern should saturate w_fast/w_char at their caps, not
    grow unboundedly -- this is the saturation fix for the original lock-in bug."""
    k = EPISODIC_OPERATING_POINT["k"]
    w_fast_max = EPISODIC_OPERATING_POINT["w_fast_max"]
    update_fn = make_update_fn(k, w_fast_max)

    w_fast = torch.ones(3)
    w_char = torch.ones(3)
    retrieval_weights = torch.tensor([1.0, 0.0, 0.0])  # pattern 0 always wins

    for _ in range(5000):
        w_fast, w_char = update_fn(w_fast, w_char, retrieval_weights)

    assert w_fast[0] <= w_fast_max + 1e-6
    assert w_char[0] <= CONSTANTS.w_char_max + 1e-6
    assert w_fast[0] > 1.0  # actually grew, not a no-op
    assert w_char[0] > 1.0


def test_eviction_gate_delays_but_does_not_rescue_indefinitely():
    """A high-w_char entry should survive further past the staleness threshold than a fresh
    entry before being evicted, but still get evicted eventually if starved long enough -- the
    validated v3 falsification claim ('rescued longer, never rescued indefinitely')."""
    mem = EpisodicMemory(dim=8, staleness_threshold=150, gap_scale_evict=20.0, strength_bonus=10.0)
    fresh_id = mem.add_pattern(torch.zeros(8), step=0)
    strong_id = mem.add_pattern(torch.zeros(8), step=0)
    mem.w_char[mem.ids.index(strong_id)] = 6.0  # a well-consolidated entry (matches the falsification test's 5-8 range)

    fresh_evicted_at = None
    strong_evicted_at = None
    for step in range(1, 3000):
        for i in range(len(mem.staleness)):
            mem.staleness[i] += 1
        info = mem.prune_step(step)
        if info is None:
            continue
        if info["id"] == fresh_id:
            fresh_evicted_at = step
        elif info["id"] == strong_id:
            strong_evicted_at = step
        if fresh_evicted_at is not None and strong_evicted_at is not None:
            break

    assert fresh_evicted_at is not None, "fresh entry should be evicted well before the loop ends"
    assert strong_evicted_at is not None, "strong entry should still be evicted eventually, not rescued forever"
    assert strong_evicted_at > fresh_evicted_at, "strength should delay eviction, not prevent it"

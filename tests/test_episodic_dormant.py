"""src/hopfield/episodic_dormant.py must reproduce the notebook toys it was promoted from EXACTLY on real
saved runs (the long world, where relinking actually happens), and each mechanism must do what it
says on small synthetic cases."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/integration/memory_limits"))

import anchored_consolidation as AC                            # noqa: E402

from src.hopfield.episodic_consolidating import (TRANSITIONAL, ConsolidatingEpisodicMemory,  # noqa: E402
                                                 GatedEpisodicMemory)
from src.hopfield.episodic_dormant import NOVEL, AnchoredConsolidatingMemory, DormantGatedMemory  # noqa: E402
from src.integration.interface import unit                     # noqa: E402

R, LW = AC.R, AC.LW
GS = 0.1


@pytest.fixture(scope="module")
def long_streams():
    runs = R.load(R.B2 / "long_world_data" / "long20_n7_seed4200[01].json.gz")
    assert len(runs) == 2, "missing long-world test data"
    rng = np.random.default_rng(0)
    return [R.build(d, wm, r, chg, 10, "H+", rng) for d, wm, r, chg in runs]


class Logging(AC.AnchoredGated):
    """The notebook memory, logging what the toy loop did at each step (state at step start, so the
    toy's post-step relinking shows up in the next step's w_char)."""
    log = []

    def step(self, query, steady=True, changing=False):
        Logging.log.append((tuple(self.mem.ids), tuple(self.mem.w_char)))
        o = super().step(query, steady=steady, changing=changing)
        Logging.log.append((o['winner'], o['created']))
        return o


def test_reproduces_the_notebook_store_exactly_on_the_long_world(long_streams):
    old = LW.GatedEpisodicMemory
    LW.GatedEpisodicMemory = Logging
    try:
        for s in long_streams:
            Logging.log = []
            ref = LW.run(s, GS, True)
            toy_log = Logging.log
            m = DormantGatedMemory(dim=s['q'].shape[1], gap_scale=GS, novel_report=False)
            log, links = [], 0
            for i, x in enumerate(torch.tensor(s['q'], dtype=torch.float32)):
                log.append((tuple(m.mem.ids), tuple(m.mem.w_char)))
                o = m.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
                log.append((o['winner'], o['created']))
                links += o['reawakened'] is not None
            assert links == ref['links'] and links > 0
            assert log == toy_log
    finally:
        LW.GatedEpisodicMemory = old


def test_without_anchoring_or_relinking_it_is_the_adopted_memory_at_that_radius(long_streams):
    for s in long_streams:
        a = DormantGatedMemory(dim=30, gap_scale=GS, novel_report=False)
        a.mem.anchor = -1.0                           # anchoring can never bite
        b = GatedEpisodicMemory(dim=30, gap_scale=GS, theta=0.8, match_floor=0.8)
        for i, x in enumerate(torch.tensor(s['q'][:400], dtype=torch.float32)):
            oa = a.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
            ob = b.step(x, steady=bool(s['steady'][i]), changing=bool(s['changing'][i]))
            if oa['reawakened'] is not None:
                break                                 # from here the two legitimately differ
            assert (oa['winner'], oa['created'], oa['report']) == (ob['winner'], ob['created'], ob['report'])


def blend(a, b, t):
    return torch.tensor(unit((1 - t) * a + t * b), dtype=torch.float32)


def test_anchoring_stops_a_slow_slide_that_plain_consolidation_follows():
    a = unit(np.isin(np.arange(30), range(0, 10)).astype(float))
    b = unit(np.isin(np.arange(30), range(3, 13)).astype(float))     # 70% overlap, cosine 0.55
    plain = ConsolidatingEpisodicMemory(30, eta=0.5, gate_content=False, match_floor=0.8)
    anch = AnchoredConsolidatingMemory(30, anchor=0.8, eta=0.5, gate_content=False, match_floor=0.8)
    for m in (plain, anch):
        m.add_pattern(torch.tensor(a, dtype=torch.float32), 0)
        for t in np.linspace(0, 1, 400):              # each query is only slightly further along
            m.consolidate(0, blend(a, b, t), 0.0)
    p0 = torch.tensor(a, dtype=torch.float32)
    assert float(plain.patterns[0] @ p0) < 0.8        # plain content slid out of its own identity
    assert float(anch.patterns[0] @ p0) >= 0.8        # anchored content stayed within the radius


def two_context_world(n_a, n_b, n_a2, rng, sigma=0.02):
    a = unit(np.isin(np.arange(30), range(0, 10)).astype(float))
    b = unit(np.isin(np.arange(30), range(10, 20)).astype(float))
    q = [unit(p + sigma * rng.standard_normal(30)) for p, n in ((a, n_a), (b, n_b), (a, n_a2)) for _ in range(n)]
    return torch.tensor(np.array(q), dtype=torch.float32)


def test_eviction_demotes_and_a_return_reawakens_with_the_old_character():
    Q = two_context_world(150, 400, 50, np.random.default_rng(0))
    m = DormantGatedMemory(dim=30, gap_scale=GS)
    demoted, reawakened, inherited = [], [], None
    for i, x in enumerate(Q):
        o = m.step(x)
        demoted += o['demoted']
        if o['reawakened'] is not None:
            reawakened.append(o['reawakened'])
            k = m.mem.ids.index(o['created']); inherited = m.mem.w_char[k]
    assert demoted == [0]                             # A's entry was demoted, not deleted
    assert reawakened == [0]                          # A's return found its dormant entry
    assert inherited > 1.5                            # and started from A's old character, not 1


def test_dormant_entries_are_pruned_once_they_carry_no_character():
    Q = two_context_world(150, 3000, 0, np.random.default_rng(1))
    m = DormantGatedMemory(dim=30, gap_scale=GS, prune_eps=1.5)   # large eps so 3000 steps of decay reach it
    seen = False
    for x in Q:
        m.step(x)
        seen |= bool(m.dormant)
        for d in m.dormant:
            assert d['w_char'] - 1 >= 1.5
    assert seen and not m.dormant                     # it went dormant, decayed, and was pruned


def test_novel_report_names_nothing_when_nothing_fits():
    a = torch.tensor(unit(np.isin(np.arange(30), range(0, 10)).astype(float)), dtype=torch.float32)
    b = torch.tensor(unit(np.isin(np.arange(30), range(3, 13)).astype(float)), dtype=torch.float32)
    for novel, expected in ((True, NOVEL), (False, 0)):
        m = DormantGatedMemory(dim=30, gap_scale=GS, novel_report=novel)
        m.step(a)                                     # entry 0 = A
        o = m.step(b, steady=False)                   # new (cosine 0.55) but not steady: can't create
        assert o['created'] is None and o['report'] == expected
        assert m.step(b, changing=True)['report'] is TRANSITIONAL   # re-learning still wins
        assert m.step(a)['report'] == 0               # a real match is still named

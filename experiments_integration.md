# Experiments Log: Integration (substrate ↔ memory)

The first phase that couples two validated parts. Hopfield / two-layer / episodic work lives in
`experiments.md`, the Brian2 substrate in `experiments_brian2.md` (index) and `docs/log/brian2/`,
and the ESN in `experiments_esn.md`. The design spec for this phase is `system_contract.md`.
Entries are newest first. State predictions before running, keep contaminated runs on record,
inspect trajectories before claiming.

## Current state (2026-09-25)

- **Coupling toy v0 is built** (offline, one-way: saved substrate runs replayed into the unchanged
  `src/hopfield` memory). Jasper gave the explicit go.
- **Established:**
  - Right after a *novel* change, the substrate's first signal is the old context's absence, not
    the new one's presence. Memory can file that transitional "not-old" query as an entry, which
    then captures later queries: it reports the two-back context, in one seed for a whole phase.
    This is specific to the substrate (the blended control shows none of it) and independent of
    strength (unchanged with the strength bias off). It's a *creation-rule* failure, not a
    violation of "strength breaks ties".
  - 1-back returns are reported within one step, via the substrate's recognition signal.
    2-back returns aren't.
  - **The complementary-systems test fails in the coupled system as built, for two reasons:**
    1. Memory's validated retention (150 steps) is shorter than a 2-back lull at a 10 s clock.
    2. Even when memory retains, it stores frozen snapshots of transitional queries, so returns
       create duplicates instead of reusing the original entry.
- **Open:** creation rule (contract Q4: the episodic layer never had one; every validated notebook
  used an oracle); the clock mapping (Q2 now has a hard constraint); content consolidation of
  memory entries (next hypothesis); readout choice for the first context.

---

## 2026-09-25 — Creation gates: a strict stability gate helps, the dip gate doesn't fire enough; two of three predictions refuted

**Code:** `notebooks/integration/coupling_v0/run_creation_gates.py` (predictions in the docstring
before the first run); `creation_gates_summary.json`. Substrate arm only, W=10 s. Retrieval,
update, prune, THETA and the grounded gap_scale are all unchanged; only *creation* is gated. An
empty memory may always create.

**Gates:**
- **G-stable:** create only if the query was steady for two steps (consecutive cosine at least
  TAU; TAU=0.9 pre-registered, sensitivity at 0.8 and 0.95).
- **G-dip:** no creation while the population rate sits below its trailing 30-window median by
  more than 3 x robust sd of its first differences. This is the contract's S4/P3 idea, the
  context-blind dip used as "transition in progress".

**Predictions and outcomes (A→B→C unless noted):**
- **GATE-P1 (G-stable at 0.9 cuts two-back reports from 124 to at most 40, and seed 32005's lock
  to at most 20): REFUTED.** 126, with 32005 still at 100. The transitional "not-B" query drifts
  slowly enough to look steady. Only the strictest listed value (0.95, just under the settled
  consecutive cosine of 0.967) cuts it: 38 total, 32005 down to 6. That's effective but fragile.
  In the v1 world, G-stable at 0.9 did halve the two-back reports (115 → 58).
- **GATE-P2 (G-dip cuts them to at most 60 and raises latency): REFUTED on the first half.** 127,
  no reduction. The dip flag is up on only about 2% of steps. It marks when a change *starts*,
  not how long the transitional state lasts. The threshold wasn't tuned afterwards. (Latency did
  rise, 3 → 6 steps.)
- **GATE-P3 (neither gate changes A→B→A accuracy by more than 0.01): marginally refuted.** Every
  gate costs 1.1-1.8%, because it delays creation at the novel swap.
- **Unexpected, checked for artifacts:** G-stable raises the A→B→A return "reuse" from 2/8 to
  8/8, but part of that is forced. The gate blocks creation for 2 steps, so memory *must*
  retrieve an old entry. Measured directly, the original A entry wins the *whole* return phase in
  5/8 seeds with G-stable versus 1/8 ungated. The rest made a duplicate after the delay. So
  delaying creation until the query settles yields a better first-context entry. Part of arc 07's
  "first context is weakly represented" is memory filing it while the substrate is still
  learning it.

**Net:** the S4 change signal isn't a usable transition flag in its simple form. A strict
stability gate helps, but it's a threshold sitting just under the settled jitter, not a
principled rule. Creation is the weak point of the coupled system.

## 2026-09-25 — Coupling toy v0: substrate → memory works for tracking and 1-back returns; fails on novel changes (two-back capture) and on bridging a 2-back lull

**Code:** `notebooks/integration/coupling_v0/run_coupling_v0.py` (design and predictions in the
docstring before the first run, then a documented revision section); `coupling_v0_summary.json`,
`coupling_v0_traces.png`; the first run's output is kept in `first_run_output.txt`.

**Setup:**
- **Worlds:** A→B→A (N=7, 2 blocks), A→B→C (novel-C) and A→B→C→A→C (v1), 8 seeds each, all from
  existing saved runs. Replay is offline and one-way; there's no feedback into the substrate.
- **Clock:** one memory step = one readout window of 10 s or 50 s.
- **Memory:** `EpisodicMemory` with the validated operating point, all constants unchanged except
  `gap_scale`, which is re-grounded by the original procedure (0.6 x the empirical top1-top2 gap
  median of the substrate query stream). The median came out 0.52-0.63, so gap_scale is
  0.31-0.38, against the old 0.1534.
- **Creation (provisional):** a new entry when the best cosine is below THETA=0.5.
- **Arms:**
  - *substrate*: the S1 readout H, label-free.
  - *clean*: the true-context prototype plus jitter matched to the substrate's step-to-step
    jitter; the memory layer's own baseline.
  - *blended*: clean prototypes mixed at the substrate's own per-window old/new fraction (the
    other session's control, isolating ambiguity from substrate-specific structure).
- **Scoring:** entries are tagged by content (their nearest context prototype); labels are used
  only for scoring.

**The first run was contaminated, caught before any claim, and fixed.** Both problems were in the
controls and scoring, not the system under test:
1. **A label leak.** Entries were tagged with the true context at creation, so an entry created
   at a novel swap scored "correct" by construction whatever its content. This is
   `principles.md` #4, the contaminated proxy. The tell was near-perfect accuracy and zero
   latency everywhere.
2. **Unfair control noise.** The controls got i.i.d. noise matched to the substrate's distance
   from the prototype, but the substrate's deviation is mostly a stable offset (consecutive
   windows correlate at 0.97). So the controls had far *more* step-to-step jitter than the
   substrate, and created 14-27 duplicate entries versus the substrate's 8.

After the fix the clean arm creates exactly one entry per context.

**Predictions on record, and outcomes (W=10 s):**
- **TOY-P1 (after novel swaps the substrate arm reports the previous context on more steps than
  clean): CONFIRMED in shape, with a different mechanism than the contract predicted.**
  - At the first novel swap, substrate and blended look alike (1-3 stale steps each, clean 0), so
    ambiguity alone explains it.
  - At the *second* novel swap (B→C), a substrate-specific failure appears: memory reports the
    **two-back context A** for 124 steps across 8 seeds (A→B→C; 110 in v1). One seed (32005) does
    so for the entire phase, and the blended arm has 0.
  - **Mechanism:** right after B→C, the B-tuned neurons lose synchronous drive and drop in rate,
    which leaves the A-holders relatively most active. So the query reads about
    [A +0.74, B −0.89, C +0.15]: "not-B", which the centered readout can't tell apart from A.
    Memory creates an entry from it (content-tagged A), and later C-leaning queries keep matching
    it through the shared −B component.
  - **Not strength:** with the strength bias off (g → 0) it's 121 steps against 124. The failure
    is creation plus content similarity, so "strength breaks ties" isn't implicated.
  - It's the confabulation failure's *shape* (confident, persistent, wrong) produced by two
    individually-validated layers.
- **TOY-P2 (1-back return reported within 1 step in at least 6/8; 2-back return not): CONFIRMED,
  both halves.**
  - The A→B→A return, and v1's A→C, are within 1 step in 8/8.
  - v1's 2-back return C→A is slow in 6/8. The two fast seeds are exactly the two with an
    A-holding residue neuron in the substrate (arc 05 v1 entry).
  - The "by reusing the old entry" part of P2 largely fails. Genuine recognition, meaning the
    original entry wins most of the return phase, is 1/8 for the A→B→A return (A is the first
    context) and 4/8 for v1's C return.

**The complementary-systems test (v1, measured as whether the entry from the original phase wins
the return phase): FAILS as built, for two separable reasons.**

| arm | A after two intervening contexts (2000 s lull), W=10 s | same, W=50 s | C after one (1000 s), W=10 s | same, W=50 s |
|---|---|---|---|---|
| clean (perfect input) | **0/8** | **8/8** | 8/8 | 8/8 |
| substrate | 1/8 | 2/8 | 4/8 | 2/8 |
| substrate + G-stable 0.9 | 2/8 | 2/8 | 3/8 | 6/8 |

1. **The clock makes memory forget.** With perfect input, memory drops A at 10 s per step and
   keeps it at 50 s per step. The validated staleness limit is 150 *steps*, so retention is
   150 x the clock: 1500 s, shorter than the 2000 s lull. Q2 isn't free. For complementary systems
   to work at all, memory's horizon must exceed the gaps it's meant to bridge, and the substrate
   is one-back whatever the clock.
2. **Memory stores bad snapshots.** Even where memory would retain (50 s), the substrate-fed
   system genuinely recognizes A in 2/8 and C in 2/8. Entries are frozen copies of whatever query
   triggered creation, often transitional or early-learning ones, so the settled context
   returning doesn't match them and a duplicate is made. In this memory, **consolidation acts on
   strength (`w_char`) only; entry content never refines after creation.** That was never tested
   with oracle-fed patterns; the substrate exposes it.

**Descriptive:**
- Memory stays bounded in every arm (substrate peak alive about 7).
- The clock also changes what memory *sees*. At 50 s the transitional period falls into one or
  two windows, so stale steps vanish, but the two-back capture persists (61 steps in v1, 47 in
  A→B→C) and accuracy is lower (0.87-0.90 vs 0.93-0.94).
- THETA sensitivity: 0.7 improves substrate accuracy (0.978-0.994) at the cost of 1.5-2x more
  entries; 0.3 lowers it. The creation threshold is a real lever, which is further evidence that
  creation is the weak point.

**Next hypothesis (not yet run):** content consolidation. Let the winning entry's content move
toward settled queries, so entries stop being frozen snapshots. The prediction is that it would
fix reason 2 but not reason 1, which is the clock and eviction.

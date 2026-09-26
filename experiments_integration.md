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
  - ~~The complementary-systems test fails in the coupled system as built~~. That was true of v0
    memory (a 150-step horizon, frozen snapshots). It's superseded by the entries above: with the
    best configuration and the handoff, the split works.
- **Content consolidation (forked memory; `src/` untouched):**
  - Alone, it fixes captured entries (two-back reports fall sharply) but not recognition.
  - Combined with a stability gate on creation, it gives reliable genuine recognition of a
    returning non-first context: 8/8 at 50 s steps.
  - It exposed that **the validated top1-top2 ambiguity gate is relative and can't see novelty.**
    A novel query forced onto the best of bad matches reads as "unambiguous", and consolidation
    runs at full rate (absorption). See the caveat under "strength breaks ties" in
    `principles.md`.
- **The two-back test passes** (v1b, pre-registered decision rule): substrate-fed memory
  recognizes a non-first context after the substrate has dropped it, 8/8 at W=50. **Transitional
  rehearsal (causal):** after each world change, while the query is unsettled and creation is
  blocked, retrieval sweeps across stored entries and resets each winner's eviction clock. A
  context survived a long lull in exactly the seeds where it won a step of such a window (6/8).
  The lesion kills survival at W=10 (6/8 → 1/8). This was first described as a "handoff" of the
  released context; a direct check showed the query points at "not-departing", not at the
  released context specifically. So memory has to bridge only from the last change where it was
  rehearsed (in practice, the substrate's release of it) to its return. **Confirmed in v1c:** with a 2000 s
  release-to-return against a 1500 s horizon, survival drops to 2/8 (v1b: 6/8 at 1000 s).
  The result holds at all 7 neighbouring parameter settings.
- **Replicated at strong_tight_gate** (the opposite-character operating point): one-back, the
  two-back pass (8/8), rehearsal (8/8 → 1/8 under the lesion) and the horizon rule all hold. The
  memory machinery matters *more* there.
- **Open, and now central:** there's no reliable signal that the world is still changing. The dip
  is too brief and the stability gate misses slow drift. It's behind the residual absorption, the
  two-back capture and creation during transitions.
- **The first-context weakness is a cold-start artifact** of the contrast readout (phase-1 query
  quality 0.31-0.34 vs 0.87-0.96 afterwards). It's deliberately not being chased.
- **Open:**
  - creation rule (contract Q4: the episodic layer never had one; every validated notebook used an
    oracle);
  - the clock mapping (Q2 now has a hard constraint);
  - the first context (still not recognized under any variant);
  - an absolute-match-quality gate: **tested.** It eliminates absorption and restores
    first-context 1-back recognition (1/8 → 5/8), at a small cost where low-match drift had been
    helping. The next refinement is to apply it only to established entries.

---

## 2026-09-26 — Replication at strong_tight_gate: all four predictions hold. Residual absorption traced to "not-departing" mixture queries; settled-only consolidation refuted because the stability gate can't see slow post-change drift

**Data:** `notebooks/brian2/stg_replication_data/`: v1b schedule (seeds 36000-36007) and v1c
schedule (37000-37007) at strong_tight_gate (10mV reference / gap_scale 1.0), 16/16 completed
(2070 s wall, 8 concurrent). The 30-input rig was calibration-checked there first against a bar
stated in advance (16.9-18.3 Hz, w_total 9.69-10.24, PASS). Readout
`notebooks/integration/two_back_test/run_stg_replication.py`, predictions STG-P1..P4 written
while the batch ran.

**Why this point:** it's the opposite character to 13mV/1.5. It's bistable at N=3 (about half
the seeds never differentiate), it's the only point with genuine ongoing identity churn, and its
reliability is dominance-based rather than suppression-based. Surviving it means the findings
belong to the architecture, not to one tidy operating point.

**Predictions and outcomes (all CONFIRMED):**
- **STG-P1, one-back:** 0-0.12 holders at every two-back return, and 2.12 at v1c's one-back
  return (speeding up +4.6 Hz).
- **STG-P2, the two-back pass:** v1b B genuine at W=50 is **8/8**.
- **STG-P3, rehearsal:** v1b B alive at W=10 is **8/8, falling to 1/8 under the lesion.** That's
  stronger than at 13mV/1.5 (6/8 → 1/8).
- **STG-P4, horizon rule:** v1c B (2000 s release-to-return) alive at W=10 is 2/8, and genuine at
  W=50 is 8/8.

**Descriptive:**
- The flagged risk (non-differentiating seeds degrading the readout) didn't materialize. Readout
  quality is 0.86-0.88 in every phase after the first. The cold start is starker here: phase 1
  reads 0.20-0.24, with some seeds near 0.
- **The machinery matters much more at this point.** Plain v0 memory drops to 0.74-0.79 accuracy
  at W=50 and recognizes v1c's B in 1/8 (best: 8/8). At 13mV/1.5 v0 had nearly kept up.

**Residual absorption, diagnosed:**
- The best configuration still shows 8 absorption events in stg v1c at W=10, despite the absolute
  floor. All happen right after a change.
- The queries genuinely match the entries (cos 0.80-0.86), and each flipped entry ends up on a
  boundary between two contexts (top-2 prototype cosines about 0.51 / 0.47).
- 6/8 of those entries had been *created* from an earlier transitional "not-C" query (birth cosine
  0.48-0.74). A later change away from C produced a similar query that pulled them across.
- 2/8 (one seed) were well-formed entries (0.76, 0.85 at birth) pulled across by such a query.
- **None were cold-start entries.** My first guess was wrong and was checked (0/8).
- Rerunning every world (below) shows the floor's "zero absorption" also doesn't hold for 13mV
  v1c (2 events). The earlier claim covered only the three worlds that existed then. It's
  corrected in `principles.md`.

**Settled-only consolidation** (`content_consolidation/run_settled_consolidation.py`, predictions
before running; all 7 worlds x 2 clocks):
- **Idea:** keep refresh on every step, but consolidate content only while the stability gate is
  open.
- **SC-P1 (absorption → 0 everywhere): REFUTED.** stg v1c W=10 is unchanged at 8.
- **SC-P2 (no key recognition drops by more than 1 seed): mostly held.** One miss: stg v1c W=50 B
  went 8 → 6.
- **Why it failed:** my diagnostic had labelled those steps "transitional" using the S1 definition
  (the first 300 s of a phase). The stability gate only closes when the query *jumps*, and the
  post-change mixture drifts slowly enough to pass as steady (the same reason TAU 0.9 didn't stop
  the capture on novel-C). The fix targeted the wrong signal. **Not adopted.**

**The common gap:** several failures now trace to one missing signal, *that the world is still
changing*:
- the two-back capture;
- the residual absorption;
- creation during transitions (Q4);
- the old episodic check (b), which was blocked on phase-awareness.

Neither candidate works as built: the swap dip fires on only about 2% of steps, and the stability
gate misses slow drift. **Candidate for next (not run):** a substrate-side "still re-learning"
signal, e.g. population weight-change rate. The re-learning neurons' weights move during the
first 60-150 s after a change and are nearly still once settled. That would tie memory's
commitment to the substrate's own plasticity.

## 2026-09-25 — The handoff horizon rule holds: memory must cover release-to-return (v1c)

**Data and code:**
- Substrate world v1c = A→B→C→A→C→B (`notebooks/brian2/v1c_schedule_data/`, seeds
  35000-35007, 8/8 completed, 6 x 1000 s; v1's frozen runner with the schedule and durations
  swapped).
- Readout `notebooks/integration/two_back_test/run_v1c_readout.py`, with predictions written
  while the batch ran, before any v1c result existed.

**Why:** the rule from the handoff lesion says memory has to bridge only the time from the
substrate *releasing* a context to its return. In v1c, B is released at C→A (refreshed then) and
returns only after A and C, so release-to-return is 2000 s, beyond the 1500 s horizon at W=10. In
v1b the same interval was 1000 s.

**Predictions and outcomes:**
- **V1C-P1 (the substrate holds nothing of B at its return): CONFIRMED.** 0 holders in 8/8.
  One-back replicates again: 3.0 holders at the one-back A→C return, speeding up +5.9 Hz, and 0 at
  the two-back C→A return.
- **V1C-P2, the rule (best config at W=10: B alive and recognized in ≤ 2/8): CONFIRMED, 2/8 and
  2/8,** against 6/8 in v1b. The only difference between the two is the release-to-return
  interval.
- **V1C-P3 (W=50: B alive 8/8, genuine ≥ 6/8): CONFIRMED,** 8/8 and 8/8.
- **V1C-P4 (at W=10 the handoff lesion changes B by ≤ 1 seed): marginally REFUTED.** 2/8 → 0/8.
  The two survivors were *also* kept alive by transitional reactivation, but at a change *after*
  the release, not at it.

**Refined rule:** a memory entry survives if some transitional reactivation happens within its
horizon before the context returns. The substrate's release is the reliable reactivation; later
changes occasionally provide another. So the Q2 requirement is to set staleness x clock to at
least the release-to-return interval the system should bridge.

**Also:**
- At W=50 with the longer interval, plain v0 memory recognizes B in only 3/8, against 8/8 for the
  best configuration. In v1b it was 7/8 vs 8/8, so the machinery matters more as gaps grow.
- The one-back return C is recognized 8/8 at W=50.
- The first context A behaves as the cold-start account predicts: 5/8 at W=50, 0/8 at W=10.

## 2026-09-25 — The two-back test passes: memory recognizes a context the substrate has dropped, and each world change opens a rehearsal window for memory (causally tested)

**Data and code:**
- Substrate world v1b = A→B→C→A→B (`notebooks/brian2/v1b_schedule_data/`, seeds 34000-34007,
  8/8 completed; v1's frozen runner with only the schedule swapped, verified).
- Readout `notebooks/integration/two_back_test/run_two_back.py`. Predictions TB-P1..P4 and a
  decision rule were in its docstring before any v1b data existed. It was dry-run on v1 data
  first, validating the exact alive-at-swap check against known results.
- Handoff lesion: `run_handoff_lesion.py`, with predictions before running.
- Memory: best configuration (stability-gated creation TAU 0.9, gated content consolidation
  η 0.1, absolute floor THETA), in the fork, with `src/` untouched.

**Why v1b:** it's the unconfounded test of the architecture's central bet (the substrate holds
one context back, so memory carries older ones). The final return is to B, which is two back
(C and A intervened) and is *not* the first context. v1's only two-back return was to A,
confounded with first-context weakness and eviction.

**Predictions and outcomes:**
- **TB-P1 (the substrate is one-back): CONFIRMED.** Holders of B at its return average 0.38 per
  seed (residue only). The substrate alone doesn't hold B.
- **TB-P2 (clock control): CONFIRMED.** Clean-input memory recognizes B 8/8 at W=50 and 0/8 at
  W=10 (evicted).
- **TB-P3, the question (substrate-fed best memory at W=50 recognizes B in ≥ 5/8): CONFIRMED,
  8/8.** By the decision rule stated in advance, the substrate/memory split works in principle.
- **TB-P4 (best beats v0 at W=50): technically confirmed, but marginal:** 8/8 vs 7/8. At a 50 s
  clock plain memory nearly suffices. The machinery matters at 10 s.

**Unpredicted, then tested causally: the handoff.**
- At W=10, clean-input memory *evicts* B (0/8 alive), yet substrate-fed best memory keeps B's
  original entry alive in 6/8 and recognizes it in 6/8.
- **Timing:** every one of B's 33 lull-period wins falls within 100 s after a world change
  (median 20 s), and none fall in settled periods.
- **Concordance:** survival matches reactivation at the C→A change exactly (8/8).
- **Proposed mechanism:** at C→A the substrate releases its one-back hold on B. For a moment its
  query points at B, because the C-tracking neurons lose drive and the B-holders are relatively
  active. With creation blocked by the stability gate, memory retrieves B's existing entry, and
  that resets its eviction clock.
- **Lesion test** (predictions before running): the identical run, except that wins on
  transitional steps (stability gate closed, label-free) don't reset staleness.

  | | W=10 B alive / genuine | W=10 A alive | W=50 B alive / genuine |
  |---|---|---|---|
  | best | 6/8 / 6/8 | 8/8 | 8/8 / 8/8 |
  | best + handoff lesion | **1/8 / 1/8** | **0/8** | 8/8 / 8/8 |

  HO-P1 (the lesion kills survival at W=10), HO-P2 (no effect at W=50, where eviction isn't in
  play) and HO-P3 (accuracy change at most 0.02; observed 0.008) are **all confirmed. The handoff
  is causal.**
- **CORRECTION (2026-09-26, direct check, `experiments_integration.md` author):** "the query
  points at B" overstates it. Measured directly at the C→A change (v1b, W=10), the query reads
  mostly "not-C" (C projection −0.42 to −0.59). A (incoming) and B (released) are both positive,
  and B is the largest in only 4/8 seeds. What actually happens: **in the first 100 s after a
  change, retrieval sweeps across several existing entries.** Original entries of A, B and C win
  20 / 16 / 12 steps, plus 32 for new entries. Every winner's eviction clock resets. **B's
  original won at least one of those steps in exactly the 6/8 seeds where it survived.** The two
  without a B win (34004, 34006) are the two that died. So each change opens a *rehearsal window*
  over stored memories, and the substrate's part is the unsettled "not-departing" query that
  partly overlaps several stored contexts, the released one included. The lesion result is
  unaffected: blocking transitional refresh kills survival. "Handoff of the released context" is
  the specific case, and "rehearsal window at every change" is the accurate general statement.
- **It's the same mechanism as the two-back capture failure** (toy v0). At a change, the query
  points at the context the substrate is holding. Brief, with creation blocked, it's a rehearsal
  that keeps the memory alive. When memory *creates* an entry from it, or it persists, it's a
  capture. The stability gate is what turns one into the other, which gives that gate a principled
  job beyond "avoid junk entries".
- **Design consequence (the rule being tested in v1c):** memory has to bridge only the time from
  the substrate *releasing* a context to its return, not its whole absence. The requirement
  becomes staleness_threshold x clock ≥ release-to-return interval.

**The first context is a cold-start artifact, not an ongoing limitation.** It was diagnosed on v1
first. With the best configuration, A's original entry is *alive* at its two-back return in 8/8
seeds but wins in only 1-2/8. The entry is a poor A prototype (cosine 0.29-0.72 to A), while the
returning settled A queries sit at about 0.88. So creation fires and a duplicate wins.
- The cause is in the readout. Settled query quality against the true context prototype is
  **0.31-0.34 in phase 1 in every world, against 0.87-0.96 in every later phase.**
- H = Σ(r_j − r̄)·w_j is a contrast. Before the first change the whole population learns the same
  pattern, so the centered rates cancel what everyone shares, and the context shows up only
  through a weak second-order covariance.
- After the first change, one-back retention keeps the population split permanently, so contrast
  is always available.
- **So retention is the price of a readable code.** The 40-55% of neurons holding the previous
  context supply the contrast the readout needs.
- The first-context weakness affects exactly one context in a lifetime, the first. It's
  deliberately **not being chased** with readout knobs.

**Robustness** (`run_two_back_sensitivity.py`, predictions before running): each best-config
setting was varied one at a time (TAU 0.8/0.95, η 0.03/0.3, floor 0.4/0.6).
- **SENS-P1 (B two-back genuine ≥ 6/8 at W=50 at every neighbour): CONFIRMED, 8/8 at all 7
  settings.**
- **SENS-P2 (W=10 survival varies more across neighbours): REFUTED.** It's 6/8 at every setting,
  so the handoff is insensitive to these parameters.
- Absorption reappears only where the absolute-floor account says it should: floor 0.4 (1 event)
  and η 0.3 (2 events).
- First-context recognition moves 2-5/8 across settings at W=50, still weak, as expected from the
  cold start.

**Net:** the complementary-systems split works in principle, with the substrate one-back and
memory carrying older contexts. It works via a handoff at release, and it's limited by memory's
horizon from release to return. What's left for the clock is a requirement Jasper sets (how long
after release should a context stay recognizable?), not a mechanism question.

## 2026-09-25 — Absolute-match gate: absorption eliminated and first-context recognition restored, at a small cost where low-match drift had been helping

**Code:** `notebooks/integration/content_consolidation/run_absolute_gate.py` (predictions in the
docstring before the first run); `absolute_gate_summary.json`. The fork gained an optional
`match_floor`: consolidate only if the winner's current content has cosine ≥ match_floor with the
query. It's additive, default off. The "combined" row reproduced the committed
`combined_summary.json` exactly, so the earlier results are unchanged. **No new parameter:**
match_floor = THETA = 0.5, the same line the creation rule uses for "novel".

**Predictions and outcomes (combined = stability gate TAU 0.9 + gated consolidation η 0.1; "+ abs"
adds the floor):**
- **AMG-P1 (absorption → 0 everywhere): CONFIRMED.** 0 in all 3 worlds x 2 clocks (combined alone
  had 1 / 6 / 3 at W=10).
- **AMG-P2 (v1 C one-back genuine ≥ 7/8 at W=50 and ≥ 5/8 at W=10): SPLIT.** 8/8 at W=50 (held);
  4/8 at W=10 (combined alone had 6/8).
- **AMG-P3 (A→B→A first-context recognition back to ≥ 5/8 at W=10): CONFIRMED,** 1/8 → 5/8. The
  lone A entry is no longer consolidated toward B queries early in phase 2, which is exactly the
  mechanism the previous entry proposed.
- **AMG-P4 (two-back no worse than 1.5x combined): 3 of 4 worlds.** A→B→C W=10 is 36 vs 21, 1.7x,
  a miss. The others are within 10%.

**Reading:** the floor removes the harmful case, where an *established* memory gets dragged toward
a query it doesn't match. But it also blocks some helpful low-match drift, where fresh
transitional entries get reshaped into the real context. That's where the two misses come from.

**Best configuration so far** (stability-gated creation + ambiguity-gated content consolidation +
absolute floor):
- no absorption;
- first-context 1-back recognition 5/8 (W=10);
- non-first-context 1-back recognition 8/8 (W=50) and 4/8 (W=10);
- the first context at two back is still not recognized (1-2/8), since at W=10 the clock/eviction
  limit also applies.

**Not run (next candidates):**
- apply the floor only to *established* entries, so unproven entries drift freely (mirrors how
  strength earns protection);
- the same absolute condition on the *strength* bias in retrieval (untested; strength-off didn't
  change the two-back capture, so lower priority).

## 2026-09-25 — Content consolidation (forked memory): fixes captured entries; combined with a creation gate it gives reliable recognition of a returning non-first context, and exposes that the ambiguity gate is blind to novelty

**Code:** `notebooks/integration/content_consolidation/`. The memory fork is
`episodic_content_fork.py`, a `ConsolidatingEpisodicMemory` subclass. **`src/hopfield/` is
untouched** (Jasper: "distinctly fork it"). The experiments are `run_content_consolidation.py` and
`run_combined.py`, with predictions in each docstring before its first run; outputs go to
`*_summary.json`.

**The fork:** after each step the winner's content moves toward the query, p ← unit(p + α(q − p)).
In the gated variant α = η(1 − g), using the same top1-top2 ambiguity gate that already governs
strength. In the ungated variant α = η. Everything else is inherited unchanged. **Sanity check:**
at η=0 the fork reproduced v0 exactly (winner ids, reports, creations) on every seed of all three
worlds at both clocks. The script asserts this before running anything else.

**Content consolidation alone. Predictions and outcomes:**
- **CC-P1 (genuine recognition rises): REFUTED.** A→B→A return 1/8 → 0/8; v1 C return (W=50)
  2/8 → 3/8; v1 A return 2/8 → 1/8. Checking what wins the return phase shows why. The first
  transitional query of a return spawns a *new* entry before the query settles, and that
  duplicate wins the return phase (A's return: 8/8 seeds at W=10, 7/8 at W=50, even though at
  W=50 the original is still alive).
- **CC-P2 (two-back reports cut to at most 40 at W=10): SPLIT.** v1 115 → 34 (confirmed), A→B→C
  126 → 77 (refuted). At W=50 the cut is large (47 → 15, 62 → 11), and accuracy rises (e.g. v1
  W=50 0.868 → 0.958). Captured "not-old" entries drift to the real context.
- **CC-P3 (ungated produces more absorption than gated): REFUTED.** Zero absorption in every
  variant alone, and ungated is as accurate or slightly more so. On its own in these worlds, the
  gate makes no difference.
- **CC-P4 (the clock problem is untouched): CONFIRMED.** Clean arm, v1, W=10: A two-back 0/8
  (evicted).

**Combined: stability gate (TAU=0.9) plus gated consolidation (η=0.1).**
- **COMBO-P1: SPLIT.**
  - Recognition of a returning *non-first* context becomes reliable: **v1 C one-back 8/8 at
    W=50 and 6/8 at W=10.** That's up from 3/8 and 4/8 with consolidation alone, and 6/8 and 3/8
    with the gate alone. It's the first reliable genuine recognition in the coupled system.
  - The *first* context still fails: A→B→A return 1/8 (the gate alone gave 5/8); v1 A two-back
    2/8 at W=50.
- **COMBO-P2 (no worse than consolidation alone on two-back): mostly.** A→B→C W=10 is 21 against
  77; v1 W=10 is 39 against 34, a marginal miss.
- **COMBO-P3 (v1 W=10 A two-back stays at most 2/8): CONFIRMED,** 1/8.

**New failure, found only in the combination: absorption (1, 6 and 3 events at W=10) is the
failure CC-P3 predicted, and the mechanism was checked directly.**
- In every event in A→B→C (16 steps), the query did *not* match the entry being consolidated.
  Best-match cosine ranged from −0.37 to +0.41, against a median of 0.99 over all consolidation
  steps. Yet the ambiguity gate read low (g 0.00-0.41), so consolidation ran at 59-100% of full
  rate.
- **The top1-top2 gate is relative, so it can't see novelty.** When a novel query arrives and the
  creation gate forces memory to retrieve anyway, the best of several bad matches looks
  "unambiguous". Several events occurred with a single entry in memory, where the toy sets g=0
  (my convention). The two-entry events show the same pattern, so it isn't only that convention.
- The validated retrieval gate never met this: with oracle creation a true match always existed.
- The same blindness applies to the validated *strength* bias. Nothing here has shown it
  mattering there yet (strength-off didn't change the two-back capture), but it's the same gate.
- Also explains, at least in part, why the combination *hurt* A→B→A first-context recognition
  (5/8 → 1/8). Early in phase 2, memory holds only the A entry, the gate forces its retrieval, and
  it's consolidated toward B.

**Net:**
- Content consolidation plus delayed creation gives reliable recognition of returning contexts,
  except the first one.
- Every remaining failure traces to three things:
  1. the first context being learned while the substrate is still forming it;
  2. the clock/eviction horizon;
  3. an ambiguity gate that can't tell "unambiguous" from "novel".
- **Next hypothesis (not run):** gate content (and strength) by *absolute* match quality as well
  as relative ambiguity, e.g. no consolidation unless best-match cosine is at least THETA. It
  should remove absorption without losing the recognition gains.

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

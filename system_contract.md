# System contract — what the whole thing has to do (DRAFT v0, 2026-09-25)

**Status:** draft for Jasper to correct. Written from `principles.md`, `CLAUDE.md`, the three logs
and the code in `src/`. The framework doc (`developing_mind_framework_v5.docx`) was *not*
consulted, so anywhere this has to guess how pieces connect, it says so and lists the options in
**Open questions** rather than picking one. Correct those against the framework doc before
building on them.

**Why this exists.** Each mechanism has been validated on its own, to a high standard. Nothing
has been tested *between* them: the Hopfield layer (PyTorch) and the spiking substrate (Brian2)
share no representation, units, or clock, and the ESN was never wired in. This file is the
end-to-end behavioural spec those parts are meant to satisfy together. It has two jobs:

1. Say what the **first cross-substrate coupling toy** will be examined on (section 4) before
   anything is built. B1-B5 are descriptive, with no thresholds (Jasper's call: exploratory work
   collects everything and claims afterwards). Only P1/P2 are predictions on record.
2. Turn "is thread X ready to graduate?" into a checklist. A thread is ready when every property
   the interface needs from it is either characterized or explicitly marked as not needed
   (section 6).

Scope: this is integration of pieces that are already validated, which is the next step in
dependency order. It is **not** the metacognitive layer, sub-brains, or curiosity. Those stay out
of scope for the reasons in `principles.md`.

---

## 1. The parts, and what each is validated to do

| part | where | validated behaviour | its clock |
|---|---|---|---|
| Spiking substrate (STDP, homeostatic scaling, ambiguity-gated lateral inhibition) | `src/brian2_stdp/`, `experiments_brian2.md` | Learns which inputs are correlated with no supervision. Population readout is stable under constant individual churn. Re-tracks a changed world within minutes. A bounded fraction of units keeps the old pattern. | simulated seconds |
| Attractor / retrieval layer (modern-Hopfield softmax over a stored pattern matrix, strength bias gated by ambiguity) | `src/hopfield/two_layer.py`, `experiments.md` | Content-addressed retrieval. Strength only breaks ties (content match 95%+ with savings kept). | steps (one query = one step) |
| Two-layer memory (w_fast τ≈50 steps, w_char τ≈2000 steps, saturating) | `src/hopfield/two_layer.py` | Graded primacy and recency, no monopolization, robust across a sweep. | steps |
| Episodic layer (grow and prune, gated eviction) | `src/hopfield/episodic.py` | Bounded size. Eviction protects consolidated entries without rescuing them forever. **Open: can't tell "abandoned" from "between its periods of relevance" (check (b)).** | steps |
| ESN | `src/esn/` | Diagnostic only. Memory horizon is about 37 steps, below every two-layer decay constant. | steps |
| Mycelium substrate, curiosity, metacog, sub-brains | — | Not built. | — |

---

## 2. The interfaces: what flows where, and how much is known

| boundary | what would flow | status |
|---|---|---|
| world → substrate | presynaptic spike trains with changing correlation structure | **Built.** `spikes.build_phase_switching_input` (2 blocks) and the local multiblock builder in `novelc_data/` already produce a world with contexts. This is ready to use as the integration toy's world. |
| substrate → attractor/memory | ? (see section 3) | **Unknown. The central gap.** |
| attractor ↔ two-layer memory | per-pattern w_fast/w_char biasing retrieval | **Validated** (same module). |
| memory → episodic write/evict | "is this new?" for creating an entry, "is this abandoned?" for evicting one | **Partly built.** Eviction is gated. What triggers *creation*, and what would tell a lull from abandonment, is undefined. |
| substrate/memory → substrate (top-down) | ? | **Not in the minimal loop.** Listed as Q3. |
| a step ↔ simulated seconds | a mapping between the two clocks | **Undefined.** See section 5. |

---

## 3. The substrate → memory interface: three readings (not decided here)

Two of the project's own statements pull in different directions. The named decision in
`principles.md` chooses the population readout "for its role *feeding* a future Hopfield layer",
which makes Hopfield a separate layer downstream. `CLAUDE.md` describes a "Hopfield attractor layer
as *emergent* self", which suggests the attractors arise from the substrate. The code has the
Hopfield layer as an explicit PyTorch pattern matrix. These readings differ:

- **(I) Substrate as encoder.** The substrate's population state at a moment (e.g. the firing-rate
  vector across its N neurons over a window) is the *query* into the Hopfield layer, and stored
  patterns are past population states. The substrate does perception and the Hopfield layer does
  memory. This is the simplest to build.
- **(II) Substrate weights as memory.** Each neuron's learned receptive field (its weight vector
  over the inputs) is a stored pattern, and the attractor layer reads and consolidates what the
  substrate learned. This is closer to "identity is substrate-modification". Retainers become
  literal stored memories.
- **(III) Attractors inside the substrate.** Recurrent spiking dynamics *are* the attractor layer,
  and the PyTorch Hopfield is a stand-in for them. This is the most literal reading of "emergent",
  and the furthest from what's built.

The first toy can provisionally pick (I) or (II) and say so. It shouldn't treat that pick as the
answer (Q1).

**A gap under all three readings: the validated "stable population readout" uses labels the
system doesn't have.** Every STDP stability result is measured as correlated-group mean weight
minus uncorrelated-group mean weight, and knowing which group is correlated is experimenter
knowledge. A downstream layer only sees label-free quantities: rate vectors, weight vectors, spike
trains. Nobody has checked whether a *label-free* readout is stable under the individual churn the
named decision tolerates. It's cheap to check on existing data before building anything:
for each saved run, is the set of neuron weight vectors (or the windowed rate vector) stable within
a phase and distinct between phases, without using block labels? (Interface property S1 below.)

**S1 result (2026-09-25, arc 07; analysis of existing runs, predictions on record first).**
- **Weights alone (reading II as one population pattern) are a history code.** The weight state
  reflects the current context *plus* the one just left, because of one-back retention. Phase-3 A
  looks more like the adjacent phase-2 B than like phase-1 A in 8/8 seeds. In A→B→C, C looks more
  like B than like A in 8/8.
- **Rates alone (reading I) say "now vs just before", not "which context".** Rates are strongly
  anti-correlated with the previous context, but their similarity to a non-adjacent context is
  noise. It ranges from −0.34 to +0.91 for the same context and up to +0.99 for a *different* one.
- **Activity projected through the learned tuning, H = Σ_j (r_j − r̄) w_j, carries context
  identity.** It has three clean blocks in the 3-block rig, no one-back contamination, and
  right-signed same-vs-different similarity in 16/16 seeds. Caveats:
  - In a 2-block world it's essentially one-dimensional, so part of its contrast there comes from
    the construction.
  - The same-rig lull test needs A→B→C→A.
  - The first context learned is represented weakly by *every* label-free readout. When all
    neurons agree there's little cross-neuron structure to read.

Implication for Q1, a suggestion and not a decision: neither reading alone. The weights hold *what
has been learned* (memory contents, e.g. per-neuron receptive fields). Activity through those
weights says *what is current* (the query). Also, the Hopfield layer has only ever been validated on
64-d random unit vectors (cosine ≈ 0 ± 0.126). These readouts are low-dimensional (effective
dimension about the number of contexts) and their similarities are nowhere near zero, so memory's
`beta` and `gap_scale` must be re-grounded from the new statistics before any behaviour is looked
at.

---

## 4. The minimal end-to-end loop, and what it has to show

**World:** a context sequence with returns and novelties, e.g. A → B → A → C → B, built from the
existing phase-switching input. **Loop:** world → substrate → label-free readout → memory
(retrieve, update w_fast/w_char, episodic create/evict) → a system-level report of "what context am
I in" at each step. No top-down feedback in v0.

Behaviours the loop is examined on. These are descriptive, with no pass bars. Each lists what
already bears on it:

| # | behaviour | what's known | unknown |
|---|---|---|---|
| B1 | **Tracking:** after a world change, the current representation follows the new world | Substrate re-tracks in 16/16 (N=3) and 8/8 (N=7), median about 85-90 s, tail to about 1000 s | Whether the *label-free* readout tracks as well (S1) |
| B2 | **Recognition:** when a known context returns, memory retrieves the right item, fast | Substrate: the retainers already hold it and speed up on its return (+5.8 Hz). Memory: retrieval by content works | Whether the query produced right after a return is clean enough to retrieve correctly |
| B3 | **Novelty without confabulation:** a new context makes a new entry, not a confident retrieval of an old one | "Strength breaks ties" holds *within* the memory layer | Whether it survives an ambiguous query coming *from the substrate* (see P1) |
| B4 | **Retention without lock-in:** a context's memory survives its lull but doesn't block new learning | Substrate: retainers hold patterns through lulls. Memory: two timescales. Episodic check (b) fails on its own | Whether the substrate's signal (P3) closes check (b) |
| B5 | **Bounded resources:** neither memory size nor substrate capacity grows or saturates without limit | Episodic size is bounded. Substrate: about a third of units retain at each swap | Whether retainers release when their pattern never returns (novel-C) |

---

## 5. Clocks: the numbers that have to line up

Nothing defines how many simulated seconds a memory step is, and the choice changes which layer is
fast relative to which. Here are the known timescales on a common axis, **if one step = 1 s** (an
assumption made to put them side by side, not a proposal):

| thing | value |
|---|---|
| w_fast time constant | ~50 steps |
| substrate re-learning after a swap (median), **weights** (phase-aligned gap reaches 0.3) | ~85-90 s |
| episodic staleness threshold | 150 steps |
| Hopfield phase-change misretrieval transient | ~250-350 steps |
| substrate swap dip recovery, **firing rate** (population rate back to baseline) | ~300 s |
| substrate settling under fixed input (13mV/1.5), **weights** | <100 s (N=3), ~250 s (N=7) |
| slowest substrate return, **weights** | up to ~977 s |
| w_char time constant | ~2000 steps |
| ESN memory horizon | ~37 steps |

At 1 step = 1 s, the substrate's post-swap recovery (~300 s) and the memory's phase-change
transient (~250-350 steps) are the same order of magnitude. At 1 step = 10 s, the substrate would
be ten times faster than memory. The integration toy should record which mapping it uses and ideally
try two (Q2).

---

## 6. What the substrate has to hand upward, and what that means for graduating STDP

**Output spec (draft).** These are the properties the interface needs from the substrate:

| # | property | status |
|---|---|---|
| S1 | A label-free readout, and its stability under churn | **Answered (arc 07; same-rig lull test passed 8/8 in v1).** All three readouts are stable within a context (≥0.95) at 10, 50 and 200 s windows. Weights alone are a *history* code (current + just-departed context). Rates alone say "now vs just before" but can't identify a context. Activity projected through tuning (H) carries context identity, with right-signed same/different similarity in 16/16 seeds. The clean same-rig lull test for H needs A→B→C→A (v1). |
| S2 | Tracking latency after a change | **Known** (one operating point, n=8). |
| S3 | Whether old patterns persist, and how many units they cost | **Known for 3 phases (novel-C).** One-back retention: the just-departed context is held by about 40-55% of units (rig-dependent), older ones are released at the next change, and a small lock-in residue (~5% of units per extra change) never releases. Whether the residue accumulates over long schedules is **unknown**. |
| S4 | A change / recognition signal | **Known, and it has to be read as a vector.** The scalar population dip marks *that* the world changed but can't tell a return from a new pattern. Recognition is per-neuron: holders speed up only when their pattern returns (p=0.00016). |
| S5 | Operating point | **Known tension.** Reliability and ongoing identity churn haven't been shown to coexist. All non-stationary data is at 13mV/1.5, so use that for the toy. |
| S6 | Timescales in seconds | **Known.** The step mapping is an interface question, not an STDP one. |

**Open STDP items that don't block the interface** (worth doing some day, but no answer changes
what gets handed up): the exact retainer fraction and its dependence on inhibition normalization;
what mediates the slow return tail; the distribution over individual-level regimes; the
`cpp_standalone` mismatch; `strong_tight_gate` at N=7 (unless the toy picks that point).

**So "is STDP ready?" comes down to two items: S1 and novel-C.** (Both are now answered, S1
provisionally. Its one missing test, a same-rig lull test for the H readout, needs the A→B→C→A
schedule, which is the coupling toy's own v1 world anyway. By this section's criterion STDP can
graduate now, with that check carried into v1. Whether to call it is Jasper's decision.) Once both
are answered, the
substrate's contribution to the first coupling toy is specified. STDP stops being the default next
thing. It isn't closed, and it reopens whenever the toy raises a substrate question.

---

## 7. Predictions the coupling toy can test that neither layer can alone

These come from combining existing findings. They were hypotheses when written. **Outcomes from
toy v0 (`experiments_integration.md`):**
- **P1:** confirmed in shape, but the mechanism differs. It isn't strength-driven: after a
  second novel change, memory captures a transitional "not-old" query and reports the *two-back*
  context, and that happens with strength switched off too.
- **P2:** confirmed. 1-back returns are reported within one step, 2-back returns aren't. But
  reusing the *original* entry mostly fails.
- **P3:** the dip-as-creation-gate version doesn't work as built.
- **P4, P5:** not tested yet.

- **P1: stale retrieval compounds across layers after a *novel* change.** Right after a swap, the
  substrate is mixed: retainers still tuned to the old pattern, trackers mid-relearn. The query it
  produces is intermediate between old and new, so it's ambiguous. Ambiguity is exactly when the
  gate hands control to strength, and strength (w_char) favours the old pattern. Result: confident
  retrieval of the context that just ended. Each layer passes its own tests, and together they
  could produce the confabulation failure `principles.md` warns about. Measure: how long after a
  novel swap retrieval stays on the old context, against **two** controls:
  1. **Clean patterns (no substrate).** This gives the memory layer's own baseline, and note that
     it already has a 250-350 step misretrieval transient on clean input.
  2. **Blended patterns at the mixing fraction the substrate actually produces** (the other
     session's suggestion). Without it you can't separate "the substrate's dynamics cause stale
     retrieval" from "ambiguity alone does".

  P1 is about the substrate only to the extent the substrate arm is worse than the blended arm.
- **P2: recognition helps after a *returning* change.** On a return, the holders speed up, which
  should sharpen the query toward the returning pattern and make B2 fast. P1 and P2 predict
  opposite behaviour for novel and returning transitions. The toy should report them separately.
- **P3: the substrate's change signal as the missing context signal for check (b).** The rate dip
  marks *that* the world changed. The holders firing up marks *which known* context came back.
  A held-but-quiet pattern (its retainers still exist) is "between beats", not abandoned. This is a
  design option for episodic create/evict: creation gated on an uncovered dip, eviction protection
  while retainers exist. **Update after novel-C:** the familiarity reading holds only per neuron.
  The scalar dip can't tell a return from a new pattern, so any version of this has to read
  per-neuron activity. That favours reading I (a rate vector) over a summed signal. One-back
  retention also caps it: only the *most recently departed* context stays held, so an older one
  (A in A→B→C→A) would be treated as new. Whether that's acceptable is a requirement for Jasper to
  set, not something the data decides.
- **P4: the readout choice decides what memory sees of suppressed units.** At 13mV/1.5 the losing
  neurons are held down by inhibition (suppression-based reliability). A rate readout (reading I)
  barely sees them, while a weight readout (reading II) sees their full, possibly stale, structure.
  Readings I and II may disagree most exactly here.
- **P5: context has to live in weights, not in raw dynamics.** The ESN result (raw recurrent memory
  of about 37 steps, shorter than any memory timescale) and the retainers (context held in
  *weights* across a ~1000 s lull) point the same way. If that holds, reading II (weights as
  memory) has an argument that reading I lacks.

---

## 8. Open questions (decide against the framework doc, not here)

- **Q1.** Interface reading I, II or III? How does "feeding a Hopfield layer" (`principles.md`)
  square with "Hopfield attractor layer as emergent self" (`CLAUDE.md`)? Is the PyTorch Hopfield
  layer the thing itself, or a stand-in for attractor dynamics?
- **Q2.** How many simulated seconds is one memory step? Or should memory run on its own event
  clock (one step per readout window, per detected change)? **Now constrained (toy v0):**
  memory's retention horizon is staleness_threshold (150 steps) x the clock. At 10 s per step,
  memory with *perfect* input forgets a context after a 2000 s lull (0/8); at 50 s it keeps it
  (8/8). The substrate is one-back whatever the clock, so memory's horizon must exceed the lulls
  the system is supposed to bridge, or complementary systems fail by construction. **Refined
  (two-back test + rehearsal lesion):** each world change opens a rehearsal window. While
  creation is blocked, retrieval sweeps stored entries and refreshes the winners, and the
  substrate's release of a context is typically one such window for it. So the horizon only has to
  cover the time from the last rehearsal (in practice, *release*) to return, not the whole absence
  (confirmed in v1c: 2000 s release-to-return against a 1500 s horizon gives 2/8 survival, vs 6/8
  at 1000 s). **What remains is a requirement for Jasper:** how long after the
  substrate lets go should a context still be recognizable? That number x the clock sets
  staleness_threshold.
- **Q3.** Does memory feed back into the substrate (top-down bias, replay, consolidation into
  weights)? It's left out of v0 deliberately. The framework doc probably says something.
- **Q4.** What decides episodic *creation*: a substrate novelty signal (P3), the curiosity layer,
  or something else? If it's curiosity, check (b) waits on a layer that's out of scope. **Now the
  central open question (toy v0):**
  - The episodic layer never had a creation rule; every validated notebook used an oracle.
  - Every failure the toy found traces to *when and from what* memory creates an entry.
  - The dip gate (P3) doesn't work in its simple form, since it flags only about 2% of steps.
  - A strict stability gate helps but is fragile.
  - Related: entry *content* never consolidates, only strength does. A forked memory with gated
    content consolidation, plus a stability gate on creation, gives reliable recognition of a
    returning non-first context. It also showed that the ambiguity gate needs an
    absolute-match condition, since it can't see novelty.
- **Q5.** Is the SNN *the* substrate, or a stand-in for the mycelium substrate the framework
  describes?

---

## 9. Suggested order

1. Jasper corrects this draft against the framework doc, especially Q1 and Q2, or picks
   provisional answers for the toy and labels them provisional.
2. ~~S1 check on existing data~~ **Done** (arc 07). H carries context; weights alone are a history
   code; rates alone are a now-vs-before contrast.
3. ~~Read novel-C~~ **Done** (arc 05). One-back retention, a small lock-in residue, per-neuron
   recognition.
4. **Coupling toy v0: DONE** (go given 2026-09-25; `experiments_integration.md`). What follows is
   the original plan, kept for reference. This needs Jasper's explicit go: `CLAUDE.md` says no cross-substrate
   building unless asked, and this draft doesn't count as asking.
   - It can be **offline**: replay the saved substrate runs into the existing `two_layer` +
     `episodic` memory. No new sims, and no live Brian2↔PyTorch coupling until feedback (Q3)
     matters.
   - Readout: H as the query, per S1.
   - Three arms: substrate readout, clean patterns, and blended patterns at the substrate's mixing
     fraction.
   - Re-ground `beta`/`gap_scale` from the new query statistics first, the same way the original
     `gap_scale` was grounded in an empirical gap median.
   - Report B1-B5 descriptively, and P1/P2 separately for novel and returning transitions.
5. **v1 world: DONE**, run as A→B→C→A→C so that swaps 3 and 4 contrast a 2-back and a 1-back
   return (arc 05). The complementary-systems test fails as built: clock/eviction plus
   frozen-snapshot entries (`experiments_integration.md`). Original plan below. This one schedule
   gives:
   - the same-rig lull test for H;
   - the one-back prediction (the substrate should *not* instantly recognize A);
   - whether the lock-in residue accumulates;
   - the complementary-systems test: does memory recognize A after the substrate has dropped it?

# Developing Mind — Design Principles

Read this when a task needs to know *why*, not just *what happened*.
CLAUDE.md has the build history. The full 14-section framework document
(developing_mind_framework_v5.docx) is not in this repo — it's the
project's primary design artifact, kept outside the codebase. This file
is the middle layer: the load-bearing commitments that should shape how
you diagnose failures and propose fixes, so a bug in one mechanism
doesn't get misread as a verdict on the whole approach, without needing
the full doc reloaded for every task.

## The core commitment

Identity here is substrate-modification through lived experience, not
weights frozen at training time and read at inference. Every mechanism
in this system should be judged against that: does it let experience
actually change the substrate, in a way that's earned and path-
dependent, or is it a workaround that only looks like memory?

## Two design principles, now empirically earned, not just asserted

**Stability-plasticity is structural, not tunable.** A single decay
constant cannot simultaneously preserve meaningful history and stay
open to new experience — this was proven directly (decay=0.01 erases
everything, decay=0.001 locks in permanently, no middle setting
escapes it). The fix is architectural (two timescales), not a better
number. If a future mechanism hits an analogous wall — one knob being
asked to do two incompatible jobs — the fix is probably splitting the
mechanism, not searching harder for the right value.

**Strength breaks ties, never overrides matches.** Any mechanism where
accumulated strength (basin depth, consolidation weight, resonance)
can influence a retrieval or write decision has a specific failure
mode: strength winning over what's actually being asked or
experienced. This produces a system that's confidently, fluently
wrong — the same shape as the confabulation failure mode, just at the
mechanism level instead of the language level. The fix, validated in
the two-layer memory work: gate strength's influence by how ambiguous
the content-level decision already is. Full influence only when
content genuinely can't decide; suppressed when it can. Expect this
principle to matter again anywhere else strength-like signals get
added — episodic write/prune decisions, thalamus gating, sub-brain
resonance activation.

## Named decision: what "stability" means for the STDP layer (2026-07-20)

**Status: provisional, scoped to the single-neuron/many-synapse population readout actually
tested — not a closed architectural commitment.** The rejection of (a) below is a fact about
the system, not scoped to any one scale, and doesn't need revisiting. The adoption of (b) is
earned at the scale tested (many synapses onto one neuron) and extrapolated, not proven, to
the scale this decision is meant to describe (many neurons forming a joint population code —
the scale that actually feeds a Hopfield layer). Revisit this entry, not just extend it, if
the shared-input competitive population experiment (flagged, not started) shows the aggregate
signal doesn't stay stable once neurons genuinely compete for shared resources — two nested
competitive dynamics instead of one, and nothing guarantees the outer one inherits the inner
one's stability property for free.

**The STDP layer's stability requirement is a population-level readout that tolerates
individual-synapse churn, not individual-synapse convergence — and the data earns this now,
it isn't a default.** Two candidate definitions were on the table, following the external
review's reframe: (a) individual-synapse convergence — a given synapse settles to a fixed
weight — versus (b) a stable population-level readout tolerating individual-unit churn,
matching real cortical representational drift (population code stable, single-unit identity
not).

(a) is not just unmet, it's unmeetable in this system as built. Reversal-frequency counting
(the Apre-sweep round, confirmed again in the population extension) found direction-reversal
frequency is essentially Apre-invariant — ~140/synapse/300s at N=1, ~500/synapse/1000s at
N=5 (the same rate once normalized for duration) — at every Apre value tested, including the
"stable" one. Every individual synapse keeps reversing direction on this timescale regardless
of Apre; low Apre just keeps each excursion small. There is no setting anywhere in the tested
range where a synapse's weight settles to a fixed value. Requiring (a) would judge the entire
tested range "unstable" and throw away the real, useful distinction already earned between
Apre=0.005 and Apre=0.02 — not a workable spec for anything downstream.

(b) is what has actually been measured as stable this whole arc, without ever being named as
the choice it was. The group-mean gap (correlated-group mean weight minus uncorrelated-group
mean weight) — a population-level readout over the synapses feeding one postsynaptic neuron —
stayed bounded away from zero in every regime tested: 0/20 replicates near zero at Apre=0.005
(settled mean +0.346, min +0.148 across the population extension), and even at Apre=0.02,
where the readout never settles and fluctuates continuously, direction never flipped across
20 replicates (closest approach +0.002, momentary). Individual synapses churn constantly
underneath this the entire time. That's stable-population/unstable-unit, exactly the
reviewer's proposed reframe — now with two independent confirmations (the 8-seed/5000s N=1
ensemble, and the 5-neuron population extension) instead of one.

**Choosing (b): this mechanism needs to guarantee a stable population-level readout, not
individual-synapse fixed points, for its role feeding a future Hopfield layer.** The empirical
case stands on its own, on the two paragraphs above alone: reversal-frequency invariance rules
out (a) as achievable at all, and the bounded group-mean gap confirms (b) at the scale tested.
The validated two-layer/episodic retrieval mechanism reads accumulated, aggregate-level
evidence — content similarity plus strength — not literal frozen per-synapse identity, which
is a second, independent reason (a) was never going to be available to hand it anyway.

One more note in the same direction, offered as motivation rather than evidence — don't read
it as adding to the empirical case above: this choice is also the better fit with the core
commitment at the top of this file, substrate that keeps reorganizing under lived experience
while its aggregate meaning persists being closer to "identity through path-dependent
modification" than further from it. That's philosophical coherence, not data; it's a nice
bonus that the choice the numbers support also happens to fit the project's stated
commitments, not a reason to believe the numbers.

**What the current dataset can't settle, stated plainly rather than glossed over:** every
check so far — including the population extension — measured a population of *synapses onto
one postsynaptic neuron*, not a population of *postsynaptic neurons* forming a joint ensemble
code. The population extension deliberately used independent replicates (each neuron its own
dedicated presynaptic block, no shared input, no competition between neurons — a shared-input
design was tried first and found not to produce any inter-neuron divergence at all, see
experiments_brian2.md) because that was the cheap, well-supported test actually available.
Whether multiple postsynaptic neurons sharing or competing over common input form a stable
joint population code — the scale that actually matters for "population code" in the
reviewer's cortical-drift sense, and the scale a Hopfield layer fed by many such neurons would
really depend on — has not been tested by anything run so far. Choosing (b) at the
single-neuron/many-synapses scale is earned by real data. Assuming it also holds at the
many-neuron/shared-population scale is an extrapolation, not a result, and should be treated
as one until the shared-input competitive population experiment (flagged, not started, see
experiments_brian2.md) actually closes that gap.

## How to fail correctly

Negative results are real data, not something to route around or
quietly reframe as positive. When something breaks:

1. **Find the actual root cause before touching scope.** "This
   mechanism failed" and "I forgot a saturation cap" are different
   claims — the second doesn't indict the first. Check the boring
   explanation (missing bound, wrong sign, contaminated metric) before
   concluding the architecture itself is wrong.
2. **Distrust a single check.** A metric that looked fine once (e.g. a
   monopolization ratio) can miss a failure happening in a different
   variable it doesn't measure. Pair structural checks with a direct
   behavioral ground-truth check before trusting either alone.
   **For any dynamical/stochastic mechanism (STDP, consolidation,
   anything evolving over many steps): a single summary number —
   endpoint value, ratio, snapshot diff — is not sufficient evidence of
   convergence or stability on its own.** Confirmed wrong in three
   independent cases this session: retrieval-bias monopolization
   ratios stayed "healthy-looking" while content-match quietly
   collapsed (Hopfield work); an unstable STDP condition's endpoint
   weight-diff read as a clean positive number purely because of which
   seed happened to be used, while its full trajectory was chaotically
   fluctuating the entire run (Brian2); and a tightly-capped STDP
   variant showed the *largest* endpoint diff of its sweep while its
   trajectory revealed a slower random walk that simply hadn't drifted
   as far in the time given, not genuine convergence (also Brian2).
   Always plot and inspect the full trajectory before drawing a
   conclusion about whether a system has settled — this is a default
   requirement for this class of mechanism, not a nice-to-have caveat
   to remember when something looks suspicious.
3. **State falsification criteria before running the experiment**, not
   after. If you don't know in advance what result would count against
   your hypothesis, you'll rationalize whatever you get.
4. **A metric that "improved" isn't automatically evidence** — check
   whether it's measuring the thing you think it's measuring, or a
   contaminated proxy (e.g. a threshold-crossing count that's really
   just measuring how close to the ceiling something started).
5. **"Stable" in a stochastic system can mean two different things —
   don't conclude the strong one from a short window.** A process can
   look stable either because it's genuinely bounded/restoring (an
   actual force resists drift) or simply because it hasn't drifted far
   enough yet by chance in the time observed — indistinguishable from a
   short trajectory alone. Check whether the underlying process has a
   real restoring force, or is just a slow random walk that hasn't
   traveled far. First identified in the STDP reversal-frequency work
   (low `Apre` "looked" converged but showed the same switching
   frequency as everywhere else, just smaller excursions) — expected to
   generalize anywhere a system is deemed stable from a bounded
   observation window.

## What's out of scope right now, and why

Not arbitrary gatekeeping — it's dependency order:

- **Metacognitive layer** needs a working, validated memory system
  beneath it before it has anything real to model. Building it now
  means it models noise.
- **Sub-brain constellation** needs a mature, stable core to orbit.
  Building it now means it has no orientation point.
- **Episodic layer pruning** needs the ambiguity-gating principle
  above proven out on retrieval first (done) before it's safe to reuse
  for eviction decisions (not done yet) — otherwise pruning inherits
  the same familiarity-bias risk from scratch.

If a task seems to require jumping ahead of this order, flag it rather
than building around the gap.

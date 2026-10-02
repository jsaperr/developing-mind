# Developing Mind — Design Principles

Read this when a task needs to know *why*, not just *what happened*.
CLAUDE.md has the build history. The full 14-section framework document
(developing_mind_framework_v5.docx) is not in this repo — it's the
project's primary design artifact, kept outside the codebase. Notes on
where the build departs from that doc (`framework_drift.md`, referenced in a
few places) are likewise kept local and not published. This file
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

*Instance found 2026-09-28 (memory toy, `experiments_integration.md`):*
w_char was being asked to carry both primacy and recency under eviction.
Recency in w_char needs an age difference comparable to its decay
constant (~2000 steps), and eviction removes an unrehearsed entry after
~150. So short phases leave B and C unordered, and long phases evict B.
No clock or phase length fixes that. The same logic as above applies: the
likely fix is carrying recency in a different variable (staleness
already records it exactly), not a better decay_char. That's flagged as
a framework question, not chosen.

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

*Caveat found 2026-09-25 (coupling toy, one experiment; a warning, not
yet a fix):* the gate as implemented, g = 1/(1 + (top1 − top2)/gap_scale),
measures **relative** ambiguity between the two best candidates. It
can't see **novelty**: a query that matches *nothing* well but whose
best (bad) match is clearly ahead of the second reads as "unambiguous",
and strength or content influence goes to full. It was validated with
oracle-created patterns, where a true match always existed. Fed by a
real substrate, with a creation gate forcing retrieval during
transitions, it consolidated entries toward queries they didn't match at
all (best-match cosine −0.37 to +0.41 at full rate) and absorbed
established memories (`experiments_integration.md`). Wherever this gate
is reused on input that can be novel, pair it with an **absolute**
match-quality condition. "Can't decide between candidates" and "no
candidate fits" are different situations, and only the first is a tie.
Tested for *content* consolidation in the same toy: requiring cosine ≥
the creation threshold before consolidating removed absorption in the
three worlds it was first tested on and restored first-context
recognition (1/8 → 5/8). **Scope correction (2026-09-26):** later worlds
(v1c at both operating points, strong_tight_gate v1b) still show 1-8
events of a different kind. Right after a change the query reads
"not-departing", which genuinely overlaps several contexts (match
0.80-0.86), so no match floor can block it. That hazard needs a signal
that the world is still changing, which the system doesn't yet have
(`experiments_integration.md`). It hasn't been tested for the strength
bias in retrieval.

## Named decision: what "stability" means for the STDP layer (2026-07-20)

**Status: (b) is now directly tested at the multi-neuron scale across a real sample, not just
extrapolated or shown twice — within a specific, stated scope, not a closed/exhaustive result.**
The rejection of (a) below is a fact about the system, not scoped to any one scale, and doesn't
need revisiting. The adoption of (b) was earned at the single-neuron/many-synapse scale when
this entry was first written, then extrapolated (not proven) to the many-neuron/shared-input
scale this decision was actually meant to describe. That extrapolation has since been tested
directly, first at n=2 and then expanded to n=7: a genuine shared-input competitive population
(all-to-all input, lateral inhibition between postsynaptic neurons) held a stable population-
level signal for 5000s in *every* differentiating seed tested, no exceptions, while individual-
level identity behavior spanned a full spectrum underneath — near-permanent lock-in, one-time
reorganization (in two distinct sub-flavors), and, most strikingly, persistent lock-in that
*never* happens at all (constant multi-way reshuffling for the entire 5000s, the highest churn
of any seeds tested). That last regime is the strongest empirical instance of (b) found so far —
a stable population readout with *zero* persistent individual identity, not even a temporary
one (see experiments_brian2.md's "Seed expansion (n=2 -> n=7)" entry for the full typology).
Remaining scope, stated precisely: n=7 differentiating seeds is a typology, not a distribution —
the relative frequency of each individual-level regime is unknown, only their existence and the
population-signal invariant across all of them are established. Still only one parameter
combination, and only the differentiating side of a genuine bifurcation this experiment found —
the other side converges to zero differentiation entirely (basin split now measured at n=14,
~50/50), so this says nothing about population-level stability where there's no differentiated
signal to be stable in the first place. Revisit this entry again if other operating points show
a structurally different picture.

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

**What's now settled at the multi-neuron scale, and what still isn't, stated plainly rather
than glossed over:** the population extension (independent replicates, no shared input, no
competition — a shared-input design was tried first there and found not to produce any
inter-neuron divergence at all) was the cheap test available at the time, and it only measured
a population of *synapses onto one postsynaptic neuron*. The competitive-population follow-up
(genuine shared input, lateral inhibition), now at n=7 differentiating seeds, closes that
specific gap with real texture: a population of *postsynaptic neurons*, genuinely sharing input
and coupled through real competition, shows a stable joint signal across a whole spectrum of
individual-level behavior — not just "churns a bit," but ranging all the way to individual
identity never locking in at all — and does so in every differentiating seed tested, not just
the original two. What's still open: this is a typology at n=7, not a measured distribution
over how common each individual-level regime is; still one parameter combination; and the whole
thing is contingent on landing in the differentiating basin at all (measured at ~50/50 across
n=14 seeds now — see experiments_brian2.md's bistability entry). A proper distributional
estimate of the individual-level regimes, and whether other operating points show the same
bifurcation structure, remain real, unstarted follow-ups.

**A real boundary condition on the identity-persistence claim, found 2026-07-23 — stated
plainly rather than left as the original broad framing.** The genuine ongoing-reorganization
version of this finding — a laggard actually re-entering the leading tier well after initial
settling, or a population that never settles into a fixed hierarchy at all — has only been
confirmed at `strong_tight_gate` (inhib=10mV, gap_scale=1.0), a marginal, near-bifurcation
operating point where differentiation itself is unreliable (~50%, see the bistability sweep).
At a more reliable setting (13mV/1.5), confirmed independently at both n_post=3 and n_post=7
(the N-scaling curve and its step-4 extension), competition instead resolves fast and
permanently within the first ~1-5% of run duration and never changes again — real seed-to-seed
variety in final hierarchy *shape*, but zero ongoing reorganization in every seed tested.
**Reliability and genuine ongoing identity-churn have not been shown to coexist at any setting
tested so far.** They may be in real tension — possibly requiring the system to sit specifically
near a differentiation bifurcation rather than comfortably reliable on one side of it, the same
"stability-plasticity is structural, not tunable" shape as the two-layer memory's own decay
constants, applied to a mechanism that hadn't been tested this precisely before. Flagged here as
a real, falsifiable, open architectural question — not a settled one, and not something to
silently assume away when reusing this mechanism elsewhere. See experiments_brian2.md's Test A
and N-scaling step-4 entries for the data this rests on.

**Named finding: reliability can come from suppression, not dominance — and the two look
identical under passive observation but diverge sharply under active perturbation, found
2026-07-23 via a three-round perturbation-testing arc (see experiments_brian2.md).** "Robust"
is underspecified without saying robust to *what*. This system produces at least two
structurally different routes to a reliably-differentiating outcome:

- **Suppression-based reliability** (13mV/1.5 in the competitive population): the excluded
  neuron gets pushed out of firing fast, via strong lateral inhibition, almost from the start
  of the run — it barely gets the chance to accumulate correlated-synapse weight at all. This
  reads as "deep, permanent lock-in" under passive observation (zero spontaneous reorganization
  in every seed tested, at two network sizes) precisely because the loser was never really in
  the competition on weight terms — it was held down by a strong hand, not out-competed.
- **Dominance-based reliability** (`strong_tight_gate`, 10mV/1.0): differentiation is less
  reliable outright (~50% of seeds even reach a stable hierarchy), but seeds that do land there
  won it on genuinely closer, more contested weight terms.

Directly perturbing the excluded neuron's own correlated-synapse weights (bypassing the
inhibitory suppression mechanism entirely, rather than waiting for spontaneous reorganization)
reveals these are different axes that don't even rank operating points the same way: the
"more reliable, deeper-locked-in" 13mV/1.5 setting was *easier* to knock into a new hierarchy
than `strong_tight_gate` (flipped in 7/8 seeds vs 5/8, at lower-or-comparable magnitude,
replicated across two independently-seeded batches at two ladder resolutions). Once the
suppression mechanism is bypassed, there's less real resistance underneath than the setting
that was less reliable but more genuinely contested. **Passive reliability (never spontaneously
reorganizes) and structural depth (resists a direct forced perturbation) are not the same
property and must not be treated as interchangeable evidence for each other anywhere this
mechanism gets reused.** See experiments_brian2.md's perturbation-testing entries (all three
rounds) for the full mechanistic trail and data.

**Named finding: under a changing world the substrate retains exactly one context back; anything
older has to live in memory (2026-09-25, via the non-stationary arc).** This was established
across three worlds and two rigs (A→B→A at N=3 and N=7, A→B→C novel-C, A→B→C→A→C v1; 8 seeds
each at N=7, 13mV/1.5), with the one-back prediction confirmed *within* seeds in v1. See
`docs/log/brian2/05_nonstationary_correlation.md`.

- **What happens:** at every change, the population re-tracks the new correlation within minutes.
  The neurons most entrenched in the *just-departed* pattern keep it and harden (about 37-55% of
  units, rig-dependent). Neurons still holding an *older* pattern release it at that change and
  learn the new one.
- **Recognition:** when the just-departed pattern returns, its holders speed up at once (+5 to
  +8 Hz). That per-neuron response, not the summed population rate, is the substrate's
  recognition signal.
- **The residue:** a few neurons (about 2-3 per 8 seeds per change) never release and keep
  hardening. Over four changes this residue didn't accumulate, but that schedule's returns
  can mask accumulation.

*Scope caveat, 2026-09-28 (generalization step 0, arc 05):* "exactly one back" holds for long
(1000 s) phases of disjoint or 50%-overlapping contexts, including a 20-phase world with no residue
build-up. It is **not** a general law.
- **What happens after short fillers:** in check (b), fillers are 300 s long and partly overlap the
  cores. Each takes 5-6 of 7 neurons, and the next core recruits back from the *filler*, so the
  just-departed filler is dropped and older core residue survives.
- **What does generalize so far:** re-assignment is triggered by the change (never between changes).
  What decides *which* context loses neurons is open, but **not phase length**: 300 s disjoint
  phases give one-back just like 1000 s (step 1, same day; 3000 s too, 2026-09-29), and so does every network
  size from 5 to 40 neurons (step 2, 2026-10-01; about half the population retains, at every N). The deviation comes with input overlap.
  At 50% overlap, a new context captures most of the population and the just-departed keeps
  only ~37% of its neurons (disjoint: 86-96%), at either phase length. **So one back is the
  disjoint-context case, and overlapping contexts get less than one back.** Memory has to carry
  whatever the substrate drops, and it already does.
- **For memory's design this changes little:** memory carries whatever the substrate doesn't, and
  rehearsal is keyed to changes.

This is not "division of labour" in the strong sense, and it isn't a capacity tax either.
Stability and plasticity are split across neurons, but the stable part only reaches one context
back. It's the same wall as "stability-plasticity is structural, not tunable", met again at the
substrate level. **Design consequence:** the substrate can't be the long-range store. Context older
than one back has to be carried by the memory layer, which is what the slow `w_char` layer was
built for. **The split works in principle (coupling toy, `experiments_integration.md`).** It was
tested with a pre-registered two-back test: substrate-fed memory recognized a non-first context
the substrate had already dropped, 8/8 at a 50 s clock. Two things make it work:

- **Every world change opens a rehearsal window for memory.** While the substrate's query is
  unsettled after a change, it reflects the departing context's *absence*, a mixture partly
  overlapping several stored contexts, including the one the substrate is releasing. If memory
  *retrieves* rather than *creates* during that window, retrieval sweeps across existing entries
  and resets each winner's eviction clock.
  - A context survived a long lull in exactly the seeds where it won at least one step of such a
    window (6/8).
  - Lesioning the refresh drops survival at a 10 s clock from 6/8 to 1/8.
  - An earlier version of this entry said the query "points at the released context". A direct
    check showed that's the special case, not the rule.
  - If memory *creates* during the window instead, the same moment produces a spurious entry: the
    "two-back capture".
- **Memory's horizon (staleness_threshold x clock) has to cover the time from release to return,
  not the whole absence.** Confirmed in v1c: with release-to-return 2000 s against a 1500 s
  horizon, survival drops from 6/8 (at 1000 s, v1b) to 2/8. The two survivors were also rescued
  by a later transitional reactivation.

Don't assume the substrate "remembers" a context it left two changes ago, and don't pick a clock
mapping without checking memory's horizon against the release-to-return intervals that matter. (The
resemblance to complementary learning systems is motivation, not evidence. The timescales here
aren't biological.)

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

## Standing operational habits

Not design principles about the architecture — mechanical hygiene about how experiments get
run, worth keeping consistent so it doesn't get rediscovered fresh each time someone hits the
same friction.

- **Background any long-running script with `python -u`** (unbuffered stdout), not plain
  `python`. Redirected stdout is block-buffered by default, not line-buffered — status prints
  sit invisibly in a buffer until it fills or the process exits, so checking a `.log` file
  mid-run shows nothing even when the script has clearly been running long enough to have
  printed several lines (hit this directly during the ESN size-scaling sweep). `-u` is free for
  the sparse status prints these scripts actually do; only matters in genuinely hot loops
  printing every iteration, which none of them are.
- **For genuinely long single runs, write structured intermediate progress to a small file**
  (a `status`/`current_step`/`progress` field, updated at each meaningful sub-step — one size
  point, one seed, one spectral-radius value), not just a start/end status. This isn't a new
  idea, it's applying the already-validated start/end pattern (the `"started"` marker written
  immediately so a hang is distinguishable from "never launched," used since the first
  multi-seed STDP ensemble) more consistently, to the middle of a run too, not just its edges.

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

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

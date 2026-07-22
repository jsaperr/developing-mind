# Developing Mind — Context for Claude Code

## Read first
Before diagnosing a failure or proposing an architectural fix, read
principles.md — it has the design commitments and failure-mode
patterns that "Known findings" below are instances of. Don't rediscover
them from scratch.

## What this is
Novel AI architecture rejecting train/deploy split. Continuous substrate
(mycelium/SNN) + Hopfield attractor layer as emergent self + two-layer
memory (fast episodic, slow character) + intrinsic curiosity + metacog
layer. Full philosophy lives in the primary design artifact
(developing_mind_framework_v5.docx), kept outside this repo — don't ask
for it unless the task needs architectural grounding, principles.md is
the load-bearing summary for day-to-day work.

## Experiment logs
Three files, split at phase boundaries: `experiments.md` for
Hopfield/two-layer-memory/episodic-layer work, `experiments_brian2.md`
for the Brian2/SNN phase, `experiments_esn.md` for the Echo State
Network phase. Check whichever matches the task.

## Current build state
Hopfield two-layer + episodic grow-and-prune work is closed out — see
Known findings below. Brian2 2.9.0 installed and validated end-to-end,
cython/MSVC compiled backend confirmed working. Shared simulation and
metric code for both mechanisms now lives in `src/` (`src/hopfield/`,
`src/brian2_stdp/`) instead of being duplicated per-notebook — check
those modules' docstrings for which notebook each was extracted from
and the validated/superseded lineage.

STDP: the original runaway is fixed via homeostatic synaptic scaling.
Standing explanation for the Apre-dependent instability — Apre
controls excursion *amplitude*, not switching *frequency* (full
reasoning in experiments_brian2.md, don't re-derive it). Apre=0.005's
stability is genuine-but-probabilistic, not absolute: bounded away
from collapse in the large majority of realizations tested, but the
settled level isn't architecturally guaranteed identical across all of
them (see experiments_brian2.md's 8-seed ensemble entry for the full
result — not duplicated here). RESOLVED, whether any of this was a
single-neuron artifact: a 5-neuron population extension (independent
replicates of the same mechanism, 4 seeds x 2 Apre values) found no
evidence it was — reversal-frequency invariance replicated
quantitatively (matched the N=1 rate almost exactly), Apre=0.005's
bounded stability replicated cleanly, and Apre=0.02's noisy-but-
directionally-robust differentiation refined rather than contradicted
the N=1 read (see experiments_brian2.md's population-extension entry).

RESOLVED, the stability-definition decision: this layer needs a
stable population-level readout tolerating individual-synapse churn,
not individual-synapse convergence — the latter is unmeetable in this
system as built (reversal frequency is Apre-invariant everywhere
tested, so no synapse ever truly settles), while the former is what's
actually been robustly measured this whole arc. TESTED DIRECTLY at the
many-neuron/shared-input scale too (not just extrapolated anymore, and
now at n=7 differentiating seeds, not just 2): a genuine shared-input
competitive population (lateral inhibition between postsynaptic
neurons, ambiguity-gated the same way as the Hopfield retrieval bias)
held a stable population-level signal for the full 5000s in every
differentiating seed tested, no exceptions — while individual-level
identity behavior spans a real spectrum underneath: near-permanent
lock-in, one-time reorganization (two distinct sub-flavors), and,
most strikingly, lock-in that never happens at all (constant
multi-way reshuffling the entire run, the highest churn of any seed
tested — the strongest instance of this stability definition found so
far). Scope stays real: n=7 is a typology, not a measured distribution
over how common each regime is; still one operating point; and only
on the differentiating side of a genuine bifurcation this same
experiment found (basin split now measured at ~50/50 across n=14
seeds) — the converging side says nothing about population-level
stability since there's no differentiated signal to be stable in the
first place. See principles.md's named decision entry for the full
defense and precise scope, and experiments_brian2.md's consolidated-
summary entry at the top of the file (or its "Seed expansion" and
"Shared-input population competition" entries directly) for the full
run.

Also RESOLVED: the positional-bias question flagged in the bimodal-
distribution entry (was the per-index floor-rate spread at Apre=0.005
a real effect or noise). Pooling 30 new single-neuron replicates with
the original 20 (n=50) moved the chi-square away from significance
(p=0.068 -> p=0.30), the signature of noise averaging out, not an
underdetected real effect. Was noise from early-relaxation-window
lock-in, not a positional bias in the generation code.

ESN (new third phase, `experiments_esn.md`): stage 1 (memory-capacity
characterization) done — a minimal 300-unit leaky-integrator reservoir
gives a real, non-degenerate memory-capacity/spectral-radius curve
(peaks ~5.75 near spectral_radius=1.1, ~37-step effective memory
horizon there). All three two-layer decay constants (`decay_fast`,
`consolidation_rate`, `decay_char`) sit beyond that horizon, even the
fastest one — the two-layer memory's persistence isn't something a
modestly-sized passive reservoir provides for free, consistent with
why a dedicated consolidation mechanism was needed. Comparison is an
order-of-magnitude sanity check, not a rigorous unit conversion — see
experiments_esn.md for the full caveat.

Three follow-ups to stage 1, RESOLVED: (1) reservoir size scaling
does NOT close the gap — total linear memory capacity plateaus
(6.63 -> 6.06 -> 5.74 across 300/1000/3000 units, within noise, not
growing) — ruling out "just make it bigger." (2) The premise that
stage 2 needs reconstruction-grade memory was wrong: a much weaker,
task-relevant classification signal ("which phase was active k steps
ago," not exact reconstruction) stays well above chance at 300 units
all the way to k=1000 (27x past the 37-step linear horizon, 2x past
the 400-step phase length stage 2 needs). (3) A multi-timescale
reservoir (half fast-leaking, half slow-leaking — the reservoir-world
version of the two-layer memory's own fast/slow split) extends that
classification signal further still (near-ceiling accuracy the whole
1-1000 range, vs. baseline's plateau ~0.6), at the cost of gutting
linear reconstruction capacity by 93% — a real, coherent trade-off,
not a free lunch. Net: stage 2 looks considerably more promising than
stage 1's raw number suggested on its own. STAGE 2 ITSELF (the actual
episodic-layer application) IS STILL EXPLICITLY NOT STARTED — deferred
pending a separate go-ahead. Don't start it without being asked.

Two real bugs caught and fixed during the follow-ups, worth knowing
about if touching src/esn/ again: a periodicity-leakage bug in the
phase-cycling task generator (a strictly periodic schedule let a
linear readout infer absolute time position instead of using genuine
memory — fixed with randomized phase order/length), and a catastrophic-
overfitting bug in the size-scaling sweep (fixed ridge_alpha=1e-6 was
essentially unregularized once reservoir size approached the training-
sample count — fixed with ridge_alpha=0.1). See experiments_esn.md for
the full diagnosis of both.

Open/blocked, not being chased right now:
- Episodic layer: variable-size-X primacy/recency ordering, blocked on
  context/phase-awareness that doesn't exist yet — ESN stage 2 is a
  candidate approach for this, not yet attempted, and now better-
  motivated than when stage 1 alone was the only data point.
- STDP: a measured distribution over the individual-level regimes
  (the above is a typology at n=7, not a frequency estimate), and
  whether the converge-vs-differentiate bifurcation appears at other
  operating points, are both real, unstarted follow-ups. The bimodal
  structure's timing dynamics (no per-synapse traces saved from the
  early ensembles) are also still open.
- ESN: the 10000-unit size-scaling point was dropped (impractically
  expensive with the current dense-matrix implementation, see
  experiments_esn.md) — the 300-3000 trend already answered the
  question, but a sparse-reservoir implementation would be needed to
  test genuinely large sizes if that's ever wanted.

NOT yet built: per-event step-size characterization for high-Apre
instability, more STDP/SNN work beyond this, BindsNET, ESN stage 2,
anything neuromorphic beyond the install check, metacog, cross-
substrate principle-transfer experiments. Don't jump ahead to those
unless explicitly asked.

## Known findings (see experiments.md for full log)
- Single-layer w with decay=0.01: full erasure of early-visited pattern
  strength once attention shifts away. No persistence.
- Single-layer w with decay=0.001: permanent primacy bias — early
  pattern never gets caught even after equal later visitation. This is
  basin lock-in, not precipitation.
- Two-layer split (fast episodic / slow character), 2026-07-17: naive version
  (unbounded consolidation growth) let the slow layer overpower content
  similarity in the retrieval softmax — 7-8% content-match rate during
  off-pattern phases. Fixed with a saturating (Oja's-rule-style) headroom term
  on w_char's growth, keeping it in the retrieval competition rather than
  decoupling it — content-match recovered to 45-78%, plasticity became real
  (pattern peaks comparable across patterns instead of one dominating).
- Follow-up, 2026-07-18: the graded persistence ordering (primacy of first
  pattern + recency of most-recent pattern, no monopolization) holds robustly
  across a parameter sweep (w_char_max 5-20, consolidation_rate 0.005-0.02) —
  not a one-setting fluke. The phase-2 misretrieval transient is a
  roughly fixed-duration relaxation (~250-350 steps) independent of phase
  length, not something that scales with it — measured, not yet fixed.
  The "savings on relearning" framing does NOT hold up: reacquisition rate
  (slope) is actually slower on return than on true cold start; the apparent
  "faster" absolute-step result is just restating persistence (shorter
  distance to climb), not a genuine relearning-speed effect. Don't report
  savings as a finding without this caveat.
- Resolved, 2026-07-18: the phase-2 transient's multi-bump shape (dip then
  re-rise) is confirmed seed noise, not real competitive oscillation —
  10-seed check found bumps in all seeds but at wildly scattered positions
  (step 21-307 out of 400, std 89.4). Dropped from findings language. The
  fixed-duration (~250-350 step) transient result itself still stands,
  corroborated by the consistent broad decay envelope across seeds.
- Rate-modulated savings attempt (v1), 2026-07-18: tried scaling increment_fast
  by w_char (k=0.5) to make w_char affect acquisition rate, not just retrieval
  bias — REOPENED lock-in, worse than the original bug, because w_fast had no
  saturation cap of its own. w_char's own ordering check didn't catch it —
  only the direct content-match diagnostic did (collapsed to 6-9%).
- Rate-modulated savings (v2, fixed), 2026-07-19: added w_fast headroom
  (mirroring w_char's fix) + an explicit cap on the modulation multiplier.
  Runaway contained (phase-1 peak 6.15 vs v1's 26.3, w_fast_max=10 ceiling
  never approached). No monopolization anywhere across a 3x3 sweep of
  (k in {0.25,0.5,1.0}, w_fast_max in {5,10,20}). Genuine slope-based savings
  appeared for the first time in this series (phase-4 slope 0.0321 > phase-1
  slope 0.0302 at k=0.5/w_fast_max=10) — but ONLY in the k>=0.5,
  w_fast_max>=10 region, which is also where content-match is most degraded
  (21-40% vs ~85-94% at k=0). Savings and clean content-addressed retrieval
  trade off against each other across this parameter space — not
  independent. The monopolization ratio metric alone is not sufficient to
  catch fidelity degradation — it stayed low (<2.5x) even at the worst
  content-match points (13.2%).
- Ambiguity-gated retrieval, same day: gated how much w_fast/w_char get to
  bias retrieval by content-similarity ambiguity (top1-top2 gap) —
  g=1/(1+gap/gap_scale), applied only to the strength terms in the softmax
  bias, update rules unchanged from v2. This decouples savings from
  content-match cleanly: at aggressive gating (gap_scale=0.0511, grounded in
  empirical gap median 0.2557), content-match recovered to 95%+ (exceeding
  the k=0 baseline) at settings that previously tanked to 21-40%, while
  savings survived in 11/12 tested combinations. Validates the "strength
  breaks ties, not clear mismatches" design principle (option (c) from the
  original two-layer failure analysis, finally tried). No final
  gap_scale/k/w_fast_max combination adopted as default — still a
  deliberate downstream choice.
- Episodic layer (variable-size X, grow-and-prune), 2026-07-19, v1-v5,
  consolidated (full arc in experiments.md): validated — pruning works
  cleanly (87-90% of created patterns evicted, bounded size, no
  monopolization); the eviction gate's coverage gap (only engaged on
  2+ simultaneous candidates, 1.7% of real evictions) was found and
  fixed by gating on distance past the eviction threshold instead of
  distance to a peer (100% of evictions now gated); the gate's core
  claim then passed a properly calibrated falsification test (a
  w_char 5-8 entry gets 14.5x the eviction-delay margin of a fresh
  one, but is not rescued indefinitely) — second confirmed instance of
  "strength breaks ties, never overrides a clear call," after
  retrieval, suggesting it's a transferable design law here, not a
  one-off fix.
- Still open, and it's a real stopping point, not an unfinished patch:
  graded primacy/recency ordering under variable-size X is unresolved
  — a core phase-dominant pattern was evicted before its own dominant
  phase began, and a creation-anchored grace-window stopgap was tried
  and mathematically confirmed to solve the wrong axis (the pattern
  wasn't new, it was old-but-between-periods-of-relevance). Genuinely
  blocked on context/phase-awareness the system doesn't have yet — not
  attempting further patches (fake phase-signal, or shrinking the test
  until it's not testing anything real). Before any future attempt:
  check whether context/phase-awareness has been built elsewhere
  (metacog, curiosity signal) first — if not, this isn't ready to
  revisit yet.

## How Jasper works
- Explain code as you go, succinct at first, verbose if asked. Trying to build fluency and understanding.
- Casual, fast-moving, thinks in leaps. Doesn't need hedged uncertainty
  language, wants direct answers.
- Validate hypotheses cheaply before building complexity. If a toy
  experiment can answer the question, don't jump to the real substrate.

## Stack
PyTorch, Brian2 (not snntorch), BindsNET, PyNN. 3050/4060 laptop GPU
locally, WWU CSCI HTCondor cluster (H100s) for heavier runs, EBRAINS
Jupyter for BrainScaleS-2 later.

## Rules
- Do not build/implement anything not explicitly requested this session.
- Flag architectural inconsistencies with the framework doc if you spot
  them — don't silently "fix" them by picking an interpretation.

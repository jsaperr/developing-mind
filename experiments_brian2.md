# Experiments Log — Brian2 / SNN phase

Hopfield / two-layer-memory / episodic-layer work lives in `experiments.md`. This file is
for the Brian2/spiking-dynamics phase specifically — different tools (differential equations
instead of tensor ops), different failure modes, split out at the natural phase boundary
rather than mixed into an already-long single log.

## 2026-07-20 — Post-hoc analysis: the group-mean-gap's "bounded stability" hides a bimodal, winner-take-most structure at the synapse level

**Analysis, not a new simulation** — pooled `final_corr_w`/`final_uncorr_w` from the existing
8-seed/5000s ensemble (`apre005_ensemble_data/apre_ensemble_seed*.json`), no new runs. Same
pattern as the amplitude-vs-frequency reframe: post-hoc analysis of data that already existed,
prompted by an informal exploratory-analysis pass, not a new experiment.

**Question:** does "group-mean gap stays bounded away from zero" (the ensemble's headline
finding) mean the correlated and uncorrelated groups cleanly separate into two tight clusters,
or is there more structure underneath the group mean than that number reveals?

**Result: final weights are bimodal in both groups — synapses land near the clip boundaries (0
or 1), rarely in between — and group identity biases which boundary, but doesn't determine it
cleanly:**

| | near ceiling (>0.9) | near floor (<0.1) | mid-range |
|---|---|---|---|
| correlated (n=80, pooled across 8 seeds) | 61% | 31% | 8% |
| uncorrelated (n=80, pooled across 8 seeds) | 20% | 55% | 25% |

Roughly a third of the "winning" correlated group's own synapses actually land at the floor,
behaving exactly like losers — not a moderately, uniformly-reinforced group. Conversely, a
fifth of uncorrelated synapses land at the ceiling.

**This refines, not contradicts, the existing group-mean-gap stability finding.** The
aggregate metric was always accurate — the mean genuinely does stay bounded away from zero,
that result stands — it's just coarser than what's actually happening at the synapse level.
"Differentiation" in this system means "most correlated synapses saturate high and most
uncorrelated synapses saturate low," not "every correlated synapse beats every uncorrelated
synapse." This quantifies what was previously only a qualitative caveat in the original
correlation-experiment writeup ("real if imperfect separation, a few defector synapses in each
group") — now with actual numbers behind it.

**Scope, matching the analysis method:** this is a post-hoc analysis of final-snapshot weights
only, from data that already existed — not a new experiment. It can't say anything about
*when* a given synapse committed to its final side, whether defectors flip back and forth or
settle early, or whether defector identity is stable across seeds (a positional-symmetry check
across the 8 seeds was inconclusive either way, too few seeds to distinguish from
uniform-random). None of the ensemble runs saved full per-synapse time traces, only group
aggregates, final snapshots, and reversal counts — see the design note added to
`src/brian2_stdp/network.py`'s docstring: future long runs should save at least a subsampled
per-synapse trace so trace-level questions like this are answerable without a full rerun.

---

## 2026-07-20 — Population extension (N=1 -> N=5): does the single-neuron STDP signature generalize, or was it an artifact?

**Data:** `notebooks/brian2/population_extension_data/run_population_seed.py` (standalone script,
8 raw result JSONs for full reproducibility). Shared rig code: `src/brian2_stdp/network.py`
(`build_population_network`), `src/brian2_stdp/spikes.py`
(`build_population_presynaptic_input`), `src/brian2_stdp/metrics.py`
(`compute_population_metrics`).

**Question, per external review feedback (relayed via Jasper):** every STDP finding so far
(reversal-frequency invariance, Apre005's genuine bounded stability, seed 2006's discrete
reorganization) used exactly one postsynaptic neuron. Could any of it be an artifact of
single-neuron zero-sum synaptic scaling specifically, rather than a general property of the
mechanism? Extend to a small population and check.

**Design pivot caught by a cheap calibration run, before committing real compute — a genuine
finding in its own right:** the first design shared one presynaptic pool across 5 postsynaptic
neurons (all-to-all) and relied on small (+/-0.02) initial-weight jitter to break the symmetry
between them. A 20s calibration run showed this **does not work**: all 5 neurons converged to
*bit-identical* final weights (max abs diff = 0.0) despite different starting points. Root
cause, confirmed by direct inspection: the LIF neuron's hard reset (`v = v_reset` on every
spike) wipes out any pre-spike membrane-potential difference every interspike interval, and
with `GMAX=6mV` the correlated group's spike bursts are large enough that postsynaptic
threshold-crossing timing is robust to +/-4% weight jitter — so every neuron saw identical
effective post-spike timing, and since STDP trace updates (`apre`/`apost`) depend only on
spike *timing* (not on the synapse's current weight), all 5 neurons received identical
trace-driven weight deltas from identical initial-weight-adjacent starting points, converging
exactly once any synapse first hit a clip boundary. **Pivoted to block-diagonal connectivity
instead**: each of the 5 postsynaptic neurons gets its own dedicated, independently-drawn
20-neuron presynaptic block (same p_share=0.9 generative process, different randomness) —
genuinely independent replicates of the validated single-neuron mechanism, run together in one
Brian2 network for efficiency, rather than literal shared-input competition. Re-calibration
confirmed real divergence (different post rates, different final weights, different reversal
counts per neuron) and **no meaningful wall-clock cost increase** — 5 replicated neurons cost
about the same as 1 (0.23s wall/simulated-second either way), since Brian2 vectorizes across
neurons/synapses.

**A second real bug caught by the same calibration run:** `build_network`'s (and the new
`build_population_network`'s) reliance on Brian2 resolving free identifiers (`tau`, `v_thresh`,
`taupre`, ...) from the *calling frame's* namespace broke the moment the network-building code
moved into `src/brian2_stdp/network.py` — a different module than wherever `run()` gets called.
Brian2's auto-resolution walks the stack from the `run()` call site, not from wherever the
object was constructed, so this was silently broken for `build_network` too (undetected by the
earlier construction-only smoke test, which never calls `run()`). Fixed by passing every free
identifier explicitly via `namespace=` in both builders — see `src/brian2_stdp/network.py`.

**Design (stated before launching the real batch):** Apre in {0.005 (single-neuron "stable"),
0.02 (single-neuron "chaotic")}, 4 independent population-seeds each (3001-3004), 1000s per run,
4 concurrent processes / 2 waves, 8 runs total (~230s wall each). Falsification stated up
front: if population replicates N=1, Apre=0.005 should show a tight, positive, bounded
group-mean-gap across all neurons/seeds; Apre=0.02 should stay noisy/non-settling; reversal
frequency should be similar in scale between the two Apre values (Apre-invariant, per the
standing explanation). Population data diverging from this — unreliable differentiation,
population-emergent Apre-dependence in reversal frequency, or Apre=0.005 crossing zero well
within 1000s — would mean the N=1 findings don't generalize.

**Result, aggregated across all 20 (4 seeds x 5 neurons) replicates per Apre value, settled
region t>=100s:**

| Apre | gap mean | gap min (any replicate, any time) | fraction crossing zero | reversals/synapse/1000s | post rate |
|---|---|---|---|---|---|
| 0.005 | +0.346 | +0.148 | 0/20 | 501.2 (std 8.0) | 18.7 Hz |
| 0.02 | +0.285 | +0.002 | 0/20 | 505.6 (std 6.7) | 18.5 Hz |

**Reversal-frequency invariance replicates cleanly, and quantitatively, not just
qualitatively:** 501.2 vs 505.6 reversals/synapse/1000s — essentially identical between Apre
values, exactly as the standing explanation predicts. This also closely matches a linear
extrapolation from the original N=1 300s finding (~140/300s -> ~467/1000s) — a real
quantitative consistency check across an entirely different network size, not just the same
qualitative shape.

**Apre=0.005's genuine bounded stability replicates cleanly, numbers included:** trajectory
plots show all 20 replicates climb from 0 to ~0.3-0.5 over the first ~100-150s then settle into
a flat band for the remaining ~850s (aggregate settled mean +0.346, matching the N=1 ensemble's
own settled-region numbers of ~0.35-0.36 closely). 2/20 replicates settle into a visibly lower
(~0.15-0.25) but still clearly positive band — the same shape as the N=1 ensemble's seed 2006
discrete reorganization, now seen twice independently in a different network. No replicate, at
any point in the full 1000s, came within even 0.1 of crossing zero.

**Apre=0.02 refines rather than contradicts the N=1 characterization -- inspected via full
trajectory plots, not the summary table alone, per the standing discipline:** visually, every
replicate fluctuates continuously and chaotically across the entire 1000s with no settling
(matches N=1's "no visible stable separation" read exactly). But the correlated>uncorrelated
**direction** never flips: all 20 replicates stay positive the entire run, and even the
closest-approach replicate (seed 3004, neuron 4, min +0.002) only brushes zero twice, briefly,
spending the overwhelming majority of 1000s clearly positive (0.15-0.5 range). N=1's original
300s snapshot never claimed the gap went negative either (its own numbers were 0.330 mid-run ->
0.228 late-run, still positive) -- "no visible stable separation" was correctly describing the
lack of *settling*, not a sign flip. The population data adds real precision N=1 didn't have at
300s: differentiation *direction* is robust even when its magnitude never settles, across 20
independent replicates and over more than 3x the original observation window.

**Honest scope caveat:** this tests whether the mechanism generalizes *across independent
instances* of itself (does replicating the exact single-neuron rig 5 times produce the same
statistical signature), not genuine population-level phenomena from neurons sharing or
competing over common input -- that design was tried first and found not to produce any
inter-neuron divergence at all (see above). A literal shared-input competitive population is a
different, unbuilt experiment.

**Verdict:** no evidence that any of the single-neuron STDP findings are single-neuron
artifacts. Reversal-frequency invariance, Apre005's genuine bounded stability, and Apre02's
noisy-but-directionally-robust differentiation all replicate across 5 independently-instantiated
neurons and 4 seeds (20 total replicates), with reversal frequency matching the N=1 rate
quantitatively. The explicit stability-definition decision (individual-synapse convergence vs.
population-level readout tolerating member churn) that this extension was meant to inform is
still open -- not decided here, flagged for the next round.

---

## 2026-07-20 — Does Apre=0.005's stability reflect genuine boundedness, or insufficient observation time? 8-seed, 5000s ensemble

**Notebook:** `notebooks/brian2/brian2_stdp_apre005_long_ensemble.ipynb` (loads results,
doesn't re-simulate — raw run script and all 8 raw result files are in
`notebooks/brian2/apre005_ensemble_data/` for full reproducibility)

**Question (per Jasper's message):** the Apre-sweep round found reversal frequency is
Apre-invariant — `Apre=0.005`'s apparent "stability" over 300s might just mean small
excursions that haven't happened to cross a boundary yet, not a genuine restoring force. Does
it hold over a much longer timescale?

**Design worked out and reported back before running anything, per explicit instruction —
this surfaced a real problem before it became a wrong result:** a calibration run found that
the literal definition given ("does any correlated synapse's weight drop below any
uncorrelated synapse's weight") is contaminated — 23% of the 100 correlated x uncorrelated
pairs already overlap at t=300s in the "stable" condition, purely from ordinary within-group
spread, not drift. Fixed by making the **group-mean gap trajectory** (not individual pairwise
crossings) the primary metric, with pairwise overlap-fraction tracked continuously as
secondary context. Back-of-envelope diffusion estimate from measured (not guessed) excursion
amplitude and reversal rate: **~1600s** naive timescale for the group-mean gap to plausibly
random-walk away — likely conservative (ignores boundary effects and the zero-sum synaptic-
scaling coupling between synapses). Design: **8 seeds x 5000s** (>3x the diffusion estimate),
run as 4 parallel processes per batch (measured 6 physical cores available, left real
headroom), each writing a seed-tagged JSON with an explicit `status` field so partial failures
are detectable rather than silently analyzed around.

**Compute-budget gate, honored as agreed:** single-run baseline was 0.2s wall-clock per
simulated second. Batch 1 (4 parallel) came back at 0.225-0.229s/simulated-second — ~13%
slower, within the pre-agreed "proceed without checking back" threshold — so batch 2 launched
at the same settings without a stop, per the explicit conditional rule set before starting.

**A second metric near-miss, caught before being reported as a finding:** the raw
`min_group_mean_gap` across batch 1 looked alarming — all 4 seeds went negative (-0.008 to
-0.021). Checked *when*: all four hit their minimum at **t=0.5s**, the trivial startup instant
before any differentiation exists (every synapse starts at `w=0.5`). Not late-run erosion,
just the naive full-trajectory minimum picking up the boring start of the run. All
"settled-region" stats below exclude `t<200s`. This is the third confirmed instance this
session of a single summary number being misleading — see `principles.md`.

**Result, all 8 seeds, settled-region (t>=200s) stats:**

| seed | gap mean | gap std | gap min | gap max | trend (early->late) |
|---|---|---|---|---|---|
| 2001 | 0.355 | 0.021 | 0.274 | 0.420 | 0.358 -> 0.358 |
| 2002 | 0.357 | 0.023 | 0.272 | 0.424 | 0.355 -> 0.359 |
| 2003 | 0.358 | 0.022 | 0.284 | 0.439 | 0.355 -> 0.356 |
| 2004 | 0.351 | 0.024 | 0.243 | 0.424 | 0.353 -> 0.347 |
| 2005 | 0.359 | 0.026 | 0.267 | 0.467 | 0.368 -> 0.353 |
| **2006** | **0.278** | **0.069** | **0.145** | 0.411 | **0.353 -> 0.211** |
| 2007 | 0.359 | 0.027 | 0.259 | 0.483 | 0.374 -> 0.352 |
| 2008 | 0.361 | 0.027 | 0.295 | 0.470 | 0.378 -> 0.352 |

**7/8 seeds show a tight, flat, consistent band** (gap mean 0.35-0.36, std 0.02-0.03, trend
essentially flat within noise) across the full 5000s — no seed among these 7 came anywhere
close to crossing zero (minimums all 0.24-0.30), directly falsifying "not genuine, just slow
drift toward zero" for the substantial majority of realizations tested.

**Seed 2006 is a real outlier, characterized precisely rather than smoothed into the average:**
the trajectory plot shows a genuine, discrete **step-down transition around t~2000-2500s** to
a new, lower band (~0.15-0.25), not gradual continuous erosion. Precisely: mean gap 0.344 in
`[0,1500]s` vs. 0.214 in `[3000,5000]s`; minimum (excluding the trivial startup) is 0.079 at
t=11.5s — **still never crosses zero**; and it still fluctuates back above 0.30 occasionally
after t=3000s, so it's not a permanently fixed new plateau either — closer to "shifted to a
new, less differentiated but still clearly non-zero regime with higher ongoing variance" than
either "erosion to collapse" or "unchanging stability."

**Verdict: neither stated falsification outcome is exactly right, again — same shape of
surprise as the Apre-sweep round.** No seed crossed or trended toward zero over 5000s (>3x
the diffusion estimate), which directly rules out "stability was just an illusion of a short
window." But one seed's real, discrete reorganization to a different (still positive) stable
level means "durable, unchanging stability" isn't quite right either. **The honest, supportable
claim: STDP produces group-level differentiation that stays bounded away from collapse over
timescales well beyond naive diffusion, in the substantial majority of realizations tested —
real and practically durable, but the specific level it settles at is itself part of the
stochastic process, not architecturally guaranteed to be a single fixed value.** This is the
`principles.md` note added this round (genuine-but-probabilistic stability, not absolute)
applying directly to its own result.

---

## 2026-07-20 — Is the Apre instability a threshold or a gradient? Neither — reversal frequency is Apre-invariant

**STANDING EXPLANATION for the Apre-dependent STDP instability — reference this, don't
re-derive it.** `Apre` controls excursion *amplitude*, not reversal *frequency* (which is
~constant, ~140/synapse/300s, across the whole tested range). "Stable" at low `Apre` means
small excursions that stay close to the current assignment, not genuine convergence to a
fixed point — the same underlying churn is present everywhere, just invisible at low
amplitude. This single finding retroactively explains why both the scaling-interval sweep and
the per-event jump-cap sweep found nothing: neither touches switching frequency, only
amplitude (jump-cap) or nothing relevant at all (interval). See the full reasoning below.

**Notebook:** `notebooks/brian2/brian2_stdp_apre_sweep.ipynb`

**Question (per Jasper's message):** two mechanisms were ruled out as the cause of
`Apre=0.02`'s instability (correction-interval timing, per-event jump-size cap). Working
hypothesis: `Apre`'s raw magnitude might act like an effective temperature parameter for this
competitive stochastic system — below some critical value, an ordered/settled phase; above
it, disordered/chaotic. **Test:** sweep `Apre` directly across `{0.005, 0.008, 0.011, 0.014,
0.017, 0.02}`, no per-event cap, `scaling_interval=500ms`, `p_share=0.9`, 300s, single fixed
seed (1000) throughout for a clean comparison. Does trajectory character change gradually, or
flip sharply at some threshold? Both are real, useful answers, stated up front.

**Workflow note, addressing the lesson from last round:** wrote the full simulation loop and
all trajectory/variance analysis together in one execution pass this time — no redundant
re-run needed for the core sweep. (One additional analysis pass *was* needed partway through,
but for a legitimate reason: the initial std-based metric turned out not to capture the right
thing, discovered only after inspecting the actual plot — see below. That's a genuinely
new-information follow-up, not the same avoidable split as before.)

**Consistency check passed exactly:** `Apre=0.02` in this sweep reproduced the jump-cap
notebook's "uncapped" result bit-for-bit (post_rate=18.46Hz, corr_w=0.580, uncorr_w=0.397,
same seed) — confirms the pipeline is fully deterministic and comparable across notebooks.

**First analysis attempt was misleading, caught before drawing a conclusion from it:** late-
window within-group weight std came out flat-to-slightly-*decreasing* as `Apre` increased
(0.005: 0.398 -> 0.02: 0.355) — directly contradicting the visual impression from the
trajectory plot (low-`Apre` panels look smoother/more directional, high-`Apre` panels look
visibly more chaotic). Rather than trust either the number or the visual impression alone,
computed a direct oscillation-frequency metric instead: mean direction reversals per synapse
(sign changes in a lightly-smoothed derivative).

**The actual result: reversal frequency is essentially Apre-invariant across the whole tested
range.** 136.8 to 147.0 reversals/synapse over the full 300s, 71.5 to 75.2 in the last 150s —
both ~5-7% total variation, no trend, well within noise. This means the switching-frequency
metric contradicts the visual "gradual chaos increase" impression too — that impression was
real but was tracking the wrong variable.

**Reframed, better-supported conclusion:** this system's underlying dynamic — individual
synapses periodically gaining and losing favor in the zero-sum competition synaptic scaling
enforces — happens at roughly the same *frequency* regardless of `Apre`. What `Apre` actually
controls is the *amplitude* of each excursion (directly, since it sets how much a single
spike-pairing event moves `w`). At low `Apre`, a synapse near the ceiling still reverses
direction ~140 times over 300s, but each reversal is small enough to stay close to the
ceiling — reading as a converged, stable state at the group-mean level and in any short
snapshot. At high `Apre`, the same reversal frequency produces excursions large enough to
traverse the entire `[0,1]` range each time, which is what reads as chaotic.

**This retroactively and cleanly explains both previous rejections, not as separate
coincidences:** the scaling-interval sweep found nothing because correction frequency has no
relationship to how often individual synapses switch sides. The per-event jump-cap sweep
found the same "random walk, just slower" behavior at every cap because capping the delta
reduces excursion *amplitude* — exactly consistent with this round's finding — without
touching reversal *frequency*, so the underlying non-convergent switching persisted regardless
of cap size.

**Neither stated outcome (gradual gradient, sharp threshold) was actually right.** Both
assumed switching *frequency* was the variable that would change across the sweep; instead
it's excursion *size* that varies, and frequency turned out to be the more fundamental,
`Apre`-independent property of this competitive system. Worth being explicit that this is a
genuine reframing, not a partial confirmation of either original option.

**One thing this reframes but doesn't resolve:** whether `Apre=0.005`'s "stable, converged"
characterization from earlier rounds should be revised. It isn't fully converged in the strict
sense — individual synapses are still switching sides at essentially the same rate as every
other `Apre` tested. It's stable in the weaker sense of staying close to its current
assignment rather than fully reversing it. Whether that weaker notion of stability is
sufficient depends on what this mechanism needs to guarantee downstream — a design question,
not something this experiment resolves on its own.

**Not testing further within this round.** This is a complete, well-supported answer to the
question asked, even though the honest answer is "neither framing was right, the wrong
variable was assumed to be changing." Flagging the reframed picture (frequency vs. amplitude)
for discussion rather than continuing to generate new hypotheses independently.

---

## 2026-07-20 — Does capping per-event STDP jump size fix Apre=0.02? Also rejected

**Notebook:** `notebooks/brian2/brian2_stdp_jump_cap.ipynb`

**Hypothesis (per Jasper's message, not assumed):** synaptic scaling only constrains the
*aggregate* weight sum — it never touches the size of a single STDP event's weight jump. At
`Apre=0.02`, an individual spike-pairing event can swing one synapse's weight by a large
amount in one step. Capping the per-event delta directly (`clip(apost/apre, -max_delta,
max_delta)` applied to the change added to `w`, independent of `Apre`'s own magnitude — a
deliberate design choice to isolate jump-size from learning-rate rather than confounding
"smaller jumps" with "slower overall accumulation" by just lowering `Apre`) might be the
actual fix the interval sweep failed to find.

**Test:** four pre-specified conditions, same seed across all four for a clean comparison —
uncapped (baseline), `max_delta` in {0.05, 0.02, 0.01}, all at `Apre=0.02, p_share=0.9`, 300s,
`scaling_interval=500ms` (reverted to the default, since the interval sweep showed it doesn't
matter).

**A workflow note before the result:** this round initially only printed endpoint summary
numbers, then required a full second 4x300s re-run to get the trajectory plots — a real,
avoidable inefficiency (each `nbconvert --execute` starts a fresh kernel with no persistence,
so appending analysis cells after inspecting a first pass forces re-running every expensive
prior cell, not just the new plotting). Should have written the simulation and the trajectory
analysis together in one pass from the start, especially since the previous round already
established that endpoint numbers alone can't be trusted here. Noted for next time.

**Result: rejected, cleanly, and it explains a superficially confusing numeric pattern:**

- **Trajectory plots:** uncapped, `max_delta=0.05`, and `max_delta=0.02` all look essentially
  identical — chaotic, full-range fluctuation for both groups across the entire 300s. At the
  tightest cap (`max_delta=0.01`), the character changes, but not toward stability: individual
  synapses move more *slowly* (smaller steps, as expected) but still wander across the *full*
  `[0,1]` range over the 300s window — a slower random walk, not a settled state. Visually and
  qualitatively different from `Apre=0.005`'s shape (correlated synapses climb and hold near
  the ceiling, uncorrelated settle into a lower band with slow, *directional* drift).
- **Late-window variance *increases* monotonically as the cap tightens** (uncapped 0.333 ->
  0.05: 0.374 -> 0.02: 0.418 -> 0.01: 0.421) — the opposite direction "capping fixes it" would
  predict. Not evidence the cap makes things worse; an artifact of the process moving slower at
  tight caps, so 300s covers less ground toward an equilibrium that, per the trajectory plot,
  doesn't appear to exist in this regime anyway.
- **The endpoint diffs would have been misleading on their own, again, the same trap confirmed
  a third time this session** (uncapped=+0.182, 0.05=+0.184, 0.02=+0.186, `0.01=+0.381`) — the
  0.01 condition's larger endpoint diff isn't evidence of better convergence, it's consistent
  with a slower random walk that simply hadn't wandered as far from its current position by the
  time the 300s cutoff landed.

**Two cleanly rejected hypotheses in a row for the same instability** (scaling-interval timing,
then per-event jump-size) is itself informative: it suggests this may not decompose into "how
often is the sum corrected" or "how big is one event's jump" at all. `Apre`'s raw magnitude may
simply act like an effective temperature parameter for this competitive, zero-sum (via
synaptic scaling) stochastic system — and no attempt to hold `Apre` fixed while isolating one
contributing factor at a time has found the actual lever, possibly because it isn't separable
that way in this system.

**Not chasing this further within this round.** Both proposed mechanisms are now ruled out.
The natural next hypothesis — does directly lowering `Apre`'s raw magnitude (accepting the
smaller-jumps/slower-accumulation confound this round tried specifically to avoid) produce a
smooth chaotic-to-stable transition, i.e. is there a threshold below which this regime changes
character — is real and testable, but a new hypothesis, not something to test unilaterally
here. Flagging for discussion rather than continuing to generate and test explanations
independently.

---

## 2026-07-20 — Is Apre=0.02 instability a timing race with the scaling correction? Hypothesis rejected

**Notebook:** `notebooks/brian2/brian2_stdp_scaling_interval.ipynb`

**Hypothesis (per Jasper's message, not assumed):** the homeostatic scaling correction only
fires every 500ms. If STDP moves weights meaningfully *within* that window at the higher
learning rate (`Apre=0.02`), the correction is structurally always chasing weights that have
already run ahead again by the next correction — a timing race between STDP's rate and how
often homeostasis checks in, not a magnitude/stability problem with the mechanism itself.

**Test:** re-ran the unstable condition (`Apre=0.02, p_share=0.9`, 300s) at a small,
pre-specified set of scaling intervals — 500ms (baseline), 100ms, 50ms, 10ms (a 50x range) —
plus the already-stable `Apre=0.005` condition at 500ms and 50ms as a neutral check.

**Confirms the hypothesis:** shortening the interval produces genuine settling, visible
directly in the trajectory plot (groups separating and *holding*, not just a
numerically-positive-but-noisy gap).
**Rejects it:** instability persists even at very short intervals.

**Result: rejected, cleanly, on both metrics used to judge it:**

- **Trajectory plots (the real test, not the summary numbers):** all four `Apre=0.02` panels
  (500/100/50/10ms) look essentially identical — continuous, chaotic, full-range fluctuation
  for both groups across the *entire* 300s, with zero visible improvement even at 10ms (a 50x
  faster correction than baseline, effectively near-continuous relative to the STDP/spike
  timescales here). Both `Apre=0.005` panels (500ms, 50ms) look qualitatively different and
  stable regardless of interval — correlated synapses climb and hold near the ceiling,
  uncorrelated synapses spread into a lower band with slow, gradual drift, not chaotic swings.
- **Quantitative check confirms the visual read:** late-window (80-100% of run) within-group
  weight std for `Apre=0.02` ranged 0.34-0.38 across all four intervals with no trend
  (500ms→0.343, 100ms→0.361, 50ms→0.384, 10ms→0.342) — flat, not decreasing. The mid-vs-late
  group-mean gap also showed no consistent trend across intervals (some increasing, some
  decreasing, no pattern tied to interval length).
- **A useful caveat surfaced by this round, not the original point of it:** final-snapshot
  diffs for `Apre=0.02` all clustered around +0.32 to +0.35 regardless of interval — including
  the 500ms condition, which with a *different* seed than the original extended run now reads
  as a "good" number (+0.348) instead of the original's declining +0.195. This is exactly the
  trap the trajectory-plotting requirement exists to catch: a single endpoint snapshot can look
  fine on a genuinely unstable, chaotically-fluctuating trajectory purely by chance of where
  the 300s cutoff happens to land.

**Not shrinking the interval further to keep chasing this** — per the stated discipline, this
was a small pre-specified set of values (500/100/50/10ms) and none showed a trend, so
continuing to shrink would be tuning-until-something-works with no remaining theoretical basis
for expecting a different answer at, say, 5ms or 1ms.

**Revised, better-supported explanation (not tested, flagged as the next hypothesis if this
thread continues):** this likely isn't about *how often* the total weight sum gets
renormalized — it's more likely about the *intrinsic step size* of individual STDP events at
`Apre=0.02`. The `on_pre`/`on_post` update applies an immediate, discrete jump to a single
synapse's weight on every relevant spike pairing; at `Apre=0.02` that jump can be large enough
to swing a synapse a substantial fraction of the full `[0,1]` range in one step. Synaptic
scaling only constrains the *aggregate sum* across all 20 synapses — it does nothing to dampen
how far any *individual* synapse can swing between pairing events, no matter how frequently
the sum gets rebalanced. That would explain the null result of this entire round: the
instability lives at the level of individual STDP events, a timescale the periodic *sum*
correction never actually touches. Whether capping the per-event weight change (independent of
`Apre`'s accumulation-rate effect), or simply reducing `Apre` itself, is what actually produces
stability is a real open question — not assumed from this result, would need its own
falsification-style test.

**Verdict:** a real, useful negative result. The hypothesis was specific and testable, and the
test cleanly rejected it rather than producing an ambiguous "maybe" — that's a clean outcome,
not a wasted one. The synaptic-scaling fix from the previous round is not invalidated by this
(it still correctly contains the total-weight runaway and broadens differentiation across the
grid); what's now clear is that scaling-interval tuning specifically is not the lever that
fixes `Apre=0.02`'s instability, so it shouldn't be pursued further down this path.

---

## 2026-07-20 — Homeostatic synaptic scaling fixes the runaway; differentiation broadens, but stability is learning-rate-dependent

**Notebook:** `notebooks/brian2/brian2_stdp_homeostatic.ipynb`

**Fix chosen (per Jasper's message): synaptic scaling**, not a target-rate feedback term.
Brian2 has an idiomatic mechanism for exactly this — a `(summed)` synapse variable
(`w_total_post`) automatically tracks the sum of `w` across all synapses onto a postsynaptic
neuron, and a periodic `run_regularly` block rescales every synapse toward a fixed target
(`20 * w_init = 10.0`, matching the original calibrated baseline) every 500ms (first pass,
not deeply tuned). This is also the closer biological analog — real cortical synaptic scaling
keeps total dendritic input roughly constant, converting "everything can grow together" into
genuine zero-sum competition between synapses. The STDP kernel itself is unchanged.

**Bundled into the same round, per the message:** (1) re-run the 6-point grid with the fix;
(2) add a true `p_share=0` null control at both `Apre` levels (8 runs total); (3) verify the
fix doesn't just relocate the problem — check total weight sum *and* postsynaptic rate, not
just individual-synapse saturation; (4) extend runtime at `p_share=0.9` to see true
convergence, since the original 60s run wasn't settled.

**A diagnostic false alarm caught before trusting anything else:** an early sanity run showed
`w_total`'s trace ranging from 0.0 to 19.45 (target 10.0) — investigated directly (plotted the
full trace, not just min/max) rather than assuming either "broken" or "fine." Root cause: a
single-sample monitor-ordering artifact at t=0 (read before the `(summed)` variable's first
computation) plus one genuine brief transient right before the first scaling event fires at
t=500ms (STDP runs fully unchecked for that first half-second). After that, the trace settles
into a tight band around 10.0 for the remainder of every run. Used the **steady-state band
(t>2s)**, not the raw full-trace min/max, as the real health metric from here on.

**Results — 8-point grid (p_share in {0.0, 0.3, 0.6, 0.9} x Apre in {0.005, 0.02}):**

| p_share | Apre | corr_w | uncorr_w | diff | post rate | steady total range |
|---|---|---|---|---|---|---|
| 0.0 | 0.005 | 0.528 | 0.479 | +0.049 | 10.6 Hz | [9.98, 10.22] |
| 0.0 | 0.02  | 0.551 | 0.470 | +0.082 | 14.2 Hz | [9.93, 10.78] |
| 0.3 | 0.005 | 0.633 | 0.371 | +0.261 | 13.3 Hz | [9.96, 10.20] |
| 0.3 | 0.02  | 0.719 | 0.277 | +0.442 | 16.1 Hz | [9.86, 10.67] |
| 0.6 | 0.005 | 0.706 | 0.292 | +0.414 | 16.8 Hz | [9.94, 10.21] |
| 0.6 | 0.02  | 0.719 | 0.257 | +0.463 | 18.0 Hz | [9.66, 10.59] |
| 0.9 | 0.005 | 0.642 | 0.359 | +0.283 | 17.6 Hz | [9.80, 10.14] |
| 0.9 | 0.02  | 0.674 | 0.320 | +0.353 | 19.2 Hz | [9.49, 10.51] |

**Fix confirmed clean, doesn't relocate the problem:** steady-state weight sum stayed within
[9.49, 10.78] (target 10.0) at every grid point; postsynaptic firing rate stayed in
10.6-19.2 Hz across the whole grid — vs. 17.8-44.4 Hz unfixed, and comfortably inside the
stated healthy band (~3-20 Hz) everywhere, not just at the points that happened to work
before.

**Differentiation broadened substantially:** all six non-null `p_share` points now show real
`corr>uncorr` separation (+0.26 to +0.46), visible directly in the weight histograms
(correlated group shifted right, uncorrelated shifted left, most clearly at `p_share=0.9,
Apre=0.005` — 9/10 uncorrelated synapses clustered in a single narrow low bin). Previously
only `p_share=0.9` differentiated at all; the other four now-differentiating points saturated
together before this fix.

**Null control validated the construction, but needed a real check, not just two data
points:** the initial 2-seed null result showed a small `corr>uncorr` bias (+0.049, +0.082) —
flagged per the stated falsification criteria as needing root-causing, not dismissal, since
group labels carry no real information at `p_share=0`. Ran 10 independent additional seeds:
mean diff -0.047, std 0.095, positive in only 3/10 — scatters around zero as genuine noise
would, not a reproducible bias. The two-seed result was noise, not a construction bug — but
this needed the actual multi-seed check to know that, not an assumption either way. Confirms
the correlation-driven effect (+0.26 to +0.46) sits clearly above the null-noise floor
(roughly 3-5x the noise std).

**Extended runtime at `p_share=0.9` (300s, both `Apre`) revealed something the summary numbers
alone would have missed — stability is learning-rate-dependent:**

- `Apre=0.005`: genuinely stable. Correlated synapses climb and plateau at the weight ceiling;
  uncorrelated synapses settle into a lower band. Group-mean gap: 0.336 at 40-60% of the run,
  0.295 at 80-100% — essentially converged, durable.
- `Apre=0.02`: the trajectory plot tells a different story than the gap-comparison number
  suggested. Both groups stay heavily intermixed and rapidly, continuously fluctuating across
  the *entire* 300s — no visible stable separation at any point, despite a numerically
  positive mean gap throughout. The declining gap (0.330 mid-run -> 0.228 late-run) is
  consistent with a noisy, non-converging regime, not with settling toward a stable value.
  **This was only caught by plotting the actual trajectories** — the before/after gap
  comparison alone reads as "maybe still converging," which understates how different this
  condition actually looks from the stable `Apre=0.005` case.

**Verdict:** the homeostatic fix is real and works — contains the runaway cleanly (weight sum
and firing rate both controlled, not just individual-synapse saturation avoided) and measurably
broadens where correlation-driven differentiation appears (all non-null grid points now show
it, not just `p_share=0.9`). But durable, *stable* differentiation is only clearly confirmed at
the lower STDP rate tested — the higher rate produces a real, above-noise signal at any given
snapshot, but doesn't look like it settles into a lasting representation. Not tuning `Apre`
further to chase stability at the higher rate; reporting precisely what is and isn't
demonstrated, per the same discipline as the rest of this thread.

---

## 2026-07-20 — First STDP experiment: correlated-vs-uncorrelated differentiation — mixed result, real mechanism identified

**Notebook:** `notebooks/brian2/brian2_stdp_correlation.ipynb`

**Question:** direct test of the framework doc's core unsupervised-plasticity claim — does
STDP alone, with zero global signal, learn which inputs matter from spike-timing statistics?
One postsynaptic LIF neuron (same validated params as the install check), 20 presynaptic
Poisson inputs through STDP synapses, all starting at the same weight. 10 form a
**rate-matched correlated group** (spikes copied from a shared underlying process with
probability `p_share`, jittered, backfilled with independent spikes so mean rate stays equal
to the uncorrelated group regardless of `p_share` — a deliberate control so any
differentiation found is attributable to timing, not rate). 10 form a fully independent
**uncorrelated group**. Standard exponential STDP kernel, `taupre=taupost=20ms`,
`Apost=-Apre*1.05` (standard slight-depression-bias for stability). No labels, no loss.

**Falsification criteria and 6-run pre-registered sweep** (`p_share` in {0.3, 0.6, 0.9} x
`Apre` in {0.005, 0.02}) stated before running — see notebook for full text. One legitimate
calibration step done first (not outcome-tuning): initial `gmax=2mV` produced zero
postsynaptic firing at any tested `w_init` — a real miscalibration, not a config issue.
Widened the calibration grid, found `gmax=6mV, w_init=0.5` gives 8.9 Hz under uniform
weights/uncorrelated input, comfortably in the moderate (1-15 Hz) target range. Locked in
before running the real grid — no further adjustment afterward.

**Result: not a clean pass on either the stated "real result" or either failure mode alone —
a genuine mix, with a clear, diagnosable mechanism, not noise:**

| p_share | Apre | corr_w mean | uncorr_w mean | diff | post rate |
|---|---|---|---|---|---|
| 0.3 | 0.005 | 0.915 | 0.947 | -0.032 | 34.8 Hz |
| 0.3 | 0.02  | 0.941 | 0.943 | -0.003 | 44.4 Hz |
| 0.6 | 0.005 | 0.691 | 0.824 | -0.133 | 23.6 Hz |
| 0.6 | 0.02  | 0.550 | 0.954 | -0.404 | 33.0 Hz |
| 0.9 | 0.005 | 0.625 | 0.458 | +0.167 | 19.5 Hz |
| 0.9 | 0.02  | 0.663 | 0.292 | +0.370 | 17.8 Hz |

- **At low-to-medium correlation (`p_share=0.3`, partially `0.6`): Failure Mode 1 (runaway)
  dominates.** The trajectory plot shows both groups' weights shoot up to the ceiling within
  the first 5-10 simulated seconds — at `p_share=0.3, Apre=0.02`, all 20 synapses (both
  groups) end up in the top histogram bin (0.9-1.0), total undifferentiated saturation, before
  any correlation-driven signal has time to develop.
- **Only at `p_share=0.9` (both `Apre` levels) does the hypothesized "real result" emerge:**
  correlated group trends up (0.625-0.663), uncorrelated trends down (0.458-0.292) — visible
  in both the final histograms (real if imperfect separation, a few "defector" synapses in
  each group) and the trajectory plot (ongoing divergent drift over the full 60s window, not
  fully converged — may need more simulated time to fully settle).
- **Secondary pattern found, not in the original criteria but clear and non-noise:**
  uncorrelated group's mean weight decreases monotonically as `p_share` increases
  (0.945 -> 0.887 -> 0.375 averaged across `Apre`), despite the uncorrelated group's own
  statistics never changing. Plausible mechanism: as the correlated group increasingly drives
  postsynaptic spike timing at higher `p_share`, uncorrelated spikes become progressively
  less likely to land favorably pre-before-post relative to the now-more-structured
  postsynaptic spikes — a real competitive effect between the groups.
- **Root cause of the runaway, matching the risk flagged before running:** postsynaptic
  firing rates in every run (17.8-44.4 Hz) came out well above the calibrated 8.9 Hz baseline
  — direct evidence that weights climbed enough during the run to meaningfully raise
  postsynaptic excitability, which the fixed 5% depression bias isn't strong enough to
  counteract once the rate itself starts climbing (positive feedback: higher weight -> more
  reliable postsynaptic spikes -> more potentiation opportunities for every synapse whose
  presynaptic spike precedes them). Same shape of failure as the `w_char` lock-in from the
  Hopfield work, now confirmed in a structurally different mechanism.

**Verdict:** the correlation signal is real and can win — clearly demonstrated at
`p_share=0.9` — but currently only when strong enough to outcompete a runaway that dominates
most of the tested parameter space. Not adjusting parameters to chase a cleaner result; this
is the pre-registered grid, reported as it came out. Flagging for discussion rather than
deciding unilaterally: next step could be a homeostatic/weight-normalization mechanism to
damp the runaway (matching how the Hopfield work needed an explicit saturation fix for the
analogous `w_char` problem), or simply confirming whether higher `p_share`/longer runtime lets
differentiation win before saturation sets in.

---

## 2026-07-20 — Brian2 install check (start of the Brian2/SNN phase)

**Notebook:** `notebooks/brian2/brian2_install_check.ipynb`

Not an experiment — a toolchain check marking the actual start of the Brian2 phase flagged in
the roadmap after the two-layer/episodic memory work closed out. Installed `brian2==2.9.0` via
conda-forge into the `developing-mind` env (pulled in `gsl`, `cython`, and the MinGW-w64 GCC
toolchain for Windows C++ code generation as dependencies — code-gen target set to `numpy` for
this check specifically, to avoid needing to validate the C++ compiler path too).

Built a single leaky-integrate-and-fire neuron under constant current drive: standard
`tau*dv/dt = -(v-v_rest) + R*I` with threshold/reset/refractory. Confirmed (1) the neuron
integrates, spikes regularly (10 spikes over 200ms at ~18.8ms intervals, clean sawtooth
membrane trace), and resets correctly; (2) an f-I curve across 7 current levels is silent
below rheobase (~200-250pA) then increases monotonically (0 -> 52 -> 70 -> 94 -> 110 Hz) —
the expected LIF shape, not just one hardcoded run happening to spike. Equations, units,
threshold/reset/refractory, monitors, and plotting all confirmed working end to end.

**Status:** Brian2 install validated, safe to build real STDP work on top of. This is a
genuinely different skill than the PyTorch/Hopfield work (differential equations instead of
tensor ops, per `principles.md`'s roadmap note) — budget real ramp-up time, don't expect
Hopfield-speed iteration here.

**Follow-up, same day — compiled backend (`cython` target) checked directly, not assumed:**
the conda install pulled in `m2w64-gcc-libs`/`gsl` as dependencies, which looked like it might
be a MinGW toolchain for compiled codegen — checked and it's **runtime DLLs only**
(`libstdc++`, `libgomp`, etc.), no actual `g++`/`gcc` binary, not usable as a compiler on its
own. What actually compiles Brian2's generated C++ is **MSVC**, auto-detected via `vswhere`
from the VS Build Tools already installed on this machine (`cl.exe` present under
`...\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\...`, not on `PATH` but found
automatically by `distutils`/`setuptools`, standard Windows behavior) — not something that
needed fixing. Confirmed directly: `prefs.codegen.target = 'cython'` compiles and runs a
single neuron, and separately a network with `SpikeGeneratorGroup` + weighted `Synapses`
(`on_pre` effects) — the shape of thing real STDP work needs. No manual toolchain setup
required. **Recommendation: default to `prefs.codegen.target = 'cython'` for real work**
going forward instead of the `'numpy'` target used for the initial install check — much
faster for anything beyond toy-scale networks.

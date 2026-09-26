# Brian2 log: Boundary-mapping sweep and perturbation testing (2026-07-23/24)

Entries moved verbatim from `experiments_brian2.md` on 2026-09-25 (no wording changed). Index: `experiments_brian2.md`.

## 2026-07-23 — Perturbation-testing redesign #2, dense-ladder rerun: backwards contrast replicates at finer resolution — candidate (1) (ceiling-saturation resolution artifact) ruled out

**Data:** `notebooks/brian2/perturbation_data/run_validation_batch_v2_dense.py`, 8 new seed JSONs
(`perturbv2dense_*`, seeds 28000-28013). Per web's read of the round-2 backwards-contrast result:
before accepting either "ladder-top saturation" or "genuinely different quantity" as the
explanation, rule out the cheap one first. Same method (`run_perturbation_seed_v2.py`, weight-nudge
on the target's correlated synapses), same two reference points, same n=4/point, `dt=0.2ms` — only
change is a denser fraction ladder confined to the lower range web specified (`0.1, 0.2, 0.3, 0.4,
0.5, 0.6` — six rungs, none at 0.75/1.0) instead of round 2's coarse `0.25/0.5/0.75/1.0`. Added
`fractions=None` as an optional parameter to `run_perturbation_seed_v2`/its CLI (defaults to the
original ladder) rather than duplicating the ~200-line script, so both ladders share one
implementation.

All 8 seeds completed cleanly (0 failures, ~5 min total).

| point | seed | settled at | threshold_frac | censored_above |
|---|---|---|---|---|
| strong_tight_gate | 28000 | 750s | 0.5 | - |
| strong_tight_gate | 28001 | 450s | - | 0.6 |
| strong_tight_gate | 28002 | 500s | 0.1 | - |
| strong_tight_gate | 28003 | 400s | - | 0.6 |
| 13mV/1.5 | 28010 | 350s | - | 0.6 |
| 13mV/1.5 | 28011 | 300s | 0.1 | - |
| 13mV/1.5 | 28012 | 400s | 0.6 | - |
| 13mV/1.5 | 28013 | 300s | 0.2 | - |

**The backwards pattern replicated, not resolved by resolution.** If candidate (1) were right
(round 2's 0.75-1.0 rungs were just uniformly strong enough to swallow real separation), this denser
low-range ladder should have shown `strong_tight_gate` flipping readily down here while 13mV/1.5
stayed robust through 0.6. That's not what happened. `strong_tight_gate` flipped in only 2/4 seeds
within 0.1-0.6 (at fractions 0.1 and 0.5; the other 2 held all the way to 0.6, unbroken).
13mV/1.5 flipped in 3/4 seeds within the same range (at fractions 0.1, 0.2, and 0.6) — flipping
*more* often, at *lower or comparable* magnitudes, than the setting expected to have the shallower
basin. Same qualitative direction as round 2's coarse ladder, now on a second, independently-seeded
batch at different resolution.

Combined across both v2 batches (8 seeds/point total, round 2 + this rerun): `strong_tight_gate`
flipped in 5/8 seeds (fractions 0.1, 0.5, 1.0, 1.0, 1.0), held unbroken through its tested range in
3/8. 13mV/1.5 flipped in 7/8 seeds (fractions 0.1, 0.2, 0.6, 0.75, 0.75, 0.75, 1.0), held unbroken in
only 1/8. The presumed-deep-basin setting is the one that breaks more often and at lower magnitude,
consistently across two independent samples and two ladder resolutions.

**Verdict: candidate (1) is ruled out, not just unconfirmed.** This wasn't a resolution problem —
finer sampling in exactly the range web specified reproduced the same backwards direction rather
than revealing hidden separation. That leaves candidate (2) — web's proposed "different quantity"
explanation (weight-injection sensitivity and spontaneous-drift resistance may just be different
properties, with stronger inhibition/tighter gap_scale making the system MORE sensitive to a direct
weight nudge even while making it LESS prone to spontaneous internal drift) — as the leading,
undisconfirmed account. Still n=4-8/point, so not calling this fully settled, but it's now a
replicated pattern across independent seed sets rather than a single small-sample result. Reporting
to web, not redesigning or extending further solo.

**Closed, per web's reply — real, replicated, mechanistically explained.** Web's read: this is
a genuine structural finding, not a loose end needing more seeds. "Resistant to spontaneous
internal drift" and "resistant to a direct forced weight injection" are different axes that were
previously conflated under one "reliable/marginal" framing, and they don't rank operating points
the same way. Mechanistic story: 13mV/1.5's reliability comes from fast inhibition-driven
suppression of the loser before it can accumulate much correlated weight at all — there was never
much genuine depth to its disadvantage in weight-space, just a strong hand holding it down. Bypass
that hand via direct weight injection and there's less real resistance underneath than
`strong_tight_gate`, where the (less reliable) competition that does resolve is won on more
genuinely contested weight terms. Web's explicit call: stop here rather than chase more seeds —
the sign of the effect is unambiguous across two independent batches already, further n would
sharpen confidence without changing the conclusion. Documented as a named, transferable finding
in `principles.md` (suppression-based vs. dominance-based reliability), not just a footnote on
this one experiment — the load-bearing rule is that passive reliability must never be treated as
evidence of structural/perturbation-resistant depth anywhere this mechanism gets reused. This
closes the perturbation-testing thread and the broader Experiment B population-competition arc
for now.

---

## 2026-07-23 — Perturbation-testing redesign #2 (weight-nudge): a genuine graded signal this time, but the expected contrast direction didn't hold — reported, not patched solo

**Data:** `notebooks/brian2/perturbation_data/run_perturbation_seed_v2.py`, `run_validation_batch_v2.py`,
8 seed JSONs (`perturbv2_*`). Per web's redesign after round 1's I_pert failure: directly nudge
the target's CORRELATED-synapse weights toward the ceiling (`wmax=1.0`), calibrated as a fraction
of the way there (0.25/0.5/0.75/1.0), rather than forcing extra firing via current injection.
Ladder magnitude is now grounded in the bimodal post-hoc analysis (winning correlated synapses
cluster near `wmax`, not an arbitrary unit). Uses the validated `dt=0.2ms` throughout per web's
instruction. Includes live monitoring (tier state printed every chunk) per web's separate ask.

**A real implementation bug caught before any real run, not after:** `syn.w[boolean_mask] =
values` (Brian2 Synapses state-variable assignment) reproducibly raised `ValueError: Provided
values do not match the size of the indices, 10 != 60` -- but only once enough simulated time had
elapsed (fails after a 450s run, works fine after 1s, identical mask/values shapes both times).
Root cause not fully chased down (plausibly an interaction between Brian2's Cython codegen index
caching and the long-running `run_regularly` scaling op), but confirmed directly that integer
indices (`syn.w[np.where(mask)[0]] = values`) work reliably at both durations -- switched to that
form, smoke-tested clean before the real batch.

**Mandatory validation (4 seeds/point, `dt=0.2ms`, ~5 min total vs round 1's ~14 min -- the
speedup is real and paying off): a genuinely different, more informative failure mode than
round 1's.** Nearly every seed flipped at SOME ladder rung this time (7/8, vs round 1's
1/8) -- and inspecting the underlying trajectory (not just the pass/fail table) shows why this
result is trustworthy in a way round 1's wasn't: the target's own gap now shows real "boost, then
partial pull-back" dynamics across the ladder. E.g. seed 27000: `corr_w` nudged to 0.840 at
frac=0.25, but by the next rung's baseline it had decayed back to 0.784 -- genuine reclamation
happening in the recovery window, not the destructive monotonic collapse round 1 showed. This is
what the method was supposed to measure.

**But the expected contrast direction didn't hold:**

| point | seed | baseline top tier | threshold_frac | censored_above |
|---|---|---|---|---|
| strong_tight_gate | 27000 | [0] | 1.0 | - |
| strong_tight_gate | 27001 | [0,2] | 1.0 | - |
| strong_tight_gate | 27002 | [0,1,2] (converged) | 1.0 | - |
| strong_tight_gate | 27003 | [0,1,2] (converged) | - | 1.0 |
| 13mV/1.5 | 27010 | [1] | 0.75 | - |
| 13mV/1.5 | 27011 | [1,2] | 0.75 | - |
| 13mV/1.5 | 27012 | [0,2] | 1.0 | - |
| 13mV/1.5 | 27013 | [1,2] | 0.75 | - |

`strong_tight_gate` needed the maximum tested fraction (1.0) to flip in 3/4 seeds (the 4th was
already fully converged, censored). 13mV/1.5 flipped at the *lower* fraction (0.75) in 3/4 seeds.
If anything, in this small sample, 13mV/1.5 -- the setting expected to be the *harder*-to-flip,
deep-basin point -- flipped more easily than the marginal, supposedly-shallow-basin
`strong_tight_gate`. Backwards from web's stated expectation, on n=4 per point.

**Verdict, stated plainly rather than either dismissed or over-interpreted:** this is a
structurally different situation from round 1. Round 1 failed because the measurement itself was
confounded -- the numbers were actively lying (deterministic, in every seed). Round 2's mechanism
looks sound on inspection (real graded dynamics, genuine pull-back, no obvious confound found so
far) but the actual DIRECTION of the reliability-vs-basin-depth relationship came out opposite to
what both of us expected going in, on a small sample. Two real possibilities, not distinguished by
this data: either the perturbation ladder's upper range (0.75-1.0, i.e. nudging most of the way to
the literal weight ceiling) is simply strong enough to overturn structure almost everywhere
regardless of true basin depth, making the ladder's TOP end uninformative and the real signal (if
any) living in a range not finely sampled here — or the relationship between inhibition strength
and basin depth genuinely isn't monotonic the way the "reliable = deep basin" framing assumed.
Not deciding between these solo. **Reporting to web before any further redesign or additional
seeds** -- two rounds is exactly the point web's own instruction named for checking in again
rather than self-approving a third attempt.

---

## 2026-07-23 — Perturbation-testing validation: FAILED, for a mechanistic reason, not a statistical one — I_pert doesn't test basin depth, it confounds two other effects

**Data:** `notebooks/brian2/perturbation_data/` (`run_perturbation_seed.py`, `run_validation_batch.py`,
8 seed JSONs, `run_dt_check.py` + `run_dt_batch.py` for the side-task). New: `src/brian2_stdp/network.py`
gained `I_pert` (per-neuron resting-potential offset, confirmed bit-identical to the original
equation at `I_pert=0` via dedicated regression test). Per web's redesign after the 600s reentry
probe's confirmed timescale limit: instead of passively waiting for spontaneous reorganization,
actively perturb the excluded neuron with escalating magnitude and measure the threshold at which
the hierarchy permanently changes — a graded basin-depth number instead of a rare-event binary.

**Side-task result first (cheap, resolved before the main experiment):** measured the actual
presynaptic ISI floor directly (0.2ms, from `spikes.dedup_spike_times`' `min_gap`) rather than
guessing a coarser dt. `dt=0.2ms` gives a real, validated 4x wall-clock speedup with zero
same-timestep spike collisions. `dt=0.5ms` (the originally-guessed value) causes hard Brian2 errors
and, separately, measurably shifts the outcome distribution at 13mV/1.5 (differentiate rate
7/8→5/8, late-window std roughly halved) — confirming the jitter-aliasing risk flagged before
running, not adopted. `dt=0.2ms` was used for validation and (see below) does not appear to be the
cause of what follows.

**Mandatory validation (per web's explicit instruction, before touching any intermediate point):**
4 seeds each at `strong_tight_gate` (10mV/1.0) and 13mV/1.5, full ladder (1/2/4/8/16mV, 50s hold +
200s recovery per rung, ~14 min total for all 8 seeds, 0 failures).

**Surface-level result:** 13mV/1.5 — 0/4 seeds flipped at any magnitude (censored above 16mV in
all four). `strong_tight_gate` — 1/4 flipped, at the *smallest* tested magnitude (1mV); the other
3/4 censored above 16mV, same as every 13mV/1.5 seed. A real but thin contrast (1/4 vs 0/4, n=4 per
point) — and on its own, not obviously distinguishable from the ALREADY-KNOWN heterogeneity at
`strong_tight_gate` itself (near-permanent lock-in, e.g. seeds 5001/6002, is one of the established
categories at that exact operating point, not a validation failure).

**Didn't stop at the surface number — inspected the actual target-neuron gap trajectory across the
magnitude ladder for all 8 seeds, and found something that invalidates the method, not just makes
the contrast thin:**

```
target_baseline -> target_hold_gap at magnitude [1, 2, 4, 8, 16 mV]
0.572 -> [0.565, 0.547, 0.521, 0.461, 0.416]   (13/1.5, seed 26010)
0.395 -> [0.380, 0.371, 0.272, 0.124, 0.164]   (13/1.5, seed 26011)
0.386 -> [0.375, 0.379, 0.314, 0.265, 0.265]   (13/1.5, seed 26012)
0.387 -> [0.389, 0.392, 0.309, 0.187, 0.070]   (13/1.5, seed 26013)
0.384 -> [0.375, 0.373, 0.365, 0.146, -0.148]  (strong_tight_gate, seed 26001)
0.421 -> [0.397, 0.368, 0.300, 0.274, 0.070]   (strong_tight_gate, seed 26002)
0.398 -> [0.396, 0.376, 0.320, 0.168, 0.003]   (strong_tight_gate, seed 26003)
```

**Every single seed shows the same monotonic pattern: the target's own correlated-vs-uncorrelated
gap DECLINES as perturbation magnitude increases, reaching near-zero or negative at 16mV.** The
perturbation was intended to help the excluded neuron build competitive strength; instead, at
sufficient magnitude, it destroys the very thing (correlated-input selectivity) that would need to
rise for it to genuinely compete. Mechanistically this makes sense on reflection: `I_pert` pins the
target's resting potential closer to threshold, making it fire more regardless of whether that
firing is triggered by correlated input specifically — STDP then potentiates correlated and
uncorrelated synapses roughly equally, collapsing the neuron's own selectivity rather than sharpening
it.

**Checked the one seed that DID flip (26000, at just 1mV) to see whether it at least demonstrates
genuine reentry through a different pathway — it doesn't.** The target's own gap stayed essentially
flat throughout (baseline 0.593 → final 0.562, no real rise). What actually happened: the *leader's*
gap declined (0.671 → 0.599) until all three converged into a single tied tier `[0,1,2]` — not "the
excluded neuron reclaimed leadership," but "the leader got pulled down toward a three-way tie."
Plausible mechanism: lateral inhibition fires per postsynaptic spike, not per rate-difference alone
(`network.py`'s `inhib` Synapses trigger `on_pre`, one event per source spike) — so even a small
`I_pert`-driven increase in the target's firing RATE directly increases the total inhibitory output
it delivers to *both* competitors, independent of anything about its own learned correlated-weight
advantage. That's a real effect, but a mechanistically different one than "how much force does the
basin resist" — it's closer to "does forcing extra firing from a follower knock the leader down via
raw inhibitory volume," which conflates the population's shared-inhibition dynamics with the
individual-neuron competitive-strength question the experiment was designed to isolate.

**Verdict: the method as implemented does not test what it was designed to test. This is not the
1/4-vs-0/4 contrast being too thin to trust statistically — it's that inspecting the underlying
mechanism shows the confound is present in literally every single tested seed, deterministically,
not probabilistically.** A resting-potential-offset perturbation confounds two effects that both
move the observable outcome in ways unrelated to genuine basin depth: (1) destroying the target's own
input-selectivity at higher magnitudes, and (2) increasing the target's raw inhibitory output to
competitors via elevated firing rate, independent of (1). Neither is "the excluded neuron rebuilding
correlated-weight strength and reclaiming the lead on its own merits," which is what the basin-depth
framing needs to mean something. **Not proceeding to any intermediate point. Reporting this to web
before attempting a redesign** — this is a design-level problem, not a parameter to retune solo.

---

## 2026-07-23 — Boundary-mapping sweep: does a reliable-and-rich region exist between strong_tight_gate and 13mV/1.5? Undetermined at 600s — the timescale itself is the obstacle, confirmed directly, not assumed

**Data:** `notebooks/brian2/boundary_sweep_data/run_boundary_sweep.py`, `analyze_boundary_sweep.py`,
128 seed JSONs, `boundary_analysis.json`. New shared code: `src/brian2_stdp/metrics.py` gained
`detect_tier_reentry` (generalizes the by-hand tier-reentry check from step 4). Per web's message:
Test A/step 4 found zero genuine reorganization at the reliable 13mV/1.5 setting at two N values —
does a region exist between that point and the known-rich-but-unreliable `strong_tight_gate`
(10mV/1.0, ~50% reliable) where both reliability and genuine reorganization coexist, or do they
trade off cleanly across the whole range?

**Design:** 4×4 grid, inhib_strength ∈ {10,11,12,13}mV × gap_scale ∈ {1.0,1.17,1.33,1.5}mV
(spans exactly between the two known points), 8 seeds/point (128 runs), 600s, N=3 (matching every
prior sweep at this scale). Two metrics tracked separately per web's instruction: reliability
(differentiate rate) and reentry_rate (fraction of ALL seeds showing genuine tier reentry via
`detect_tier_reentry`, not swap-count).

**A real bug caught before trusting the first result, then a second, deeper one caught after
"fixing" the first — worth walking through both, not just reporting the final number:**

1. **First pass:** reentry_rate came back suspiciously high and uniform across the *entire* grid
   (12-62%), including at the exact 13mV/1.5 endpoint where Test A/step 4 had already found zero
   reentry in 8 seeds. That contradiction was the tell. Inspected two flagged seeds directly:
   both were still visibly in the middle of the initial differentiation transient (gap values
   monotonically rising from their common starting point, one neuron catching up to another over
   250-400+s) when `detect_tier_reentry`'s fixed 100s washout treated a too-early window as the
   "settled baseline" — the exact false-positive class already caught by hand in step 4, just not
   carried into this function's defaults when it was generalized into reusable code.
2. **Fix (round 1):** replaced the fixed washout with adaptive per-seed settling detection — find
   the first point where the top tier stays IDENTICAL for 3 consecutive windows, use that as the
   baseline. Re-verified against the same two seeds: fixed one, but the other (24124) *still*
   showed a false positive. Root cause: a slow, continuous convergence can keep each individual
   window-to-window step under the 0.03 threshold while still accumulating real drift across the
   whole "stable" stretch — top-tier-set stability alone isn't sufficient evidence of an actual
   plateau.
3. **Fix (round 2):** required the full per-neuron VALUE range across the whole stable stretch
   (not just pairwise adjacent-window steps) to stay under threshold. Both regressions now pass
   as dedicated unit tests (`test_detect_tier_reentry_does_not_flag_slow_initial_convergence_as_reentry`),
   not just fixed ad hoc.

**Re-ran the analysis (no re-simulation needed) with the corrected metric: reentry_rate dropped
to 0.00 at 13 of 16 grid points, non-zero (0.12, i.e. 1/8 seeds) at exactly 3.** Inspected all
three non-zero cases directly rather than accepting the number. All three show the *same* pattern
as the two already-fixed false positives: slow, extended, non-plateauing drift continuing for
most of the 600s run, with no clean, sharp "settled, then something new happened later" structure
— ambiguous at best, not a confirmed instance of genuine reentry. **Decisive check: even
`strong_tight_gate` itself (10mV/1.0) — included in this grid, the one point already known with
certainty to show real reorganization at full 5000s duration — reads reentry_rate=0.00 here.**
That is the clean confirmation, not an assumption: if the reference point that *is* rich reads as
zero, this design cannot be trusted to detect richness anywhere else in the grid either.

**Verdict: the boundary-mapping question is undetermined at 600s — not because of a remaining
bug, but because the settling process itself, in this system, does not reliably finish within a
600s window for a meaningful fraction of seeds.** The pre-stated caveat (genuine reorganization
in the original typology took 1000-2600s, past 600s) turns out to understate the problem: it's
not just that late reorganization is missed, it's that the network can still be in its *initial*
relaxation for most or all of a 600s run, leaving no reliable "settled" baseline to check reentry
against in the first place. No amount of smarter windowing fixes that within a fixed 600s budget
-- the fix here catches false positives, it can't manufacture time that wasn't simulated.

**What this sweep DOES answer cleanly:** reliability itself (0.62-1.00 across all 16 points, no
sharp cliffs) and disorder_rate (0.00 at every single point in this range) — the "never settles
at all" pattern doesn't appear anywhere between the two known operating points, which at least
rules out that specific failure mode across this whole region. What it cannot answer: whether the
5003-style one-time-late-reorganization pattern exists anywhere in this range, reliable or not —
that requires runs long enough for the network's own relaxation to actually finish, which 600s
frequently isn't.

**Decision (web, after review): do not rerun at a guessed longer duration.** A 1500-2000s rerun
would itself be an unvalidated duration — the exact same category of problem this round just
surfaced with 600s, just kicked down the road rather than solved. The one duration already known
to be sufficient is 5000s (validated by Test A and step 4). If this boundary is ever mapped
properly, the right approach is a handful of specific intermediate points extended to the full
5000s the way Test A and step 4 were, not a new "probably long enough" guess at cheap scale.

**Status: closed as honest, precisely-scoped debt, not chased further right now.** The boundary
zone's shape (needle-thin point vs. real region) is genuinely unknown. The cheap 600s probe
methodology has a confirmed timescale limit and cannot answer this question at any tuning of the
detection algorithm — two rounds of fixing `detect_tier_reentry` establish that directly, not by
assumption. Answering it properly requires full 5000s runs at a handful of intermediate points
between `strong_tight_gate` and 13mV/1.5, a real cost, not undertaken tonight since nothing
currently depends on resolving it. This closes out the population-competition thread for now —
reliability (bistability sweep), hierarchy shape (N-scaling curve), and the reliability-vs-
reorganization tension (Test A, step 4, this boundary probe) are all now characterized as
precisely as tonight's tools allow, with what's still unknown stated plainly rather than glossed.

---


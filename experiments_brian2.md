# Experiments Log — Brian2 / SNN phase

Hopfield / two-layer-memory / episodic-layer work lives in `experiments.md`. This file is
for the Brian2/spiking-dynamics phase specifically — different tools (differential equations
instead of tensor ops), different failure modes, split out at the natural phase boundary
rather than mixed into an already-long single log.

## 2026-07-21 — STDP/SNN arc: consolidated summary

Closing summary for the Brian2/STDP thread — every entry below is kept in full as the detailed
record; this is a synthesis on top, not a replacement. Fifteen-plus entries deep now (runaway →
two rejected hypotheses → amplitude-vs-frequency reframe → long-run stability check →
population extension → shared-input competition → bistability → n=7 seed expansion →
positional-bias resolution) — read this for the throughline, follow the links below for
mechanism and data.

**What's validated:**

- **The core claim holds, robustly, at every scale tested.** STDP alone, zero supervision,
  learns to differentiate correlated from uncorrelated input from spike-timing statistics —
  confirmed at a single postsynaptic neuron, at an independent-replicate population (N=5), and
  at a genuine shared-input competitive population (N=3, lateral inhibition). Not a
  single-neuron artifact, not an artifact of any one network configuration (see the "First STDP
  experiment", "Population extension", and "Shared-input population competition" entries).
- **The runaway problem and its fix.** Unconstrained STDP produces positive-feedback lock-in —
  higher weight → more reliable postsynaptic spikes → more potentiation, both groups saturate
  together before any real signal develops. Homeostatic synaptic scaling (a `(summed)` Brian2
  variable rescaling each postsynaptic neuron's own incoming weights toward a fixed target)
  contains this cleanly without eliminating the differentiation signal — confirmed via total
  weight sum and postsynaptic firing rate, not just individual-synapse saturation (see
  "Homeostatic synaptic scaling" entry).
- **The stability finding, precisely stated.** Individual synapses never truly converge —
  reversal frequency is essentially constant (~140-500/synapse, scaled by duration) regardless
  of learning rate, at every `Apre` tested. `Apre` controls excursion *amplitude*, not switching
  *frequency* — this is the STANDING EXPLANATION, don't re-derive it (see "Is the Apre
  instability a threshold or a gradient" entry). Despite that, group-level differentiation stays
  genuinely bounded away from collapse over timescales far exceeding naive diffusion estimates,
  in the large majority of realizations tested — genuine-but-probabilistic stability, not
  absolute (see the 8-seed/5000s ensemble entry).
- **The multi-neuron result — the strongest one, and the actual payoff of this arc.** A stable
  population-level signal ("someone represents the correlated pattern") holds even when
  individual identity churns underneath it, across a real n=7 sample in a genuine shared-input
  competitive population: ranging from near-permanent individual-level lock-in, through
  one-time reorganization (two distinct sub-mechanisms), to seeds that never lock into a
  hierarchy at all yet still produce a rock-solid population signal — the strongest instance of
  this property found. This directly validates the population-readout stability definition in
  `principles.md`, tested (not just extrapolated) at the multi-neuron scale that decision was
  originally missing (see "Seed expansion (n=2 -> n=7)" entry).
- **The bimodal winner-take-most structure.** "Group-mean gap stays bounded" was always
  accurate but coarser than reality — final weights are bimodal (synapses cluster near the clip
  boundaries, not the middle), and roughly a third of the "winning" correlated group's own
  synapses land at the floor, behaving like losers. Confirmed independently across two
  structurally different datasets (see the bimodal-distribution entry).

**What's still open:**

- **The bimodal structure's timing dynamics are unknown.** None of the early ensemble/
  population runs saved full per-synapse time traces (only aggregates, finals, and reversal
  counts) — so *when* a synapse commits to its final side, and whether "defectors" flip back
  and forth or settle early, isn't answerable without a rerun. Future long runs should save at
  least a subsampled per-synapse trace (see the design note in `src/brian2_stdp/network.py`).
- **Whether the ~50/50 bistability rate at `strong_tight_gate` holds at other operating
  points is untested.** Only one inhibition-strength/gap_scale combination has been
  characterized in any depth; whether other combinations show the same bifurcation structure,
  a different split ratio, or no bistability at all is a real, unstarted follow-up.
- **The n=7 seed-expansion result is a typology, not a distribution** — the relative frequency
  of each individual-identity regime (near-permanent lock-in / one-time reorganization / never
  locks in) is observed at this sample size, not established as a stable rate. Stated explicitly
  in `principles.md` rather than implied.

---

## 2026-07-23 — Population-competition bistability sweep: reliability rises with inhib_strength, richness doesn't trade off against it

**Data:** `notebooks/brian2/bistability_sweep_data/run_sweep.py` (orchestrator), `analyze_sweep.py`
(analysis), 128 per-seed result JSONs, `sweep_analysis.json` (aggregated grid). Shared code:
`src/brian2_stdp/metrics.py` gained `classify_differentiation` (the late-window
cross-neuron-gap-std>0.03 basin classifier, extracted from the informal criterion used ad hoc for
the original `strong_tight_gate` calibration/seed-expansion, now reusable and tested). Per web's
design message: `strong_tight_gate` (inhib=10mV, gap_scale=1.0) showed ~50/50
converge/differentiate across 14 seeds — before treating that as *the* operating point, map the
actual shape of the tradeoff, and specifically whether reliability and the rich individual-identity
dynamics from the n=7 extension can coexist or trade off, the way savings-vs-content-fidelity did
in the Hopfield work.

**Question:** across a grid of `inhib_strength` x `gap_scale`, does a region exist where
differentiation is *reliable* (most seeds differentiate) AND the individual-identity spectrum
(some seeds locking in fast/hard, others staying loose/noisy) still shows up — or does reliability
only rise by making differentiation uniformly rigid/winner-take-all, a real tradeoff?

**Design:** 4x4 grid, `inhib_strength` in {6, 8, 10, 13} mV x `gap_scale` in {0.5, 1.0, 1.5, 2.0}
(spans the known converged corner at 6/2.0 up through the known ~50/50 point at 10/1.0), 8 fresh
seeds per point (128 runs total, seed block 21000+, no overlap with any prior batch), 600s each
(calibration scale, matching the original characterization — full 5000s extension deliberately
deferred to whichever region looks promising, not run at every point). Tracked two things per
point, not a single collapsed number: **reliability** (fraction differentiating, via
`classify_differentiation`) and **richness** (among differentiating seeds only: holder-identity
swap count within the 600s window, via the already-validated `count_identity_swaps` — an explicit
proxy for the identity-churn spectrum, not the full frozen/reorganization/never-locks-in
classification, which needs the full 5000s to resolve).

**Falsification criteria, stated before running (in `analyze_sweep.py`, written before the sweep
launched):** the interesting outcome is a region with reliability >=80% where richness
(swap-count spread across differentiating seeds, `std_swaps`) stays real, not collapsed toward
zero. A region where reliability only rises as richness drops would be a genuine tradeoff finding,
reported as such, not fished into a "best of both" reading.

**Infrastructure note:** built a Python-level batch orchestrator (`run_sweep.py`, concurrency-
limited `subprocess` pool) rather than bash `&`/`wait` loops — 128 jobs is too many to manage
safely with shell backgrounding at this scale, and this project already hit a real bash
subshell-variable-scoping bug once at much smaller scale. Calibrated wall-clock first (6-job batch,
~300s/round at 6-way concurrency, matching the machine's physical core count) before committing to
the full grid; actual full run finished in 4568s (~76 min, faster than the ~107 min estimate — later
rounds ran closer to ~165s/job as the OS-level page/disk cache warmed up). 0/128 runs failed.

**Two apparent anomalies checked before trusting anything else, not glossed over:**
1. `strong_tight_gate` itself (10mV/1.0) landed at 1/8 (12.5%) differentiating in this fresh
   sample — well under the established ~50% (n=14) rate. Checked directly: all 8 seeds'
   `late_window_std` values sit far from the 0.03 threshold in either direction (converging seeds
   0.0005-0.0069, the one differentiating seed 0.0913) — a clean classification, not a
   threshold-boundary artifact. Binomial variance at n=8 against a true ~50% rate lands at <=1
   success about 3.5% of the time; with 16 grid points swept, seeing one point that low somewhere
   isn't remarkable. Doesn't revise the established ~50% figure (n=14 remains the better-powered
   estimate for this exact point) -- flagged here rather than silently smoothed over.
2. `medium` (6mV/2.0), previously characterized as reliably converging, differentiated in 5/8
   fresh seeds here (62%) -- checked the same way, all classifications clean (converging seeds
   std 0.003-0.006, differentiating seeds 0.047-0.131, nothing near the boundary). This one *does*
   revise the prior picture, honestly: `medium`'s "fully converges" characterization rested on a
   single calibration seed, never actually replicated at n>1 the way `strong_tight_gate` was --
   this is the same "small n can mislead" trap the project already learned once with
   `strong_tight_gate` itself, now caught at `medium` too before it got treated as a settled fact.

**Result: reliability rises clearly with `inhib_strength`; richness does not collapse as it
does.** Average reliability by row: 6mV=0.41, 8mV=0.59, 10mV=0.50 (dragged down by the anomaly
above), 13mV=**0.88**, consistently high (0.75-1.00) across all four `gap_scale` values tested at
13mV -- the clearest single-parameter driver in the grid. Correlation across all 15 grid points
with >=2 differentiating seeds: `corr(reliability, std_swaps) = +0.10` -- richness's *spread*
across seeds does not shrink as reliability rises, no collapse into uniform rigidity.
`corr(reliability, mean_swaps) = -0.47` -- a real but moderate tendency for the *average* seed to
lock in a bit faster/sharper at higher reliability, but that's a shift in the average, not a loss
of variety: individual seeds at every reliability level still range from near-immediate lock-in to
persistent churn.

**Standout point: inhib=13mV, gap_scale=1.5 -- reliability=1.00 (8/8 differentiate), and the
richest spread in the "reliable" set.** Per-seed identity-swap counts: 3, 96, 17, 62, 11, 100, 70,
112 -- a genuine spread from fast/hard lock-in to persistent reshuffling, all within seeds that
differentiated cleanly (`late_window_std` 0.086-0.264, nowhere near the 0.03 boundary). This is
the target outcome stated before running: reliable AND rich, not a coin flip and not a rigid
monoculture either.

**Verdict:** the pessimistic tradeoff hypothesis (reliability only bought at the cost of rigid,
uniform winner-take-all dynamics) is not what this grid shows -- richness survives, even
strengthens in absolute terms, in the region that differentiates most reliably. `inhib_strength`
(not `gap_scale`) is the primary lever for reliability. **Next step, not run this round per the
original scoping:** extend inhib=13mV (gap_scale=1.5 specifically, or the row more broadly) to the
full 5000s duration, the way 5001/5003 were extended from the original bistability check -- that's
the deliberate follow-up once a promising region is identified, not something to run at every grid
point.

---

## 2026-07-21 — Positional-bias follow-up: the p=0.068 chi-square washes out to noise with more data, resolved

**Data:** `notebooks/brian2/apre005_ensemble_data/run_positional_bias_replicate.py`,
`positional_bias_extension/seed7001.json` through `seed7030.json`. The fallback queued while
Jasper was away (never triggered then, since the competitive-population calibration cleared its
bar) — picked back up now that the population-competition thread reached its own stopping
point, per the stated priority order.

**Question:** the population extension's EDA found a suggestive-but-not-significant spread in
per-correlated-index floor rates at Apre=0.005 (15%-55% across the 10 indices, chi-square
p=0.068, n=20) — plausibly early random luck locking in during the ~24-98s relaxation window
before 20 replicates was enough to average it out, but explicitly flagged as unresolved rather
than either dismissed or oversold.

**Design:** 30 new independent single-neuron (N=1) replicates at Apre=0.005, 1000s each (`
build_network` as-is, no new mechanism), reusing the fully-validated single-neuron rig. Wall-
clock measured directly (0.1911s/s on a 200s test run, not assumed) before committing to the
batch size — ~191s/replicate, 30 replicates at 5-concurrent/6-wave ≈ 19-22 min, matching the
concurrency level already characterized rather than pushing higher and risking another
contention surprise. Pooled with the original 20 into one real n=50 combined chi-square, not
two separate checks compared informally, per explicit instruction.

**Result: the spread washes out with more data — moves away from significance, not toward it,
exactly what noise looks like:**

| sample | n | chi-square | p | floor-rate range |
|---|---|---|---|---|
| original (population extension) | 20 | 15.93 | 0.068 | 15%-55% |
| new (single-neuron replicates) | 30 | 9.41 | 0.400 | 16.7%-46.7% |
| **pooled** | **50** | **10.61** | **0.304** | **18%-42%** |

The new 30 alone show no structure at all (p=0.40) and don't even agree with the original 20 on
*which* index looks elevated (original: index 4 highest at 55%; new: index 5 highest at 46.7%)
— a real positional effect should reproduce roughly the same shape on a fresh independent draw;
a noise artifact should not, and does not. Pooling to n=50 narrows the range further (18%-42%,
down from the original 20's 15%-55%) and pushes p further from significance, not closer — the
signature of averaging out chance fluctuation, not of an underdetected real effect gaining
power.

**Verdict: resolved.** The per-index floor-rate spread was noise from the early-relaxation-
window lock-in mechanism already hypothesized, not a positional effect in the mechanism. No
change needed anywhere else in the codebase or log — `spikes.py`'s generation process was
already known not to treat indices differently, and this confirms there's no hidden effect it
was missing. Closes the "open, unresolved" item flagged in the bimodal-distribution entry below.

---

## 2026-07-21 — Seed expansion (n=2 -> n=7): individual-identity behavior is a spectrum, population signal is the invariant across all of it

**Data:** `notebooks/brian2/competitive_population_data/competitive_seed6001.json`,
`competitive_seed6002.json`, `competitive_seed6007.json`, `competitive_seed6008.json`,
`competitive_seed6009.json` (full traces, same format as 5001/5003). Follow-up to the entry
below, per Jasper + the external review instance: the original result rested on exactly 2 seeds
that happened to land in the differentiating basin — before treating it as a settled result
rather than "we saw it twice," it needed a real sample.

**Design:** same `strong_tight_gate` combo (inhib=10mV, gap_scale=1.0), 600s calibration-scale
check across 10 fresh seeds (6001-6010, no overlap with 5001-5004), sorted into the two known
basins by the same numeric criterion as before (late-window cross-neuron gap std > 0.03 =
differentiate). Every differentiating seed extended to the full 5000s; converging seeds not
extended (nothing to track over time in that branch).

**Basin split confirms the original ratio wasn't noisy at small n:** 5/10 differentiate (6001,
6002, 6007, 6008, 6009), 5/10 converge (6003, 6004, 6005, 6006, 6010) — landing almost exactly
on the original 4-seed ratio (2/4 = 50%). With n=4 alone the true rate could plausibly have been
anywhere from ~20-80%; with n=14 total now sitting at ~50%, that's a meaningfully tighter bound.
Stated as its own small result, not just background on the way to the extension numbers — and
explicitly *not* claimed as a precise rate, just a much narrower plausible range than n=4 gave.

**Wall-clock, checked rather than assumed:** running all 5 extensions in parallel (vs. 2 for the
original pair) came in at ~1465-1496s per seed (avg ~1480s) — about 25% slower than the
2-parallel baseline of ~1180s. Real degradation going from 2 to 5 concurrent processes on 6
physical cores, not the clean linear scaling assumed going in. Worth having for sizing future
batches at this concurrency level.

**A false alarm caught before it went in the log:** seed 6002 initially showed
`population_max_gap` min=-0.0081 after excluding t<5s, which looked like the first real
zero-crossing among any differentiating seed. Checked precisely: it's a single sample at
exactly t=5.0s, the tail edge of the initial climb from zero, not a genuine late-run crossing.
Not an exception — flagged and dropped before being reported as one.

**Result: population-level signal held in all 7 differentiating seeds, no exceptions.** Every
seed's `population_max_gap` stayed clearly positive for the full 5000s once the trivial startup
transient is excluded. That's the core, invariant finding, and it's the same across everything
described below.

**What varies is how much individual-level structure ever locks in, and for how long -- a
spectrum, not a clean binary, and richer than the original two categories anticipated (this is
the ambiguous-middle case explicitly flagged as worth its own honest treatment rather than being
forced into an existing bucket):**

- **Near-permanent individual lock-in (5001, 6002):** a hierarchy forms early (~t=150s) and
  never changes structurally for the rest of the run. The only churn is noise-level rank-trading
  between two neurons close enough that it doesn't matter (5001's original 863-swap case).

- **Individual lock-in, but not permanent -- one real structural transition, then a new
  hierarchy locks in and holds (5003, 6007, 6001):** worth keeping the sub-flavors visible
  rather than flattening them, since they're genuinely different mechanisms:
  - *Laggard promoted* (5003): a clear laggard (~0.35-0.40) rises to join the leader tier
    (~0.58) at t~2600s, late, well past relaxation.
  - *Intermediate promoted, earlier* (6007): a neuron starting at an intermediate level (~0.47,
    between the laggard at ~0.40 and the leader at ~0.58) rises to join the leader by t~1000s;
    the actual laggard is untouched throughout.
  - *Leader-pair-internal merge* (6001): one leader starts elevated (~0.73) and decays down to
    match its partner (~0.58) by t~2000-2500s; the laggard is untouched throughout -- a
    reorganization within the leader tier, not a laggard promotion at all.

- **Individual lock-in never happens, ever (6008, 6009) -- and this is the strongest instance
  of the stability-definition decision found so far, not just an odd third case:** no hierarchy
  forms at any point across the full 5000s. 6009's per-neuron gaps are already near-identical
  (0.58/0.58/0.59) in the very first sampled window (t=100-500s) and stay that way the entire
  run -- never shows a laggard phase at all. 6008 starts closer together than any other
  differentiating seed (0.53/0.52/0.58 early) and locks into near-parity by t~1000s. Both then
  show constant, large-amplitude reshuffling among all three neurons for the rest of the run --
  1203 and 1353 holder-identity swaps respectively, the two highest counts of all 7 seeds.
  Genuinely different from the *converged* seeds (which lock tightly together at every instant,
  cross-neuron std ~0.007-0.01): 6008/6009 keep real ongoing spread at any given moment, they
  just never let any one neuron settle into a stable lead or lag. **A stable population-level
  readout with *zero* persistent individual identity, not even a temporary one** -- if the
  population-readout framing in `principles.md` were challenged on "sure, but individual units
  mostly do settle down eventually," 6008/6009 are the direct counterexample.

**Framing, stated precisely rather than oversold:** this is a typology observed at n=7
(2 near-permanent-lock-in, 3 one-transition in two sub-flavors, 2 never-locks-in), not a
distribution. Do not read "3/7 show reorganization" as a stable ~40% rate -- the relative
frequency of each regime is unknown at this sample size, only the existence of all three regimes
and the invariance of the population-level signal across them are established. The one precise
rate this entry does support is the ~50% differentiate-vs-converge basin split (see above), which
is a different question from the typology within the differentiating basin.

**Verdict:** strengthens, and adds real texture to, the multi-neuron closure of the
stability-definition decision (`principles.md`). Population-level stability isn't just
compatible with individual-identity churn -- it's compatible with a whole spectrum of
individual-level behavior, from near-total identity permanence through one-time reorganization
to individual identity never resolving at all, with the population signal equally solid in every
case tested. Not yet known: the relative frequency of each regime, or whether other operating
points (beyond `strong_tight_gate`) show the same spectrum or something structurally different.

---

## 2026-07-21 — Competitive-population extension to 5000s: population-level signal stays stable while individual identity genuinely churns

**Data:** `notebooks/brian2/competitive_population_data/competitive_seed5001.json`,
`competitive_seed5003.json` (full per-synapse traces, per-neuron spike-rate bins, r-traces,
metadata). Follow-up to the bistability entry directly above — seeds 5001 and 5003 (the two of
four calibration seeds that landed in the differentiating basin at `strong_tight_gate`) re-run
fresh at the full 5000s duration, per the external review instance's decision to extend known-
differentiating seeds rather than gamble the full budget on fresh draws or chase parameters
toward more reliable differentiation (see the entry above for the full reasoning and the
explicit flag that this was decided autonomously while Jasper was away).

**Question, restated precisely:** does the population-level signal — is the correlated pattern
represented by *someone* — stay stable over a long run even as *which* neuron holds that role
drifts or swaps? This is the actual multi-neuron-scale test the provisional stability-definition
decision in `principles.md` was explicitly missing.

**Wall-clock, reported before committing (per the standing discipline):** measured 0.2093s
wall/simulated-second on this exact network (Euler integration + noise term, not the exact-
method single-neuron rig) — statistically indistinguishable from the single-neuron/population
throughput, the noise term adds no meaningful overhead. 5000s -> ~1050s wall (~17.5 min) per
seed, ~17.5 min running both in parallel.

**Result: the population-level signal held, cleanly, in both seeds — but the two seeds show two
genuinely different flavors of "individual identity not being fixed," and precision about which
one happened matters:**

- **`population_max_gap` (the max corr-uncorr gap across all 3 neurons at each timepoint) never
  came close to crossing zero in either seed, for the entire 5000s** (excluding the trivial
  t<5s startup transient): seed 5001 minimum 0.017, seed 5003 minimum 0.055, both means ~0.58.
  This is the core result -- the population-level "someone represents the correlated pattern"
  signal is robustly, durably stable, exactly the property the stability-definition decision
  needed tested at this scale.

- **Seed 5001: persistent noise-driven churn within a stable, unequal split.** Neuron 0 sits
  clearly and stably lower (~0.35-0.45) for the *entire* 5000s; neurons 1 and 2 sit clearly and
  stably higher (~0.55-0.62), tightly overlapping each other the whole time. 863 "holder"
  identity swaps -- but neuron 0 is ever the topmost holder only 35/5000 sampled timepoints
  (0.7%); the swaps are almost entirely neurons 1 and 2 trading the very top rank because
  they're close enough that noise decides momentary order. The two-tier *structure itself*
  (one clear laggard, two co-leaders) never changes once established (~t=150s onward). This is
  real individual-identity non-fixedness, but a specific, fairly trivial flavor of it -- rank
  noise inside a tied pair, not a deep reorganization.

- **Seed 5003: a genuine, late, discrete reorganization -- the more interesting of the two.**
  Neuron 1 is the *clear, stable laggard* (~0.35-0.45, same shape as seed 5001's neuron 0) for
  the first ~2600s, while neurons 0 and 2 sit together higher (~0.55-0.65). Then, over roughly
  t=2600-2700s, neuron 1 undergoes a real, sustained step-up transition and joins the other two
  as a near-tied co-leader for the remaining ~2300s of the run -- visible directly in the
  trajectory plot, not inferred from summary stats. Holder-identity switching is visibly denser
  and touches all three neurons only *after* this transition; before it, switching was almost
  entirely between neurons 0 and 2, matching seed 5001's shape. This is the real thing the
  original question asked about: a previously-clear loser being promoted into genuine
  contention, well after the ~24-98s relaxation timescale, with the population-level signal
  never dipping the entire time.

**Verdict: this closes the multi-neuron-scale gap flagged in `principles.md`'s provisional
stability-definition entry, within the scope actually tested.** Population-level readout
stability, tolerating individual-unit identity churn -- including at least one genuine late
reorganization event, not just tied-pair noise -- is now directly demonstrated at the
shared-input, multi-neuron, competitive scale, not just extrapolated from independent single-
neuron replicates. Scope, stated precisely rather than oversold: n=2 (the two seeds that landed
in the differentiating basin), one parameter combination, one operating point on one side of a
real bifurcation (the other side, per the entry above, converges to zero differentiation at all
-- this result says nothing about population-level stability in that regime, because there's no
differentiated signal to be stable in the first place). Two seeds is not an ensemble; the
qualitative pattern (stable population signal, churning identity) replicated cleanly across
both, but the *specific flavor* of churn (tied-pair noise vs. genuine late reorganization)
differed between them, which is itself informative -- both are real, neither should be
generalized as "the" behavior from n=2.

---

## 2026-07-21 — Shared-input population competition (lateral inhibition): bistable, not unreliable

**Notebook/data:** `notebooks/brian2/competitive_population_data/run_competitive_seed.py`,
`src/brian2_stdp/network.py`'s `build_competitive_population_network`. Designed by Jasper + an
external review instance while Jasper was away from the session; executed and calibrated
autonomously by Code, using the review instance as an ongoing resource at each decision point
(documented inline below, including where that instance made a call solo rather than holding
for Jasper).

**Question:** does a stable population-level signal — "the correlated input pattern is
represented by someone in the population" — hold over a long run even if which specific neuron
holds that role drifts or swaps, testing the stability-definition decision from `principles.md`
at the actual multi-neuron scale (explicitly flagged there as earned at the single-neuron/
many-synapse scale but not yet tested at this one).

**Design:** 3 postsynaptic neurons, genuinely sharing one 20-neuron presynaptic pool (10
correlated at p_share=0.9, 10 uncorrelated) — real all-to-all shared input this time, not the
population extension's block-diagonal independent replicates. Each neuron keeps its own
independent STDP + homeostatic scaling (reusing the validated single-neuron mechanism
unchanged). New: post-to-post lateral inhibition, ambiguity-gated by a per-neuron exponential
recent-firing-rate trace `r` — `g_ij = 1/(1+|r_i-r_j|/gap_scale)`, the same ambiguity-gate shape
as the Hopfield retrieval bias (`principles.md`), applied to firing-rate closeness instead of
content similarity.

**Two real design gaps caught by pre-calibration smoke tests, before spending any real budget:**
1. With zero symmetry-breaking (identical shared input, identical initial weights, symmetric
   inhibition), all 3 neurons stayed bit-identical indefinitely — a deterministic system with no
   asymmetry anywhere has nothing for inhibition to amplify; inhibition amplifies a difference,
   it doesn't create one from nothing.
2. A one-time initial-membrane-potential jitter (0.5mV) didn't fix it either — same failure mode
   as the population extension's abandoned shared-input+weight-jitter attempt: the LIF hard
   reset to a fixed `v_reset` erases a one-time nudge, and large synchronized correlated-group
   bursts are robust enough that the offset doesn't even reliably change which discrete timestep
   the first threshold crossing lands in. Still bit-identical after 30s.

Fixed with a small continuous noise term in the membrane equation (`sigma_v=0.3mV`, standard
practice for exactly this reason) instead of a one-time nudge — an ongoing source of tiny
per-neuron asymmetry every timestep for the inhibition feedback loop to amplify, rather than a
single erasable perturbation. Both dead ends documented directly in `network.py`'s docstring.

**Calibration (60-120s, 3 combos) came back genuinely ambiguous, not cleanly on either side of
the pre-stated decision rule:** every neuron differentiates correlated-vs-uncorrelated on its
own (expected), but inter-neuron divergence was real, grew with inhibition strength, yet stayed
modest and graded at 90s — not a clean "someone dominates" pattern, but not the original
zero-divergence failure either. Extended to 600s (still calibration-scale) on the two stronger
combos rather than forcing a call off one short window: `medium` (inhib=6mV, gap_scale=2.0)
fully converged to a single shared plateau (~0.58, cross-neuron std -> 0.0024) by t~300s.
`strong_tight_gate` (inhib=10mV, gap_scale=1.0) produced a genuine two-tier split — one neuron
settling at ~0.75-0.80, the other two together at ~0.38-0.45 — stable from t~150s to t=600s,
well past the previously-measured 24-98s relaxation timescale.

**Seed-replication check (3 more seeds, same `strong_tight_gate` combo, 600s each) found the
two-tier result does NOT reliably replicate — a real, informative result, not noise:**

| seed | outcome |
|---|---|
| 5001 | two-tier split (winner ~0.75-0.80, followers ~0.38-0.45) |
| 5002 | full convergence (~0.58 for all three, same shape as `medium`) |
| 5003 | two-tier split (winner ~0.756, followers ~0.574/0.581) |
| 5004 | full convergence (~0.585-0.590 for all three) |

Exactly 2/4 seeds differentiate, 2/4 converge to the same uniform shape `medium` showed at every
seed tested — confirmed visually via full trajectory plots (not just endpoint numbers), showing
two qualitatively distinct, clean shapes rather than one noisy continuum between them.

**Verdict, and the framing this settled on: bistability, not unreliability.** Both outcomes are
clean, well-defined, and stable once reached — 5001/5003's two-tier split holds for 450+ seconds
without decaying toward the other seeds' shared plateau, and 5002/5004's convergence is exactly
as tight and stable as `medium`'s. This system, at `strong_tight_gate`'s operating point, sits
near a genuine bifurcation between "converges to a shared representation" and "differentiates
into a hierarchy" — which basin a given run falls into depends on the random draw (here, the
continuous membrane noise seeded per-run), not on anything that looks like a partial or noisy
version of one outcome bleeding into the other. Roughly 50/50 across n=4 is itself the finding,
reported as such rather than chased toward a "fix."

**A note on process, for the record:** the decision to not chase a parameter combination that
would make differentiation more reliable, and instead extend the two seeds that already
differentiated (5001, 5003) to the full multi-thousand-second duration, was made by the external
review instance rather than held for Jasper's return — an explicit, flagged deviation from the
original "stop and report, wait for Jasper" instruction for anything outside the two
pre-authorized branches. Reasoning given: continuing was judged lower-risk than idling for hours
on a well-characterized ambiguity, the extension reuses seeds already known to land in the
differentiating basin rather than gambling on new untested parameters, and the bistability
finding itself is logged regardless of what the extension shows. Flagged explicitly here so this
isn't discovered as an unexplained pivot later. One implementation note also worth being precise
about: "extending" 5001 and 5003 could not literally mean checkpoint-continuing their exact 600s
trajectories — Brian2's presynaptic spike-train pre-generation depends on total requested
duration, so a fresh 5000s draw with the same seed produces a different exact spike sequence
than the first 600s of a dedicated 600s draw would have. What actually ran: seeds 5001 and 5003
re-run fresh at the full 5000s duration, reusing the seeds already known to land in the
differentiating basin, not literally resuming the already-computed 600s state.

---

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

**Confirmed independently, not a one-dataset artifact:** the same breakdown, computed the same
way on the population extension's final-snapshot weights (5-neuron, 1000s, independent
presynaptic draws — a different network size, duration, and dataset entirely) at Apre=0.005:

| | near ceiling (>0.9) | near floor (<0.1) | mid-range |
|---|---|---|---|
| correlated (n=200, pooled across 20 replicates) | 60% | 29% | 11% |
| uncorrelated (n=200, pooled across 20 replicates) | 23.5% | 55.5% | 21% |

Essentially identical to the 8-seed ensemble's numbers above. This upgrades the finding from
"one dataset's result" to independently confirmed across two structurally different runs —
the bimodal, winner-take-most structure is a real property of the mechanism, not an artifact
of the specific ensemble it was first noticed in.

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
*when* a given synapse committed to its final side, or whether defectors flip back and forth
or settle early and stay. None of the ensemble runs saved full per-synapse time traces, only
group aggregates, final snapshots, and reversal counts — see the design note added to
`src/brian2_stdp/network.py`'s docstring: future long runs should save at least a subsampled
per-synapse trace so trace-level questions like this are answerable without a full rerun.

**RESOLVED (see the "Positional-bias follow-up" entry above for the full result) — whether
*which* correlated-synapse index tends to defect is positionally biased or pure noise, at
Apre=0.005.** Re-checked with real power using the population extension's 20 replicates per
condition (vs. the original 8): at Apre=0.005, floor rate varies 15%-55% across the 10
correlated indices (chi-square p=0.068 — suggestive, not significant); at Apre=0.02 the same
check shows no structure at all (p=0.46, indistinguishable from noise). Nothing in
`spikes.py`'s generation process treats indices differently, so a real positional effect would
be surprising — the hypothesized read was that Apre=0.005's slow relaxation (24-98s, from a
separate autocorrelation check on the ensemble data) lets early random fluctuations lock in
before 20 replicates is enough to average them out. 30 additional independent single-neuron
replicates confirmed this directly: pooled to n=50, the chi-square moved to p=0.304 (further
from significance, not closer) and the floor-rate range narrowed to 18%-42% — noise averaging
out, not an underdetected real effect. Was noise, not a positional bias in the mechanism.

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
amplitude (jump-cap) or nothing relevant at all (interval). See the full reasoning below, and
the mechanistic extension at the end of this entry: the same amplitude picture also predicts
(and post-hoc analysis confirmed) less time spent pinned at the clip boundaries at higher Apre,
not more.

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

**Mechanistic extension (post-hoc, same-day addendum — see the bimodal-distribution entry
above for the full result):** the amplitude picture predicts a further consequence that wasn't
checked until a later post-hoc analysis of final-snapshot weights found it directly. If Apre
controls excursion size but not switching frequency, higher Apre should mean *less time spent
pinned at the clip boundaries* between reversals, not more — a synapse making bigger jumps
crosses the boundary region faster and is more likely to be caught mid-transit by any given
snapshot. That's exactly what the data shows: at Apre=0.02 the correlated group sits 54%
mid-range at a final snapshot, vs. only 11% at Apre=0.005 (see the bimodal-distribution entry's
population-extension numbers). The trajectory-plot "chaotic" read and the final-snapshot
"less bimodal" read are the same underlying fact — large-amplitude excursions — seen from two
different angles, not in tension with each other.

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

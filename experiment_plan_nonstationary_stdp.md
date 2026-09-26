# Experiment Plan — Non-stationary correlation structure (input-driven perturbation)

**Status:** RUN. A→B→A at N=3 (both operating points) and N=7 (13mV/1.5) done and logged; N=7
follow-up analyses (slow tail, swap signal) done; novel-C (A→B→C) done and logged. The results live
in the log, not here: `docs/log/brian2/05_nonstationary_correlation.md` (index:
`experiments_brian2.md`). This file is the original plan plus a Round 2 addendum at the bottom;
it's kept for the design reasoning.
**Phase:** Brian2 / SNN (in-phase continuation of the perturbation-testing thread — NOT a
cross-substrate transfer experiment; see Scope below).

---

## One-line question

When the *input's* correlation structure changes partway through a run — a different group of
presynaptic neurons becomes the correlated one — does the homeostatically-scaled, laterally-
inhibited competitive population re-assign its representation to the new structure, and how does
that depend on whether the operating point's reliability is suppression-based or dominance-based?

## Why this experiment, why now

Every STDP run in this project so far has held input statistics **fixed** for the whole
simulation (one `p_share`, one correlated group, start to finish). The entire stability-plasticity
story on the Hopfield side is about *time* — primacy/recency, lock-in vs. erasure, persistence
across phases — and none of that has ever been asked of the spiking substrate.

The perturbation-testing thread (now closed) got close but perturbed through a different channel:
it injected drive/weights directly (`I_pert`, direct correlated-weight nudges) to measure basin
depth. Its named finding — **suppression-based reliability (13mV/1.5) vs. dominance-based
reliability (`strong_tight_gate`, 10mV/1.0) are different axes** — was established by *forcing*
the substrate. It explicitly flagged that revisiting needs "a different premise, not a third
[variant of the same manipulation]."

This is that different premise. Instead of changing the substrate, **change the world** — flip
which presynaptic block is correlated — and let the weights follow the input statistics on their
own. That's the ecologically real version of the disturbance the perturbation thread reached for
with direct injection, and it directly tests the project's core commitment in the SNN substrate:
is identity here path-dependent *but still earned by current experience*, or does early input
freeze the representation?

## Rig and operating points (existing code, no rig changes)

- Network: `src/brian2_stdp/network.py :: build_competitive_population_network`
  (shared-input all-to-all population, per-neuron STDP + homeostatic scaling, ambiguity-gated
  lateral inhibition). **Do NOT use `I_pert`** — leave it at its default `0*mV`. The whole point
  is that this experiment perturbs input, not drive. Keeping `I_pert=0` also keeps the membrane
  equation bit-identical to every prior competitive run.
- Input: `src/brian2_stdp/spikes.py :: build_presynaptic_input` (single shared pool; correlated
  block is presynaptic indices `[0, n_corr)`, uncorrelated `[n_corr, n_pre)`).
- Two operating points, both already characterized, contrasted deliberately:
  - **`strong_tight_gate` = inhib=10mV, gap_scale=1.0** — dominance-based reliability; reliability
    comes from genuinely weight-earned competition near the differentiate/converge bifurcation.
  - **`13mV/1.5` = inhib=13mV, gap_scale=1.5** — suppression-based reliability; reliability comes
    from fast inhibitory suppression, not weight-earned depth.
- `apre_val = 0.005` (the genuine-but-probabilistic stable regime; do not use 0.02).
- `n_post`: start at 3 (the characterized scale). N-scaling is a later question, not this one.

## Protocol

Three phases, one continuous run (the substrate is never reset between phases — that's the
experiment):

- **Phase 1 (learn):** correlated block = group A (indices `[0, n_corr)`), `p_share=0.9`.
  Run until settled. Prior settling transients ran 250–400s in some seeds
  (`metrics.detect_tier_reentry` docstring) — budget **≥600s**, confirm settling per-seed from the
  trajectory rather than assuming a fixed cutoff.
- **Phase 2 (swap the world):** correlated block = group B (indices `[n_corr, n_pre)`); group A
  becomes independent. Rates held equal across the swap (rate-matched construction already
  guarantees this) so *only timing structure moves*, never rate. Run ≥600s.
- **Phase 3 (return):** correlated block = group A again. Probes the other horn — does A's phase-1
  history leave any trace (savings-like reacquisition) or was it cleanly erased? Run ≥600s.

Seeds: enough for a distribution, not a point estimate — start **8 seeds per operating point**
(matching the perturbation batches), expand if the phase-2 behavior looks bimodal.

### New glue code required before running (small, belongs in `src/`, not the notebook)

1. **Phase-switching input builder** in `spikes.py`: generate per-phase segments with the
   correlated block placed on whichever indices that phase designates, offset each segment's spike
   times by the phase start, concatenate. Keep it a pure function returning `(idx, t)` like the
   existing builders so it stays notebook-independent.
2. **Phase-aligned gap metric** in `metrics.py`: the existing `compute_competitive_metrics`
   hard-codes correlated = `[0, n_corr)`. After a swap that's wrong for phase 2. Add a variant that
   takes the *currently-correlated* block per phase and reports
   `phase_aligned_gap = mean(w_current_corr) - mean(w_current_uncorr)`, which should climb positive
   in **every** phase if the network is tracking. Tracking latency = time after each swap for this
   to re-cross the differentiation threshold.
3. **Save full (subsampled) per-synapse weight traces**, not just group aggregates — required to
   see the flip trajectory, and exactly the lesson `network.py`'s design note already records from
   the post-hoc-analysis miss. Sample fine enough to resolve a swap that may complete in <100s.

## What to capture — everything

Per Jasper's call: gather all the data, make claims after. Log, at minimum, for every seed ×
operating point:

- Full per-synapse `w(t)` traces (all synapses, all phases).
- Per-neuron `r(t)` traces (lets `g_ij(t)` be reconstructed post-hoc, per network.py's note).
- Postsynaptic spike trains / rates.
- `w_total(t)` health band (confirm homeostatic scaling still holds through the swaps — a swap is
  a new stressor it's never been checked against).
- Phase-aligned gap, per-neuron gap, `population_max_gap`, `holder_identity` across the *whole*
  run (through both swaps), plus `count_identity_swaps` / `full_rank_swap_count` per phase.

## Analyses to run once data is in (exploratory — not a pre-commitment)

- Tracking latency per swap, per operating point, per seed — distribution, not a mean.
- Does the phase-2 representation reach phase-1-comparable separation, or a degraded one?
- Phase-3 reacquisition **slope** vs. phase-1 cold-start slope (the Hopfield-side savings caveat
  applies directly: report slope, not absolute level — "faster to reach" can just be restating a
  shorter climb, not genuine savings).
- Full-trajectory inspection is mandatory before any convergence/settling claim (standing project
  rule — a single endpoint snapshot on a chaotic trajectory has burned this project ≥3 times).
- Does homeostatic scaling survive the swap, or does the transient break the weight-sum band?

## One prediction on record (optional — Jasper's call whether to keep)

Flagging the tension honestly rather than silently dropping it: `principles.md` states falsification
criteria should go on record *before* running. Jasper's preference this round is to collect all
data first and claim at the end, which is the right call for the **exploratory** surface here (no
one has run non-stationary input through this rig — boxing in the phenomenology in advance would be
premature). So the doc keeps everything open **except** the single directional contrast that rides
on an *already-established* named finding, where post-hoc storytelling is the real risk:

> **Prediction:** under an input-driven world-switch, `13mV/1.5` (suppression-based) tracks the
> swap *fast* — the inhibitory machinery simply re-suppresses the now-uncorrelated group —
> whereas `strong_tight_gate` (dominance-based) *lags*, because genuinely weight-earned dominance
> resists the world changing under it.

Note this is predicted **opposite in direction** to the perturbation-testing result (there,
suppression-based `13mV/1.5` was *easier* to flip by direct injection, 7/8 vs 5/8). Same two
operating points, two different perturbation channels, predicted to rank oppositely. If that holds,
it corroborates suppression-vs-dominance from a fully independent channel; if it doesn't, it
exposes that finding as partly an artifact of the injection method. Either outcome is worth having.
This is the one thing worth writing down first; everything else stays open.

## Scope notes

- **In-phase Brian2 work**, framed as a continuation of the (closed) perturbation thread via a new
  perturbation channel — deliberately NOT framed as "strength-breaks-ties transfers to the SNN
  substrate." Cross-substrate principle-transfer is on `CLAUDE.md`'s "NOT yet built / don't jump
  ahead" list; this experiment stands entirely on its own inside the SNN phase and doesn't need
  that framing to be worth running.
- Does not touch `episodic.py`, the Hopfield layer, or any cross-layer integration.
- Reuses the existing rig unmodified except for the three additive helpers above (input builder,
  phase-aligned metric, trace saving) — no change to the STDP/scaling/inhibition mechanism itself,
  so it remains comparable to every prior competitive-population run.

## Open decisions for Jasper before a run

1. Keep the single prediction-on-record, or go fully exploratory with zero pre-commitment?
2. Is a 3-phase run (with the phase-3 return/savings probe) worth the compute up front, or run
   phases 1–2 first and only add phase 3 if the swap behavior is interesting?
3. Does this read as "related but not pursuing the others," or too close to the closed perturbation
   thread to run without a fresh web-side sign-off? (My read: distinct channel, opposite predicted
   ranking, answers something injection structurally can't — but it's your call.)

---

# Round 2 addendum (2026-09-25) — results so far + novel-C follow-up

## What the N=3 and N=7 runs established
- Prediction on record (suppression tracks fast / dominance lags) **refuted**: re-learning time
  ~equal at both operating points; input-driven and injection-driven perturbation do NOT rank the
  two points the same. So "suppression-based vs dominance-based reliability" is a property of the
  *injection* channel, not substrate-general — a real footnote on the perturbation thread's finding.
- Unplanned result: a bounded fraction of neurons **retains** the old pattern at each swap while the
  rest re-learn. N-scaling adjudicated the "dedicated keeper vs proportion" fork → **proportion**
  (~1/3; 33% at N=3, 37% at N=7, count bounded away from both 1 and N — never 0/1/>3 at N=7).
  Dedicated-role specialization is out; **graded basin-depth inertia** is in. The specific value
  "1/3" rests on two N points at n=8 with an inhibition-normalization confound — treat "a bounded
  middle fraction" as supported, "one third" as not yet a law.
- Retainer = previous top-tier neuron: clean at 13mV/1.5 (p=0.0004), only suggestive at
  strong_tight_gate (wide ties). Correctly scoped to 13mV/1.5.

## Slow second-swap tail — mechanism analysis (done, on existing N=7 data)
Script: `notebooks/brian2/nonstationary_data/analyze_slow_tail.py` (analysis only, no new sim). Reproduces the reported
62–977s swap-2 return tail (15 returners across 8 seeds). Discriminated two mechanisms for the slow
return:
- **Candidate A (retainer-suppression — the pattern is already covered, so weak pressure to also
  return): SUPPORTED.** Per-seed (non-pseudoreplicated): every seed where **3** neurons already
  hold the returned-to pattern has a slower slowest-returner than every seed where only **2** do —
  perfect separation, MWU p=0.018 (the floor for 3-vs-5). More of the population already covering
  the pattern → the rest are slower to also return.
- **Candidate B (returners competing among themselves): not supported** (lat ~ n_co_returners
  rho=−0.19, p=0.51).
- **Caveat — mechanism localization is NOT established.** The ambiguity-gate (`g_ij`) correlations
  during the climb were weak/mixed (returners if anything slightly *more* rate-locked to
  co-returners than to holders; the one significant gate correlation points the "wrong" way for a
  simple story). So "coverage slows return" is a clean seed-level fact; that it's *mediated by the
  r-gate inhibition specifically* is a hypothesis this data can't confirm. Needs a direct test
  (gate lesion, or the novel-C run below).
- **Reframes the phase-3 "benefit":** the same retention that makes the returned-to pattern
  instantly available *suppresses the rest of the population's return to it*. Instant-representation
  and slow-return are two faces of one mechanism, not a clean win.

## Novel-C experiment — DONE (seeds 32000–32007; results in the arc 05 log, not here)

(Note: the "Slow second-swap tail" section above called candidate A "SUPPORTED". Novel-C then
showed ~700 s slow commits with *no* coverage, so coverage isn't the general cause. See the log.)
**Question:** is the retained fraction adaptive division of labour, or a capacity tax? The existing
runs return phase 3 to A, so retainers holding A "paid off." Send phase 3 to a **novel** pattern C
(neither A nor B) and anyone still holding A or B is pure dead weight.

**Rig:** 3 disjoint presynaptic blocks of 10 (30 inputs), one correlated per phase, corr blocks
`[0,1,2]`. **Drive matched to the 20-input rig by holding `target_total=10` (w_init=1/3)** — the
homeostatic scaling pins total synaptic drive to this regardless of `n_pre`, so `gmax` is unchanged.
N=7, 13mV/1.5, 8 seeds, 1000s/phase, dt=0.2ms, Cython. Everything else identical to
`run_nonstationary_seed.py`; `I_pert=0`. Code: `notebooks/brian2/novelc_data/` (the multiblock input
builder is kept local, not in `src/spikes.py`, until the experiment is worth keeping).

**Calibration confirmed before the batch** (`notebooks/brian2/novelc_data/calibrate_novelc.py`, stated bar: per-neuron
3–20 Hz and w_total ~9.5–10.5): PASS — 15.0/17.3/16.0 Hz, w_total [9.74,10.20]. No `gmax` re-tuning,
so this is a strict extension of the known-good rig, not a re-tune.

**Read once results land (exploratory — no pre-committed bar):**
- Phase-3 (novel C): what fraction of neurons commit to C vs. keep holding a now-never-returning
  stale pattern? If ~1/3 still hold A/B → retention is a capacity tax (dead weight), not adaptive.
  If the population commits to C → the earlier phase-3 payoff was conditional on the world returning.
- Compare the C-commit trajectory to the A→B tracker trajectories — is moving to a *novel* pattern
  slower/harder than moving to a *previously-seen* one?
- Confirm homeostatic scaling holds through both swaps at 30 inputs (new stressor).

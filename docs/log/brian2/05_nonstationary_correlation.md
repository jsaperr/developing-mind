# Brian2 log: Non-stationary correlation: world swaps A->B->A, N=3 and N=7 (2026-09-25)

Entries moved verbatim from `experiments_brian2.md` on 2026-09-25 (no wording changed). Index: `experiments_brian2.md`.

## 2026-09-25 — Non-stationary correlation (world swaps A→B→A): the population always tracks, but a subset keeps the old pattern (1 of 3 at N=3, 2-3 of 7 at N=7) — and the on-record prediction was not supported

**Data:** `notebooks/brian2/nonstationary_data/` (16 seed JSONs, seeds 30000-30007 at 13mV/1.5 and
30100-30107 at strong_tight_gate; `run_nonstationary_seed.py`, `run_nonstationary_batch.py`,
`analyze_nonstationary.py`; figures `nonstationary_*.png`). Per `experiment_plan_nonstationary_stdp.md`.
New glue in `src/` with tests: `spikes.build_phase_switching_input`,
`metrics.phase_aligned_per_neuron_gap`. One continuous run per seed, never reset: 3 phases x 1000s,
correlated block A (presynaptic 0-9) → B (10-19) → A, rates matched across blocks and phases, N=3,
dt=0.2ms, Cython backend, I_pert=0, everything else identical to prior competitive runs. Full
per-synapse w(t) at 1s, r(t), w_total(t), 1s spike bins saved. Exploratory: trajectories inspected
first, claims after. 16/16 completed, 0 failures, ~18 min. n=8 per point, so counts below are
descriptive, not distributional estimates.

**Prediction on record before running (from the plan): 13mV/1.5 (suppression-based) tracks the
swap FAST, strong_tight_gate (dominance-based) LAGS. Not supported.** Re-learning latency (first
time a tracking neuron's phase-aligned gap reaches 0.3 after a swap, excluding neurons that were
already positive because they retained the returning pattern): median 92s (13mV/1.5, n=24) vs 82s
(strong_tight_gate, n=29), Mann-Whitney p=0.50. If anything the reliable point had more slow
re-learners (>=100s: 10/24 vs 5/29, Fisher p=0.069, suggestive at best), the opposite direction
from the prediction. Worth noting this also does not repeat the perturbation-testing ranking
(there 13mV/1.5 was the easier one to flip by direct weight injection): under input change the two
points look alike at the population level.

**What actually happened — a division of labour, in both points, in nearly every swap.**
- Phase 1: all three neurons learn A within ~100-200s (phase-aligned gap 0.55-0.8; some tier
  structure, e.g. 0.56 vs 0.75).
- Each swap: every neuron's phase-aligned gap drops to about -0.6 to -0.8 at once (weights
  unchanged, world inverted), then within ~50-100s the neurons that re-learn the new pattern climb
  to ~0.77. Typically two of three do; some are staggered much later (up to 374s at 13mV/1.5, one
  545s outlier at strong_tight_gate).
- **Exactly one neuron does not re-learn: it keeps the old pattern and its weights harden toward
  saturation** (in most cases the aligned gap drifts from about -0.8 toward -1.0, i.e. old-pattern
  weights to the ceiling, new-pattern weights to the floor; a few stay near -0.77). At 13mV/1.5 this is 16/16 swaps (exactly one retainer
  every time); at strong_tight_gate 13/16, the other 3 being slots where all three neurons re-learned
  (seeds 30103 and 30107 at swap 1, 30106 at swap 2).
- **The retainer is a member of the previous phase's top tier:** 15/16 (13mV/1.5) and 13/13
  (strong_tight_gate) of the single-retainer swaps, against about 9/16 and 7.3/13 expected by chance
  given tier sizes. **Correction made after a cleaner check:** that top-tier count is the weaker
  test where tiers are wide (strong_tight_gate has many near-ties). Comparing previous-phase gaps
  at swap 1 only (where every neuron starts positive, so the comparison is not biased by the
  neuron that held the returning pattern): 13mV/1.5 retainers 0.752 vs trackers 0.607 (Mann-Whitney
  p=0.0004, clear); strong_tight_gate 0.753 vs 0.698 (p=0.34, not distinguishable at n=6 vs 18).
  So "the most entrenched neuron is the one that does not follow the world" is supported at
  13mV/1.5 and only suggestive at strong_tight_gate. (`analyze_retainers.py`; the swap-2 version of
  this comparison is deliberately not reported, it is biased.)
- Retainers are NOT silenced: late-phase rate ~12-15 Hz vs ~18 Hz for the trackers. They keep firing
  on their old (now uncorrelated) inputs while holding the old weights.
- **Consequence at the return swap (phase 3):** whenever a retainer existed in phase 2, the returning
  pattern was already represented, so recovery was instant for that neuron: 8/8 at 13mV/1.5, 6/8 at
  strong_tight_gate (both misses are the two seeds where no neuron retained A through phase 2). The
  neuron that tracked B in phase 2 then becomes the new retainer of B. So after two swaps the
  population holds both patterns at once.
- Homeostatic scaling survived the swaps: `w_total` stayed within 9.6-10.4 (target 10) in every seed,
  max deviation 0.40 within 60s of a swap. The swap is not a new failure mode for it.

**What this does and doesn't say.**
- At the population level the system is NOT locked in: in 16/16 seeds at both points the
  correlated-input representation was re-acquired after each swap within minutes. The
  fast-permanent-settling seen at 13mV/1.5 under stationary input (Test A, N-scaling step 4) does not
  mean it cannot follow a changing world; settling was permanence of a hierarchy under fixed input,
  not inability to relearn.
- At the neuron level the most-entrenched neuron does lock in, and that is what supplies memory of
  the earlier pattern. Stability and plasticity end up divided between neurons rather than traded
  off within one — nothing in the mechanism was designed to do this.
- Not a savings claim: the instant recovery is retained weights, and the slope-versus-cold-start
  comparison the plan asked for was not computed. Descriptive at n=8 per point; no test of why a
  neuron becomes the retainer beyond prior strength.

**N=7 follow-up (13mV/1.5 reference, inhibition scaled with `scale_inhib_for_n` to 4.33 mV
per connection as in the N-scaling runs; seeds 31000-31007, same 3x1000s protocol, 8/8 completed,
~9 min).** Question: does "exactly one retainer" hold at larger N, or scale with population?
- **It is not one, and it does not scale linearly either: 2 or 3 of 7 neurons keep the old pattern
  at every swap** (7 swaps with 2, 9 with 3, none with 0, 1 or more than 3; mean 2.56 of 7 = 37%,
  vs 1 of 3 = 33% at N=3). Roughly a constant fraction of the population, not a fixed count of one
  and not everyone. Tight across seeds.
- Same selection rule: at swap 1 the retainers were the stronger neurons beforehand (previous-phase
  gap 0.772 vs 0.661 for the trackers, n=21 vs 35, Mann-Whitney p<0.0001). Same firing signature:
  retainers ~12.2 Hz vs ~17.9 Hz, active but slower.
- Instant recovery at the return swap: 2-3 neurons already held the returning pattern in 8/8 seeds,
  and that number equals the seed's swap-1 retainer count every time. The old pattern is carried
  intact by the retainers through the whole intervening phase.
- Re-learning speed: median 84s, similar to N=3 (92s/82s). But the slow tail is much longer at N=7:
  18/50 re-learners took >=100s and the slowest 977s (nearly the whole phase). It is concentrated at
  swap 2: all swap-1 re-learners at N=7 finished within 181s, whereas swap-2 re-learners (only 1-2
  neurons per seed) took 62-977s, median ~210s. The few neurons that must move a second time are the
  slow ones. (Swap-2 re-learners were also slower at N=3, e.g. 239-374s, but the effect is larger
  here.)
- Not established: why the fraction is about a third (e.g. how it depends on the inhibition
  normalization or on N beyond 3 and 7), and only the 13mV/1.5-equivalent point was run at N=7. The
  swap-2 slowness is descriptive; nothing here tests its cause. n=8.

---


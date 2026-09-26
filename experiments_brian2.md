# Experiments Log: Brian2 / SNN phase (index and current state)

Hopfield / two-layer-memory / episodic-layer work lives in `experiments.md`, the Echo State Network
work in `experiments_esn.md`. This file is the entry point for the Brian2/spiking phase: a current-
state summary (what is established, what is closed, what is open) and an index into the full
entries, which live in `docs/log/brian2/` grouped by arc. Entries there are verbatim, newest-first
within each file. Raw data and the scripts that made it: `notebooks/brian2/README.md`.

**Adding a new entry:** write it at the top of the matching arc file (or start a new numbered arc
file), add one line to the index below, and update the state summary here if it changes what is
established or open. Keep the `CLAUDE.md` bullet to two lines and point here. State any prediction
before running; save full traces for exploratory runs; inspect trajectories before claiming.

*State as of 2026-09-25.*

## Current state

**Established**
- **The core claim holds.** STDP alone, with no supervision, learns to separate correlated from
  uncorrelated input from spike-timing statistics: single neuron, an N=5 replicate population, and a
  shared-input competitive population with lateral inhibition (N=3 through N=10). Arc 01, 02.
- **Runaway is fixed by homeostatic synaptic scaling.** Arc 01.
- **Standing explanation, do not re-derive:** `Apre` sets the amplitude of individual-synapse
  excursions, not their reversal frequency (about constant per synapse at every `Apre`). Arc 01,
  "threshold or a gradient" entry. `Apre=0.005` stability is genuine but probabilistic, not absolute.
- **Stability means a population-level readout that tolerates individual-synapse churn**, not
  individual convergence (named decision in `principles.md`), confirmed at n=7 at `strong_tight_gate`.
  Final weights are bimodal, winner-take-most. Arc 01, 02.
- **Positional bias was noise.** The apparent position effect (p=0.068) washed out to p=0.30 at
  n=50. Arc 01.
- **Competitive population (N=3):** reliability of differentiation rises with inhibition strength (the
  13 mV row is 75-100%); richness does not trade off against it; `13mV/gap_scale 1.5` is the standout.
  `strong_tight_gate` (10 mV / 1.0) is bistable, about 50/50 between converging and differentiating.
  Under fixed input, `13mV/1.5` settles fast (<100 s) and permanently; the 600 s swap-count proxy did
  not hold at 5000 s. Arc 02.
- **N-scaling (Experiment B):** reliability holds through N=10 with inhibition normalized to keep
  total drive constant. Hierarchy shape is non-monotonic (a minimum at N=7, gradual creep back up by
  N=10); at N=7 all seeds settle permanently by about 250 s. Arc 03.
- **Suppression-based versus dominance-based reliability (named finding, `principles.md`):**
  "never spontaneously reorganizes" and "resists a direct forced perturbation" are different axes.
  `13mV/1.5` was easier to knock over by direct weight injection than `strong_tight_gate`. Reached
  through three perturbation designs; the first two were rejected for confounds found by inspecting
  trajectories. Arc 04.
- **Non-stationary input (correlated block A -> B -> A):** the population always re-tracks the new
  correlation within minutes, in 16/16 (N=3) and 8/8 (N=7) seeds. Stability and plasticity divide
  between neurons: a subset keep the old pattern and harden, the rest re-learn. One neuron of three
  at N=3, 2-3 of seven at N=7 (about a third). Retainers are the previously strongest neurons (clear
  at `13mV/1.5`, only suggestive at `strong_tight_gate`), stay active (12-15 Hz vs 18 Hz), and mean
  the returning pattern is already held at the return swap. The prediction on record
  (`13mV/1.5` tracks faster) was not supported. Arc 05.
- **Novel-C (A→B→C, C never seen; 30-input 3-block rig, N=7, `13mV/1.5`): one-back retention.**
  At a new change, most neurons holding an older pattern release it (28/31) and the neurons that
  learned the just-departed pattern keep it. Stale neurons stay at about 45% and don't accumulate
  over 3 phases. A small residue (3 neurons in 3/8 seeds) never releases and hardens: genuine
  lock-in. The retainer fraction is 55% at the first swap in this rig against about 37% in the
  20-input rig, so it's rig-dependent, not "a third". Arc 05.
- **v1 (A→B→C→A→C) confirms one-back retention within seeds:** at a 2-back return about 0.25
  neurons per seed already hold the pattern, against 3.0 at a 1-back return (8/8). Holders speed
  up +4.7 to +8.3 Hz at the 1-back return. The residue doesn't accumulate over four changes
  (0, 0, 2, 0, 0), though A's return masks accumulation. In 2 seeds a residue neuron carried A
  through two intervening contexts and recognized its return. The H readout passes the same-rig
  lull test 8/8, which completes S1. Arcs 05, 07. What memory does with this:
  `experiments_integration.md`. v1b (A→B→C→A→B) replicates one-back (0.38 holders at a two-back
  return). The H readout's weak first context is a cold-start artifact (query quality 0.31-0.34
  in phase 1 vs 0.87-0.96 after): retention supplies the contrast. Arc 07.
- **World changes are visible in the substrate's own firing rate:** population rate dips about 3 Hz
  in the first 10 s after a swap, with no dip at non-swap times (p=4e-6). **The scalar dip doesn't
  tell a return from a new pattern.** It's smaller at the second swap whether the pattern is
  returning or new (novel-C control), so the earlier "smaller on return" reading was swap order.
  **Recognition is per-neuron:** neurons holding a pattern speed up at swap 2 only when that
  pattern returns (per-seed means +4.7 to +7.7 Hz vs -0.9 to +2.1 Hz, perfect separation,
  p=0.00016, cross-rig). Arc 05.
- **Label-free readout (S1, analysis of the N=7 runs):** without experimenter labels, the weight
  state alone encodes recent history (current plus the just-departed context, failing a lull test
  in 0/8 seeds), and rates alone encode "now vs just before" without identifying a context.
  Activity projected through the learned tuning, Σ (r_j − r̄) w_j, carries context identity, with
  right-signed same-vs-different similarity in 16/16 seeds. Its clean same-rig lull test needs
  A→B→C→A. Window size (10-200 s) barely matters. Arc 07.
- **Performance:** runs use the Cython runtime; about 99% of time is the simulation, bound by
  per-object dispatch, so there is no hotspot. About 6 physical cores; 8-9 concurrent jobs is the
  sweet spot. `dt=0.2 ms` is validated (0.5 ms is not). `cpp_standalone` is about 90x faster but
  gave different outcomes at `strong_tight_gate`, so it is not a drop-in. Arc 06.

**Closed**
- Reservoir-based forecasting for the episodic layer (see `experiments_esn.md`).
- The perturbation-testing thread and the boundary-mapping sweep (undecidable at 600 s: settling
  itself takes most of that budget). Whether any setting gives reliability together with genuine
  spontaneous reorganization was not answered; the thread ended with the finding above instead.
- Apre instability hypotheses (scaling-interval race, jump cap): both rejected.

**Open**
- What sets the retainer fraction. It isn't constant: about 37% in the 20-input rig at N=7, 55%
  in the 30-input novel-C rig. Candidates are correlated fraction, `n_pre` and inhibition
  normalization. Only the `13mV/1.5`-equivalent point was run at N=7.
- The distribution over individual-level regimes (n=7 at `strong_tight_gate` is a typology, not a
  frequency).
- ~~Whether to add the non-stationary finding to `principles.md`~~ Done 2026-09-25 (Jasper's
  go): named finding "the substrate retains exactly one context back; anything older has to live
  in memory".
- The cause of the `cpp_standalone` mismatch at `strong_tight_gate`.
- One-back retention over longer schedules: does the lock-in residue accumulate across many
  changes, and is release triggered by the change itself or just slow (more than 1000 s)? It
  predicted that A→B→C→A should *not* recognize A instantly, and v1 confirmed that. Still open:
  whether the residue accumulates when contexts never return (needs an all-novel schedule with
  more blocks), and whether release is triggered by the change or just slow.
- Slow commits (up to about 1000 s): in A→B→A they went with how many neurons already held the
  target, but novel-C shows them with no coverage at all. The cause is open; the ambiguity gate
  isn't supported by the saved r traces.


## Index of entries

Each line is the entry's own heading (its conclusion is in the title).


**Archive (superseded)**: [`docs/log/brian2/00_archive_2026-07-21_summary.md`](docs/log/brian2/00_archive_2026-07-21_summary.md)

- 2026-07-21 — STDP/SNN arc: consolidated summary

**01 STDP foundations (2026-07-20/21)**: [`docs/log/brian2/01_stdp_foundations.md`](docs/log/brian2/01_stdp_foundations.md)

- 2026-07-21 — Positional-bias follow-up: the p=0.068 chi-square washes out to noise with more data, resolved
- 2026-07-20 — Post-hoc analysis: the group-mean-gap's "bounded stability" hides a bimodal, winner-take-most structure at the synapse level
- 2026-07-20 — Population extension (N=1 -> N=5): does the single-neuron STDP signature generalize, or was it an artifact?
- 2026-07-20 — Does Apre=0.005's stability reflect genuine boundedness, or insufficient observation time? 8-seed, 5000s ensemble
- 2026-07-20 — Is the Apre instability a threshold or a gradient? Neither — reversal frequency is Apre-invariant
- 2026-07-20 — Does capping per-event STDP jump size fix Apre=0.02? Also rejected
- 2026-07-20 — Is Apre=0.02 instability a timing race with the scaling correction? Hypothesis rejected
- 2026-07-20 — Homeostatic synaptic scaling fixes the runaway; differentiation broadens, but stability is learning-rate-dependent
- 2026-07-20 — First STDP experiment: correlated-vs-uncorrelated differentiation — mixed result, real mechanism identified
- 2026-07-20 — Brian2 install check (start of the Brian2/SNN phase)

**02 Population competition (2026-07-21/23)**: [`docs/log/brian2/02_population_competition.md`](docs/log/brian2/02_population_competition.md)

- 2026-07-23 — Test A: does the 600s swap-count proxy hold at full 5000s duration? Partially — laggard exclusion is fast and permanent, but the "richness" it measured isn't what the proxy suggested
- 2026-07-23 — Population-competition bistability sweep: reliability rises with inhib_strength, richness doesn't trade off against it
- 2026-07-21 — Seed expansion (n=2 -> n=7): individual-identity behavior is a spectrum, population signal is the invariant across all of it
- 2026-07-21 — Competitive-population extension to 5000s: population-level signal stays stable while individual identity genuinely churns
- 2026-07-21 — Shared-input population competition (lateral inhibition): bistable, not unreliable

**03 Experiment B, N-scaling (2026-07-23)**: [`docs/log/brian2/03_n_scaling_experiment_b.md`](docs/log/brian2/03_n_scaling_experiment_b.md)

- 2026-07-23 — Experiment B step 4: N=7 at full 5000s — extends Test A's finding, not a contradiction. Reliable competition still means fast, permanent settling, not genuine reorganization
- 2026-07-23 — Experiment B step 3 follow-up: is N=10's shape reversal real or n=8 noise? Partially real — it's a gradual creep from a real minimum at N=7, not a sharp cliff
- 2026-07-23 — Experiment B step 3: population-competition N-scaling curve — reliability holds through N=10, hierarchy shape doesn't move monotonically

**04 Boundary mapping and perturbation testing (2026-07-23/24)**: [`docs/log/brian2/04_perturbation_and_boundary.md`](docs/log/brian2/04_perturbation_and_boundary.md)

- 2026-07-23 — Perturbation-testing redesign #2, dense-ladder rerun: backwards contrast replicates at finer resolution — candidate (1) (ceiling-saturation resolution artifact) ruled out
- 2026-07-23 — Perturbation-testing redesign #2 (weight-nudge): a genuine graded signal this time, but the expected contrast direction didn't hold — reported, not patched solo
- 2026-07-23 — Perturbation-testing validation: FAILED, for a mechanistic reason, not a statistical one — I_pert doesn't test basin depth, it confounds two other effects
- 2026-07-23 — Boundary-mapping sweep: does a reliable-and-rich region exist between strong_tight_gate and 13mV/1.5? Undetermined at 600s — the timescale itself is the obstacle, confirmed directly, not assumed

**05 Non-stationary correlation (2026-09-25)**: [`docs/log/brian2/05_nonstationary_correlation.md`](docs/log/brian2/05_nonstationary_correlation.md)

- 2026-09-25 — v1 world (A→B→C→A→C): one-back retention confirmed within seeds, the residue doesn't grow over four changes, and a residue neuron recognizes a 2-back return
- 2026-09-25 — Novel-C (A→B→C, C never seen): held patterns are released at the next change (one-back retention, a small permanent residue), and recognition lives in which neurons speed up, not in the population dip
- 2026-09-25 — N=7 follow-up analyses: the slow return is tied to how many neurons already hold the pattern, and the population's firing rate distinguishes a new pattern from a returning one
- 2026-09-25 — Non-stationary correlation (world swaps A→B→A): the population always tracks, but a subset keeps the old pattern (1 of 3 at N=3, 2-3 of 7 at N=7) — and the on-record prediction was not supported

**06 Performance audit (2026-09-25)**: [`docs/log/brian2/06_performance_audit.md`](docs/log/brian2/06_performance_audit.md)

- 2026-09-25 — Simulation performance audit: where the time goes, and why `cpp_standalone` isn't a drop-in

**07 Interface readout, S1 (2026-09-25)**: [`docs/log/brian2/07_interface_readout.md`](docs/log/brian2/07_interface_readout.md)

- 2026-09-25 — The H readout's first-context weakness is a cold-start artifact; later contexts read at about 0.9 because retention supplies the contrast
- 2026-09-25 — S1 completed: the H readout passes the same-rig lull test (8/8)
- 2026-09-25 — S1: no single label-free readout carries context on its own. Weights alone encode recent history, rates alone encode "now vs just before", and activity projected through tuning carries context identity

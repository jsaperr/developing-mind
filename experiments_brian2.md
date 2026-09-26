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
- Why the retainer fraction is about a third, and whether it depends on inhibition normalization or N
  beyond 3 and 7. Only the `13mV/1.5`-equivalent point was run at N=7.
- The distribution over individual-level regimes (n=7 at `strong_tight_gate` is a typology, not a
  frequency).
- Whether adding the non-stationary division-of-labour finding to `principles.md` is wanted
  (proposed, not decided).
- The cause of the `cpp_standalone` mismatch at `strong_tight_gate`.
- Another session has `notebooks/brian2/novelc_data/` in progress; not yet logged here.


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

- 2026-09-25 — Non-stationary correlation (world swaps A→B→A): the population always tracks, but a subset keeps the old pattern (1 of 3 at N=3, 2-3 of 7 at N=7) — and the on-record prediction was not supported

**06 Performance audit (2026-09-25)**: [`docs/log/brian2/06_performance_audit.md`](docs/log/brian2/06_performance_audit.md)

- 2026-09-25 — Simulation performance audit: where the time goes, and why `cpp_standalone` isn't a drop-in

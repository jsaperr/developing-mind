# Brian2 log: Interface readout (S1): what a downstream layer could read from the substrate (2026-09-25)

Newest first. Index: `experiments_brian2.md`. Context: `system_contract.md` (S1, section 3).

## 2026-09-26 — Correction: the cold start and the overlap weakness belong to the contrast readout H, and a rectified readout H+ fixes both

`notebooks/integration/set_worlds/compare_readouts.py` (see `experiments_integration.md`).
H+ = Σ max(r_j − r̄, 0) w_j reads phase 1 at 0.96-0.97 (H: 0.20-0.34). It reads 50% overlapping
contexts at 0.96 (H: 0.46) and leaks ~0 of the previous context (H: about −0.7). So the entry
below ("retention supplies the contrast; retention is the price of a readable code") holds only
for H, and "not chased" is superseded. The one-back retention finding itself (arc 05) is
unaffected.

## 2026-09-25 — The H readout's first-context weakness is a cold-start artifact; later contexts read at about 0.9 because retention supplies the contrast

Diagnosed while running the coupling toy (`experiments_integration.md`, two-back entry). This is
an analysis of existing runs.
- **Data:** settled substrate query (H, centered and unit-normalized) cosine to the true-context
  prototype, by phase. **Phase 1: 0.31 / 0.32 / 0.34** (A→B→A / A→B→C / A→B→C→A→C). **Every later
  phase: 0.87-0.96.** The per-seed ranges don't overlap (phase 1 at most 0.44, later phases at
  least 0.84).
- **Mechanism, from the readout's definition:** H = Σ_j (r_j − r̄)·w_j. When every neuron is tuned
  to the same pattern, Σ(r_j − r̄) = 0 cancels the shared component. The context then appears only
  through the weak covariance between rate and tuning strength. That's the situation only before
  the first change.
- After the first change, one-back retention (arc 05) keeps 40-55% of neurons on the previous
  context, so the population is permanently split and the contrast exists.
- **Retention is what makes the current context readable.** A population that re-learned
  completely at every change would read like phase 1, at about 0.3.
- **Consequence:** the first context ever learned gets a poor memory entry (cosine to its
  prototype 0.29-0.72 against 0.88 for its later settled queries). At its return, memory creates
  a duplicate instead of recognizing it. This affects exactly one context per lifetime, so it's a
  boot-up transient. Deliberately not chased with readout knobs.

## 2026-09-25 — S1 completed: the H readout passes the same-rig lull test (8/8)

The one missing S1 test was a return to a non-adjacent context *in the same rig*. The v1 world
(A→B→C→A→C, arc 05) provides it. With phase-mean settled H readouts, A4 is closest to A1 (rather
than B2 or C3) in 8/8 seeds (+0.24 to +0.47), and C5 is closest to C3 in 8/8 (+0.66 mean). Both
were predicted before launch (V1-P3 in `analyze_v1.py`). The first-context weakness persists
(A4-A1 +0.33 vs C5-C3 +0.66). By `system_contract.md`'s section-6 checklist, every property the
interface needs from the substrate is now characterized.

## 2026-09-25 — S1: no single label-free readout carries context on its own. Weights alone encode recent history, rates alone encode "now vs just before", and activity projected through tuning carries context identity

**Data:** existing runs only, no new simulation. A→B→A N=7 13mV/1.5
(`nonstationary_data/nonstat_n7_reliable_13_1p5_*`, 20 inputs, 8 seeds) and A→B→C N=7 13mV/1.5
(`novelc_data/novelc_n7_reliable_13_1p5_*`, 30 inputs, 8 seeds). Script:
`notebooks/brian2/interface_readout/analyze_s1_readout.py`; figure `s1_similarity_matrices.png`.
The predictions below were written into the script's docstring before its first run. The
dimensionality and per-seed checks further down were run afterwards, prompted by the results.

**Why.** Every STDP result so far, including the phase-aligned gap and the retainer/tracker
definitions, is measured with experimenter labels (which block is correlated). A downstream layer
only sees rates, weights and spikes. So whether the "stable population readout" survives *without
labels* is the untested assumption under the whole interface. The other session raised this for its
own results too.

**Readouts** (label-free; labels used only to score them), per non-overlapping window of 10, 50 or
200 s:
- **R**, the per-neuron rate vector (dim 7).
- **W**, the full weight state (dim 140 / 210).
- **H** = Σ_j (r_j − r̄)·w_j, the input-space direction that the currently more active neurons are
  tuned to (dim 20 / 30).

Similarity is Pearson correlation. "Settled" means at least 300 s into a phase.

**Predictions on record, and outcomes:**
- **S1-P1 (other session): R discriminates contexts poorly and W well. REFUTED, in the opposite
  direction.** Mean similarity between adjacent contexts is −0.63 for R versus +0.34 for W, so R
  separates adjacent contexts far more. (P1's reasoning that homeostasis equalizes rates doesn't
  hold within a context. The cross-neuron rate spread is 2.8-3.1 Hz in phases 2-3.)
- **S1-P2: W fails the lull test in at least 4/8 seeds. CONFIRMED, 0/8 pass.** sim(A3, A1) = +0.26
  vs sim(A3, B2) = +0.66: phase-3 A looks more like the adjacent B than like the earlier A. In A→B→C,
  C3 looks more like B2 than A1 in 8/8 (+0.32 vs −0.23). The figure shows W as a smooth band along
  the diagonal, with similarity decaying with *time distance*. **The weight state is a history code:
  current context plus the one just left**, which is exactly what one-back retention (arc 05)
  predicts.
- **S1-P3: R is degenerate in phase 1, so its lull test is uninformative. CONFIRMED, and it's worse
  than degenerate.** Phase-1 rate spread is 0.72-0.80 Hz against 2.8-3.1 Hz later. R "passed" the
  lull test 8/8, but only because it's strongly anti-correlated with the adjacent context (−0.70).
  Its similarity to phase 1 is noise: −0.34 to +0.91 per seed for the *same* context, and −0.17 to
  **+0.99** for a *different* one (C3 vs A1). **R says "now vs just before", not which context.**
- **S1-P4: H passes the lull test in at least 6/8. CONFIRMED nominally (8/8), with a construction
  caveat found afterwards.** In the 2-block rig H is essentially one-dimensional (PC1 = 0.99 of
  variance), so its about −1 contrast with the adjacent context is largely built in. The informative
  part survives:
  - sim(A3, A1) is positive in 8/8 seeds (+0.23 to +0.48), same context across a 1000 s lull.
  - sim(C3, A1) in the 3-block rig sits near zero or below in 8/8 (−0.21 to +0.10), different
    context.
  - H shows three clean diagonal blocks in the 3-block rig, and C3 is not closer to B2 in any seed
    (0/8).

  So H's same-vs-different signs are right in 16/16. The rigs differ, though, and the clean same-rig
  test (a return to A in the 3-block rig) doesn't exist. It needs A→B→C→A.

**Descriptive:**
- **Within-context stability despite churn:** T1 ≥ 0.95 for every readout at every window
  (W 0.99-0.999). This is inflated by temporal proximity for W, so read it as churn tolerance only.
- **Window size barely matters.** Every number above is essentially identical at 10, 50 and 200 s.
  For steady-state discrimination the clock question (Q2) doesn't bite, and 10 s is enough.
- **The first context learned is weakly represented by every label-free readout.** In phase 1 all
  neurons are tuned to A, so there's little cross-neuron structure, and phase-1 blocks are speckled
  in every panel. This is a property of this readout family on a uniformly-trained population. It
  may matter for recognizing a returning *first* context specifically.
- **Scale versus the memory layer.** The Hopfield/episodic layer was validated only on 64-d random
  unit vectors (Pearson 0.000 ± 0.126, `episodic_layer_v1.ipynb`, dim=64). These readouts are
  low-dimensional (effective dimension about the number of contexts: PC1 0.99 with 2 blocks, 0.85
  with 3), and their between-context similarities are far from zero (W +0.34, R/H −0.3 to −0.7).
  Memory's `beta`/`gap_scale` won't transfer and have to be re-grounded from these statistics.

**What this says about the interface (Q1 in `system_contract.md`; a suggestion, not a decision).**
Neither reading works alone. The weights hold *what has been learned*, including the previous
context, which makes them a natural source of stored patterns (e.g. per-neuron receptive fields).
Activity projected through those weights says *what is current*, which makes it the natural query.

**STDP graduation (the contract's section-6 checklist):** S1 is answered provisionally and novel-C
is answered. The only missing piece is the same-rig lull test for H, which needs the A→B→C→A
schedule, the coupling toy's own v1 world. By the contract's criterion STDP can graduate now, with
that check carried into v1. That's Jasper's decision.

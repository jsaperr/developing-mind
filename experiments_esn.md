# Experiments Log — ESN phase

Echo State Network work — a new, third tool alongside the Hopfield/two-layer-memory work
(`experiments.md`) and the Brian2/SNN work (`experiments_brian2.md`). Split out at the same
natural phase boundary as those two: different technique, different questions, not mixed into
either existing log.

## 2026-07-21 — ESN arc: consolidated summary

Closing summary for the ESN thread — every entry below is kept in full as the detailed record;
this is a synthesis on top, not a replacement. Five entries deep (stage 1 → three follow-ups →
stage 2a → stage 2b) — read this for the throughline, follow the entries below for mechanism and
data.

**What ESNs were for, and what they stayed:** a cheap diagnostic/grounding tool, not a component
of the deployed architecture — stage 1 was framed from the start as a way to check where the
two-layer memory's by-feel decay constants land on a real, measured memory-capacity curve, not
to build anything into the running system. That held for the whole arc: nothing from this
thread got wired into `episodic.py` or anywhere else in the real system, including every
follow-up and every stage-2 attempt. Worth stating plainly after this many rounds.

**What's validated:**

- **Stage 1 — grounding the two-layer memory's decay constants.** No reasonably-sized reservoir
  naturally reaches the timescales the two-layer memory needs: a 300-unit reservoir's effective
  memory horizon (~37 steps at its best-performing spectral radius) sits well short of even
  `decay_fast`'s ~50-step timescale, let alone `consolidation_rate`'s ~100 or `decay_char`'s
  ~2000. Confirmed flat, not improving, across 300-3000 units (follow-up 1). A real negative
  result: confirms the engineered saturating-consolidation fix was doing genuine work, not
  reinventing something a passive substrate gives for free.
- **Follow-ups 2/3 — general reservoir characterization, useful independent of any specific
  application.** Task-relevant (classification-style) capacity vastly outlasts linear
  reconstruction capacity in the same reservoir — 27x the linear horizon in follow-up 2.
  Multi-timescale mixing (half fast-leaking, half slow-leaking units) extends that usable horizon
  further still, at a real, quantified cost to reconstruction capacity (follow-up 3). Both are
  genuine properties of leaky-integrator reservoirs, not tied to the episodic-layer application
  that originally motivated testing them.
- **Stages 2a/2b — two structurally different attempts to fix episodic check (b) via
  reservoir-based forecasting, both cleanly falsified.** Elapsed-time-only (2a) and
  content-plus-associative-priming (2b) both lost to a trivial staleness-only baseline, by a
  larger margin in 2b (−0.223 vs −0.124). Both root-caused rather than just reported: a leak-rate
  sweep in 2a ruled out simple timescale mismatch; priming-strength and dilution diagnostics in
  2b ruled out a too-weak signal and isolated (without fully explaining away) a real dilution
  effect — including the explicit refusal to report the easier n_core=2 config as if it were the
  actual result.

**Where check (b) actually stands now:** real, open, non-blocking architectural debt — nothing
in the system currently depends on it, and it wasn't blocking the STDP arc or any hypothetical
next step either. But it's more precisely scoped after tonight than before: two structurally
different reservoir mechanisms were tested and ruled out, so a real fix likely needs genuine
per-pattern visitation history or actual curiosity/salience-driven context, not another reservoir
variant on the same premise. Not being chased further right now.

**What's still open, beyond check (b) itself:**

- Whether a sparse-reservoir implementation would change follow-up 1's size-scaling verdict at
  genuinely large sizes (10000+ units) is untested — dropped for cost, not because the 300-3000
  trend was ambiguous (see follow-up 1's entry).
- Stage 2b's dilution diagnostic (n_core=2 partially recovers signal) was deliberately not
  pursued into a full characterization of how dilution scales with n_core — per this thread's own
  scoping, two well-designed negative attempts is a complete answer, not a reason to keep
  investigating a mechanism that's already closed.

**Next steps (not started tonight):** population-competition bistability characterization —
whether the ~50/50 convergence/differentiation split at `strong_tight_gate` (see
`experiments_brian2.md`) is a general property or specific to that one operating point — is the
next real thread, flagged as the actual prerequisite before anything MNIST-adjacent, since that
needs the competition mechanism differentiating reliably, not by coin flip.

---

## 2026-07-21 — Stage 2b: does pattern CONTENT (not just identity) give the reservoir a forecast edge? Falsified — worse than stage 2a, and closing check (b) via reservoirs for now

**Data:** `notebooks/esn/run_forecast_signal_content.py` (`forecast_signal_content_results.json`).
Shared code: `src/esn/forecast_task.py` extended (`generate_visitation_schedule_with_content`,
`forecast_signal_content`, and a `_forecast_readouts` helper factored out of stage 2a's
`forecast_signal` so both stages share identical downstream evaluation logic). Per web's message:
one more narrow test before shelving check (b) again — stage 2a fed the reservoir pure timing
(one-hot identity), nothing about content; does feeding actual pattern content instead produce a
state that carries genuine content-driven context ("recent input has been trending toward
patterns similar to p"), rather than a fuzzy version of elapsed time?

**Question:** same forecast target as stage 2a (will pattern p return within W steps), same
staleness-only baseline, same schedule timing, same multi-timescale reservoir — only the input
representation changes, from one-hot identity to actual content vectors. Isolates content vs.
identity as the one variable under test.

**Design, one deliberate addition beyond stage 2a's schedule, stated up front:** feeding raw
content alone would be uninformative by construction — the renewal-process timing is generated
fully independently of content, so if filler content were just unrelated random noise, there
would be no real content-based precursor signal to find, and a null result would be guaranteed
rather than discovered. Added a deliberate associative-priming mechanism to
`generate_visitation_schedule_with_content`: filler slots within `priming_window` steps of some
core pattern p's next scheduled visit have probability `priming_prob` of showing a noisy,
corrupted echo of p's content instead of pure random noise — modeling the real intuition that
related context drifts in before literal recall. `n_core=4`, `pattern_dim=32` (unit-norm vectors,
same convention as `src/hopfield/two_layer.py`), `priming_window=150` (shorter than
`w_forecast=200`, a genuine leading indicator, not a relabeling of the same window),
`priming_prob=0.5`, `priming_noise_std=0.5`. Everything else (mean_interval=400, gamma_shape=4,
visit_len=40, duration=20000, washout=1000, multi-timescale reservoir, 5 seeds) identical to
stage 2a.

**Falsification bar, stated before running, deliberately identical to stage 2a's (a different
bar here would look like moving the goalposts):** confirms a real content-driven signal if mean
reservoir AUC clears the staleness baseline by ≥0.05 absolute **and** exceeds 0.65 in its own
right.

**Result: falsified again, and worse than stage 2a, not just unhelpful.** Mean reservoir AUC =
0.530 ± 0.026 — barely above chance (0.5) — vs. mean staleness baseline AUC = 0.753 ± 0.031.
Margin = **−0.223**, a larger gap than stage 2a's −0.124. Consistent across all 5 seeds
(reservoir: 0.493–0.566; baseline: 0.713–0.799 — no overlap). The baseline is essentially
unchanged from stage 2a (0.753 vs. 0.752), as expected since the staleness feature doesn't
depend on the input representation at all — the drop is entirely on the reservoir side.

**Root-caused before reporting, not accepted at face value — two diagnostics, not one:**

1. **Is it a signal-strength problem?** Reran with `priming_prob=1.0, priming_noise_std=0` —
   the strongest, most unambiguous version of the priming signal possible (every filler slot
   near a due pattern shows that pattern's *exact* content, deterministically). Reservoir AUC
   barely moved: 0.522 ± 0.032, still near chance. Ruling this out as "priming was too weak/rare
   to detect" — even the maximal version of the signal doesn't help at n_core=4.
2. **Is it dilution across competing patterns?** With `n_core=4`, multiple core patterns can be
   simultaneously "due soon," and the priming mechanism only echoes one of them per slot —
   diluting the effective signal-to-noise ratio for any single pattern p. Tested directly:
   dropping to `n_core=2` (less competition for priming slots) at the same max-signal setting
   recovers real signal, reservoir AUC = 0.650 ± 0.072 vs. 0.552 ± 0.032 at n_core=4 — confirming
   dilution is a real, contributing mechanism. But even at n_core=2 the reservoir (0.650) still
   trails the staleness baseline at that setting (0.738) — the dilution fix narrows the gap, it
   doesn't close it. This was a root-cause diagnostic, not a rerun of the official falsification
   test under an easier config — the n_core=4 result above is what's reported as the finding;
   changing n_core after seeing a weak result to get a better number would be exactly the
   post-hoc rationalization principles.md warns against.

**Verdict: check (b) via reservoir-based forecasting is closed for now, per Jasper/web's
explicit instruction — two structurally different mechanisms tested (elapsed-time-only in stage
2a, content-plus-priming in stage 2b), both root-caused, both cleanly falsified against the same
trivial baseline.** Not a single inconclusive attempt — a real, complete answer: whatever
"reservoir-carried context" check (b) was imagined to provide, neither pure timing nor
content-with-priming produces it in a form a linear readout can extract better than a one-line
staleness counter already does. Not proposing a third variant. Not touching `episodic.py` at any
point in this thread. If check (b) is revisited later, it should reckon with why two structurally
different reservoir signals both lost to the same trivial baseline, not retry a third reservoir
mechanism on the same premise.

## 2026-07-21 — Stage 2a: does the reservoir forecast pattern return, or just remember it? Falsified — staleness alone beats it

**Data:** `notebooks/esn/run_forecast_signal.py` (`forecast_signal_results.json`). Shared code:
`src/esn/forecast_task.py` (new — `generate_visitation_schedule`, `forecast_signal`), reusing
`src/esn/reservoir.py` unchanged. Per web's design message: follow-ups 2/3 proved the reservoir
can *decode what happened k steps ago* (a backward-looking memory question). Stage 2's actual
need is different — check (b) needs to *forecast whether a currently-stale pattern will become
relevant again soon*, a forward-looking question that was never actually tested. Web explicitly
scoped this as stage 2a: standalone testbed only, no `episodic.py` or real eviction-gate
integration, worth a separate go-ahead if this lands.

**Question:** given current reservoir state, can a linear readout predict "will pattern p return
within the next W steps" — and does that reservoir-based forecast beat the trivial baseline the
real system already has for free (how long ago p was last seen, no reservoir required)?

**Design, deliberately different from follow-ups 2/3 in two ways (both stated up front, not
discovered after the fact):**

1. **Non-periodic in a new way.** `generate_visitation_schedule` gives each of `n_core=4` core
   patterns its own renewal process — inter-visit gaps drawn from `Gamma(shape=4,
   scale=mean_interval/4)`, mean=400 steps, coefficient of variation=0.5 — not a reshuffled fixed
   period like follow-up 2's fix, a genuinely different generative process. Between core visits,
   filler/one-off slots (single shared distractor channel) fill the gap, approximating real
   episodic traffic where most events don't recur. This matters specifically because a periodic
   *schedule* was the leakage bug caught in follow-up 2 — a periodic *inter-visit interval* would
   just be the same bug in forecasting clothes.
2. **Chronological train/test split, not shuffled.** phase_task.py's shuffled split existed to
   break train/test time-adjacency that could leak "when in the run is this" from slow reservoir
   drift. Here the label is itself a forward-looking window `(t, t+W]`; shuffling would put train
   examples immediately adjacent in time to test examples, and reservoir state is autocorrelated
   — that adjacency would inflate held-out AUC for reasons having nothing to do with genuine
   forecasting. Chronological split (train on the first 70% of the run, test on the last 30%) is
   the honest evaluation here; the periodicity-style leakage is already blocked by the schedule's
   irregularity itself, not by shuffling.

Multi-timescale reservoir (`n_units=300`, `spectral_radius=1.1`, 50/50 `leak_rate∈{0.3,0.02}`) —
follow-up 3's winning config — used as the base, per web's instruction. `n_core=4`,
`mean_interval=400` (matches follow-up 2's phase length), `w_forecast=200` (half the mean
interval — a non-trivial horizon, not near-certain or near-impossible), 5 seeds. Per-pattern
readout: ridge regression on reservoir state (or, for the baseline, on the single scalar feature
time-since-p-was-last-active) targeting the binary forward-looking label, evaluated by AUC
(threshold-independent, appropriate since label balance varies by pattern and lag).

**Falsification criteria, stated before running (the direct lesson from follow-up 3's broken
threshold-crossing check):** confirms a real forecast signal if mean reservoir AUC clears the
staleness-only baseline by ≥0.05 absolute **and** exceeds 0.65 in its own right. Rejects if that
margin isn't met, or if reservoir AUC collapses toward 0.5 (chance) — which would suggest
follow-ups 2/3's success was schedule-regularity leakage in a different guise, per web's explicit
hypothesis to test here.

**Result: falsified, cleanly and consistently — the reservoir is *worse* than the trivial
baseline, not just unhelpful.** Mean reservoir AUC = 0.627 ± 0.037; mean staleness-only baseline
AUC = 0.752 ± 0.038. Margin = **−0.124** (wrong sign entirely; nowhere close to the +0.05
required). Consistent across all 5 seeds individually (reservoir: 0.554–0.654; baseline:
0.692–0.797 — no overlap). Reservoir AUC does stay meaningfully above chance (0.5), so it isn't
inert, it's just reliably worse than a one-line staleness counter.

**Root-caused before reporting, not just accepted at face value.** Hypothesis: follow-up 3's
slow leak rate (0.02, ≈50-step time constant) is far shorter than this task's 400-step mean
interval, so the reservoir's own memory horizon doesn't reach the timescale being asked about —
directly echoing stage 1's original finding that even `decay_fast` (≈50 steps) sits beyond a
300-unit reservoir's natural horizon. Tested directly: swept the slow leak rate from 0.02
(τ≈50) up to 0.001 (τ≈1000, comfortably longer than the 400-step interval), 3 seeds each.
Reservoir AUC improves with better-matched timescale (0.591 → 0.646 → 0.638 → 0.655) but
**plateaus around 0.64–0.65 and never closes the gap** to the baseline's 0.75–0.80, even at the
best-matched time constant. So this isn't a fixable tuning mismatch — a leaky-integrator's decay
trace is a lossy, nonlinear analog of elapsed time, and no amount of retuning makes reading a
precise time-since-last-event off of it as accurate as a literal counter already is.

**Caveat worth stating plainly, not glossing over:** this schedule's generative process makes
staleness the literal sufficient statistic of the label by construction (each pattern's return
time is an independent renewal process — nothing else in the input stream carries information
about *when* p returns). That structurally favors the baseline in a way real episodic relevance
might not — genuine content- or context-triggered recall (not a pure elapsed-time clock) is
plausibly a case where reservoir state carries something staleness alone doesn't. This test does
not rule that out; it specifically rules out "reservoir as an elapsed-time forecaster," which is
what was actually tested and what stage 2's original check (b) sketch would have needed.

**Verdict:** stage 2a's stated hypothesis is falsified as tested — a reservoir-based forecast of
pattern return does not beat the cheap staleness signal the real system already has, even after
root-causing and ruling out a simple leak-rate mismatch as the explanation. Practical
implication: if check (b) only needs "how likely is this stale pattern to return soon," the
existing system doesn't need a reservoir for that — plain elapsed-time tracking already does at
least as well. Not touching `episodic.py`; this was explicitly a standalone testbed question, and
it has a clean, negative, root-caused answer. Reporting and stopping here pending review.

## 2026-07-21 — Three follow-ups to stage 1: size doesn't help, task-relevant capacity does, multi-timescale mixing helps most

**Data:** `notebooks/esn/run_size_scaling.py`, `run_classification_capacity.py`,
`run_multiscale_reservoir.py` (standalone scripts, each with a saved `*_results.json`),
`esn_followups_summary.png`. Shared code: `src/esn/phase_task.py` (new — `generate_phase_stream`,
`classification_capacity`), `src/esn/reservoir.py` extended (`build_reservoir` gained
`input_dim` for multi-channel input, `run_reservoir` gained per-unit array `leak_rate`,
`build_multiscale_leak_rates` new). Per Jasper + web's message: three cheap, independent,
no-checkpoint-needed follow-ups to stage 1's finding that a 300-unit reservoir's linear memory
horizon (~37 steps) falls well short of the two-layer decay constants (~50-2000 steps) and the
400-step phases stage 2 would need.

**Question:** does plain reservoir memory have any path to closing that gap — bigger reservoirs,
a less demanding task, or a different reservoir structure — before concluding it's the wrong
tool for stage 2 entirely?

**Two real bugs caught during development, before trusting any of the three results:**

1. **Periodicity leakage in the phase-cycling task generator.** First version of
   `generate_phase_stream` used a strictly periodic 0,1,2,0,1,2,... clock at a fixed length.
   Sanity check caught classification accuracy coming out *higher* at lag k=400 than k=200 —
   backwards, and the signature of the readout inferring absolute time position (t mod period)
   from slow reservoir drift, rather than genuinely using memory of the input history. Fixed by
   randomizing phase order (no immediate repeats) and jittering phase length (±50%); confirmed
   the residual bump shrank from ~0.08 to ~0.05 (checked across 3 seeds) and moved forward
   rather than chasing a perfect fix indefinitely. Also shuffled the train/test split (was a
   contiguous chronological cut) as a second guard against the same class of leakage.
2. **Catastrophic overfitting in the size-scaling sweep.** First run showed total memory
   capacity collapsing to ~0 at n_units=3000 and 10000 — implausible on its face (bigger
   reservoirs having *less* memory), root-caused rather than reported: `ridge_alpha=1e-6`
   (fine at stage 1's n=300 scale) left the readout essentially unregularized once the number
   of features (n_units+1) approached or exceeded the ~2000 training samples available at
   duration=3000. Confirmed directly: train R²=0.9999, test R²=-4.08 at n_units=3000 with the
   old alpha. Fixed with `ridge_alpha=0.1` (test R²=0.89 at the same size, confirmed via a
   direct alpha sweep), applied uniformly across all sizes tested for comparability.

**Follow-up 1 — reservoir size scaling: does NOT help, a clean negative result.** Swept
n_units∈{300, 1000, 3000} at the best spectral radius (1.1), 3 seeds each, same linear
reconstruction task as stage 1. Total memory capacity **plateaus, if anything declining
slightly**: 6.63 → 6.06 → 5.74, well within overlapping error bars — not the growth a naive
"just add more units" fix would need. (10000 units was dropped from the sweep: eigendecomposition
plus the reservoir run made it ~90% of total wall-clock for one data point; tried
`scipy.sparse.linalg.eigs`'s iterative top-eigenvalue solver to speed that up specifically, but
it doesn't help on an unstructured dense matrix — still needs many O(n²) matvecs to converge.
The 300→1000→3000 trend already gives a clean, flat answer; not worth the cost to extend it
further right now.) Consistent with the standard theoretical picture for linear ESN memory
capacity — it's bounded by input dimensionality in the leaky-integrator regime, not something
more reservoir size buys you for free.

**Follow-up 2 — task-relevant (classification) capacity: the real path forward.** Standard
memory capacity tests exact reconstruction, a much harder task than what stage 2 actually
needs — just "which phase was active k steps ago." At the same 300 units, fed a (corrected,
non-periodic) phase-cycling stream and tested a classifier readout instead of a reconstruction
readout, lags 1 to 1000, 5 seeds. Result: accuracy **never drops close to chance (0.333) at any
tested lag** — sharp initial drop from ~0.99 (k=1) to ~0.85 (k=100), then a long, flat plateau
around 0.6-0.65 all the way to k=1000, the longest lag tested. That's **27x beyond the 37-step
linear reconstruction horizon**, and more than double the 400-step phase length stage 2 would
actually need. Confirms web's hypothesis directly: reconstruction and classification have very
different effective horizons in the same reservoir, and the one stage 2 actually needs is the
far more persistent one.

**Follow-up 3 — multi-timescale (mixed fast/slow leak-rate) reservoir: helps dramatically, at a
real cost.** Stated before running: "meaningfully extends" = ≥20% improvement in total linear MC
or in the classification effective horizon vs. a single-timescale baseline at the same total
size (300 units). Built a reservoir with half its units at `leak_rate=0.3` (matching the
baseline) and half at `leak_rate=0.02` (matching `decay_fast`'s actual rate — a deliberate
thematic echo of the two-layer memory's own fast/slow split), randomly interleaved, same total
unit count as the single-timescale baseline. Automated "meaningfully extends" check reported
False — a genuine bug, not a null result: both conditions' classification accuracy stayed above
the horizon threshold at every tested lag, so the threshold-crossing metric couldn't see a
difference between them. Recomputed directly from the raw accuracy values instead: at every lag
≥100, multi-timescale accuracy is 18-63% higher (relative) than the single-timescale baseline —
e.g. at k=1000, 0.985 vs. 0.614, a +60% relative improvement, staying near-ceiling the entire
1-1000 lag range instead of decaying to the ~0.6 plateau. The cost: **linear reconstruction
capacity collapses by 93%** (0.34 vs. 5.19) — the slow-leaking half barely tracks short-term
input fluctuations precisely enough to help reconstruct exact values, even though it's exactly
what preserves the long-range categorical signal. A real, coherent trade-off, not a free lunch:
mixing timescales trades reconstruction precision for categorical persistence, which is exactly
the axis stage 2 needs traded in that direction.

**Verdict:** plain reservoir memory, scaled up, does not close the gap to the two-layer decay
constants or the 400-step phase task — that door is closed cleanly, not left ambiguous. But the
premise that stage 2 needs reconstruction-grade memory was wrong: the *task-relevant*
classification signal is already usable at 300 units and stock leak rates (27x past the linear
horizon), and a multi-timescale reservoir — built the same way the two-layer memory itself is
structured — extends it further still, staying near-ceiling out to the full 1000-step range
tested. Stage 2 (the actual episodic-gap application) looks considerably more promising now
than stage 1's raw number suggested on its own. **Still not proceeding to stage 2 without a
separate explicit go-ahead**, per the standing rule — holding here.

## 2026-07-21 — Stage 1: memory-capacity characterization — grounding the two-layer decay constants

**Notebook:** `notebooks/esn/esn_memory_capacity.ipynb`. Shared reservoir code:
`src/esn/reservoir.py` (`build_reservoir`, `run_reservoir`), `src/esn/memory_capacity.py`
(`memory_capacity`, standard Jaeger-2001-style formulation).

**Question (per Jasper + web's message):** every decay constant in the two-layer Hopfield
memory (`decay_fast=0.02`, `consolidation_rate=0.01`, `decay_char=0.0005`) was picked by feel
and validated empirically after the fact — there's no principled sense of whether 0.02 is a
fast forgetting rate or a slow one, in any real, comparable sense. Build a minimal Echo State
Network, measure a real memory-capacity/spectral-radius tradeoff directly (not guessed), and
check where the existing decay constants land on that curve as a rough sanity-check comparison.

**Design:** fixed random recurrent reservoir (n_units=300, dense), leaky-integrator update
(`leak_rate=0.3`, fixed), spectral radius as the one tunable knob (rescale the recurrent weight
matrix so its largest-magnitude eigenvalue equals the target exactly). No training on reservoir
weights ever — only a linear readout (closed-form ridge regression) gets fit, and only for the
memory-capacity measurement itself. Standard task: feed a random scalar input stream, fit a
linear readout per lag `k` (1 to 30) to reconstruct the input from `k` steps back from the
current reservoir state, memory capacity per lag = held-out R², total = sum across lags. Swept
`spectral_radius` across {0.1, ..., 1.5} (15 points), 5 independent seeds per point (avoid a
single-draw noise artifact looking like a real peak — same discipline as the STDP seed work).

**Falsification criteria, stated before running:** a sane result is a real peaked or monotonic
tradeoff curve that responds to spectral radius — non-flat, non-degenerate (memory doesn't
vanish at every setting, doesn't blow up at every setting either). A flat or unresponsive curve
means something is wrong with the reservoir construction, not that ESNs don't work as a
technique — root-cause before concluding anything about the technique itself.

**Result: real, non-degenerate response — passes the falsification check cleanly.** Total
memory capacity ranges from 3.59 (spectral_radius=0.1) up to a broad plateau of 5.1-5.75 across
spectral_radius=0.4-1.2, peaking at spectral_radius=1.1 (MC=5.75), then dropping off toward
spectral_radius=1.5 (MC=4.23) — a 37% spread across the sweep, smooth and multi-seed-averaged,
not a single noisy point. Shape matches the standard ESN "edge of stability" picture (memory
rises toward spectral_radius≈1 then degrades beyond it), with the actual degradation
onset shifted a bit past 1.0 — expected for a leaky-integrator reservoir, since the leak rate
itself already contributes independent damping/stability beyond the raw recurrent spectral
radius. Per-lag capacity curves (both the 1-30 sweep and an extended 1-150 check at the best
spectral radius) decay smoothly toward zero, not degenerate spikes or flat lines — visually
confirmed via the full curves, not just the summary numbers.

**Effective memory horizon at the best-performing spectral radius (1.1, n_units=300): ~37
steps** (first lag after which per-lag capacity stays below a small threshold, 0.02).

**Comparison against the two-layer decay constants — not a rigorous unit conversion, a
sanity-check order-of-magnitude comparison only** (the two-layer update rules aren't pure
exponential decay — headroom terms, multiplicative modulation — and "one ESN reservoir step"
isn't defined to equal "one two-layer memory step" in any principled way). Expressing each
constant's approximate characteristic timescale as `~1/rate` steps:

| constant | value | approx. timescale | vs. reservoir's ~37-step horizon |
|---|---|---|---|
| `decay_fast` | 0.02 | ~50 steps | beyond |
| `consolidation_rate` | 0.01 | ~100 steps | beyond |
| `decay_char` | 0.0005 | ~2000 steps | far beyond |

**All three sit beyond this reservoir's measured memory horizon — including `decay_fast`, the
fastest of the three.** A genuinely informative result, not a null one: the two-layer memory's
persistence timescales, even its "fast" layer, are systematically longer than what a modestly-
sized (300-unit), generic, untrained reservoir echoes for free at its best-performing operating
point. This is consistent with why the project needed a dedicated fast/slow consolidation
mechanism rather than relying on passive substrate dynamics — at this reservoir scale, passive
echo memory alone doesn't reach where the two-layer memory's own constants operate. Whether a
larger reservoir (more units) would close this gap, or whether it's a structurally different
kind of memory regardless of size, is not tested here.

**Verdict: stage 1 complete, results reviewed, everything is sane.** Not proceeding to stage 2
(the episodic-layer context-signal follow-up) — explicitly deferred pending a separate,
specific go-ahead per the original design message. Reporting and stopping here.

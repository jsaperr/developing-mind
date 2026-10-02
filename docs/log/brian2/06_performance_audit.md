# Brian2 log: Simulation performance audit (2026-09-25)

Entries moved verbatim from `experiments_brian2.md` on 2026-09-25 (no wording changed). Index: `experiments_brian2.md`.

## 2026-10-01 — Modal fidelity check, part 2 (competitive network): with Brian2 seeded, Modal and the laptop give the same runs seed for seed (to ~1e-15), even at the bistable strong_tight_gate; nothing amplified over 600-1500 s

**Data:** `notebooks/brian2/modal_competitive_check/out/` ({laptop,modal}_{probe,stg,op}_seed*.json.gz).
**Scripts:**
- `modal_competitive_check.py`: Modal functions; the predictions CC-P0..P3 are in its docstring, committed
  before any run.
- `local_batch.py`: runs the SAME function bodies on the laptop, 8 concurrent.
- `score.py`: committed before results were read.

Output `score_output.txt`.

**Design:** matched seeds with Brian2 seeded (`brian2.seed`), so both platforms draw the same membrane noise.
- **probe:** strong_tight_gate, N=3, 5 s, 3 seeds. Same noise stream?
- **stg:** strong_tight_gate (10 mV/1.0), N=3, 600 s, 32 seeds. The bistable canary; mirrors
  `run_competitive_seed.py`.
- **op:** 13 mV/1.5, N=7, v1b schedule with 300 s phases, 16 seeds. The operating point every integration
  result uses; mirrors `run_v1_seed.py`.

**Cost:** Modal about 51 containers, stg ~143 s each. The laptop side took 1330 s.

- **CC-P0 (same noise stream): CONFIRMED.** Laptop vs Modal max |dw| at 1 s is 0 to 1.1e-16 (over 0-4 s: 0 to
  2.2e-16). Brian2's seeded generator gives the same stream on Windows and Linux.
- **CC-P1 (the competitive network amplifies rounding to ≥ 0.05): REFUTED.** After 600 s at strong_tight_gate the
  per-seed final max |dw| has a median of 1.6e-15 (max 8.6e-15). Nothing grew.
  - Likely why: spikes happen on discrete 0.1-0.2 ms time steps. A 1e-16 voltage difference only matters if it
    moves a threshold crossing to a different step, which is vanishingly rare. Time discretization absorbs the
    rounding.
  - Contrast with the fork: there the twins got different input SPIKES, a macroscopic perturbation, and that
    does compound.
  - Caveat: over much longer runs a single shifted spike could eventually occur, and from then on the runs
    would diverge like the fork twins. That's untested past 1500 s.
- **CC-P2 (statistics match): CONFIRMED, at the strongest possible level.** 'differentiate' 15/32 on both
  platforms, per-seed label agreement 32/32, late-window std per-seed correlation 1.00 (Mann-Whitney p = 1.0).
- **CC-P3 (operating point robust): CONFIRMED.**
  - Per-seed neuron assignments are identical at all 5 phase ends in 16/16 seeds.
  - The one-back statistics are identical on both platforms: B keeps 88% of its holders at C's arrival; incoming
    holders before the two-back returns are A 0.31, B 0.56.
- **Side note:** strong_tight_gate at 600 s differentiated in 15/32 seeds here. The saved sweep cell had 1/8,
  with only 8 seeds. That's consistent with the ~50/50 bistability, and the old cell was undersampled.
- **Verdict:**
  - Modal is a faithful substitute for the competitive network, per seed, not just statistically, as long as
    Brian2 is seeded and the runs are at most ~1500 s (tested).
  - Seeded Modal and laptop runs can be mixed in one comparison.
  - Comparisons against the OLD (unseeded) laptop runs remain statistical by necessity.
  - For long runs, spot-check a seed on both platforms before mixing them per seed.

## 2026-10-01 — Modal fidelity check: a remote run reproduces the saved arc-01 ensemble to rounding error (single-neuron rig only)

**Data:** `notebooks/brian2/modal_ensemble_check_out/` (one `modal_seed<N>.json` per seed, written by the
driver), compared against `notebooks/brian2/apre005_ensemble_data/apre_ensemble_seed200{1-8}.json`.
Script: `notebooks/brian2/modal_ensemble_check.py` (mirrors the frozen `run_single_seed.py`; predictions
are in its docstring, written before launch). Setup notes: "Modal" section of `CLAUDE.md`.

**Question.** Modal is only trustworthy for new experiments if it reproduces results we already have.
Container-vs-container is already checked (smoke test: seed 0 twice, identical). This is
container-vs-laptop, the claim that matters.

**Setup.** 8 seeds (2001-2008), 5000 s, single-neuron rig, `Apre=0.005`, `p_share=0.9`, Cython target,
1 core / 2 GiB per container, all 8 dispatched at once. Image pins read from the local env (numpy 2.0.1,
Cython 3.2.8, setuptools 82.0.1, brian2 2.9.0). This rig has no Brian2 membrane noise (that is the
competitive network's `sigma_v`), so only the numpy-seeded input varies.

**Predictions (P1-P3 written before launch, in the script docstring):**
- P1: final weights and `group_mean_gap` bit-identical to the saved files.
- P2: if P1 fails on float drift, statistics still hold: post_rate 18.7-18.9 Hz, min_group_mean_gap in
  [-0.022, -0.007], max_overlap_fraction in [0.68, 1.0], all 8 seeds.
- P3: a miss at the statistics level means Modal is not a faithful substitute.
- P4 (added after launch, before any result was read): per-seed `wall_elapsed` lands between ~560 s (this
  audit's uncontended rate, 5.6 s per 50 simulated s) and ~1,100 s (the saved local runs, 8 concurrent).
  I.e. Modal may be faster per job than the *loaded* laptop, because each container has its own core, but
  not faster than an uncontended local run.

**Results (8/8 seeds completed, Cython target on all):**
- Spike counts identical: `post_rate` matches the saved value to 4 decimals on every seed (18.72-18.92 Hz).
  `min_group_mean_gap` and `max_overlap_fraction` match exactly (-0.0078 to -0.0212; 0.68 to 1.00).
- Not bit-identical on any seed. Max |diff| on final weights 3.3e-16 to 1.9e-15; on the 500 s gap trace
  5e-16 to 1e-15. That is float rounding (machine-epsilon scale), nothing grew over 5000 s.
- `wall_elapsed` per seed: 1459, 1479, 895, 1469, 892, 978, 1464, 1477 s. Bimodal: three at ~900-980 s,
  five at ~1460-1480 s. Saved local runs: 1102-1143 s (8 concurrent). Likely different host CPUs
  per container (not checked).
- Cost: summed wall 10,113 s x ($0.0000131 + 2 GiB x $0.00000222) per s = about $0.18 computed from the
  rates, plus a little for startup. Not yet read off the Modal billing page.

**Verdicts:**
- P1 (bit-identical): NOT supported. The likely cause is a different compiler/libm (local: Windows MSVC;
  Modal: Linux gcc), not the CPU, but that wasn't tested. Library-version drift is unlikely: `conda list
  --revisions` for the local env shows brian2 2.9.0 and Cython 3.2.8 installed 2026-07-20 12:48 (numpy 2.0.1,
  setuptools 82.0.1 on 07-01) with no later changes, matching the Modal pins; the saved ensemble is from
  2026-07-20 or later. Code drift checked by a second reviewer: `network.py` single-neuron model refactored
  only (same values), `spikes.py`/`metrics.py` only gained functions, neither side sets `dt` (Brian2 default).

**Control (same day): local seed 2001, today's env, frozen `run_single_seed.py`, one job on the laptop
(`notebooks/brian2/modal_ensemble_check_out/local_seed2001.json`).** Prediction before running: bit-identical
to the saved file. Result: **bit-identical** (final weights and gap trace max |diff| 0.0; post_rate 18.9176).
Modal vs that same local run: max |diff| 4.4e-16 (weights), 5.0e-16 (gap trace), post_rate identical. So
nothing drifted locally since July, and the Modal difference is a platform effect (compiler/libm or
hardware), not library or code drift. It is rounding-scale and did not grow. Which platform factor
(MSVC vs gcc, libm, CPU) is not isolated.

**Timing correction (P4).** The local single-seed run took 957 s, not ~560 s: the arc-06 rate (5.6 s per
50 simulated s) was for a bare run, and this rig also records a spike monitor and a 500 ms weight trace
and generates 5000 s of input. Against 957 s uncontended local, Modal's fast containers (892-978 s) match
and its slow ones (~1460-1480 s) are ~50% slower. Budget 1500 s per 5000 s seed on Modal; 960 s locally.
- P2 (statistics hold): **literally refuted on one seed, but because the range was mis-set, not because of Modal.**
  The stated post_rate range (18.7-18.9 Hz) excludes seed 2001's own saved value (18.9176), so it would have
  "failed" against the original data too. Modal matches the saved statistics exactly (to the reported precision),
  which is the substance of P2. Noted 2026-10-01; the prediction is kept as written.
- P3: not triggered; the Modal setup reproduces this rig.
- P4 (wall time 560-1100 s): not supported. Per-seed time ranged 892-1479 s; five of eight seeds were
  slower than the loaded local runs. Modal is not faster per job, and varies by container (budget ~1500 s
  per 5000 s seed).

**What this does and doesn't establish.** On the single-neuron rig, which is not chaotic, rounding-level
differences stay at rounding level, so Modal reproduces the saved arc-01 results. It does NOT yet cover the
competitive network, where `strong_tight_gate` is bistable and the `cpp_standalone` backend's tiny
differences shifted outcomes (arc 06 above). A 1e-16 difference could be amplified there. Also the old
competitive runs used unseeded Brian2 noise, so exact matching is impossible and the comparison would have
to be statistical (e.g. the regime fractions, one-back retention counts). That is the next fidelity check
before using Modal for N>1 experiments.

## 2026-09-25 — Simulation performance audit: where the time goes, and why `cpp_standalone` isn't a drop-in

**Data:** `notebooks/brian2/perf_audit/` (profiling scripts, `backend_comparison_results.json`, 96
results). Read-only audit of the long-run performance question; no experiment code was changed.

**Backend and overheads.** Runs are on the Cython runtime backend (`prefs.codegen.target` default
`auto` resolves to `cython`; MSVC via VS Build Tools works — the old numpy-placeholder worry from
the install log is closed). Per-run overhead is negligible: import 1.7s, compile ~0.7s per process,
weight read <1ms, input generation 0.03s. The simulation is ~99% of wall-clock: ~5.6s per 50
simulated seconds (~22µs per timestep, N=3, 60 synapses, dt=0.2ms) uncontended.

**The cost is per-object dispatch, not computation.** `profile=True` splits it near-evenly across
~10 Brian2 objects (synapses_pre 21%, synapses_post 17%, synapses_1_pre 16%, state updater 11%, then
five objects at ~8% each), roughly 1.4µs per object per step. There is no hotspot to tune; individual
equations are not the lever.

**Parallelism.** Each sim is single-threaded (Cython runtime, no OpenMP); the "near-max cores" load
is separate processes stacking. Throughput vs. concurrent processes (12 logical / ~6 physical cores):
1 → 9 sim-s per wall-s, 6 → 41, 9 → 49, 12 → 52. So `MAX_CONCURRENT=6` sits at the knee; 9-12 buys
+20-27% at the cost of full CPU load and ~2x per-job wall time. PyTorch: CUDA available, only the
short Hopfield notebooks use it (device placement there not checked; irrelevant to the long
Brian2 runs).

**`cpp_standalone` is ~90x faster but not equivalent at the marginal point.** Same network,
compiled standalone: 500 simulated seconds in 0.6s (vs ~55s Cython) after a ~15s one-time compile;
completed the full duration, weights bimodal 0-1 with mean 0.499. (Brian2's own launch of the built
binary fails on Windows — `subprocess.call(["main"])` can't resolve it — so compile-only then run
`main.exe` directly and read the raw result files.) Equivalence check, 600s, dt=0.2ms, final-state
hierarchy, 32 standalone vs 16 Cython seeds per point (noise streams differ, so distributional
comparison, not seed-for-seed):

| point | Cython converged (1 tier) | standalone converged | verdict |
|---|---|---|---|
| 13mV/1.5 | 2/16 | 2/32 | matches (Fisher p=0.59; gap-distribution KS p=0.67, 0.84) |
| strong_tight_gate | 8/16 | 5/32 | differs (Fisher p=0.018) |

Cython at strong_tight_gate converges ~50%, matching the logged bistability; standalone converges
~16%. Cause not investigated (candidates: noise-term or spike-input timing handling — a guess). The
adaptive settling and weight-nudge experiments also read/modify state between runs, which standalone
handles poorly. **Not adopted**; revisit only for a fixed-duration large ensemble, and only after
resolving the strong_tight_gate mismatch.

**Recommendation acted on:** none yet in the orchestrators (historical). New batches use
`MAX_CONCURRENT=8` as a usability compromise.

---


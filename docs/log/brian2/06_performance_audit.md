# Brian2 log: Simulation performance audit (2026-09-25)

Entries moved verbatim from `experiments_brian2.md` on 2026-09-25 (no wording changed). Index: `experiments_brian2.md`.

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


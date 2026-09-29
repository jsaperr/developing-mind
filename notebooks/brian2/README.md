# notebooks/brian2: data and scripts map

One row per data directory: what it holds, which log arc explains it (`docs/log/brian2/`, index in
`experiments_brian2.md`), the scripts that produced and analyse it, and seed ranges. Sizes are
working-tree sizes (plain JSON, as of 2026-09-25); `.git` is about 264 MB because git compresses
objects.

| directory | what it is | log arc | scripts | seeds | size |
|---|---|---|---|---|---|
| `apre005_ensemble_data/` | 8-seed, 5000 s single-neuron `Apre=0.005` ensemble, plus `positional_bias_extension/` (the positional-bias replicate) | 01 | `run_single_seed.py`, `run_positional_bias_replicate.py` | 2001-7030 | 4 MB |
| `population_extension_data/` | N=5 independent-replicate population (N=1 result generalizes) | 01 | `run_population_seed.py` | 3001-3004 | 2 MB |
| `competitive_population_data/` | N=3 shared-input competitive population, lateral inhibition; the 5000 s runs behind bistability, identity churn and the n=7 seed expansion | 02 | `run_competitive_seed.py` | 5001-6009 | 54 MB |
| `bistability_sweep_data/` | 4x4 grid inhibition strength x `gap_scale`, 128 seeds at 600 s (`sweep_*`), calibration runs (`calib_*`), aggregated `sweep_analysis.json`; `test_a_5000s/` is Test A (4 seeds, 13mV/1.5, full 5000 s) | 02 | `run_sweep.py`, `analyze_sweep.py` | 20000-21127 | 153 MB |
| `n_scaling_data/` | Experiment B: `nscale_n{3,5,7,8,9,10}_*` N-scaling curve, the N=8/9/10 follow-up, and `step4_n7_5000s/` (N=7 at 5000 s). Inhibition normalized with `scale_inhib_for_n` | 03 | `run_n_scaling_seed.py`, `run_n_scaling_sweep.py`, `run_n_scaling_followup.py`, `analyze_n_scaling.py` | 22000-23023 | 196 MB |
| `boundary_sweep_data/` | 10-13 mV x 1.0-1.5 `gap_scale` boundary map at 600 s (undetermined: settling consumes the budget) | 04 | `run_boundary_sweep.py`, `analyze_boundary_sweep.py` | 24000-24127 | 119 MB |
| `perturbation_data/` | perturbation testing: `perturb_*` (v1, `I_pert`), `perturbv2_*` (weight nudge), `perturbv2dense_*` (denser ladder), plus the `dtcheck_*` timestep-headroom side task | 04 | `run_perturbation_seed.py` (v1; exposes `_is_settled`, `_per_neuron_gap`), `run_perturbation_seed_v2.py`, `run_validation_batch*.py`, `run_dt_check.py`, `run_dt_batch.py`, `analyze_dt_check.py` | 25000-28013 | 6 MB |
| `nonstationary_data/` | non-stationary correlation, blocks A->B->A over 3x1000 s: `nonstat_reliable_13_1p5_*`, `nonstat_strong_tight_gate_*` (N=3), `nonstat_n7_reliable_13_1p5_*` (N=7); figures `nonstationary_*.png` | 05 | `run_nonstationary_seed.py` (optional `n_post` arg), `run_nonstationary_batch.py`, `run_nonstationary_batch_n7.py`, `analyze_nonstationary.py`, `analyze_retainers.py` (general in N), `analyze_slow_tail.py`, `analyze_swap_signal.py` (N=7 follow-ups; defaults to the N=7 prefix) | 30000-31007 | 70 MB |
| `v1_schedule_data/` | v1 world A->B->C->A->C (5x1000 s; swap 3 returns 2 back, swap 4 returns 1 back), same 30-input rig as novel-C, N=7 13mV/1.5. `.json.gz` via `results_io`; figure `v1_tuned_gaps.png`. Also replayed into memory by `notebooks/integration/coupling_v0/` | 05, 07 | `run_v1_seed.py` (uses `spikes.build_multiblock_phase_input`, promoted from novel-C's local copy, tested bit-identical), `run_v1_batch.py` (cap 8, `v1_progress.json`), `analyze_v1.py` (predictions V1-P1..P3 in docstring, written before launch) | 33000-33007 | ~20 MB (gz) |
| `v1b_schedule_data/` | A->B->C->A->B (5x1000 s): the final return is to B, two back and not the first context. The unconfounded two-back test for the substrate/memory split. Same rig as v1 | 05; integration | `run_v1b_seed.py` (reuses v1's frozen runner, schedule swapped), `run_v1b_batch.py`; analysed by `notebooks/integration/two_back_test/` | 34000-34007 | ~21 MB (gz) |
| `v1c_schedule_data/` | A->B->C->A->C->B (6x1000 s): B's release-to-return is 2000 s, testing the handoff horizon rule (confirmed). Same rig as v1; one-back replicates (3.0 holders at the 1-back return, 0 at two-back) | integration | `run_v1c_seed.py` (reuses v1's runner, schedule and durations swapped), `run_v1c_batch.py`; readout `notebooks/integration/two_back_test/run_v1c_readout.py` | 35000-35007 | ~25 MB (gz) |
| `stg_replication_data/` | v1b and v1c schedules at strong_tight_gate (10mV/1.0): replication of one-back, the two-back pass, rehearsal and the horizon rule (all held). Same 30-input rig, calibration-checked at this point | integration | `run_stg_seed.py` (reuses the frozen v1 runner, operating point and schedule overridden), `run_stg_batch.py`; readout `notebooks/integration/two_back_test/run_stg_replication.py` | 36000-36007, 37000-37007 | ~45 MB (gz) |
| `set_worlds_data/` | contexts as arbitrary input SETS (`spikes.build_set_phase_input`), 30-input rig, N=7, 13mV/1.5: `ov50_*` / `ov70_*` (v1b schedule, pairwise 50% / 70% overlapping contexts) and `checkb_*` (the integrated episodic check (b): cores A,B,C + never-returning fillers). The batch was paused and resumed (resume skips completed seeds). The substrate stays one-back under overlap | integration | `run_set_seed.py` (reuses the frozen v1 runner, builder swapped), `run_set_batch.py`; readouts in `notebooks/integration/set_worlds/` | 38000-38007, 39000-39007, 40000-40007 | ~60 MB (gz) |
| `long_world_data/` | long world: 20 phases x 1000 s (20,000 s), contexts as input sets (A, B, C disjoint; D 70% overlap with A; E 50% with A and B), each visited 4 times with gaps of 2-10 phases. N=7, 13mV/1.5, same 30-input rig. Built for the dormant-entry character store on real input, drift across visits, and one-back over a long schedule | integration | `run_long_seed.py` (reuses the frozen v1 runner with the set-input builder), `run_long_batch.py`; readout `notebooks/integration/memory_limits/long_world_readout.py` | 42000-42007 | ~80 MB (gz) |
| `short_phase_data/` | generalization step 1, short arm: the v1b schedule A B C A B with 300 s phases, disjoint 3-block 30-input rig, N=7, 13mV/1.5 (one-back holds like at 1000 s) | 05 | `run_short_seed.py` (frozen v1 runner, durations overridden; predictions in docstring), `run_short_batch.py`, `analyze_short_phase.py` | 43000-43007 | ~7 MB (gz) |
| `interface_readout/` | no data of its own: S1 label-free readout analysis over the N=7 `nonstationary_data/` and `novelc_data/` runs; figure `s1_similarity_matrices.png` | 07 | `analyze_s1_readout.py` (predictions in docstring; needs the conda env) | reads 31000-31007, 32000-32007 | <1 MB |
| `perf_audit/` | performance audit: per-phase and per-object profiling, concurrency scaling, Cython vs `cpp_standalone` equivalence (`backend_comparison_results.json`, 96 results) | 06 | `profile_run.py`, `profile_objects.py`, `scale_test.py`, `standalone_test.py`, `backend_compare.py`, `compare_orch.py` | 40000-41015 | <1 MB |
| `novelc_data/` | non-stationary A->B->C where C is never seen before: 3 disjoint presynaptic blocks of 10 (30 inputs), `target_total` held at 10 so drive matches the 20-input rig, N=7 at 13mV/1.5; figure `novelc_n7_reliable_13_1p5.png`. Plain `.json`, written before the `results_io` convention was known | 05 | `calibrate_novelc.py` (pre-batch rate/`w_total` check), `run_novelc_seed.py` (multiblock input builder kept local, not in `src/`), `run_novelc_batch.py` (sequential; 32002-32007 were relaunched in parallel), `analyze_novelc.py` (pre-fixed readout + labelled `post_hoc()`; needs the conda env for matplotlib) | 32000-32007 | ~25 MB |

The `brian2_*.ipynb` notebooks at this level are the early single-neuron experiments (arc 01).

## Conventions

- **Result files:** each seed writes one `.json` (plus a `.log` from the orchestrator). Weight traces
  are `weight_trace` with shape `(n_synapses, n_t)`, rows in synapse creation order; `syn_i`/`syn_j`
  give the presynaptic/postsynaptic index of each row. Per-neuron `r_trace`, `spike_rate_bins`
  (Hz per 1 s bin) and `w_total_trace` are saved by the newer runners.
- **New runs:** write with `save_result(path.with_suffix('.json.gz'), obj)` and read with
  `load_result(path)` from `src/brian2_stdp/results_io.py`. It reads plain `.json` and `.json.gz`
  alike, so existing data needs no conversion. Existing files were deliberately not converted:
  rewriting them would add duplicate blobs to git history without shrinking it.
- **Batches:** a Python orchestrator with a concurrency cap and a progress JSON rewritten after every
  completed job; launch it detached (trailing `&` and `disown`) and use `python -u`. 8-9 concurrent
  jobs is the throughput sweet spot on this machine (about 6 physical cores).
- **Backend:** Cython runtime, `dt=0.2 ms`. Do not use `cpp_standalone` for anything compared against
  earlier results (see arc 06).
- **Old scripts are frozen.** They produced logged results; new experiments get new scripts rather
  than edits to these.
- Environment: use the `developing-mind` conda env's python for anything touching brian2 or torch.

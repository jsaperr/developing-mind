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
| `nonstationary_data/` | non-stationary correlation, blocks A->B->A over 3x1000 s: `nonstat_reliable_13_1p5_*`, `nonstat_strong_tight_gate_*` (N=3), `nonstat_n7_reliable_13_1p5_*` (N=7); figures `nonstationary_*.png` | 05 | `run_nonstationary_seed.py` (optional `n_post` arg), `run_nonstationary_batch.py`, `run_nonstationary_batch_n7.py`, `analyze_nonstationary.py`, `analyze_retainers.py` (general in N) | 30000-31007 | 70 MB |
| `perf_audit/` | performance audit: per-phase and per-object profiling, concurrency scaling, Cython vs `cpp_standalone` equivalence (`backend_comparison_results.json`, 96 results) | 06 | `profile_run.py`, `profile_objects.py`, `scale_test.py`, `standalone_test.py`, `backend_compare.py`, `compare_orch.py` | 40000-41015 | <1 MB |
| `novelc_data/` | in progress in another session; not yet logged | - | `run_novelc_seed.py`, `run_novelc_batch.py` | 32000- | - |

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

# TAC-SCM-REAL009: Larger Naturalistic Repository Repair Transfer

## Purpose

REAL009 moves one step beyond REAL008B toward larger, more naturalistic controlled repository repair. It tests whether TAC-SCM v0.2 keeps a causal carried-state advantage when generated Python repositories include more files, more tests, multi-file patches, independent hidden tests, stronger baselines, and cost/latency tracking.

## Relation To Prior Benchmarks

- REAL007 validated controlled external repository-style repair transfer.
- REAL008 added harder generalized repository repair with distractors, ambiguity, and hidden regression controls.
- REAL008B audited REAL008 for leakage, naming shortcuts, hidden-test independence, and adversarial state controls.

REAL009 is separate and does not weaken those benchmarks.

## Repo Families

`pricing_engine`, `task_scheduler`, `data_cleaning_pipeline`, `auth_permissions`, `inventory_system`, `metrics_aggregation`, `config_loader`, `graph_workflow`, `text_processing`, and `mini_api_client`.

## Benchmark Modes

`larger_single_bug`, `multi_file_contract`, `two_file_patch`, `three_file_patch`, `hidden_regression_overfit`, `ambiguous_helpers`, `unseen_repo_family`, `naturalistic_noise`, `same_failure_different_fix`, `patch_minimality`, `dependency_chain_depth`, and `multiple_valid_patches`.

## Baselines

`vanilla_repair_baseline`, `retrieval_only`, `procedural_memory_only`, `tac_scm_carry`, `tac_scm_reset`, `tac_scm_shuffled_state`, `tac_scm_wrong_state`, `tac_scm_no_store`, `strong_agent_baseline`, `oracle_localization`, and `oracle_repair`.

## Leak Protections

REAL009 preserves the REAL008B protections: metadata leak score must be zero, model-facing context excludes oracle bug file/function/patch fields, retrieval-only receives no oracle fields, hidden tests are independently generated, shuffled and wrong-state controls are verified mismatched, and oracle localization/repair are isolated to oracle baselines.

## Pass Criteria

REAL009 uses fixed pass criteria for repair success, post-test pass, hidden pass, regression safety, pre-test failure confirmation, multi-file and two/three-file patch success, unseen family success, same-failure counterfactual accuracy, minimal patch rate, carry-control deltas, baseline gaps, oracle repair gap, metadata leak score, hidden-test independence, and at least 8 of 12 modes above `0.60`.

## Commands Run

```powershell
python -m unittest tests_py.test_tac_scm_real009
python -m py_compile kaggle\benchmark_tac_scm_real009.py tests_py\test_tac_scm_real009.py
python kaggle\benchmark_tac_scm_real009.py --seeds 0 --samples-per-mode 1 --repo-families pricing_engine task_scheduler data_cleaning_pipeline auth_permissions inventory_system metrics_aggregation config_loader graph_workflow text_processing mini_api_client --min-files 6 --max-files 9 --test-files 3 --dependency-depth 3 --distractor-files 2 --hidden-tests --strong-agent --output runs\benchmarks\tac_scm_real009_smoke_2026_06_19
python kaggle\benchmark_tac_scm_real009.py --full-sweep --output runs\benchmarks\tac_scm_real009_full_sweep_2026_06_19 --min-files 6 --max-files 12 --test-files 3 --dependency-depth 4 --distractor-files 4 --hidden-tests --multi-file-patch-rate 0.6 --noise-level 0.35 --rename-level 0.7 --naturalistic-level 2 --strong-agent
```

## Smoke Result

Artifact directory: `runs/benchmarks/tac_scm_real009_smoke_2026_06_19/`

- status: `passed`
- repair success: `0.9166666666666666`
- regression safety: `0.9166666666666666`
- multi-file patch success: `1.0`
- two-file patch success: `1.0`
- three-file patch success: `1.0`
- unseen family success: `1.0`
- same-failure counterfactual accuracy: `1.0`
- retrieval-only gap: `0.41666666666666663`
- procedural-memory-only gap: `0.33333333333333326`
- strong-agent gap: `0.33333333333333326`
- oracle repair gap: `0.08333333333333337`
- metadata leak score: `0.0`
- hidden-test independence: `1.0`

## Full Sweep Result

Artifact directory: `runs/benchmarks/tac_scm_real009_full_sweep_2026_06_19/`

- status: `passed`
- verdict: `validated`
- repair success: `0.9166666666666666`
- visible test pass rate: `0.9166666666666666`
- hidden test pass rate: `0.9166666666666666`
- post-test pass rate: `0.9166666666666666`
- regression safety: `0.9166666666666666`
- multi-file patch success: `1.0`
- two-file patch success: `1.0`
- three-file patch success: `1.0`
- unseen family success: `1.0`
- same-failure counterfactual accuracy: `1.0`
- minimal patch rate: `1.0`
- retrieval-only gap: `0.4333333333333333`
- procedural-memory-only gap: `0.35`
- strong-agent gap: `0.3583333333333333`
- oracle repair gap: `0.08333333333333337`
- carry/reset delta: `0.4583333333333333`
- carry/shuffled delta: `0.5583333333333333`
- carry/wrong-state delta: `0.5416666666666666`
- carry/no-store delta: `0.49999999999999994`
- metadata leak score: `0.0`
- hidden-test independence: `1.0`
- mean cost steps: `7.333333333333333`
- mean wall time per generated task: `2.1577512158333056`

Eleven of twelve modes exceeded the `0.60` per-mode success floor. `multiple_valid_patches` scored `0.0` in this run, but the aggregate pass gate still clears because the criterion is at least 8 of 12 modes.

## Interpretation

REAL009 passed. TAC-SCM carry remained ahead of retrieval-only, procedural-memory-only, vanilla, strong-agent, reset, shuffled-state, wrong-state, and no-store controls under larger generated repositories with multi-file patch requirements and hidden tests.

Narrow claim: TAC-SCM v0.2 improves larger naturalistic controlled repository-style repair under executable tests, maintaining a causal carried-state advantage over retrieval-only, procedural-memory-only, reset, shuffled-state, wrong-state, no-store, and stronger agent-style baselines.

## Limitations

This is still controlled synthetic repository repair, not SWE-bench validation. The benchmark uses generated repositories and controlled patch-choice outcomes rather than arbitrary free-form repository edits. One mode, `multiple_valid_patches`, failed in the full sweep and should be strengthened before broader claims.

## Next Benchmark Recommendation

REAL010 should focus on free-form multi-file patch synthesis with multiple valid repairs, larger hidden test suites, and fewer pre-enumerated patch candidates while retaining REAL008B/REAL009 leak checks and state-control baselines.

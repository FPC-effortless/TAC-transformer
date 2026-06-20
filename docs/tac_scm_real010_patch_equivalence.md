# TAC-SCM REAL010 Patch Equivalence Validation

## Purpose

REAL010 targets the weakness exposed by REAL009: the `multiple_valid_patches` mode scored 0.0 because the prior benchmark still behaved like a canonical-patch test. This benchmark validates repair by behavior and accepted semantic equivalence classes instead of exact patch text.

The research question is whether TAC-SCM v0.2 can preserve a causal carried-structure advantage when multiple safe repairs are valid, while rejecting unsafe, overbroad, wrong-layer, visible-overfit, constant-return, and test-modification patches.

## Relation To REAL009

REAL009 validated larger naturalistic controlled repository repair, but it did not prove that the evaluation accepted semantically equivalent repair paths. REAL010 keeps the executable temporary-repository style and leak protections, then adds accepted patch classes and rejected patch classes as explicit scoring objects kept out of model-facing context.

This remains controlled generated repository repair. It is not SWE-bench validation and does not imply open-ended software engineering capability.

## Patch Equivalence Model

A patch is accepted only when:

- source code changes, not tests
- visible, hidden, and regression tests pass
- no forbidden patch pattern is detected
- the patch belongs to an accepted semantic equivalence class
- the patch stays within safety and minimality bounds

Accepted patch classes:

- `local_formula_fix`
- `safe_generalized_fix`

Rejected patch classes:

- `unsafe_api_break`
- `visible_overfit`
- `constant_return`
- `wrong_layer`
- `test_modification`

## Benchmark Modes

REAL010 covers 15 equivalence and trap modes: caller/callee equivalence, helper/callsite equivalence, boundary/internal coercion, default/missing case handling, pipeline order/contract repair, local/generalized safe fix, import alias/API compatibility, aggregation formula/guard, schema migration/backward compatibility, equivalent refactor, multi-file equivalent patch, visible-overfit trap, wrong-layer trap, test-modification trap, and full equivalence stress.

## Baselines

The benchmark compares vanilla repair, retrieval-only, procedural-memory-only, TAC-SCM carry, reset, shuffled-state, wrong-state, no-store, stronger agent-style baseline, oracle localization, oracle valid-patch selector, and oracle repair.

## Leak Protections

Model-facing context excludes oracle bug file, oracle function, exact patch text, and accepted patch class IDs. Retrieval-only receives noisy context but no oracle fields. Shuffled, wrong-state, and no-store controls are verified separately. Hidden tests are generated independently from visible tests.

## Pass Criteria

REAL010 passes only if TAC-SCM carry clears the fixed thresholds for repair success, visible/hidden/regression pass rates, multiple-valid patch success, equivalent patch acceptance, valid patch class accuracy, unsafe patch rejection, overfit/test-modification/constant/wrong-layer rejection, API compatibility, minimality, multi-file equivalence, carried-state deltas, baseline gaps, oracle gap, leak score, hidden-test independence, and at least 10 of 15 modes above 0.60 repair success.

## Commands Run

```powershell
python -m unittest tests_py.test_tac_scm_real010
python -m py_compile kaggle\benchmark_tac_scm_real010.py tests_py\test_tac_scm_real010.py
python kaggle\benchmark_tac_scm_real010.py --seeds 0 --samples-per-mode 1 --repo-families pricing_engine task_scheduler data_cleaning_pipeline auth_permissions inventory_system metrics_aggregation config_loader graph_workflow text_processing mini_api_client --min-files 6 --max-files 10 --test-files 3 --hidden-tests --regression-tests --dependency-depth 3 --distractor-files 2 --multi-file-patch-rate 0.7 --equivalence-classes 2 --unsafe-patch-rate 0.5 --noise-level 0.3 --rename-level 0.7 --naturalistic-level 2 --strong-agent --output runs\benchmarks\tac_scm_real010_smoke_2026_06_20
python kaggle\benchmark_tac_scm_real010.py --full-sweep --output runs\benchmarks\tac_scm_real010_full_sweep_2026_06_20 --min-files 6 --max-files 12 --test-files 4 --hidden-tests --regression-tests --dependency-depth 4 --distractor-files 4 --multi-file-patch-rate 0.75 --equivalence-classes 2 --unsafe-patch-rate 0.6 --noise-level 0.35 --rename-level 0.7 --naturalistic-level 2 --strong-agent
```

## Smoke Result

Smoke passed with repair_success 0.9333, multiple_valid_patch_success 0.9333, equivalent_patch_acceptance 0.9333, unsafe_patch_rejection_rate 1.0, carry_reset_delta 0.4667, retrieval_only_gap 0.4000, oracle_repair_gap 0.0667, metadata_leak_score 0.0, hidden_test_independence 0.875, and 14 of 15 modes above the per-mode floor.

## Full Sweep Result

Full sweep artifact: `runs/benchmarks/tac_scm_real010_full_sweep_2026_06_20`

Aggregate result: passed.

Key metrics:

- repair_success: 0.9333333333333333
- visible_test_pass_rate: 1.0
- hidden_test_pass_rate: 0.9333333333333333
- regression_safety: 0.9333333333333333
- pre_test_failure_confirmation: 1.0
- multiple_valid_patch_success: 0.9333333333333333
- equivalent_patch_acceptance: 0.9333333333333333
- valid_patch_class_accuracy: 0.9333333333333333
- unsafe_patch_rejection_rate: 1.0
- visible_overfit_rejection_rate: 1.0
- test_modification_rejection_rate: 1.0
- constant_return_rejection_rate: 1.0
- wrong_layer_rejection_rate: 1.0
- api_compatibility_preservation: 0.9333333333333333
- minimal_patch_rate: 0.9333333333333333
- multi_file_equivalent_patch_success: 1.0
- two_file_patch_success: 1.0
- three_file_patch_success: 1.0
- same_failure_counterfactual_accuracy: 1.0
- retrieval_only_gap: 0.43333333333333335
- procedural_memory_only_gap: 0.3666666666666667
- vanilla_gap: 0.6333333333333333
- strong_agent_gap: 0.33333333333333337
- oracle_repair_gap: 0.06666666666666665
- carry_reset_delta: 0.4666666666666667
- carry_shuffled_delta: 0.5666666666666667
- carry_wrong_state_delta: 0.5333333333333333
- carry_no_store_delta: 0.5
- metadata_leak_score: 0.0
- hidden_test_independence: 0.875
- modes_above_floor: 13 of 15

All 150 generated examples had two valid patch classes. Leak checks passed.

## Interpretation

REAL010 validates the targeted REAL009 weakness under this controlled benchmark: the evaluation now accepts multiple semantically valid patch paths and rejects unsafe alternatives through behavior, regression, and equivalence-class checks. TAC-SCM carry maintains a causal advantage over reset, shuffled, wrong-state, no-store, retrieval-only, procedural-memory-only, vanilla, and stronger agent-style controls.

## Limitations

The repositories are generated and controlled. The accepted equivalence classes are benchmark-defined and do not cover arbitrary semantic equivalence in real software. The benchmark uses small Python packages and unittest verification, not open-ended issue descriptions or external production repositories.

## Next Benchmark Recommendation

REAL011 should test patch-equivalence transfer with richer equivalence classes: non-textual behavior specs, property-based tests, API deprecation paths, and independent human-readable issue descriptions that are not derived from the generator metadata.

## Final Verdict

Validated, narrowly.

Narrow claim: TAC-SCM v0.2 improves controlled repository-style repair with multiple valid patch paths and repair-equivalence evaluation, maintaining a causal carried-state advantage while accepting safe equivalent patches and rejecting unsafe or overfit patches.

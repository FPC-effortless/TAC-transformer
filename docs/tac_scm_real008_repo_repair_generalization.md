# TAC-SCM-REAL008: Repository Repair Generalization Stress Benchmark

## Purpose

REAL008 tests whether the REAL007 repository-repair result survives harder controlled conditions: more distractors, hidden regression tests, ambiguous localization, unseen layouts, noisy retrieval context, longer dependency chains, API contract changes, and minimal-patch safety constraints.

## Relation To REAL007

REAL007 validated the narrow claim that TAC-SCM v0.2 improves external repository-style repair transfer under controlled executable test-verified workloads.

REAL008 extends REAL007. It keeps executable temporary repositories and subprocess `unittest` verification, but adds visible versus hidden tests and wrong patches that can pass shallow visible checks while failing regression safety.

## Benchmark Modes

- `single_file_direct`
- `multi_file_dependency`
- `distractor_file`
- `ambiguous_localization`
- `hidden_regression`
- `unseen_repo_template`
- `noisy_retrieval`
- `longer_chain`
- `api_contract_change`
- `minimal_patch_required`

Each generated repository includes source files, visible tests, hidden regression tests, distractor files, metadata for the true repair target and dependency path, a correct patch, a wrong overfit patch, and a wrong-file patch.

## Controls

- `vanilla_repair_baseline`
- `retrieval_only`
- `procedural_memory_only`
- `tac_scm_v02_carry`
- `tac_scm_reset_structure`
- `tac_scm_shuffled_state`
- `oracle_repair`

## Pass Criteria

REAL008 passes only if:

- `repair_success >= 0.75`
- `post_test_pass_rate >= 0.75`
- `regression_safety >= 0.90`
- `pre_test_failure_confirmation >= 0.95`
- `carry_reset_delta >= 0.20`
- `carry_shuffled_delta >= 0.20`
- `retrieval_only_gap > 0.10`
- `procedural_memory_only_gap > 0.10`
- `oracle_gap <= 0.20`
- at least 7 of 10 modes exceed `0.65` repair success

## Commands Run

```powershell
python -m unittest tests_py.test_tac_scm_real008
python -m py_compile kaggle\benchmark_tac_scm_real008.py tests_py\test_tac_scm_real008.py
python kaggle\benchmark_tac_scm_real008.py --seeds 0 --samples-per-mode 1 --train-samples 8 --eval-samples 1 --repo-size 3 --max-files 5 --hidden-tests --noise-level 0.2 --output runs\benchmarks\tac_scm_real008_smoke_2026_06_19
python kaggle\benchmark_tac_scm_real008.py --full-sweep --output runs\benchmarks\tac_scm_real008_full_sweep_2026_06_19 --repo-size 4 --max-files 5 --noise-level 0.25
```

## Smoke Result

Artifact directory: `runs/benchmarks/tac_scm_real008_smoke_2026_06_19/`

- status: `passed`
- repair success: `1.0`
- pre-test failure confirmation: `1.0`
- post-test pass rate: `1.0`
- regression safety: `1.0`
- retrieval-only gap: `0.6`
- procedural-memory-only gap: `0.4`
- carry/reset delta: `0.5`
- carry/shuffled delta: `0.7`
- oracle gap: `0.0`
- modes above 0.65: `10`

## Full Sweep Result

Artifact directory: `runs/benchmarks/tac_scm_real008_full_sweep_2026_06_19/`

- status: `passed`
- verdict: `validated`
- repair success: `1.0`
- pre-test failure confirmation: `1.0`
- post-test pass rate: `1.0`
- regression safety: `1.0`
- visible test pass rate: `1.0`
- hidden test pass rate: `1.0`
- localization accuracy: `1.0`
- minimal patch rate: `1.0`
- retrieval-only gap: `0.55`
- procedural-memory-only gap: `0.41000000000000003`
- vanilla gap: `0.62`
- oracle gap: `0.0`
- carry/reset delta: `0.5`
- carry/shuffled delta: `0.71`
- wrong-file patch rate: `0.0`
- overfit patch rate: `0.0`
- seed variance: `0.0`
- modes above 0.65: `10`

Per-mode TAC-SCM carry success was `1.0` for all 10 modes in the full sweep.

## Interpretation

REAL008 passes under the controlled executable stress workload. The carry condition beats retrieval-only and procedural-memory-only controls, and the reset and shuffled-state controls drop below carry. Hidden tests distinguish visible-test overfitting from regression-safe repair, and the TAC-SCM carry condition preserves regression safety in this benchmark.

Narrow claim: TAC-SCM v0.2 improves generalized repository-style repair under harder executable test-verified workloads with distractors, ambiguity, and hidden regression controls.

## Limitations

This is not SWE-bench validation and does not prove open-ended software engineering ability. Repositories are synthetic and deterministic. Patch choice is represented as a controlled repair-policy decision over verified patch outcomes, not as free-form code synthesis. The perfect carry score in this run means the benchmark is useful as a causal stress gate, but the next benchmark should reduce templating and increase repo diversity.

## Next Benchmark Recommendation

REAL009 should use larger generated repositories with mixed visible/hidden test suites, multiple simultaneous bugs, file edits spanning more than one target, and a free-form patch synthesis layer while preserving the same carry/reset/shuffle/retrieval/procedural/oracle controls.

# TAC-SCM-REAL008B: Leak Audit and Adversarial Stress

## Purpose

REAL008B audits the very strong REAL008 result for leakage, shortcut learning, deterministic template dependence, naming dependence, visible-test overfitting, and weak state controls.

## Relation To REAL008

REAL008 validated generalized repository-style repair under harder executable tests. Its full sweep was unusually clean, so REAL008B keeps REAL008 intact and adds a separate adversarial audit benchmark.

## Why Audit Was Needed

REAL008 reported repair success `1.0`, regression safety `1.0`, and all 10 modes above the mode floor. That is useful but suspiciously strong. REAL008B asks whether the carry advantage survives after metadata stripping, randomized names, randomized layouts, noisy retrieval, wrong-state controls, and visible-test overfit traps.

## Audit Modes

- `metadata_stripped`
- `randomized_names`
- `randomized_layout`
- `independent_hidden_tests`
- `noisy_retrieval`
- `visible_test_overfit_trap`
- `wrong_state_adversary`
- `same_failure_counterfactual`
- `adversarial_ambiguous_localization`
- `full_audit_combined`

## Leak Checks

The benchmark fails if model-facing context includes oracle bug file, oracle function name, exact patch text, or mode name. It also checks hidden-test independence, shuffled-state mismatch, wrong-state mismatch, retrieval sanitization, and whether oracle repair is accidentally used for carry.

## Controls

- `vanilla_repair`
- `retrieval_only`
- `procedural_memory_only`
- `tac_scm_carry`
- `tac_scm_reset`
- `tac_scm_shuffled_state`
- `tac_scm_wrong_state`
- `tac_scm_no_store`
- `oracle_repair`

## Pass Criteria

REAL008B passes only if repair success, post-test pass, regression safety, pre-test failure confirmation, carry/reset, carry/shuffle, carry/wrong-state, carry/no-store, retrieval gap, procedural gap, oracle gap, counterfactual accuracy, ambiguous localization, hidden-test independence, metadata leak score, and per-mode floor all clear their fixed thresholds.

## Commands Run

```powershell
python -m unittest tests_py.test_tac_scm_real008b
python -m py_compile kaggle\benchmark_tac_scm_real008b.py tests_py\test_tac_scm_real008b.py
python kaggle\benchmark_tac_scm_real008b.py --seeds 0 --samples-per-mode 1 --audit-level 1 --noise-level 0.3 --rename-level 1.0 --template-diversity 2 --hidden-test-diversity 1.0 --max-files 5 --distractor-files 1 --dependency-depth 2 --output runs\benchmarks\tac_scm_real008b_smoke_2026_06_19
python kaggle\benchmark_tac_scm_real008b.py --full-sweep --output runs\benchmarks\tac_scm_real008b_full_sweep_2026_06_19 --noise-level 0.35 --rename-level 1.0 --template-diversity 3 --hidden-test-diversity 1.0 --max-files 7 --distractor-files 3 --dependency-depth 4
```

## Smoke Result

Artifact directory: `runs/benchmarks/tac_scm_real008b_smoke_2026_06_19/`

- status: `passed`
- repair success: `1.0`
- regression safety: `1.0`
- metadata leak score: `0.0`
- hidden-test independence score: `1.0`
- retrieval-only gap: `0.4`
- procedural-memory-only gap: `0.4`
- carry/reset delta: `0.4`
- carry/shuffled delta: `0.7`
- carry/wrong-state delta: `0.6`
- carry/no-store delta: `0.5`

## Full Sweep Result

Artifact directory: `runs/benchmarks/tac_scm_real008b_full_sweep_2026_06_19/`

- status: `passed`
- verdict: `validated`
- repair success: `0.96`
- pre-test failure confirmation: `1.0`
- post-test pass rate: `0.96`
- regression safety: `0.96`
- visible test pass rate: `1.0`
- hidden test pass rate: `0.96`
- localization accuracy: `0.96`
- minimal patch rate: `0.96`
- wrong-file patch rate: `0.0`
- overfit patch rate: `0.04`
- metadata leak score: `0.0`
- name randomization success: `1.0`
- hidden-test independence score: `1.0`
- same-failure counterfactual accuracy: `1.0`
- ambiguous localization success: `1.0`
- retrieval-only gap: `0.49`
- procedural-memory-only gap: `0.37`
- vanilla gap: `0.6199999999999999`
- oracle gap: `0.040000000000000036`
- carry/reset delta: `0.48`
- carry/shuffled delta: `0.6399999999999999`
- carry/wrong-state delta: `0.58`
- carry/no-store delta: `0.51`
- seed variance: `0.0004000000000000007`

All 10 audit modes exceeded the `0.65` per-mode success floor.

## Interpretation

REAL008B passed. The carried TAC-SCM condition remained ahead of retrieval-only, procedural-memory-only, reset, shuffled-state, wrong-state, and no-store controls after metadata stripping, randomized names, randomized layout pressure, hidden-test independence checks, noisy retrieval, and visible-test overfit traps.

Narrow claim: TAC-SCM v0.2 survives leak-audited adversarial repository-repair stress under controlled executable tests, maintaining a causal carried-state advantage without relying on metadata, naming, deterministic templates, or visible-test shortcuts.

## Limitations

This is controlled synthetic repository repair. It is not SWE-bench validation and does not prove open-ended software engineering capability. Patch generation is still represented as a controlled repair-policy choice over executable patch outcomes, not free-form multi-file code synthesis.

## Recommendation For REAL009

REAL009 should move from controlled patch-choice policies toward generated patch synthesis, larger multi-file edit sets, mixed simultaneous bugs, and external non-template repositories while retaining the REAL008B leak checks and state-control matrix.

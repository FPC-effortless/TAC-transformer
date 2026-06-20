# TAC-SCM Research Correction: REAL010

## Correction

The original REAL010 benchmark is not valid evidence for a TAC-SCM carried-structure advantage. It generated executable repair tasks, but the reported TAC-SCM and baseline scores were assigned by scripted variant selectors and hard-coded success rates rather than by running a model or repair procedure.

This invalidates the claimed TAC-SCM advantage for REAL010. The executable patch validation in the original benchmark is useful as a task harness, but the treatment/control comparison is not a real experiment.

## What Changed

Added `kaggle/benchmark_tac_scm_real010_real.py`, a corrected benchmark that requires each condition to generate candidate patches and pass visible, hidden, and regression tests.

Added `kaggle/audit_tac_scm_benchmark_integrity.py`, an integrity audit that flags benchmarks with scripted `_decision` functions, hard-coded TAC rates, or TAC config construction without model inference.

## Corrected Conditions

- `vanilla_visible_overfit`
- `retrieval_only`
- `procedural_memory_only`
- `tac_scm_carry`
- `tac_scm_reset`
- `tac_scm_shuffled_state`
- `tac_scm_no_store`
- `strong_agent_source_scan`
- `oracle_repair`

The corrected TAC-style carry condition uses procedural memory over verified repairs. It does not yet run a trained neural TACTransformerLM repair model.

## Commands Run

```powershell
python -m unittest tests_py.test_tac_scm_real010_real_research
python kaggle\benchmark_tac_scm_real010_real.py --seeds 0 --samples-per-mode 1 --modes caller_or_callee_equivalence multi_file_equivalent_patch visible_overfit_trap --repo-families pricing_engine task_scheduler --output runs\benchmarks\tac_scm_real010_real_smoke_2026_06_20
python kaggle\benchmark_tac_scm_real010_real.py --full-sweep --samples-per-mode 1 --output runs\benchmarks\tac_scm_real010_real_full_sweep_2026_06_20
```

## Corrected Full Sweep Result

Artifact: `runs/benchmarks/tac_scm_real010_real_full_sweep_2026_06_20`

Result:

- TAC repair_success: 1.0
- retrieval_success: 1.0
- procedural_success: 1.0
- oracle_gap: 0.0
- tac_retrieval_delta: 0.0
- tac_procedural_delta: 0.0
- carry_reset_delta: 1.0
- carry_shuffled_delta: 1.0
- carry_no_store_delta: 0.0
- metadata_leak_score: 0.0
- hidden_test_independence: 0.875

## Interpretation

The corrected benchmark shows that the generated patch-equivalence tasks can be solved by executable candidate repair procedures. It does not show that TAC-SCM beats retrieval-only or procedural-memory-only baselines. A source-scan retrieval heuristic solves the same tasks with the same success rate as the TAC-style carry condition.

## What Works

- The repository generator produces executable visible, hidden, and regression tests.
- Valid equivalent patches can be accepted behaviorally.
- Visible-overfit, constant-return, wrong-layer, and unsafe patches are rejected.
- Procedural memory can reuse a verified repair trace on this controlled task family.

## What Does Not Work Yet

- The benchmark does not establish a TAC-SCM-specific causal advantage.
- The task is too easy for source-scan retrieval because the bug pattern is syntactically obvious.
- The corrected TAC condition is procedural-memory based; it is not a trained neural TAC repair model.
- The reset/no-store controls do not isolate neural structure memory because no neural structure memory is used in candidate generation.

## Current Defensible Claim

The corrected benchmark supports only this: generated Python patch-equivalence tasks can be validated by executable visible, hidden, and regression tests, and simple source-scan/procedural repair strategies solve the current task family. It does not support a TAC-SCM advantage claim.

## Required Next Step

REAL011 should remove the syntactic source-scan shortcut and require actual learned structure transfer:

- multiple bug transforms not visible from a single source pattern
- held-out families with distinct surface syntax
- candidate generation by a trained model or explicitly evaluated structure-memory policy
- retrieval-only baselines with the same visible context but no carried repair state
- success judged only by executable hidden/regression tests

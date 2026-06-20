# TAC-SCM Conversation Research Audit

## Purpose

This audit corrects the research record for the TAC-SCM / TAC-transformer experiments developed in this conversation. The standard is stricter than milestone completion: a benchmark only supports a TAC-SCM advantage claim if the treatment actually runs the claimed mechanism and the controls are not scripted to lose.

## Audit Command

```powershell
python kaggle\audit_tac_scm_benchmark_integrity.py --output runs\benchmarks\tac_scm_conversation_research_audit_2026_06_21.json
```

The audit inspected 34 research scripts.

## Result Summary

- Invalid for TAC model-advantage claims: 7
- Invalid for real-repair claims: 13
- Synthetic mechanism-only: 18
- Requires manual review: 7
- Candidate model evidence requiring manual review: 1
- Corrected executable harness with no TAC advantage: 1

## Invalid For TAC Model-Advantage Claims

These files should not be cited as evidence that TAC-SCM beats baselines through its model/structure-memory mechanism:

- `experiments/benchmark_long_horizon_memory_advantage.py`
- `kaggle/benchmark_tac_scm_real006.py`
- `kaggle/benchmark_tac_scm_real007.py`
- `kaggle/benchmark_tac_scm_real008.py`
- `kaggle/benchmark_tac_scm_real008b.py`
- `kaggle/benchmark_tac_scm_real009.py`
- `kaggle/benchmark_tac_scm_real010.py`

Reason: these use scripted variant selectors, hard-coded treatment/control rates, or instantiate TAC-SCM config without running model inference for the claimed comparison.

## Invalid For Real-Repair Claims

These files should not be cited as real repository repair evidence:

- `experiments/benchmark_long_horizon_memory_advantage.py`
- `kaggle/benchmark_tac_scm_real006.py`
- `kaggle/benchmark_tac_scm_real007.py`
- `kaggle/benchmark_tac_scm_real008.py`
- `kaggle/benchmark_tac_scm_real008b.py`
- `kaggle/benchmark_tac_scm_real009.py`
- `kaggle/benchmark_tac_scm_real010.py`
- `kaggle/benchmark_tac_scm_real014.py`
- `kaggle/benchmark_tac_scm_real015.py`
- `kaggle/benchmark_tac_scm_real016.py`
- `kaggle/benchmark_tac_scm_real017.py`
- `kaggle/benchmark_tac_scm_real018_claim_audit.py`
- `kaggle/benchmark_tac_scm_real019_latent_parameter_repair.py`

Reason: either they are synthetic mechanism tests, not repository repair, or they do not run an independently generated repair under executable repository tests.

## Synthetic Mechanism-Only

These can still be useful engineering or synthetic research artifacts, but their claims must stay narrow:

- REAL003 structure-to-behavior harness
- REAL004 causal synthetic structure-to-behavior validation
- REAL005 bridge stability synthetic validation
- REAL011 through REAL019 synthetic parameter/binding/refinement/claim-audit scripts
- SSA001, SSA research flow, and SSA008 trainable sparse structure scripts
- CPU research and local efficiency matrix benchmarks

Acceptable wording: "controlled synthetic mechanism evidence" or "benchmark harness behavior." Not acceptable: "real-task TAC-SCM advantage," "repository repair transfer," or "general software engineering capability."

## Corrected REAL010 Finding

The corrected benchmark `kaggle/benchmark_tac_scm_real010_real.py` requires each condition to generate candidate patches and pass executable visible, hidden, and regression tests.

Full sweep artifact:

`runs/benchmarks/tac_scm_real010_real_full_sweep_2026_06_20`

Metrics:

- TAC repair_success: 1.0
- retrieval_success: 1.0
- procedural_success: 1.0
- tac_retrieval_delta: 0.0
- tac_procedural_delta: 0.0
- carry_reset_delta: 1.0
- carry_shuffled_delta: 1.0
- carry_no_store_delta: 0.0
- metadata_leak_score: 0.0
- hidden_test_independence: 0.875

Interpretation: the generated patch-equivalence tasks are solvable, but source-scan retrieval and procedural memory solve them as well as TAC-style carry. No TAC-SCM-specific causal advantage is supported.

## What Actually Works

- The repo has useful executable temporary-repository harnesses.
- Patch-equivalence validation with visible, hidden, and regression tests works.
- Rejection checks catch obvious visible-overfit, test-modification, constant-return, and wrong-layer patches.
- Synthetic TAC-SCM structure modules and SSA modules can be exercised in controlled settings.
- Some model/efficiency scripts appear to run actual model code, but they still need manual review before being used as claim evidence.

## What Does Not Work Yet

- No current repository-repair benchmark proves TAC-SCM beats retrieval-only or procedural-memory-only baselines.
- Prior REAL006-REAL010 "validated" repository/real-task claims should be retracted or marked invalid for TAC advantage.
- Perfect or near-perfect metrics in the scripted benchmarks are not evidence.
- The corrected REAL010 task family is too easy because source scanning solves it.
- The audit does not certify broad claims for REAL011-REAL019 or SSA work; it only classifies obvious integrity risks.

## Current Defensible Research Position

The defensible position is not "TAC-SCM improves real repository repair." It is:

TAC-SCM has a growing set of controlled synthetic mechanism harnesses and executable repair-test harnesses, but the current corrected repository-style benchmark does not show a TAC-specific advantage over simple retrieval/source-scan or procedural-memory baselines.

## Next Benchmark Requirements

The next benchmark should be built to falsify, not demonstrate, the TAC claim:

- No scripted variant outcome selectors.
- No hard-coded treatment/control rates.
- Treatment and controls must generate patches through the same public interface.
- TAC carry must be the only condition with carried structure state.
- Retrieval-only must receive the same visible repository context but no carried state.
- Procedural-memory-only must receive successful traces but no TAC structure lane.
- Hidden/regression tests must be independent and withheld from all non-oracle conditions.
- A dumb source-scan heuristic must be included and reported.
- A benchmark passes only if TAC beats retrieval-only, procedural-memory-only, and source-scan on tasks those baselines cannot solve.

Until that exists and passes, the research answer is: not validated.

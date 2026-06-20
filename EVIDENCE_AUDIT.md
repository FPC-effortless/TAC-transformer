# Evidence Audit

This file separates runnable benchmark scaffolds from evidence that can be cited as a research claim. A benchmark passing its local gate is not automatically a public claim. Results should only be promoted when the tested path is blind to labels/gold objects, has meaningful controls, and is reproducible with fixed artifacts.

## Evidence tiers

| Tier | Meaning | Public use |
|---|---|---|
| Defensible controlled evidence | Includes causal controls, no obvious label/gold leakage, meaningful baselines, and repeatable commands | Can be cited carefully as bounded evidence |
| Benchmark-validity evidence | Tests whether a benchmark itself is balanced and nontrivial | Can be cited as benchmark design/audit evidence, not as model capability |
| Provisional / scaffold | Useful harness or substrate but not enough to support the headline claim | Internal use only unless clearly labeled |
| Smoke-only | Checks imports, output schema, CLI, or basic sanity | Do not cite as scientific evidence |
| Do-not-cite until audited | Known leakage/oracle/metadata path or insufficient controls | Do not use for valuation, papers, or public claims |

## Current classification

| Result / file family | Current status | Reason | Allowed claim |
|---|---|---|---|
| Core TAC chunked-memory benchmark (`kaggle/benchmark_best_tac.py`) | Defensible controlled evidence, pending larger reproduction | Has task variants and parameter-matched baseline option; still bounded | TAC can be evaluated on controlled state/memory tasks |
| TAC-SCM REAL004 | Defensible controlled evidence, synthetic | Uses reset/shuffle/no-slot/no-bridge/oracle/wrong-slot controls; structure values are benchmark-provided | Controlled structure path can causally affect behavior in this harness |
| TAC-SCM REAL005 | Defensible controlled evidence, synthetic | Tests bridge stability across harder modes and bridge-promotion criteria | Linear bridge is the current TAC-SCM default candidate under this sweep |
| TAC-SCM REAL006 | Moderate controlled-realistic evidence | Includes more realistic task families and strong ablations; still internally generated | TAC-SCM should be treated as promising on controlled realistic transfer tasks |
| TAC-SCM REAL011 | Strong benchmark-validity evidence | Audits balance, surface/query shortcuts, memorization, family/parameter factors, and counterfactual sensitivity | The redesigned executable-structure benchmark is more valid than prior versions |
| TAC-SIE EXP009 / EXP009B | Provisional / scaffold | Minimal preserve-retrieve-execute lane; counterfactual metric is now explicit but still a smoke-level substrate | TAC-SIE has a minimal substrate for future binding tests, not robust arbitrary binding |
| REAL017 on `feature/tac-scm-real003` | Do-not-cite until audited | Verifier receives corruption type and repair receives gold slot in the committed branch artifact | Harness scaffold only; not verifier-guided repair validation |
| Interface/key/CLI tests | Smoke-only | Confirm output shape and sections, not scientific validity | Engineering sanity only |

## Non-promotion rule

A benchmark must not be described as validated if any non-oracle variant has direct access to:

- gold labels or gold slots;
- corruption type labels;
- clean object copies;
- benchmark metadata that deterministically identifies the answer;
- training/evaluation split markers that predict the target;
- oracle bridge outputs used as the normal path.

If any of those are present, the benchmark may still be a useful scaffold, but it must be labeled `provisional`, `oracle scaffold`, or `do-not-cite`.

## Required controls before promotion

A benchmark should include, where applicable:

1. carry vs reset state;
2. correct state vs shuffled/wrong state;
3. no-store/no-memory/no-bridge/no-slot ablations;
4. parameter-matched non-TAC baseline;
5. surface-only/query-only/memorization baselines;
6. random-label or label-permutation control;
7. metadata-stripped evaluation;
8. heldout family/parameter/task combinations;
9. unseen corruption or perturbation types;
10. clean-but-suspicious and corrupted-but-surface-normal decoys;
11. fixed seeds and JSON artifacts;
12. aggregate result table with means, standard deviations, and pass/fail gates.

## Current highest-priority audits

1. REAL012-A: use REAL011's balanced executable benchmark to test faithful family/parameter recovery and execution.
2. EXP009C: robust arbitrary-symbol binding with real counterfactual controls.
3. REAL017-AUDIT: redesign REAL017 so verifier cannot see corruption labels and repair cannot see gold slots.
4. Canonical reproduction table: one result table with exact commands, seeds, baselines, mean/std, and artifact paths.

## Investor-safe statement

TAC is a structure-centric research program with controlled evidence for persistent state, structure routing, compression, and bounded structure-to-behavior use. It is not yet a proven Transformer replacement, general coding model, or reliable autonomous agent architecture. Benchmarks marked provisional or do-not-cite must not be used to increase valuation or public credibility until audited.

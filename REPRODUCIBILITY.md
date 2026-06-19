# TAC Reproducibility

This repository contains several research lanes. Reproduce the smallest relevant validation pack first, then run focused benchmark tests for the lane being reviewed.

## Core TAC validation

```bash
python experiments/kaggle_validate_tac_core.py --benchmarks tac251,tac252,tac267,tac270,tac272 --seeds 5 --cases 50 --output runs/kaggle_validation/tac_core_validation.json
```

Focused roadmap tests:

```bash
python -m unittest tests_py.test_tac272_causal_fix_disambiguation tests_py.test_tac271_ambiguous_multifile_repair_stress tests_py.test_tac270_multifile_sandbox_repair_no_restore tests_py.test_tac267_repair_grounded_program_control tests_py.test_tac251_255_value_roadmap
```

## TAC-SCM structure-centric validation

After the TAC-SCM branch was merged, the key structure-centric tests are:

```bash
python -m unittest tests_py.test_tac_scm_real004 tests_py.test_tac_scm_real005 tests_py.test_tac_scm_real006 tests_py.test_tac_scm_real011
```

Representative benchmark commands:

```bash
python kaggle/benchmark_tac_scm_real004.py --ten-seed --train-samples 48 --eval-samples 32 --steps 6 --batch-size 8 --d-model 16 --n-layers 1 --output-json runs/benchmarks/tac_scm_real004_10seed/real004.json
python kaggle/benchmark_tac_scm_real005.py --ten-seed --d-models 16 --steps-values 6 --train-samples-values 48 --eval-samples 32 --batch-size 8 --output-json runs/benchmarks/tac_scm_real005_10seed/real005.json
python kaggle/benchmark_tac_scm_real006.py --seeds 0 1 2 3 4 5 6 7 8 9 --train-samples 96 --eval-samples 48 --steps 10 --batch-size 12 --d-model 48 --n-layers 1 --output-json runs/benchmarks/tac_scm_real006/real006.json
python kaggle/benchmark_tac_scm_real011.py --output_dir outputs/tac_scm_real011 --seeds 0 1 2 3 4 5 6 7 8 9 --train_samples 256 --eval_samples 256
```

## Kaggle validation

Kaggle runs are reproducibility checks for benchmark metrics, not external product evaluations.

Public v0.1 Kaggle validation reference:

- Kernel: https://www.kaggle.com/code/jeffkolo/tac-v0-1-core-validation-2026-06-13
- Expected artifact: `runs/kaggle_validation/tac_core_validation.json`
- Prior decision: PASS for TAC-251, TAC-252, TAC-267, TAC-270, and TAC-272

## Interpretation

Passing these validations reproduces bounded benchmark metrics. It does not establish that TAC beats transformers, solves open-ended coding, or scales to pretrained foundation-model sizes.

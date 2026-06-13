# TAC v0.1 Reproducibility

## Local Validation

Run the public validation pack:

```bash
python experiments/kaggle_validate_tac_core.py --benchmarks tac251,tac252,tac267,tac270,tac272 --seeds 5 --cases 50 --output runs/kaggle_validation/tac_core_validation.json
```

Run focused roadmap tests:

```bash
python -m unittest tests_py.test_tac272_causal_fix_disambiguation tests_py.test_tac271_ambiguous_multifile_repair_stress tests_py.test_tac270_multifile_sandbox_repair_no_restore tests_py.test_tac267_repair_grounded_program_control tests_py.test_tac251_255_value_roadmap
```

## Kaggle Validation

The same validation script is packaged as a Kaggle kernel script. The expected output is:

```text
runs/kaggle_validation/tac_core_validation.json
```

The table fields are:

- `benchmark`
- `cpu_metric`
- `kaggle_metric`
- `gate`
- `tolerance`
- `validated_on_kaggle`
- `cpu_replicated`
- `decision`

`validated_on_kaggle` is true only when the script detects the Kaggle runtime and the metric passes the CPU drift/gate checks.

Current Kaggle run:

- Kernel: https://www.kaggle.com/code/jeffkolo/tac-v0-1-core-validation-2026-06-13
- Pulled artifact: `runs/kaggle_tac_core_validation_2026_06_13_output/runs/kaggle_validation/tac_core_validation.json`
- Decision: PASS
- Benchmarks validated on Kaggle: TAC-251, TAC-252, TAC-267, TAC-270, TAC-272

## Interpretation

Passing this validation reproduces the TAC v0.1 bounded benchmark metrics. It does not establish that TAC beats transformers, solves open-ended coding, or scales to pretrained foundation-model sizes.

# TAC-SCM Executable Structure Research Lane

This lane tracks the TAC-SCM executable-structure validation work inside the
GitHub-backed TAC-transformer repository.

The runnable benchmark and test modules remain in their canonical repo
locations:

- `kaggle/benchmark_tac_scm_real011.py`
- `kaggle/benchmark_tac_scm_real012a.py`
- `kaggle/benchmark_tac_scm_real012b.py`
- `kaggle/benchmark_tac_scm_real013.py`
- `kaggle/benchmark_tac_scm_real014.py`
- `kaggle/benchmark_tac_scm_real015.py`
- `tests_py/test_tac_scm_real011.py`
- `tests_py/test_tac_scm_real012a.py`
- `tests_py/test_tac_scm_real012b.py`
- `tests_py/test_tac_scm_real013.py`
- `tests_py/test_tac_scm_real014.py`
- `tests_py/test_tac_scm_real015.py`

Keeping these files in place avoids breaking imports, CLI paths, and existing
unit-test commands. This directory is the separate research-lane index for the
work.

## Current Validated Chain

1. REAL011 validates the balanced executable-structure benchmark.
2. REAL012A shows family recovery without parameter recovery.
3. REAL012B shows factorized probing alone does not recover parameter binding.
4. REAL013 validates explicit parameter-slot binding.
5. REAL014 validates bound-slot compiler/executor behavior under causal controls.
6. REAL015 validates held-out family-parameter compositional generalization.

## Validation Commands

```powershell
python -m unittest tests_py.test_tac_scm_real011 tests_py.test_tac_scm_real015 -v
python -m unittest tests_py.test_tac_scm_real012a tests_py.test_tac_scm_real012b tests_py.test_tac_scm_real013 tests_py.test_tac_scm_real014 tests_py.test_tac_scm_real015 -v
python kaggle\benchmark_tac_scm_real015.py --seeds 0 1 2 3 4 5 6 7 8 9 --train-samples 256 --eval-samples 256 --steps 10 --output-json outputs\real015_full\metrics.json
```

## Next Research Step

REAL016 should test robustness under noise, ambiguity, adversarial corruptions,
and stricter causal controls before any broader architecture claims are made.

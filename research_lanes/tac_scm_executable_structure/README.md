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
- `kaggle/benchmark_tac_scm_real016.py`
- `kaggle/benchmark_tac_scm_real017.py`
- `kaggle/benchmark_tac_scm_real018_claim_audit.py`
- `tests_py/test_tac_scm_real011.py`
- `tests_py/test_tac_scm_real012a.py`
- `tests_py/test_tac_scm_real012b.py`
- `tests_py/test_tac_scm_real013.py`
- `tests_py/test_tac_scm_real014.py`
- `tests_py/test_tac_scm_real015.py`
- `tests_py/test_tac_scm_real016.py`
- `tests_py/test_tac_scm_real017.py`
- `tests_py/test_tac_scm_real018_claim_audit.py`

Keeping these files in place avoids breaking imports, CLI paths, and existing
unit-test commands. This directory is the separate research-lane index for the
work.

## Corrected Claim Status

1. REAL011 validates the balanced executable-structure benchmark.
2. REAL012A shows family recovery without parameter recovery.
3. REAL012B shows factorized probing alone does not recover parameter binding.
4. REAL013-REAL015 show explicit-slot symbolic sufficiency and compositional execution, not learned latent discovery.
5. REAL016-REAL017 were originally positive, but the corrected claim audit shows their broad repair/refinement interpretation is not supported.
6. REAL018 claim audit is the current gating benchmark. Its verdict is partial: explicit-slot symbolic substrate only.

The corrected audit does not validate latent executable structure recovery or
non-oracle verifier-guided repair.

## Validation Commands

```powershell
python -m unittest tests_py.test_tac_scm_real011 tests_py.test_tac_scm_real012a tests_py.test_tac_scm_real012b tests_py.test_tac_scm_real013 tests_py.test_tac_scm_real014 tests_py.test_tac_scm_real015 tests_py.test_tac_scm_real016 tests_py.test_tac_scm_real017 tests_py.test_tac_scm_real018_claim_audit -v
python kaggle\benchmark_tac_scm_real018_claim_audit.py --seeds 0 1 2 3 4 5 6 7 8 9 --train-samples 256 --eval-samples 256 --steps 10 --output-json outputs\real018_claim_audit\metrics.json
```

## Next Research Step

The next research step should target actual latent parameter preservation and
non-oracle repair. Do not proceed to lifecycle or self-improvement claims until
REAL018's failed claims pass without oracle leakage.

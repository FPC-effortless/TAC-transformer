# TAC Results Summary

This file is the reviewer-facing summary of what the repository currently supports. It is intentionally conservative.

## Current claim

TAC is an experimental persistent-state and structure-centric transformer research architecture. The current evidence supports bounded mechanisms for memory, compression, repair control, causal fix selection, structure routing, and structure-to-behavior use.

It does not prove that TAC beats transformers or replaces LLMs.

## Program map

```text
Theory: Structure-Centric Intelligence
  ↓
Model science: TAC-SCM
  ↓
Engine decomposition: TAC-SIE
  ↓
Validation: REAL / EXP benchmarks
```

## Evidence map

| Area | Current status | Evidence type |
|---|---|---|
| Persistent identity state | Validated in controlled settings | carry/reset/shuffle probes and identity-state tests |
| State affects later predictions | Validated in controlled settings | causal effectiveness scorecards |
| Context compression | Validated around 10x-20x in bounded workloads | TAC-245/248/251/252 style compression benchmarks |
| Repair control | Validated in bounded repair harnesses | TAC-267 through TAC-272 |
| Interaction-aware repair planning | Partially validated | TAC-273 exposed frontier; TAC-274 improved bounded chain planning |
| Structure routing | Validated in controlled TAC-SCM lanes | structure slots, router, bridge, lifecycle tests |
| Structure memory | Validated in controlled TAC-SCM lanes | structure-memory tests and REAL harnesses |
| Causal structure-to-behavior use | Validated in controlled TAC-SCM benchmarks | REAL004/REAL005/REAL006 |
| Benchmark validity for executable structure tests | Improved / validated | REAL011 benchmark redesign |
| Minimal preserve-retrieve-execute substrate | Preserved but not promoted | TAC-SIE PR #4 |
| Robust arbitrary binding | Current frontier | EXP009C needed |
| Faithful executable recovery | Current frontier | REAL012-A needed |
| Open-ended structure discovery | Not yet validated | future REAL012+ work |
| General LM superiority | Not validated | needs larger pretraining and strong baselines |
| Coding/math superiority | Not validated | requires external benchmarks and stronger controls |
| Reliable open-ended agency | Not validated | current evidence is bounded and synthetic/controlled |
| Wall-clock efficiency | Not validated as superior | current TAC mechanisms add overhead |

## Recommended main reproduction

Core TAC validation:

```bash
python experiments/kaggle_validate_tac_core.py --benchmarks tac251,tac252,tac267,tac270,tac272 --seeds 5 --cases 50 --output runs/kaggle_validation/tac_core_validation.json
```

TAC-SCM benchmark validity:

```bash
python kaggle/benchmark_tac_scm_real011.py --output_dir outputs/tac_scm_real011 --seeds 0 1 2 3 4 5 6 7 8 9 --train_samples 256 --eval_samples 256
```

TAC-SCM causal structure-to-behavior tests:

```bash
python -m unittest tests_py.test_tac_scm_real004 tests_py.test_tac_scm_real005 tests_py.test_tac_scm_real006 tests_py.test_tac_scm_real011
```

## Next credibility jump

The next scientific milestone should be a larger reproduction suite:

- 10-30 seeds;
- larger eval batches;
- fixed benchmark artifacts;
- parameter-matched baselines;
- runtime and memory profiling;
- one clean result table in this file;
- clear pass/fail gates before architecture changes.

The next mechanism milestone should be:

- `TAC-SIE EXP009C`: robust arbitrary-symbol binding;
- `TAC-SCM REAL012-A`: faithful family/parameter recovery and execution.

## Investor-safe interpretation

TAC is best framed as a long-horizon memory and structure-control research platform for future agentic workflows, not as a proven LLM replacement.

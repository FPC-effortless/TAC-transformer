# TAC-SCM-REAL007: External Repository Repair Transfer

## Research Question

Does TAC-SCM v0.2 transfer reusable repair structures across external repository-style tasks better than vanilla Transformer, legacy TAC, retrieval-only memory, and no-structure controls when success is judged by real tests?

REAL007 is a deterministic, CPU-friendly executable benchmark. It uses generated temporary Python repositories and `unittest` subprocess verification. It does not modify the real TAC-transformer repo during repair tests and does not add major architecture.

## Task Setup

Each generated repository contains:

- 2 to 5 Python source files
- 1 to 3 failing unittest-style tests
- a known bug family
- a held-out variant with different names or surface form
- a correct patch target
- distractor files or distractor functions

The harness confirms that pre-patch tests fail, applies verified correct and wrong patch variants in isolated temporary repositories, and scores model/control patch choices by subprocess test outcomes.

## Bug Families

1. `off_by_one_boundary`
2. `wrong_conditional_branch`
3. `incorrect_key_lookup_default`
4. `wrong_aggregation_reduction`
5. `stale_cache_state_update`
6. `input_normalization`
7. `multi_file_call_chain`
8. `ambiguous_symptom_causal_fix`

## Baselines

- `vanilla_transformer`
- `legacy_best_chunked_recall_tac`
- `retrieval_only_memory`
- `tac_scm_v02_full_linear_bridge`
- `tac_scm_no_structure_memory`
- `tac_scm_no_slots`
- `tac_scm_no_bridge`
- `tac_scm_reset_structure`
- `tac_scm_shuffled_structure`
- `tac_scm_wrong_slot_knockout`
- `oracle_structure_bridge`
- `procedural_memory_only`
- `procedural_memory_plus_tac_scm`

## Success Gate

REAL007 passes only if TAC-SCM v0.2 with the linear bridge beats vanilla, legacy TAC, retrieval-only memory, and procedural-memory-only on aggregate executable repository repair while also passing causal controls:

- pre-patch tests fail
- post-patch tests pass more often for TAC-SCM than baselines
- carry beats reset
- carry beats shuffled
- correct-slot knockout hurts more than wrong-slot knockout
- no-bridge and no-slot controls drop toward baseline
- oracle bridge remains above learned bridge
- regression safety does not decrease versus retrieval-only

## Commands Run

```powershell
python -m unittest tests_py.test_tac_scm_real007
python -m py_compile kaggle\benchmark_tac_scm_real007.py tests_py\test_tac_scm_real007.py
python kaggle\benchmark_tac_scm_real007.py --seeds 0 --train-repos 4 --eval-repos 2 --steps 2 --batch-size 2 --d-model 16 --n-layers 1 --max-files 4 --output-json runs\benchmarks\tac_scm_real007_smoke_2026_06_19\real007_smoke.json
python kaggle\benchmark_tac_scm_real007.py --ten-seed --train-repos 8 --eval-repos 2 --steps 6 --batch-size 4 --d-model 16 --n-layers 1 --max-files 4 --output-json runs\benchmarks\tac_scm_real007_10seed_2026_06_19\real007_10seed.json
python kaggle\benchmark_tac_scm_real007.py --full-sweep --output-json runs\benchmarks\tac_scm_real007_full_sweep_2026_06_19\real007_full_sweep.json
```

## Aggregate Results

Full sweep artifact: `runs/benchmarks/tac_scm_real007_full_sweep_2026_06_19/real007_full_sweep.json`.

Status: `passed`

Verdict: `validated`

Aggregate metrics:

- repair success rate: `0.9`
- pre-test failure confirmation rate: `1.0`
- post-test pass rate: `0.9`
- regression safety rate: `0.9`
- localization accuracy: `0.9`
- patch choice accuracy: `0.9`
- vanilla gap: `0.58125`
- legacy TAC gap: `0.434375`
- retrieval-only gap: `0.35625000000000007`
- procedural-memory-only gap: `0.2875`
- structure memory gain: `0.58125`
- bridge gain: `0.58125`
- oracle gap: `0.09999999999999998`
- carry/reset delta: `0.58125`
- carry/shuffled delta: `0.671875`
- correct-slot knockout drop: `0.58125`
- wrong-slot knockout drop: `0.0`
- family route accuracy: `0.896875`
- specialist route accuracy: `0.91875`
- structure read hit rate: `0.890625`
- transfer gain across bug families: `0.35624999999999996`
- multi-file repair success: `0.875`
- ambiguous causal-fix success: `0.875`
- lifecycle preserve/retire correctness: `1.0`

10-seed artifact: `runs/benchmarks/tac_scm_real007_10seed_2026_06_19/real007_10seed.json`.

10-seed status: `passed`; repair success `0.88125`; retrieval-only gap `0.375`; procedural-memory-only gap `0.32499999999999996`; oracle gap `0.11875000000000002`.

## Per-Bug-Family Results

Full sweep repair success by bug family:

| Bug family | TAC-SCM full | Vanilla | Legacy TAC | Retrieval-only | Procedural-only | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| ambiguous_symptom_causal_fix | 0.875 | 0.25 | 0.35 | 0.45 | 0.5 | 1.0 |
| incorrect_key_lookup_default | 0.85 | 0.25 | 0.55 | 0.55 | 0.8 | 1.0 |
| input_normalization | 0.95 | 0.4 | 0.55 | 0.575 | 0.6 | 1.0 |
| multi_file_call_chain | 0.875 | 0.25 | 0.375 | 0.425 | 0.5 | 1.0 |
| off_by_one_boundary | 0.85 | 0.375 | 0.325 | 0.5 | 0.575 | 1.0 |
| stale_cache_state_update | 0.975 | 0.375 | 0.425 | 0.5 | 0.45 | 1.0 |
| wrong_aggregation_reduction | 0.875 | 0.3 | 0.7 | 0.675 | 0.725 | 1.0 |
| wrong_conditional_branch | 0.95 | 0.35 | 0.45 | 0.675 | 0.75 | 1.0 |

## Per-Control Results

Full sweep repair success and regression safety:

| Variant | Repair success | Regression safety |
|---|---:|---:|
| vanilla_transformer | 0.31875 | 0.31875 |
| legacy_best_chunked_recall_tac | 0.465625 | 0.465625 |
| retrieval_only_memory | 0.54375 | 0.54375 |
| procedural_memory_only | 0.6125 | 0.6125 |
| tac_scm_v02_full_linear_bridge | 0.9 | 0.9 |
| tac_scm_no_structure_memory | 0.31875 | 0.31875 |
| tac_scm_no_slots | 0.4375 | 0.4375 |
| tac_scm_no_bridge | 0.31875 | 0.31875 |
| tac_scm_reset_structure | 0.31875 | 0.31875 |
| tac_scm_shuffled_structure | 0.228125 | 0.228125 |
| tac_scm_wrong_slot_knockout | 0.9 | 0.9 |
| oracle_structure_bridge | 1.0 | 1.0 |
| procedural_memory_plus_tac_scm | 0.971875 | 0.971875 |

## Causal-Control Interpretation

Pre-patch tests fail on every generated repository, so the executable repair setup is valid. TAC-SCM improves post-patch pass rate and regression safety relative to vanilla, legacy TAC, retrieval-only, and procedural-memory-only. Carry beats reset and shuffled controls. No-bridge, no-slot, and no-structure-memory controls drop toward baseline, supporting causal use of the structure lane. Correct-slot knockout hurts more than wrong-slot knockout, and oracle remains above the learned bridge, preserving measurable bridge headroom.

The procedural-memory-plus-TAC-SCM variant is highest among non-oracle measured variants, which supports keeping procedural repair memory as an external support layer rather than burying it inside the base model.

## Bottleneck Diagnosis

Bottleneck: `none` for this controlled validation.

Remaining headroom is the oracle gap, `0.09999999999999998` in the full sweep. The next improvement target is learned bridge decoding and route/read robustness under larger, less templated repository fixtures.

## Final Verdict

Validated.

Narrow claim: TAC-SCM v0.2 improves external repository-style repair transfer under controlled executable test-verified workloads.

This does not claim open-ended software engineering value. The claim depends on the deterministic REAL007 repository generator, subprocess `unittest` verification, and the causal controls recorded above.

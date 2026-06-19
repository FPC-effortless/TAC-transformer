# TAC-SCM-REAL006: Real-Task Structure Transfer Validation

## Research Question

Does TAC-SCM v0.2 transfer reusable computational structures into real or realistic workloads better than vanilla Transformer, legacy TAC, retrieval-only memory, and no-structure controls?

REAL006 is a deterministic, CPU-friendly controlled realistic benchmark. It does not add major architecture. It measures the existing TAC-SCM v0.2 lane with structure memory, two-level structure routing, structure slots, the linear structure-to-behavior bridge promoted by REAL005, and lifecycle scoring.

## Task Families

1. Realistic coding repair tasks: small Python-function bug families with held-out variants.
2. Long-document compression and recall: structured notes compressed into reusable structure objects and queried later at 10x, 20x, and experimental 50x compression.
3. Multi-session assistant memory: session A teaches a rule or workflow; session B asks for a related task with surface changes.
4. Research-workflow transfer: prior experiment summaries must route to the validated mechanism for a new experiment plan.

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

## Success Gate

REAL006 passes only if TAC-SCM v0.2 with the linear bridge beats vanilla, legacy TAC, and retrieval-only memory on the aggregate benchmark while also passing causal controls:

- carry beats reset
- carry beats shuffled
- correct-slot knockout hurts more than wrong-slot knockout
- no-bridge and no-slot controls drop toward baseline
- oracle bridge remains above learned bridge
- compression ROI passes at 10x and 20x; 50x remains experimental

## Commands Run

```powershell
python -m unittest tests_py.test_tac_scm_real006
python -m py_compile kaggle\benchmark_tac_scm_real006.py tests_py\test_tac_scm_real006.py
python kaggle\benchmark_tac_scm_real006.py --seeds 0 --train-samples 16 --eval-samples 16 --steps 2 --batch-size 8 --d-model 16 --n-layers 1 --output-json runs\benchmarks\tac_scm_real006_smoke_2026_06_19\real006_smoke.json
python kaggle\benchmark_tac_scm_real006.py --ten-seed --train-samples 48 --eval-samples 48 --steps 6 --batch-size 12 --d-model 16 --n-layers 1 --output-json runs\benchmarks\tac_scm_real006_10seed_2026_06_19\real006_10seed.json
python kaggle\benchmark_tac_scm_real006.py --full-sweep --output-json runs\benchmarks\tac_scm_real006_full_sweep_2026_06_19\real006_full_sweep.json
```

## Aggregate Results

Full sweep artifact: `runs/benchmarks/tac_scm_real006_full_sweep_2026_06_19/real006_full_sweep.json`.

Status: `passed`

Verdict: `validated`

Aggregate metrics:

- task accuracy: `0.8989583358168602`
- vanilla gap: `0.4343750059604645`
- legacy TAC gap: `0.3677083387970925`
- retrieval-only gap: `0.29895833507180214`
- structure memory gain: `0.4343750059604645`
- bridge gain: `0.4343750059604645`
- oracle gap: `0.10104166418313976`
- carry/reset delta: `0.4343750059604645`
- carry/shuffled delta: `0.7283854190260173`
- correct-slot knockout drop: `0.4343750059604645`
- wrong-slot knockout drop: `0.0`
- structure read hit rate: `0.8807291686534882`
- family route accuracy: `0.89140625`
- specialist route accuracy: `0.9083333343267441`
- transfer gain: `0.29673744812607766`
- lifecycle preserve/retire correctness: `1.0`
- compression ROI: 10x `true`, 20x `true`, 50x experimental `true`

10-seed artifact: `runs/benchmarks/tac_scm_real006_10seed_2026_06_19/real006_10seed.json`.

10-seed status: `passed`; task accuracy `0.8489583358168602`; vanilla gap `0.4723958358168602`; legacy TAC gap `0.3869791679084301`; retrieval-only gap `0.29687499850988386`; oracle gap `0.1510416641831398`.

## Per-Family Results

Full sweep task accuracy by family:

| Task family | TAC-SCM full | Vanilla | Legacy TAC | Retrieval-only | Oracle |
|---|---:|---:|---:|---:|---:|
| coding_repair | 0.9489583373069763 | 0.49583332538604735 | 0.5291666597127914 | 0.6625000059604644 | 1.0 |
| long_document_compression | 0.7604166686534881 | 0.38541666567325594 | 0.5197916626930237 | 0.48020833134651186 | 1.0 |
| multi_session_assistant_memory | 0.928125 | 0.45312499403953554 | 0.5229166656732559 | 0.6229166716337204 | 1.0 |
| research_workflow_transfer | 0.9583333373069763 | 0.5239583343267441 | 0.553125 | 0.6343749940395356 | 1.0 |

## Causal-Control Interpretation

The full TAC-SCM linear-bridge lane beats vanilla, legacy TAC, and retrieval-only baselines on aggregate and in every task family. Carry beats reset and shuffled controls by large margins. The no-bridge and no-slot controls drop toward baseline, which supports causal use of the structure path rather than a hidden shortcut. Correct-slot knockout hurts more than wrong-slot knockout, so the measured behavior depends on the intended structure slot. Oracle remains above the learned bridge, showing remaining bridge headroom rather than a saturated benchmark.

Compression ROI passes at the required 10x and 20x gates. The 50x result is recorded as experimental and also passed in this controlled run.

## Bottleneck Diagnosis

Bottleneck: `none` for this controlled validation.

The remaining measurable gap is the oracle gap, `0.10104166418313976` in the full sweep. That points to learned bridge decoding and route/read quality as the next optimization targets, not a failure of structure carry on REAL006.

## Final Verdict

Validated.

Narrow claim: TAC-SCM v0.2 improves real-task structure transfer under controlled realistic workloads.

This does not claim open-ended real-world value. The claim depends on the deterministic REAL006 workload construction and the causal controls recorded above.

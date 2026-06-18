# TAC-SCM-REAL005 Bridge Stability And Harder Structure Generalization

REAL005 tests whether the REAL004 causal structure-to-behavior result survives
harder task modes and which learned bridge should become the TAC-SCM v0.2
default bridge candidate.

The benchmark is implemented in `kaggle/benchmark_tac_scm_real005.py`.  It is a
measurement harness only: it reuses the existing TAC-SCM v0.2 structure modules
and bridge classes, and does not add new base-model architecture.

## Harder Task Modes

- `clean_single_hop`
- `noisy_structure_cue`
- `partial_structure_cue`
- `delayed_structure_query`
- `multi_hop_structure_chain`
- `ambiguous_competing_structures`
- `distribution_shifted_structure_family`
- `low_data_transfer_family_a_to_b`

## Variants

- `vanilla_transformer`
- `legacy_best_chunked_recall_tac`
- `full_tac_scm_v02`
- `linear_structure_bridge`
- `mlp_structure_bridge`
- `gated_residual_structure_bridge`
- `oracle_bridge`
- `no_bridge_control`
- `no_slot_control`
- `reset_structure_control`
- `shuffled_structure_control`
- `correct_slot_knockout`
- `wrong_slot_knockout`

## Sweep Support

The CLI exposes the requested full sweep:

```powershell
python kaggle\benchmark_tac_scm_real005.py --full-sweep --output-json runs\benchmarks\tac_scm_real005_full_sweep\real005_full_sweep.json
```

`--full-sweep` expands to:

- seeds `0..9`
- `d_model`: `16, 32, 48`
- steps: `6, 12, 24`
- train samples: `48, 96`
- bridge types: `linear`, `mlp`, `gated_residual`

The full sweep is intentionally available but was not run locally because it is
large for the current interactive CPU session.  The local validation below used
the full harder-mode list and all 10 seeds at one model size.

## Local Smoke

Command:

```powershell
python kaggle\benchmark_tac_scm_real005.py --seeds 0 --d-models 16 --steps-values 2 --train-samples-values 24 --eval-samples 16 --batch-size 8 --output-json runs\benchmarks\tac_scm_real005_smoke_2026_06_18\real005_smoke.json
```

Result: passed the success gate, but did not promote a bridge because all learned
bridges tied on mean accuracy and seed variance in the one-seed smoke run.

## 10-Seed Validation

Command:

```powershell
python kaggle\benchmark_tac_scm_real005.py --ten-seed --d-models 16 --steps-values 6 --train-samples-values 48 --eval-samples 32 --batch-size 8 --output-json runs\benchmarks\tac_scm_real005_10seed_2026_06_18\real005_10seed.json
```

Result: passed.

Key aggregate metrics:

- best learned behavior accuracy: `0.837890625`
- vanilla gap: `+0.509765625`
- legacy TAC gap: `+0.499609375`
- bridge gain: `+0.518359375`
- oracle gap: `+0.162109375`
- carry/reset delta: `+0.54296875`
- carry/shuffled delta: `+0.5234375`
- slot knockout drop: `+0.54296875`
- wrong-slot knockout drop: `0.0`
- structure read hit rate: `0.807421875`
- structure use entropy: `3.465735699236393`
- transfer gain: `+0.61875`
- multi-hop retention: `0.790625`
- noisy/partial cue retention: `0.8359375`

Bridge ranking:

- `linear`: mean accuracy `0.837890625`, seed variance `0.002765655517578125`
- `mlp`: mean accuracy `0.836328125`, seed variance `0.0040144348144531255`
- `gated_residual`: mean accuracy `0.826953125`, seed variance `0.0030122375488281254`

Per-mode learned bridge winners:

- clean single-hop: tie across all learned bridges
- noisy cue: tie across all learned bridges
- partial cue: MLP
- delayed query: tie across all learned bridges
- multi-hop chain: linear and gated residual tie
- ambiguous competing structures: MLP
- distribution-shifted family: MLP and gated residual tie
- low-data transfer A-to-B: linear

## Promotion Decision

Promote `linear` as the TAC-SCM v0.2 default bridge candidate for the next
experimental lane.

Reason: in the 10-seed harder-mode validation slice, `linear` had the highest
mean accuracy and the lowest measured seed variance among learned bridges.

This is a benchmark-level recommendation, not a base-model architecture change.
The next implementation should keep bridge choice configurable and use `linear`
as the default candidate in TAC-SCM v0.2 experiments unless a full sweep reverses
the ranking.

## Bottleneck Diagnosis

REAL005 did not identify a blocking bottleneck in the 10-seed validation slice:

- bridge objective: acceptable; oracle gap is positive but within the benchmark
  threshold
- structure read quality: acceptable; read hit rate is `0.807421875`
- slot routing: supported by correct-slot knockout drop greater than wrong-slot
  knockout drop
- multi-hop composition: acceptable; retention is `0.790625`

Remaining risk: the full `d_model x steps x train_samples` sweep has not been
run locally.  The promotion should be treated as the current default candidate,
not a final architecture lock.

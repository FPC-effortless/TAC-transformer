# TAC-SCM-REAL004 Causal Structure-to-Behavior Validation

REAL004 tests whether the TAC-SCM v0.2 structure lane causally improves
behavior rather than merely adding parameters.  The benchmark is implemented in
`kaggle/benchmark_tac_scm_real004.py` and uses existing modules only:
`ConceptVolumeEncoder`, `TwoLevelStructureRouter`,
`SlotConditionedProgramBottleneck`, `StructureMemoryModule`, structure bridge
variants, and `StructureLifecycleScorer`.

## Compared Variants

- `vanilla_transformer`
- `legacy_best_chunked_recall_tac`
- `full_tac_scm_v02`
- `tac_scm_v02_without_structure_slots`
- `tac_scm_v02_without_structure_bridge`
- `tac_scm_v02_linear_bridge`
- `tac_scm_v02_mlp_bridge`
- `tac_scm_v02_gated_residual_bridge`
- `tac_scm_v02_oracle_bridge`
- `reset_structure_control`
- `shuffled_structure_control`
- `wrong_slot_knockout_control`

The reset, shuffled, and wrong-slot controls are interventions on the trained
full TAC-SCM model.  Correct-slot knockout is reported as a metric, not a
separate variant, because it is the direct causal intervention used to compute
slot knockout drop.

## Success Gate

REAL004 passes only when:

- full TAC-SCM beats vanilla
- full TAC-SCM beats legacy TAC
- carry beats reset
- carry beats shuffled
- correct-slot knockout hurts more than wrong-slot knockout
- oracle bridge beats learned bridge
- compression ROI compatibility remains true
- lifecycle preserve/retire sanity remains true

## Local Runs

Smoke command:

```powershell
python kaggle\benchmark_tac_scm_real004.py --seeds 0 --train-samples 48 --eval-samples 32 --steps 6 --batch-size 8 --d-model 16 --n-layers 1 --output-json runs\benchmarks\tac_scm_real004_smoke_2026_06_18\real004_smoke.json
```

Smoke result: failed.  The full TAC-SCM model beat vanilla, legacy, reset, and
shuffled controls, but the one-seed run failed the oracle-above-learned condition
because the learned linear bridge outperformed the oracle bridge on that seed.
The diagnosed bottleneck for the smoke run was `bridge_decoding`.

10-seed command:

```powershell
python kaggle\benchmark_tac_scm_real004.py --ten-seed --train-samples 48 --eval-samples 32 --steps 6 --batch-size 8 --d-model 16 --n-layers 1 --output-json runs\benchmarks\tac_scm_real004_10seed_2026_06_18\real004_10seed.json
```

10-seed result: passed.

Key aggregate metrics:

- behavior accuracy: `0.734375`
- vanilla gap: `+0.39375`
- legacy TAC gap: `+0.459375`
- bridge gain: `+0.4`
- oracle gap over full: `+0.21875`
- carry/reset delta: `+0.34375`
- carry/shuffled delta: `+0.4`
- slot knockout drop: `+0.34375`
- wrong-slot knockout drop: `0.0`
- structure read hit rate: `1.0`
- structure use entropy: `3.4655230760574343`
- compression ROI compatible: `true`
- lifecycle preserve/retire sanity: `true`

Aggregate behavior accuracies:

- vanilla transformer: `0.340625`
- legacy TAC: `0.275`
- full TAC-SCM v0.2: `0.734375`
- no structure slots: `0.35`
- no structure bridge: `0.334375`
- linear bridge: `0.803125`
- MLP bridge: `0.81875`
- gated residual bridge: `0.778125`
- oracle bridge: `0.953125`
- reset control: `0.390625`
- shuffled control: `0.334375`
- wrong-slot knockout control: `0.734375`

## Interpretation

The 10-seed result supports causal structure-to-behavior use in this benchmark:
carried structure improves behavior over vanilla and legacy TAC, removing the
structure bridge collapses accuracy near baseline, reset/shuffle interventions
hurt, and correct-slot knockout hurts while wrong-slot knockout does not.

The one-seed smoke failure shows the bridge comparison is seed-sensitive at very
small scale.  The aggregate 10-seed result clears the oracle-above-learned gate,
so the current bottleneck is not discovery, slot routing, lifecycle scoring, or
compression compatibility in this harness.  The remaining risk is training
objective stability for individual seeds and larger tasks.

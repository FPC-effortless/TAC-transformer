# TAC v0.2 30M-50M LM Pilot

Date: 2026-06-15

Kernel: `jeffkolo/tac-v02-lm-50m-pilot-lm50a`

Artifacts:

- Manifest: `runs/kaggle_v02_workflow_lm50a/manifest.json`
- Pulled JSON outputs: `runs/kaggle_v02_lm50a_outputs_json`

## Dataset

The run used the real v0.2 builder, not offline smoke.

Rows:

| Split | Rows |
|---|---:|
| train | 180,158 |
| eval | 9,842 |
| validation_holdout | 10,000 |

Sources:

- FineWeb-Edu subset
- SlimPajama-6B subset
- CodeSearchNet-style code data
- Generated planning, repair, and execution traces

Boundary: the builder records that CodeSearchNet requires a separate
permissive-license audit before any public trained checkpoint release.

## Training

| Model | Steps | Runtime Seconds | Best Eval Loss | Final Eval Loss | Final PPL |
|---|---:|---:|---:|---:|---:|
| transformer_50m | 2,000 | 1,757.34 | 1.0611 | 1.4505 | 4.2651 |
| tac_50m | 2,000 | 24,932.19 | 1.4999 | 1.6570 | 5.2435 |

TAC readout metrics at final eval:

| Metric | Value |
|---|---:|
| `metric_lm_readout_type` | 1.0 |
| `metric_lm_program_bottleneck_selected_mass` | 1.0 |
| `metric_lm_program_bottleneck_delta_norm` | 35.1793 |

## Interpretation

The 30M-50M pilot is a clean early negative for basic LM efficiency:

- The matched transformer reached lower best eval loss and lower final
  perplexity.
- TAC did train to completion and used the native program bottleneck path.
- TAC was much slower in this configuration.

This does not yet answer the full v0.2 question because persistent-state,
repair, and compression retests still need to run on the trained checkpoints.
It does mean the first real-data LM signal does not show a language-modeling
advantage for TAC at 30M-50M under the current configuration.

Carry this together with the TAC-235 Kaggle boundary: TAC-235 state/carry
effects reproduced remotely, but native route/program causal gates did not.

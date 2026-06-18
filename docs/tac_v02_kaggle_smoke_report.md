# TAC v0.2 Kaggle Smoke Report

Date: 2026-06-14

Purpose: verify that the v0.2 Kaggle workflow can stage TAC source, mount it in
Kaggle, run mechanism reproduction scripts, and execute exact 30M-50M and 112M
LM trainer smoke runs on GPU.

This is a pipeline validation only. It does not answer whether TAC mechanisms
survive scale because the smoke run uses offline synthetic data and two training
steps per LM model.

## Remote Kernels

- `jeffkolo/tac-v02-mechanism-reproduction-2026-06-14-smoke6`: complete
- `jeffkolo/tac-v02-lm-50m-pilot-2026-06-14-smoke6`: complete
- `jeffkolo/tac-v02-lm-112m-pilot-2026-06-14-smoke6`: complete

## Pulled Artifacts

- `runs/kaggle_v02_smoke6_outputs/mechanism/tac235/native_program_bottleneck_antibypass.json`
- `runs/kaggle_v02_smoke6_outputs/mechanism/tac242/tac242_algorithm_distillation.json`
- `runs/kaggle_v02_smoke6_outputs/mechanism/tac272/tac272_causal_fix_disambiguation.json`
- `runs/kaggle_v02_smoke6_outputs/lm50m/transformer_50m/metrics_v02.json`
- `runs/kaggle_v02_smoke6_outputs/lm50m/tac_50m/metrics_v02.json`
- `runs/kaggle_v02_smoke6_outputs/lm112m/transformer_112m/metrics_v02.json`
- `runs/kaggle_v02_smoke6_outputs/lm112m/tac_112m/metrics_v02.json`

## LM Smoke Metrics

| Model | Steps | Best Eval Loss | Checkpoints |
|---|---:|---:|---|
| transformer_50m | 2 | 2.0022 | disabled for smoke |
| tac_50m | 2 | 3.1270 | disabled for smoke |
| transformer_112m | 2 | 0.8313 | disabled for smoke |
| tac_112m | 2 | 0.9122 | disabled for smoke |

## Pipeline Fixes Validated

- Kernel metadata titles now use unique slugs, avoiding Kaggle title collisions.
- Kernel bootstrap accepts both zipped source bundles and Kaggle-mounted source
  trees.
- Serial push mode waits for each kernel to complete before starting the next,
  avoiding the account's two batch-GPU-session limit.
- Smoke LM runs use `--no-save-checkpoints`, keeping remote output payloads
  small.
- Successful kernels remove the unpacked source tree from `/kaggle/working`;
  mechanism smoke also removes TAC-272 sandbox workspaces before output pull.

## Next Required Run

Run the non-smoke workflow on streamed real data:

```powershell
python experiments\stage_v02_kaggle_workflow.py --serial-push --kernel-wait-seconds 43200 --date-slug 2026-06-14-full --output-root runs --owner jeffkolo --push
```

The full run must train the matched transformer first, then TAC, on the same
tokens and compute budget before any claim is made about mechanism survival.

## 2026-06-15 Bottleneck Smoke

After V02-006, `smoke7` reran the full smoke workflow with the v0.2 TAC configs
using `lm_readout_type=slot_conditioned_program_bottleneck`.

Remote kernels:

- `jeffkolo/tac-v02-mechanism-reproduction-2026-06-15-smoke7`: complete
- `jeffkolo/tac-v02-lm-50m-pilot-2026-06-15-smoke7`: complete
- `jeffkolo/tac-v02-lm-112m-pilot-2026-06-15-smoke7`: complete

LM readout check:

| Model | Steps | Eval Loss | LM Readout | Selected Mass | Delta Norm |
|---|---:|---:|---:|---:|---:|
| transformer_50m | 2 | 2.0022 | n/a | n/a | n/a |
| tac_50m | 2 | 8.9387 | 1.0 | 1.0 | 19.6061 |
| transformer_112m | 2 | 0.8313 | n/a | n/a | n/a |
| tac_112m | 2 | 8.9029 | 1.0 | 1.0 | 22.6327 |

Interpretation: the remote TAC LM smoke uses the native program bottleneck. The
loss values are not scientific evidence because the run is only two steps on
offline synthetic data.

# TAC-281 Mechanism Sharpening Before Scale

Date: 2026-06-18

Status: complete. Combined decision: `do_not_scale_yet`.

## Purpose

TAC-280 produced the first real Stage-2 win: TAC lost the plain LM comparison,
but the trained 50M checkpoint retained persistent-structure behavior and a
large native bottleneck knockout delta.

TAC-281 is the next gate:

> Keep the TAC mechanism advantage while reducing the LM and speed penalty.

Do not launch 112M until this gate is answered.

## Variants

All variants stay in the 30M-50M parameter band.

| Variant | Model Name | Estimated Params | Intent |
|---|---|---:|---|
| Late-bottleneck TAC | `tac_50m_late_bottleneck` | 41,073,846 | Early layers behave more like a normal Transformer; TAC identity computation activates only from layer 3 onward. |
| Small TAC adapter | `tac_50m_small_adapter` | 30,341,598 | Keep the LM path mostly normal with 12 lower-rank programs, late TAC activation, hidden LM readout, and a small memory/procedure adapter. |
| Auxiliary-mechanism TAC | `tac_50m_auxiliary_mechanism` | 41,073,846 | Keep the current bottleneck path, but apply stronger mechanism auxiliary pressure while leaving LM loss dominant. |

Matched transformer reference:

| Model | Estimated Params |
|---|---:|
| `transformer_50m` | 41,752,704 |

## Implemented Changes

- `TACConfig.tac_active_layer_start` controls where TAC identity computation
  starts.
- Inactive early TAC layers run the normal attention/MLP path and emit zeroed
  identity diagnostics, avoiding early routing/program-expert compute.
- `scripts/train_v02_lm.py` can train the three TAC-281 variants.
- `scripts/run_v02_checkpoint_mechanism_retests.py` can retest any TAC variant
  checkpoint against the matched transformer checkpoint.
- `scripts/summarize_tac281_variants.py` enforces the scale/no-scale rule.
- `experiments/stage_v02_kaggle_workflow.py` can stage a `tac281-variants`
  Kaggle kernel.

## Decision Rule

Scale to 112M only if at least one TAC-281 variant satisfies all checks:

- TAC mechanism wins remain at least 3 / 4 families.
- TAC carry advantage remains positive.
- Bottleneck knockout delta remains clearly positive.
- LM loss gap shrinks by at least 30%.
- Speed penalty is reduced versus LM50A.

No variant passed. Continue 50M sharpening or redesign TAC-Prime before
spending a 112M run.

## Current Split-Kernel Results

### `small_adapter`

Kaggle kernel:
`jeffkolo/tac-v02-tac281-small-adapter-tac281a-small`

Decision: `not_scale_ready`.

| Check | Result |
|---|---:|
| Transformer best eval loss | 1.061079 |
| Small-adapter best eval loss | 1.082974 |
| LM gap shrink vs LM50A | 95.01% |
| Speed penalty | 4.35x |
| Mechanism win families | 3 / 4 |
| Carry-positive families | 3 / 4 |
| TAC carry advantage | 0.044389 |
| Bottleneck knockout delta | 0.000021 |

Interpretation: the small adapter sharply reduces the LM loss gap and cuts the
speed penalty from the original 14.19x to 4.35x, while preserving a positive
carry signal and 3 / 4 mechanism-family wins. It fails the scale gate because
the bottleneck knockout effect is effectively gone. This points to a useful
efficiency direction, but not a 112M-ready mechanism-preserving variant yet.

### `late_bottleneck`

Failed infrastructure run:
`jeffkolo/tac-v02-tac281-late-bottleneck-tac281a-late`

Failure: inactive TAC layers emitted sequence-collapsed identity context
(`program_identity` shaped `[batch, d_model]`) while identity-first attention
requires token-shaped context (`[batch, seq, d_model]`). This is an
implementation failure, not a valid scale-gate result.

Fixed Kaggle rerun:
`jeffkolo/tac-v02-tac281-late-bottleneck-tac281b-latefix`

Decision: `not_scale_ready`.

| Check | Result |
|---|---:|
| Transformer best eval loss | 1.061079 |
| Late-bottleneck best eval loss | 1.534723 |
| LM gap shrink vs LM50A | -7.94% |
| Speed penalty | 6.96x |
| Mechanism win families | 3 / 4 |
| Carry-positive families | 3 / 4 |
| TAC carry advantage | 0.026638 |
| Bottleneck knockout delta | 5.777481 |

Interpretation: late bottleneck preserves the native bottleneck mechanism and
keeps a positive carry signal, but it does not reduce the LM gap. It therefore
fails the scale gate.

### `auxiliary_mechanism`

Kaggle kernel:
`jeffkolo/tac-v02-tac281-auxiliary-mechanism-tac281a`

Decision: `not_scale_ready`.

| Check | Result |
|---|---:|
| Transformer best eval loss | 1.061079 |
| Auxiliary-mechanism best eval loss | 1.507470 |
| LM gap shrink vs LM50A | -1.73% |
| Speed penalty | 13.22x |
| Mechanism win families | 3 / 4 |
| Carry-positive families | 2 / 4 |
| TAC carry advantage | 0.038671 |
| Bottleneck knockout delta | 5.523225 |

Interpretation: auxiliary mechanism also preserves bottleneck causality and
wins 3 / 4 mechanism families, but it remains too slow and does not shrink the
LM gap enough to justify 112M.

## Final Gate

| Variant | Gate Result | Main Lesson |
|---|---|---|
| `small_adapter` | fail | efficient LM path, but bottleneck causality disappears |
| `late_bottleneck` | fail | bottleneck causality survives, but LM gap worsens |
| `auxiliary_mechanism` | fail | bottleneck causality survives, but LM gap worsens |

The current v0.2 variants do not justify 112M scaling. The next architecture
work should combine the small adapter's efficient LM path with a causal bridge
that restores intervention-sensitive persistent computation.

## Launch Command

Preferred split-kernel launch:

```powershell
python experiments\stage_v02_kaggle_workflow.py `
  --serial-push `
  --kernel-wait-seconds 43200 `
  --kernels tac281-late-bottleneck,tac281-small-adapter,tac281-auxiliary-mechanism `
  --date-slug tac281a `
  --output-root runs `
  --owner jeffkolo `
  --push
```

The combined `--kernels tac281-variants` path is also available for smoke tests
or high-wall-time environments. The split path is preferred for full Kaggle
runs because each variant can finish independently under Kaggle wall-time
limits.

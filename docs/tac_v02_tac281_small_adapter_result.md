# TAC-281 Small Adapter Result

Date: 2026-06-16

Kernel: `jeffkolo/tac-v02-tac281-small-adapter-tac281a-small`

Decision: `not_scale_ready`

## Summary

The `small_adapter` variant is the best efficiency result so far, but it does
not pass the TAC-281 scale gate.

It reduced the LM gap from the LM50A baseline by 95.01% and reduced the speed
penalty from 14.19x to 4.35x. It also kept positive carry advantage and won 3
of 4 mechanism families.

The failure is bottleneck causality: the aggregate bottleneck knockout delta is
only 0.000021, below the required clearly-positive threshold.

## Metrics

| Metric | Value |
|---|---:|
| Transformer best eval loss | 1.061079 |
| Small-adapter best eval loss | 1.082974 |
| Current LM gap | 0.021894 |
| Original LM50A gap | 0.438800 |
| LM gap shrink fraction | 0.950104 |
| Transformer runtime seconds | 1919.68 |
| Small-adapter runtime seconds | 8350.26 |
| Current speed penalty | 4.349811 |
| Original speed penalty | 14.187459 |
| TAC mechanism win families | 3 / 4 |
| TAC carry-positive families | 3 / 4 |
| TAC carry advantage | 0.044389 |
| Bottleneck knockout delta | 0.000021 |

## Gate Checks

| Gate | Passed |
|---|---|
| Mechanism wins >= 3 / 4 | yes |
| Carry advantage positive | yes |
| Knockout delta clearly positive | no |
| LM gap shrinks enough | yes |
| Speed penalty reduced | yes |

## Interpretation

Small adapter changes the pitch in the useful direction: TAC can be much closer
to a Transformer LM while retaining some persistent-state behavior.

It does not yet preserve enough causal bottleneck dependence to justify 112M
scaling. The next TAC-281 split run is `late_bottleneck`, which tests whether
delaying TAC activation can keep stronger bottleneck causality while still
improving LM efficiency.

# TAC-281 Final Gate

Date: 2026-06-18

Decision: `do_not_scale_yet`

Combined artifact:
`runs/kaggle_v02_tac281_combined_decision/tac281_variant_decision.json`

TAC-281 tested three 30M-50M variants before spending another 112M run:

| Variant | Status | LM Gap Shrink | Speed Penalty | Mechanism Wins | Carry Families | Knockout Delta | Primary Failure |
|---|---|---:|---:|---:|---:|---:|---|
| `small_adapter` | `not_scale_ready` | 95.01% | 4.79x | 3 / 4 | 3 / 4 | 0.000021 | lost bottleneck causality |
| `late_bottleneck` | `not_scale_ready` | -7.94% | 6.96x | 3 / 4 | 3 / 4 | 5.7775 | LM gap did not shrink |
| `auxiliary_mechanism` | `not_scale_ready` | -1.73% | 13.22x | 3 / 4 | 2 / 4 | 5.5232 | LM gap did not shrink |

No variant passed all TAC-281 checks.

## Interpretation

TAC-281 gives a clear pre-112M answer:

- Persistent-computation mechanisms can remain measurable after real 50M LM
  training.
- Strong bottleneck causality and LM efficiency are still in conflict.
- The only efficient variant, `small_adapter`, lost the bottleneck mechanism.
- The mechanism-preserving variants, `late_bottleneck` and
  `auxiliary_mechanism`, did not improve the LM gap enough to justify 112M.

This is not evidence that TAC mechanisms disappear at scale. It is evidence
that the current v0.2 architecture has not found a scale-ready configuration
that preserves mechanisms while remaining competitive enough as a language
model.

## Stage-Gate Action

Do not launch a new 112M run from the current v0.2 variants.

The next technical step is a 30M-50M redesign pass, not larger scaling:

- keep the efficient hidden-readout path from `small_adapter`;
- add a causal mechanism bridge that restores knockout sensitivity;
- avoid forcing all token prediction through the expensive native bottleneck;
- retest with the same TAC-281 gate before any 112M spend.

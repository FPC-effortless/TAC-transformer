# Do Persistent Computational States Survive Scale?

Date: 2026-06-18

Status: pre-112M gate report

## Question

TAC v0.2 asked whether TAC's validated mechanisms survive the move from small
benchmarks to real autoregressive language-model training.

The planned 112M run is not counted: Kaggle reports
`jeffkolo/tac-v02-lm-112m-pilot-lm112a` as
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`.

The usable evidence is therefore the 30M-50M real-data LM pilot, TAC-280
checkpoint mechanism retests, and TAC-281 mechanism-sharpening gate.

## Evidence

Baseline 50M LM pilot:

| Model | Best Eval Loss | Final Perplexity | Runtime Seconds |
|---|---:|---:|---:|
| `transformer_50m` | 1.0611 | 4.2651 | 1,757.34 |
| `tac_50m` | 1.4999 | 5.2435 | 24,932.19 |

The first TAC 50M model lost the plain LM comparison.

TAC-280 checkpoint retests showed mechanism retention in that losing TAC
checkpoint:

| Metric | Result |
|---|---:|
| Mechanism family wins | 3 / 4 |
| Carry-positive families | 2 / 4 |
| Bottleneck knockout delta | 5.4773 |
| Overall TAC carry advantage | 0.0726 |

TAC-281 then tested three 30M-50M variants before another 112M spend:

| Variant | LM Gap Shrink | Mechanism Wins | Carry Families | Knockout Delta | Gate |
|---|---:|---:|---:|---:|---|
| `small_adapter` | 95.01% | 3 / 4 | 3 / 4 | 0.000021 | fail |
| `late_bottleneck` | -7.94% | 3 / 4 | 3 / 4 | 5.7775 | fail |
| `auxiliary_mechanism` | -1.73% | 3 / 4 | 2 / 4 | 5.5232 | fail |

## Finding

Persistent computational mechanisms do survive in some real 50M LM
checkpoints, but the current v0.2 designs do not preserve those mechanisms and
close the LM-efficiency gap at the same time.

The failure is not "state does not work." The failure is a tradeoff:

- Efficient LM variant: `small_adapter` nearly closes the LM gap but loses
  causal bottleneck dependence.
- Mechanism-preserving variants: `late_bottleneck` and `auxiliary_mechanism`
  keep strong knockout sensitivity but do not improve the LM gap.

## Decision

Do not scale the current TAC v0.2 variants to 112M.

The credible next step is a 30M-50M redesign that keeps the efficient
small-adapter LM path while adding a causal bridge that restores intervention
sensitivity. A new 112M run should only happen after that design passes the
same TAC-281 gate.

## Claims Not Supported

These results do not prove better language modeling, coding, mathematics,
planning, reasoning, agents, or multi-hop reasoning.

They support a narrower statement:

> TAC-style persistent computation remains measurable after real LM training,
> but the current v0.2 architecture has not made it scale-ready.

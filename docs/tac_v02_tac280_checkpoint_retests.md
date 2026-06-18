# TAC-280 Checkpoint Mechanism Retests

Date: 2026-06-15

Kernel: `jeffkolo/tac-v02-checkpoint-retests-tac280a`

Source checkpoints: `jeffkolo/tac-v02-lm-50m-pilot-lm50a`

Artifacts:

- Workflow manifest: `runs/kaggle_v02_workflow_tac280a/manifest.json`
- Pulled JSON outputs: `runs/kaggle_v02_tac280a_outputs`
- Primary result: `runs/kaggle_v02_tac280a_outputs/tac280/tac280_checkpoint_mechanism_retests.json`

## Purpose

TAC-280 answers the immediate post-LM50A question:

> After a negative plain LM comparison, does the trained TAC checkpoint still
> show persistent-state, repair, compression, retrieval, carry/reset, or
> bottleneck-knockout mechanism advantages?

This is intentionally not a new architecture change and not a bigger scaling
run.

## Method

The suite evaluates the trained `transformer_50m` and `tac_50m` checkpoints
from LM50A on deterministic text probes:

- persistent-state carry
- repair trace reuse
- compression / structure reuse
- noisy-key retrieval
- reset versus carried-state ablation
- native program bottleneck knockout

Transformer is evaluated with full-context loss and reset loss. TAC is
evaluated with prefix-carried identity state, reset state, and a zeroed native
program bottleneck knockout.

Boundary: positive loss deltas show checkpoint sensitivity to the tested
mechanisms. They do not prove product-grade coding, agent, planning, or
open-ended reasoning performance.

## Decision

TAC-280 status: `mechanism_advantage`

| Decision Field | Value |
|---|---:|
| Families | 4 |
| TAC win families vs transformer full-context | 3 |
| Carry-positive families | 2 |
| Carry state survives | true |
| Bottleneck matters | true |
| TAC beats transformer by family-count rule | true |

## Overall Metrics

Lower loss is better. Positive deltas favor TAC or the tested mechanism.

| Metric | Value |
|---|---:|
| Transformer full-context loss | 3.3358 |
| Transformer reset loss | 3.0059 |
| TAC carry loss | 3.5336 |
| TAC reset loss | 3.6062 |
| TAC bottleneck-knockout loss | 9.0109 |
| TAC vs transformer full-context delta | -0.1977 |
| TAC carry advantage over reset | 0.0726 |
| Bottleneck knockout delta | 5.4773 |

Interpretation: TAC wins 3 of 4 mechanism families and the bottleneck knockout
effect is large, but TAC does not win the aggregate average loss because the
persistent-state family is much harder for this checkpoint than for the
transformer full-context baseline.

## Family Metrics

| Family | Transformer Full | TAC Carry | TAC Reset | TAC Knockout | TAC vs Transformer Delta | Carry Advantage | Knockout Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| compression_structure_reuse | 2.3750 | 2.2062 | 2.1510 | 9.0109 | 0.1687 | -0.0552 | 6.8047 |
| noisy_key_retrieval | 3.3849 | 3.0033 | 3.0732 | 9.0109 | 0.3816 | 0.0699 | 6.0076 |
| persistent_state_carry | 5.3479 | 6.9881 | 7.2730 | 9.0109 | -1.6402 | 0.2850 | 2.0228 |
| repair_trace_reuse | 2.2357 | 1.9367 | 1.9276 | 9.0109 | 0.2989 | -0.0091 | 7.0742 |

## Interpretation

LM50A remains a negative result for plain next-token learning efficiency:
Transformer reached lower LM loss, lower perplexity, and much faster runtime.

TAC-280 changes the story from "TAC loses LM, therefore stop" to:

> TAC sacrifices plain LM efficiency in this configuration, but the trained
> checkpoint retains measurable sensitivity to carried state and native program
> bottleneck structure.

The result is positive enough to continue mechanism-retention validation, but
not clean enough to claim that TAC is a better language model or a better
general coding/reasoning model. The immediate claim should stay narrow:

> TAC is a research architecture for persistent structure memory, procedural
> repair, and long-horizon agent state, currently validated against transformer
> baselines.

## Next Gate

Do not scale bigger solely from the LM50A training result. Before a 112M run,
keep the next gate focused on mechanism retention:

- repeat TAC-280 with more holdout rows and less templated probes;
- separate aggregate-loss and family-count decision rules;
- add transformer reset/full-context controls to the public table;
- keep TAC-235 Kaggle route/program causality caveats visible.


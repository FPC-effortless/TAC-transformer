# TAC v0.2 Success Criteria

TAC v0.2 asks one question:

> Do the TAC mechanisms survive scale?

The v0.2 gates must be defined before the scaling run starts.

## Required Conditions

TAC v0.2 validates only if all of these are true:

1. The TAC model has at least 100M trainable parameters.
2. A parameter-matched vanilla control is trained or evaluated under a comparable budget.
3. A TAC reset-state control is trained or evaluated.
4. Carried IdentityState beats reset IdentityState on the target workflow.
5. Program knockout measurably harms the matching capability.
6. Context compression advantage remains after scaling.
7. Repair/control metrics remain positive outside bounded toy simulations.
8. Gains are not explained by retrieval-only or prompt-only controls.

## Minimum Stage-4 Metrics

| Metric | Gate |
|---|---:|
| TAC parameters | `>= 100M` |
| Carried-state advantage over reset | `> 0.05` |
| Program knockout drop | `> 0.10` |
| Compression matched-accuracy ratio | `>= 10x` |
| Real-repository task improvement over reset | `> 0.05` |
| Regression avoidance | `>= 0.90` |
| First-pass repair success | `>= 0.60` |
| Checkpoint resumability | required |

## Non-Validation Outcomes

TAC v0.2 is not validated if:

- the 100M model trains but state carry does not beat reset;
- program knockouts stop mattering;
- compression disappears outside synthetic tasks;
- all gains can be reproduced with retrieval-only controls;
- the model cannot finish or resume cleanly on Kaggle;
- the only positive result is lower training loss.

## Interpretation Boundary

Passing TAC v0.2 would not prove that TAC beats frontier LLMs. It would prove a
narrower and more important Stage-4 point:

> Persistent state, routed programs, and repair/control mechanisms survive a
> materially larger learned system and remain causally useful under controls.


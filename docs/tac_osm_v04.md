# TAC-OSM v0.4 falsification benchmark

## Purpose

v0.4 is a controlled 2x2 experiment for temporal persistent state and
action-conditioned transition prediction.

The benchmark was redesigned after a harness audit found four critical
problems in the earlier runner: P1 was not temporal, the cells were not
capacity matched, a direct policy head bypassed forecasting, and stochastic
oracle evaluation contaminated labels and regret.

## Causal task

Each episode has two observations:

- **t0:** the observation contains a task context vector.
- **t1:** the query observation contains no context; the same context is hidden.
- **decision:** candidate actions are forecast and ranked using the fixed objective.

The primary mechanism is:

`t0 context -> persistent state -> t1 representation -> candidate transition
-> predicted outcome -> fixed objective -> selected action`

P1 carries information from t0 to t1. P0 cannot access t0 after the query
starts.

The context also modulates the action effect in the environment, so a model
cannot solve the held-out decision by using only the current query state.

## 2x2 factors

| Cell | Temporal state | Action-conditioned transition |
|---|---:|---:|
| P0-A0 | off | off |
| P0-A1 | off | on |
| P1-A0 | on | off |
| P1-A1 | on | on |

All cells instantiate the same parameterized network. Factor-off conditions
mask the corresponding information path. Parameter counts must therefore match.

## Oracle protocol

The environment is deterministic for the mechanism benchmark. The oracle is
used to construct training transition targets and independent evaluation
metrics. Oracle optimal-action labels and scores are never model inputs.

A second environment regime is held out for evaluation. The held-out regime
changes the latent state persistence coefficient while preserving action semantics; it is not present during training.

## Primary metrics

- held-out all-action forecast MSE
- held-out planning accuracy
- achieved objective
- oracle objective
- objective regret
- parameter count
- action forecast sensitivity
- reset/shuffle/corruption state sensitivity

## Required falsification probes

1. **Action probe:** same state/context, vary only candidate action. A1 must
   change forecasts; A0 should be approximately invariant.
2. **Reset probe:** remove t0 state before t1. P1 performance should degrade.
3. **Shuffle probe:** carry another example's t0 state. P1 performance should
   degrade.
4. **Corruption probe:** perturb carried state. P1 predictions should change.
5. **Held-out regime:** evaluate under a regime not used for training.
6. **Parameter equality:** all four cells must have identical trainable counts.

## What this does not establish

This remains a small synthetic one-step decision benchmark. Passing it does
not establish general planning, world-model capability, autonomy, continual
learning, or real-world operational competence.

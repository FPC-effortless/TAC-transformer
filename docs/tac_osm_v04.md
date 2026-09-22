# TAC-OSM v0.4 controlled experiment

## Purpose

v0.4 is the first controlled training experiment for the proposed
**forecast -> candidate outcomes -> decision** loop.

It isolates two factors:

1. **Persistent state (P):** whether a bounded structural state is carried and
   retrieved across an observation.
2. **Action-conditioned transition (A):** whether the transition predictor
   receives the candidate action.

This gives a 2x2 matrix:

| Condition | Persistent state | Action-conditioned transition |
|---|---:|---:|
| P0-A0 | no | no |
| P0-A1 | no | yes |
| P1-A0 | yes | no |
| P1-A1 | yes | yes |

The P1-A1 cell is the minimal TAC-OSM hypothesis under test; it is not
treated as evidence of planning or counterfactual reasoning by construction.

## Training

Each training sample contains:

- current state S_t
- sampled intervention A_t
- oracle next state S_(t+1) from the synthetic operational world
- oracle optimal action label, computed independently from the transition oracle

The model is trained only on S_t, A_t, and the resulting next-state/action
targets. Oracle action scores are not model inputs.

The objective is:

L = MSE(predicted_next_state, observed_next_state) + CE(predicted_action, optimal_action)

No reinforcement learning is used.

## Evaluation

The runner reports:

- all-action transition forecast MSE
- direct action decision accuracy
- achieved one-step operational objective
- oracle objective
- objective regret
- parameter count
- training wall-clock
- final and midpoint training loss

For action-conditioned cells, candidate actions are scored using the model's
predicted next state and the same fixed operational objective. For
non-action-conditioned cells, the direct action head supplies the decision.

## Controls and limitations

- independent deterministic seeds are used for train/evaluation generation
- all four cells use the same hidden/input dimensions and optimizer settings
- this is a synthetic linear world with four discrete interventions
- the current benchmark tests one-step intervention outcomes
- persistent state is reset at the start of each independent evaluation batch
- it does not yet test long-horizon state accumulation, held-out dynamics,
  constraints, uncertainty calibration, repair, or continual acquisition

A result should therefore be reported as evidence about the tested
factorization, not as evidence for a general world model or autonomous agent.

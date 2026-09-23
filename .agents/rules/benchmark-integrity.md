# Benchmark Integrity Rules

These rules govern experiments intended to support claims about TAC-OSM, persistent computation, causal action conditioning, forecasting, planning, or decision mechanisms.

## 1. No temporal persistence claim without temporal evidence

A persistent-state factor is valid only if information is introduced at an earlier timestep and is unavailable from the current observation at the decision timestep.

Required controls:
- carry: previous state is available;
- reset: previous state is removed;
- shuffle: previous states are mismatched across examples;
- corruption: stored information is perturbed;
- oracle-memory: optional upper bound with exact hidden context.

A model that writes the current observation and immediately reads it is not a temporal-persistence test.

## 2. Capacity must not explain the factor

All cells in a factorial comparison must have matched trainable parameter counts, or use an explicit capacity-matched control.

Unused factor modules may exist for parameter matching, but their outputs must be masked so the factor is genuinely absent.

Report parameter counts per cell.

## 3. Primary decision must flow through the proposed mechanism

For forecast-to-decision experiments:
`observation -> persistent state -> action-conditioned transition -> predicted outcome -> fixed objective -> selected action`

Do not train a direct `state -> optimal action` head in the primary condition.

A direct policy head may exist only as a separately labeled shortcut control.

## 4. Oracle separation

The model must not receive oracle action scores, oracle optimal-action labels as inputs, hidden environment parameters, evaluation-only metadata, or future observations.

Targets may be derived from the environment oracle, but the derivation must be explicit and independent of model computation.

## 5. Deterministic or common-random-number evaluation

For mechanism tests, prefer deterministic expected dynamics. If stochastic evaluation is needed, use common random numbers across candidate actions and report the noise protocol.

Never compare model regret using unrelated random draws that add evaluation noise to the metric.

## 6. Generalization must be explicit

Do not use identical train and evaluation dynamics as the only evidence for causal/action-conditioned generalization.

At least one held-out condition should vary a meaningful environment property: intervention effects, dynamics coefficient, regime, state distribution, or action/effect composition.

Report exactly what is held out.

## 7. Intervention probes are mandatory

For a trained model, test:
- same observation, different action;
- same current observation, state reset;
- same current observation, state shuffled;
- same current observation, state corrupted.

Compare predicted changes with the true causal changes.

## 8. Harness tests are part of the experiment

Tests must check benchmark integrity, not only tensor shapes.

Include tests for temporal information absence, deterministic oracle consistency, parameter equality, no oracle-score access, action sensitivity, state intervention sensitivity, and held-out evaluation construction.

## 9. Claim discipline

Do not relabel bounded latent memory as temporal persistence, direct policy learning as planning, one-step prediction as a world model, synthetic transfer as real-world transfer, or optimization speed as continual learning.

## 10. Run gate

Do not launch the full multi-seed benchmark until the harness-integrity tests pass.

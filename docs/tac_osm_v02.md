# TAC-OSM v0.2 — action-conditioned forecasting and decision

v0.2 adds a small experimental surface for action-conditioned future-state
prediction and finite candidate-action comparison.

Inspired by open research systems such as WorldAgen and A2World, this is not a
copy of either architecture. WorldAgen jointly models world dynamics and
actions and uses test-time world-model training; A2World studies transferable
action-conditioned dynamics priors. Both support the motivation for testing
P(S_future | S_t, A). These references do not establish TAC-OSM capability.

## New surfaces

- ActionConditionedForecaster: predicts future observations conditioned on an
  explicit action sequence.
- ActionDecisionHead: scores candidate trajectories from terminal predicted
  state.
- action_logits: direct action proposal from the reusable computation.
- action_value: relative score over explicitly supplied candidate actions.

Existing v0.1 calls remain valid.

## Training objective

For observed transition data, initially use supervised losses:

L = L_forecast + lambda_decision L_decision + lambda_verify L_verify

Forecast loss compares predicted future states with observed outcomes.
Decision loss compares candidate selection with an environment-derived target.
This deliberately avoids RL until causal access and outcome labeling are
validated.

## What this does not establish

This code does not establish planning, valid counterfactuals, autonomous
agency, world-model competence, or real action execution. The new pathway is
an experimental interface until trained and evaluated.

## Next experiment

Use a tiny synthetic operational environment with continuous state, 4–8
discrete actions, stochastic transitions, a known objective, and held-out
dynamics combinations.

Compare forecast-only, action-conditioned, action-conditioned plus persistent
state, and full TAC-OSM. Track forecast error, action-selection accuracy,
objective achieved, constraint violations, compute, and capability acquisition
speed.

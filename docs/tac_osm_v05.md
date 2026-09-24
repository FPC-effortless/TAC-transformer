# TAC-OSM v0.5 — causal intervention benchmark

v0.5 is the Stage 3 follow-on to the v0.4 temporal/action-conditioning
substrate. It adds an explicit intervention evaluation protocol without adding
a new model mechanism.

## Question

When the query state and candidate action are held fixed, does changing the
context carried from t0 change the predicted consequence in the same direction
as the independently computed environment intervention?

## Protocol

Training remains observational. Evaluation constructs matched pairs with the
same query state and candidate action but different t0 contexts. Context is
never supplied at t1.

For every action a, compare:

predicted_delta = f(S(context_B), x_t1, a) - f(S(context_A), x_t1, a)

against:

true_delta = T(x_t1, a, context_B) - T(x_t1, a, context_A)

The primary metric is normalized intervention contrast MSE. A secondary metric
checks whether context-induced changes in the optimal action are reproduced.

## Promotion requirements

1. Five independent seeds.
2. Identical held-out query states across interventions.
3. Frozen model during intervention evaluation.
4. No oracle action labels in training.
5. Held-out dynamics regime.
6. Positive intervention sensitivity.
7. Lower normalized contrast MSE than a context-blind/null control.
8. CI reproduction.

A successful result would support causal validity of this synthetic pathway.
It would not establish general causal reasoning, planning, or real-world
causal competence.

No persistence expansion, verifier, recurrence, or RL is introduced by v0.5.

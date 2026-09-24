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


## Diagnostic interpretation

The initial five-seed CI reproduction is a closed diagnostic rather than a promotion result. Seed 1 reached normalized intervention MSE 0.38614 with 0.9668 context-flip recall; seeds 0, 2, 3, and 4 remained at approximately the context-blind null (normalized MSE ≈ 1 and flip recall ≈ 0). The next experiment therefore diagnoses the learning path without changing the v0.5 gate or adding causal supervision.

The diagnostic reports observational training fit, t0 persistent-state context separation, t1 retrieved-state separation, hidden-state separation, and forecast separation. These measurements distinguish context encoding, retrieval, and forecast-binding failures.

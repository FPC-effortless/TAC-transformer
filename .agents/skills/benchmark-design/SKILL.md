---
name: benchmark-design
description: Design, audit, and repair TAC-OSM experiments so causal factors are isolated and models cannot exploit oracle leakage, temporal shortcuts, capacity differences, direct-policy bypasses, stochastic evaluation noise, or train/eval contamination.
compatibility: Python, PyTorch, TAC-transformer repository
---

# Benchmark Design

Use this skill whenever changing or reviewing a TAC-OSM benchmark, especially before a multi-seed run or when a result may support a research claim.

## Procedure

1. Read the experiment runner, environment, model, tests, and benchmark document.
2. Write the claimed mechanism in one causal chain.
3. Identify every input available to the model at each timestep.
4. Mark every oracle-derived value and verify that it is used only for targets/evaluation.
5. Check whether each factorial condition differs only in the intended factor.
6. Count trainable parameters for every condition.
7. Check whether the proposed mechanism can be bypassed by a direct supervised head.
8. Check train/eval separation: state distribution, dynamics, intervention effects, regimes, and action/effect combinations.
9. Check stochasticity and evaluation noise.
10. Add intervention probes and harness-integrity tests.
11. Run focused tests before any expensive training.
12. Only then run the planned seed matrix.

## Required TAC-OSM v0.4 design

The benchmark must contain a sequence:

`t0: context is visible -> t1: context is absent -> action is selected`

The persistent condition carries the context from t0 to t1. The non-persistent condition receives the same t1 observation but no carried context.

The primary decision path is:

`S_t + carried_state + candidate_action -> predicted_outcome -> fixed_objective -> argmax`

Do not train a direct optimal-action head in the primary path.

All four P/A cells use the same parameterized architecture. Factor-off cells mask the corresponding information rather than deleting modules.

The oracle should provide deterministic expected transitions and deterministic objective scores. Stochastic dynamics may be added later as a separate robustness experiment.

## Falsification matrix

| Probe | Expected if mechanism is real |
|---|---|
| P1 vs P0 with context hidden at t1 | P1 retains useful performance; P0 degrades |
| P1 reset | Performance approaches P0 |
| P1 shuffle | Performance degrades toward mismatched-memory control |
| P1 corruption | Performance changes predictably with corruption |
| A1 same state, different action | Forecast changes in the direction of true causal effect |
| A0 same state, different action | Forecast is invariant to action |
| held-out regime | Mechanism retains non-trivial performance without memorizing training dynamics |
| parameter counts | Equal across all cells |

## Stop conditions

Stop before training if:
- parameter counts differ without a justified matched control;
- current observations contain the information claimed to be persistent;
- direct action supervision can solve the primary decision without forecasting;
- evaluation uses noisy oracle labels when deterministic expected labels are available;
- the model can read evaluation-only oracle information;
- the held-out split is not actually different from training;
- integrity tests do not establish the intended intervention.

## Evidence record

Every benchmark result should record git commit, exact command, Python/PyTorch versions, device, all seeds, training steps, optimizer settings, parameter counts, train/eval environment definitions, primary/control metrics, integrity-test result, and deviations from the preregistered protocol.

Do not upgrade the claim after seeing the result.

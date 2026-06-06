# Agentic Objective Experiments

Date: 2026-05-28

Goal: test whether agent-layer ideas from the proposed formal loop are worth adding to the current best TAC architecture for a small-lab, data/energy-conscious agentic platform.

The full agent formula contains many layers:

```text
Backbone + Routing + Memory + WorldModel + Planner + Tools + Reflection + Orchestration
```

This experiment intentionally starts with the layers that can be tested inside the model without building a full external agent runtime:

- action/tool policy
- carried memory retrieval
- world prediction auxiliary head
- reward prediction auxiliary head
- reflection/calibration auxiliary head

Planner, tool execution, and multi-agent orchestration remain system-level experiments for a later harness.

## Task

The benchmark is a no-leak action/tool-selection task:

```text
context chunk: [context_token, key, hidden_value, noise...]
query chunk:   [query_token, key, observation, action_token, noise...]
target:        choose the correct action/tool
```

The hidden value is removed from the query input. Correct action selection depends on both the hidden value and the query observation, so the model must carry useful memory from context into the query. With four actions, random chance is `25%`.

## Tested Variants

| Variant | Description |
| --- | --- |
| `policy` | Action head over the best TAC hidden state. |
| `world` | Policy plus next-observation/world-model auxiliary head. |
| `world_reward` | World plus reward auxiliary head. |
| `reflection` | World/reward plus reflection-calibration auxiliary head. |
| `memory_policy` | TAC-native memory-action readout: carried program memory feeds the action policy directly. |
| `memory_world` | Memory-action readout plus world auxiliary head. |
| `memory_reflection` | Memory-action readout plus world/reward/reflection heads. |

## Naive Agent Heads

The first matrix tested policy/world/reward/reflection heads without a direct memory-action readout.

| Variant | Effective | Carry acc | Reset acc | Shuffled acc | Baseline acc | TAC-baseline gap | Carry-reset delta | Train TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| policy | 0/3 | 0.2552 | 0.2461 | 0.2721 | 0.2422 | 0.0130 | 0.0091 | 0.4818 |
| world | 0/3 | 0.2435 | 0.2565 | 0.2695 | 0.2552 | -0.0117 | -0.0130 | 0.4689 |
| world_reward | 0/3 | 0.2422 | 0.2539 | 0.2786 | 0.2383 | 0.0039 | -0.0117 | 0.5023 |
| reflection | 0/3 | 0.2383 | 0.2435 | 0.2773 | 0.2578 | -0.0195 | -0.0052 | 0.4743 |

Result: naive world/reward/reflection heads are not worth adding to the base architecture yet. They stayed near chance and did not create a reliable carry advantage.

Artifacts:

- `runs/benchmarks/agentic_objectives_2026_05_28/RESULTS.md`
- `runs/benchmarks/agentic_objectives_2026_05_28/aggregate_agentic_objectives.json`

## Memory-Conditioned Action Policy

The second matrix tested a TAC-native action policy where carried `program_memory` is read by key and routed directly into the action/tool head.

| Variant | Effective | Carry acc | Reset acc | Shuffled acc | Baseline acc | TAC-baseline gap | Carry-reset delta | Train TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| memory_policy | 0/3 | 0.2747 | 0.2448 | 0.2826 | 0.2409 | 0.0339 | 0.0299 | 0.4706 |
| memory_reflection | 1/3 | 0.2617 | 0.2409 | 0.2630 | 0.2513 | 0.0104 | 0.0208 | 0.4848 |
| memory_world | 0/3 | 0.2591 | 0.2357 | 0.2747 | 0.2513 | 0.0078 | 0.0234 | 0.5026 |
| policy | 0/3 | 0.2552 | 0.2461 | 0.2721 | 0.2422 | 0.0130 | 0.0091 | 0.4779 |

Result: `memory_policy` is the best tested way to apply the agentic loop inside TAC so far. It improves carry accuracy and TAC-baseline gap over policy-only, but it still fails the shuffled-state test because shuffled accuracy remains high. That means the model is not yet using identity-state content reliably enough for this agentic task.

Artifacts:

- `runs/benchmarks/agentic_memory_objectives_2026_05_28/RESULTS.md`
- `runs/benchmarks/agentic_memory_objectives_2026_05_28/aggregate_agentic_memory_objectives.json`

## Decision

Do not promote world-model, reward, reflection, or multi-agent machinery into the default TAC architecture yet.

The commercially sensible path is:

```text
1. Keep best_tac_config as the base.
2. Keep AgenticController as an experimental adapter.
3. Continue with memory-conditioned action policy, not naive world/reward/reflection heads.
4. Only promote an agentic head when carry beats reset, carry beats shuffled, and TAC beats baseline across seeds.
```

The current best agentic addition is `memory_policy`, but it is not ready as a default. It is a research candidate for the next iteration.

## Full Layered Stack Matrix

After the first two matrices, the harness was extended to cover the remaining model-level pieces from the formal architecture:

- hybrid recurrent state;
- sparse modular cognition experts;
- four-store memory projections for working/episodic/semantic/procedural-style reads;
- planner/value action head;
- critic-weighted multi-agent orchestration;
- all-agentic stack with memory, recurrent, modular, planner, orchestration, world, reward, and reflection enabled together.

| Variant | Effective | Carry acc | Reset acc | Shuffled acc | Baseline acc | TAC-baseline gap | Carry-reset delta | Shuffled penalty | Train TPS ratio | Enabled layers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| memory_policy | 0/3 | 0.2747 | 0.2448 | 0.2826 | 0.2409 | 0.0339 | 0.0299 | -0.0078 | 0.4739 | memory_action |
| all_agentic | 0/3 | 0.2695 | 0.2773 | 0.2734 | 0.2604 | 0.0091 | -0.0078 | -0.0039 | 0.5604 | memory_action, recurrent, modular, memory_stores, planner, orchestration, world, reward, reflection |
| planner | 1/3 | 0.2617 | 0.2539 | 0.2734 | 0.2474 | 0.0143 | 0.0078 | -0.0117 | 0.4795 | planner |
| memory_stores | 1/3 | 0.2578 | 0.2513 | 0.2539 | 0.2357 | 0.0221 | 0.0065 | 0.0039 | 0.4733 | memory_action, memory_stores |
| policy | 0/3 | 0.2552 | 0.2461 | 0.2721 | 0.2422 | 0.0130 | 0.0091 | -0.0169 | 0.4747 | policy_only |
| modular | 0/3 | 0.2513 | 0.2786 | 0.2487 | 0.2409 | 0.0104 | -0.0273 | 0.0026 | 0.5106 | modular |
| hybrid | 1/3 | 0.2422 | 0.2383 | 0.2474 | 0.2357 | 0.0065 | 0.0039 | -0.0052 | 0.5301 | recurrent |
| orchestration | 0/3 | 0.2409 | 0.2591 | 0.2669 | 0.2318 | 0.0091 | -0.0182 | -0.0260 | 0.5088 | orchestration |

Result: adding everything is not better. `all_agentic` ranks second by carry accuracy but fails carry-vs-reset and carry-vs-shuffled, and its TAC-baseline gap is much weaker than `memory_policy`.

The conclusion is stronger now:

```text
For the current model-level benchmark, the only agentic layer worth continuing is
memory-conditioned action policy. Planner, orchestration, recurrent hybrid state,
modular cognition, world, reward, and reflection should stay outside the default
base model until the memory-action path passes state-content validation.
```

Artifacts:

- `runs/benchmarks/agentic_full_layers_2026_05_28/RESULTS.md`
- `runs/benchmarks/agentic_full_layers_2026_05_28/aggregate_agentic_full_layers.json`
- `runs/benchmarks/agentic_full_layers_2026_05_28/per_seed_agentic_full_layers.json`

## Next Experiments

The next tests focused on making memory-action use real rather than decorative:

- add a direct memory-action contrastive loss where correct carried state helps and wrong shuffled state hurts;
- train longer budget curves for `policy` vs `memory_policy`;
- test easier two-action and harder eight-action tool spaces;
- add explicit planner/tool simulation only after memory-conditioned action selection passes the carry/reset/shuffle bar;
- compare against small recurrent baselines once the task shows a reliable TAC signal.

## Memory-Action Validation

The validation matrix added:

- `memory_contrastive`: `memory_policy` plus carried-vs-shuffled contrastive loss;
- budget curves at 60, 120, and 240 steps for 4-action routing;
- 2-action, 4-action, and 8-action tool spaces at 120 steps;
- a small GRU recurrent baseline on every run.

### Budget Curves, 4 Actions

| Variant | Steps | Effective | Carry | Reset | Shuffled | Vanilla | Recurrent | Carry-reset | Carry-shuffled | TAC-vanilla | TAC-recurrent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| memory_contrastive | 60 | 1/3 | 0.2526 | 0.2474 | 0.2487 | 0.2292 | 0.2201 | 0.0052 | 0.0039 | 0.0234 | 0.0326 |
| memory_policy | 60 | 1/3 | 0.2487 | 0.2526 | 0.2383 | 0.2292 | 0.2201 | -0.0039 | 0.0104 | 0.0195 | 0.0286 |
| policy | 60 | 1/3 | 0.2370 | 0.2513 | 0.2344 | 0.2253 | 0.2747 | -0.0143 | 0.0026 | 0.0117 | -0.0378 |
| memory_contrastive | 120 | 0/3 | 0.2669 | 0.2422 | 0.2969 | 0.2409 | 0.2344 | 0.0247 | -0.0299 | 0.0260 | 0.0326 |
| memory_policy | 120 | 0/3 | 0.2747 | 0.2448 | 0.2826 | 0.2409 | 0.2344 | 0.0299 | -0.0078 | 0.0339 | 0.0404 |
| policy | 120 | 0/3 | 0.2552 | 0.2461 | 0.2721 | 0.2422 | 0.2643 | 0.0091 | -0.0169 | 0.0130 | -0.0091 |
| memory_contrastive | 240 | 0/3 | 0.2422 | 0.2396 | 0.2773 | 0.2747 | 0.2539 | 0.0026 | -0.0352 | -0.0326 | -0.0117 |
| memory_policy | 240 | 0/3 | 0.2383 | 0.2487 | 0.2682 | 0.2747 | 0.2539 | -0.0104 | -0.0299 | -0.0365 | -0.0156 |
| policy | 240 | 0/3 | 0.2500 | 0.2695 | 0.2474 | 0.2708 | 0.2878 | -0.0195 | 0.0026 | -0.0208 | -0.0378 |

### Action-Space Sweep, 120 Steps

| Variant | Actions | Chance | Effective | Carry | Reset | Shuffled | Vanilla | Recurrent | Carry-reset | Carry-shuffled | TAC-vanilla | TAC-recurrent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| memory_contrastive | 2 | 0.5000 | 0/3 | 0.4596 | 0.5091 | 0.5013 | 0.4948 | 0.5000 | -0.0495 | -0.0417 | -0.0352 | -0.0404 |
| memory_policy | 2 | 0.5000 | 0/3 | 0.4701 | 0.5000 | 0.5078 | 0.4948 | 0.5000 | -0.0299 | -0.0378 | -0.0247 | -0.0299 |
| memory_contrastive | 4 | 0.2500 | 0/3 | 0.2669 | 0.2422 | 0.2969 | 0.2409 | 0.2344 | 0.0247 | -0.0299 | 0.0260 | 0.0326 |
| memory_policy | 4 | 0.2500 | 0/3 | 0.2747 | 0.2448 | 0.2826 | 0.2409 | 0.2344 | 0.0299 | -0.0078 | 0.0339 | 0.0404 |
| memory_contrastive | 8 | 0.1250 | 0/3 | 0.1224 | 0.1224 | 0.1380 | 0.1276 | 0.1224 | 0.0000 | -0.0156 | -0.0052 | 0.0000 |
| memory_policy | 8 | 0.1250 | 0/3 | 0.1250 | 0.1276 | 0.1302 | 0.1276 | 0.1224 | -0.0026 | -0.0052 | -0.0026 | 0.0026 |

Result: the direct contrastive loss did not solve the problem. It sometimes improved carry-vs-reset, but shuffled state remained as good as or better than carried state. Longer training did not help, and the two-action/eight-action sweeps did not reveal a reliable memory-action signal.

This blocks planner/tool simulation promotion under the stated rule: do not add explicit planner/tool simulation until memory-conditioned action selection passes carry/reset/shuffle. The GRU recurrent baseline is also a warning sign: TAC is not yet clearly better on this agentic control task.

Artifacts:

- `runs/benchmarks/agentic_memory_validation_2026_05_28/RESULTS.md`
- `runs/benchmarks/agentic_memory_validation_2026_05_28/aggregate_agentic_memory_validation.json`
- `runs/benchmarks/agentic_memory_validation_2026_05_28/per_seed_agentic_memory_validation.json`

Updated decision:

```text
Do not promote memory_policy, memory_contrastive, planner, tool simulation,
or orchestration into the default architecture yet.

The best commercial path remains:
best_tac_config for the base model,
agentic behavior handled mostly outside the base model,
and a new training task/objective before another agentic-model promotion attempt.
```

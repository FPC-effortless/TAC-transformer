# TAC v0.1 Demo Script

Target length: 3 to 5 minutes.

Goal: explain TAC v0.1 as a reproducible research asset, not as a claim of
transformer replacement.

## 0:00 - 0:30 Problem

Long-horizon AI agents run into a recurring problem: they lose useful working
state, bloat context, and often struggle to repair failures over multiple steps.

TAC asks whether a model can carry persistent computational state and route work
through reusable programs, instead of relying only on a growing text context.

## 0:30 - 1:10 What TAC Is

Show: `docs/tac_v0_1_architecture_diagram.md`

TAC v0.1 has a simple conceptual flow:

```text
User Query
  -> Routing
  -> Programs
  -> Persistent State
  -> Repair Planning
  -> Output
```

The public claim is deliberately narrow:

> TAC is an experimental persistent-state architecture for long-horizon AI
> agents, with validated mechanisms for memory, compression, control, repair,
> and causal fix selection in bounded benchmarks.

## 1:10 - 1:50 Reproducibility

Show: `REPRODUCIBILITY.md`

The core validation pack runs locally and on Kaggle:

```bash
python experiments/kaggle_validate_tac_core.py --benchmarks tac251,tac252,tac267,tac270,tac272 --seeds 5 --cases 50 --output runs/kaggle_validation/tac_core_validation.json
```

Kaggle result:

- decision: PASS
- execution_environment: kaggle
- validated_on_kaggle: true for TAC-251, TAC-252, TAC-267, TAC-270, and TAC-272

## 1:50 - 2:40 Evidence Chain

Show: `runs/benchmarks/benchmark_summary_tac235_tac272.md`

Key milestones:

- TAC-235 and TAC-236: causal program mechanisms and reproduction.
- TAC-251 and TAC-252: context-compression and cost proxy.
- TAC-267: repair-grounded control.
- TAC-270: no-restore multi-file repair.
- TAC-272: causal fix disambiguation.

## 2:40 - 3:35 Failure And Recovery

Show TAC-273 first.

TAC-273 is important because it did not fail broadly. It passed:

- root-cause set
- regression avoidance
- repair-step budget
- state continuity

It failed chain completion:

```text
0.6335 < 0.70
```

Then show TAC-274.

TAC-274 added interaction-aware repair planning:

- dependency-graph planning
- patch-order prediction
- interaction tracking
- premature-fix avoidance

It improved chain completion:

```text
TAC-273: 0.6335
TAC-274: 0.7306
```

while maintaining regression avoidance:

```text
0.9557
```

## 3:35 - 4:25 Limitations

Show: `LIMITATIONS.md`

TAC v0.1 does not prove:

- TAC beats transformers.
- TAC is an open-ended software engineer.
- TAC works at foundation-model scale.
- TAC mechanisms survive large-scale pretraining.

The current strongest interpretation is:

> TAC v0.1 is a reproducible research asset for persistent-state agent
> architecture.

## 4:25 - 5:00 Future Work

Show: `docs/tac_v0_2_stage4_roadmap.md`

The next real question is:

> Do TAC mechanisms survive scaling?

TAC v0.2 should focus on:

- 100M+ parameter model validation
- real repository repair corpus
- multi-session software tasks
- external benchmark comparisons
- ablations proving the persistent-state mechanisms still matter


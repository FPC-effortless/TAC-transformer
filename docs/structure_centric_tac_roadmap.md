# Structure-Centric TAC Roadmap

## Thesis

TAC-274 changes the main research object from memory alone to reusable
computational structure.

Old framing:

```text
experience -> identity state -> programs -> routing -> output
```

Structure-centric framing:

```text
experience -> structure memory -> concept volume -> program family -> specialist route -> output
```

A reusable structure is not a single embedding vector. The practical TAC
definition should be:

```text
Structure = concept volume + program family + usage history
```

Where concept volume stores geometry, program family stores executable
specialists, and usage history stores when the structure works, fails, survives,
transfers, or should be retired.

## Research Grounding

The exact names `Neural Survival Field`, `DPSL`, and `USEF-X` are project-local
or user-local labels in this repository context. I did not find reliable
external sources using those exact names as standard ML terms.

The technical prior art that supports the direction is:

- Gaussian embeddings: concepts as learned distributions rather than points.
- Order embeddings: hierarchy as a partial order.
- Box and probabilistic box embeddings: containment, overlap, and disjointness.
- Modular deep learning and MoE: separate computation, routing, aggregation, and
  transfer.
- Dynamic modularity and continual learning: sparse reusable modules can reduce
  interference and forgetting.

## TAC-274 Result

TAC-274 validated the structure-representation layer:

- adaptive diagonal Gaussian volumes train;
- anisotropic volumes beat fixed isotropic regions on held-out likelihood;
- hierarchy, overlap, and disjoint relation losses are differentiable and stable.

What it does not prove:

- route selectivity in a behavior task;
- causal program usefulness;
- reset degradation;
- program knockout drop;
- full LM stability.

## TAC-275 Result

TAC-275 tested whether routing directly through concept volumes improves
behavior under few-shot related concepts.

Result: not validated.

Key metrics from the relation-weight sweep:

| Relation weight | Target gain | Adaptive target acc | Point target acc | Hierarchy transfer | Structure reuse |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | -0.3625 | 0.0458 | 0.4083 | 0.8708 | 0.0875 |
| 0.03 | -0.3625 | 0.0458 | 0.4083 | 0.8708 | 0.0875 |
| 0.20 | -0.3625 | 0.0458 | 0.4083 | 0.8708 | 0.0875 |

Interpretation:

- Volume routing preserved parent/overlap structure well.
- Exact child concept behavior collapsed.
- The failure is architectural, not only a relation-weight setting.

Conclusion:

```text
Concept volumes should not be the final executable route.
They should be the first-stage structure-family route.
```

## Proposed Optimal Architecture

### 1. Structure Volume Bank

Each structure has:

- `mu`: center;
- `log_var` or richer covariance;
- relation edges: same, child_of, parent_of, overlaps, disjoint,
  transfer_related;
- calibration statistics.

### 2. Program Family

Each structure volume owns or indexes a family of executable programs:

- parent/generalist program;
- child/specialist programs;
- overlap adapters;
- analogy/transfer adapters.

### 3. Structure Memory

Structure memory is not fact memory.

It should store:

- task descriptors where the structure worked;
- success and failure counts;
- reset sensitivity;
- knockout sensitivity;
- transfer edges;
- survival score;
- mutation, merge, split, and retirement history.

### 4. Two-Level Router

Use volumes for family routing:

```text
p(structure | z) = softmax(-Mahalanobis(z, mu_s, Sigma_s))
```

Then use a specialist router inside the selected structure family:

```text
p(program | z, structure, task) = specialist_gate(z, structure_memory, task_cue)
```

This solves the TAC-275 failure: parent volume reuse can remain high while child
behavior is selected by a specialist route instead of being collapsed into the
parent region.

## Unified Loss

The next full objective should be:

```text
L =
  L_LM
  + alpha * L_adaptive_volume
  + beta * L_relation
  + gamma * L_structure_family_route
  + delta * L_specialist_route
  + eta * L_structure_reuse
  + theta * L_survival
  + kappa * L_knockout_causality
```

Where:

- `L_adaptive_volume`: TAC-274 Mahalanobis/NLL volume loss.
- `L_relation`: hierarchy, overlap, disjointness, transfer relation loss.
- `L_structure_family_route`: selected family should match concept volume.
- `L_specialist_route`: selected executable program should match child behavior.
- `L_structure_reuse`: related tasks should reuse parent/overlap structures
  before creating new ones.
- `L_survival`: structures should remain recoverable under noise, memory
  attack, distribution shift, and program knockout.
- `L_knockout_causality`: removing the selected structure/program should
  degrade the behavior it claims to support.

## Stage Roadmap

### Stage 1: Structure Representation

Question: Can structures exist?

- TAC-274: adaptive concept volumes.
- TAC-275: volume-aware routing behavior.
- TAC-276: two-level structure-family plus specialist routing.
- TAC-277: structure memory data model.

### Stage 2: Structure Survival

Question: Do structures persist?

- TAC-S001: noise attack.
- TAC-S002: memory attack.
- TAC-S003: distribution shift.
- TAC-S004: program knockout.

Metrics:

- retention;
- recovery;
- stability;
- survival score;
- reset degradation.

### Stage 3: Structure Reuse

Question: Can structures transfer?

- TAC-S101: A -> B transfer.
- TAC-S102: A -> B -> C chain.
- TAC-S103: few-shot transfer.
- TAC-S104: cross-domain transfer.

Metrics:

- reuse;
- transfer gain;
- learning speed;
- negative transfer;
- structure reuse score.

### Stage 4: Structure Evolution

Question: Can structures improve themselves?

- TAC-S201: structure mutation.
- TAC-S202: structure merge.
- TAC-S203: structure split.
- TAC-S204: structure retirement.

Metrics:

- adaptation speed;
- survival score;
- task improvement;
- mutation acceptance rate;
- retirement precision.

## Validated Local Model

The current best local model is:

```text
adaptive concept volume
  -> structure-family route
  -> specialist executable route
  -> Structure Memory update
```

Validated evidence:

- TAC-274 validates adaptive volume geometry.
- TAC-275 falsifies direct volume routing as sufficient for behavior.
- TAC-276 validates the two-level structure-family plus specialist route.
- TAC-277 validates Structure Memory records derived from TAC-276 behavior.
- TAC-S001 validates controlled noise survival for the two-level router.
- TAC-S002 validates bounded Structure Memory attack recovery.
- TAC-S003 validates coherent distribution-shift survival.
- TAC-S101 validates A -> B structure transfer.
- TAC-S102 validates A -> B -> C structure-transfer chains.

TAC-276 tested:

- family route accuracy;
- specialist route accuracy;
- child exact behavior;
- parent/overlap reuse;
- reset degradation;
- family knockout drop;
- specialist knockout drop;
- source retention.

Validation should require both:

- high structure-family reuse;
- high exact child behavior.

That is the first clean bridge from TAC-274 representation to DPSL-style
structure reuse.

## Next Priority

The next implementation can start the first USEF-X-style evolution gates,
because the initial representation, behavior, memory, survival, transfer,
replication, baseline, and bridge gates now validate locally.

Recommended order:

- TAC-S201: structure mutation.
- TAC-S202: structure merge.
- TAC-S203: structure split.
- TAC-S204: structure retirement.

Evolution gates must keep the current validated controls:

- compare against fresh/direct baselines;
- report reset and knockout sensitivity;
- update Structure Memory;
- reject changes that improve one target while damaging source retention or
  transfer-chain reuse.

## Next-Phase Validation

TAC-S010 through TAC-S013 completed the requested next phase locally.

TAC-S010 replicated the full structure suite across five seeds:

- Benchmark pass rate: `1.0000`.
- Benchmarks passed: `6 / 6`.
- Mean structure advantage: `0.2809`.
- Mean knockout drop: `0.4032`.
- Mean survival score: `0.6943`.
- Mean transfer gain: `0.2325`.
- Ablation failure rate: `1.0000`.
- Replication score: `0.9764`.

TAC-S011 compared the structure chain against controlled same-task proxy
baselines:

- Structure TAC score: `0.7847`.
- Best proxy baseline score: `0.3385`.
- TAC margin over best proxy baseline: `0.4462`.
- Baseline win rate: `1.0000`.

Boundary: this is not yet a trained same-size transformer/MoE checkpoint
comparison. It is a controlled proxy baseline matrix over the same local
structure task.

TAC-S012 bridged the structure chain to repository-grounded repair-control
signals:

- Structure route-to-repair accuracy: `0.7865`.
- Baseline repair success: `0.4485`.
- Structured repair success: `0.6602`.
- Targeted repair gain: `0.2117`.
- Bridge transfer gain: `0.2997`.
- Real-task bridge score: `0.4854`.

Boundary: this is a repository-grounded repair-control bridge, not live
code-editing with the volume router inside a trained LM.

TAC-S013 added a Kaggle-ready structure-suite validation pack:

- Remote pack decision: `PASS`.
- Execution environment: `kaggle`.
- `validated_on_kaggle`: `true`.
- Staged package:
  `runs/kaggle_tac_structure_suite_validation_2026_06_14`.
- Pulled remote artifact:
  `runs/kaggle_tac_structure_suite_validation_2026_06_14_output/runs/kaggle_validation/tac_structure_suite_validation.json`.
- Local preflight artifact:
  `runs/kaggle_validation/tac_structure_suite_validation.json`.

Kaggle auth repair:

- The default CLI path was trying OAuth `access_token` introspection and failed.
- A temporary `KAGGLE_CONFIG_DIR` containing only the classic `kaggle.json`
  allowed `kaggle kernels list`, `kaggle kernels push`, status polling, and
  output pull to succeed.
- The temporary credential copy was removed after the run.

Remote benchmark rows all passed:

- `tacs010_replication_score`: `0.9770`.
- `tacs010_benchmark_pass_rate`: `1.0000`.
- `tacs011_baseline_margin`: `0.4471`.
- `tacs012_real_task_bridge_score`: `0.4853`.
- `tacs102_chain_transfer_gain`: `0.1805`.

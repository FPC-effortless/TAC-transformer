# Program Specialization Solution Research

Date: 2026-06-01

## Question

Why do TAC programs show weak category specialization, and what is the best working solution?

## Diagnosis

The core problem was observability plus routing, not proof that all programs are identical.

Run 3 used `routing_type="base"`, so selected programs were assigned mainly by a BASE token schedule. The old report looked only at the final token, and for a 256-token window with 32 programs that naturally points at program 31. That made program 31 look dominant even though full-token diagnostics and forced routing showed a richer state:

- BASE selected routes are balanced across token positions, but not category-conditioned.
- Raw activation distributions are high-entropy rather than collapsed.
- Forced routing changes loss substantially, so program identity already changes computation.
- The missing piece is a router that makes task/content signals affect selected programs.

The project state changed from "the architecture learned no specialization" to "the architecture has differentiated programs, but BASE routing does not expose category-conditioned use."

## Implemented Solution Surface

### Token-Level Observability

Added token-level diagnostics so specialization is measured over every token:

- `token_program_activations`
- `token_selected_program_mask`
- token-level mutual information
- activation and selected-route category/program matrices
- activation sparsity, category selectivity, and knockout selectivity

This fixes the final-token artifact in BASE routing.

### Hard `base_semantic` Routing

Added `routing_type="base_semantic"`.

This preserves the BASE scheduled program as a stable anchor, then adds activation-conditioned semantic programs up to `routing_top_k`. The purpose is to keep the stable memory compute path while letting token semantics influence which extra program executes.

### Labeled Category Routing Objectives

Added category-labeled JSONL training support and two objectives:

- fixed category-to-program target: `category_route_objective="fixed"`
- differentiable mutual-information objective: `category_route_objective="mi"`

The MI objective is the stronger option because it encourages category-program dependence without requiring stable hand-assigned program IDs.

### Category-Conditioned Knockouts

Extended `kaggle/benchmark_program_specialization_objectives.py` with `--include-knockouts`, which disables every program in turn and measures per-category loss deltas. This is the functional specialization test.

## Full Matrix Results

### Harder Memory Routing Matrix

Artifact:

`runs/benchmarks/program_specialization_routing_full_matrix_2026_06_01/aggregate_harder_research_matrix.json`

Settings:

- variants: `current_best`, `content_synthesis_k2`, `content_synthesis_semantic_k2`, `base_semantic_k2`, `base_semantic_balanced_k2`, `base_semantic_balanced_k3`, `sparse_ensemble_k2`
- seeds: 11, 23, 37
- tasks: longer single key, multi key, delayed query, noisy key, multi hop
- 120 steps, batch size 32, 8 eval batches

| Variant | Mean carry | Effective | Task wins |
| --- | ---: | ---: | ---: |
| `current_best` | 0.5099 | 15/15 | 3 |
| `content_synthesis_k2` | 0.5042 | 15/15 | 1 |
| `content_synthesis_semantic_k2` | 0.5029 | 15/15 | 1 |
| `base_semantic_balanced_k3` | 0.0526 | 15/15 | 0 |
| `base_semantic_balanced_k2` | 0.0503 | 15/15 | 0 |
| `base_semantic_k2` | 0.0503 | 15/15 | 0 |
| `sparse_ensemble_k2` | 0.0495 | 15/15 | 0 |

Conclusion:

- Route-only semantic candidates destroy the harder memory behavior.
- The content-addressed/synthesis memory stack must remain intact.
- The specialization solution must be applied to the full content-memory architecture, not as a pure routing replacement.

### Multi-Seed Objective Matrix

Artifact:

`runs/benchmarks/program_specialization_objectives_full_matrix_2026_06_01/objectives.json`

Settings:

- seeds: 11, 23, 37
- hard-agentic labeled JSONL
- 120 steps, batch size 8
- 96 train records/category, 24 eval records/category

| Variant | Eval loss | Raw NMI | Selected NMI | Selected MI | Selected entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_semantic_mi_0p5` | 4.2899 | 0.01326 | 0.00947 | 0.02209 | 2.4113 |
| `base_semantic_mi_0p1` | 4.2899 | 0.01326 | 0.00933 | 0.02171 | 2.4245 |
| `base_semantic` | 4.2901 | 0.01342 | 0.00902 | 0.02101 | 2.4125 |
| `base_semantic_supervised_0p5` | 4.3233 | 0.01356 | 0.00882 | 0.02140 | 2.4541 |
| `base_semantic_supervised_0p2` | 4.2953 | 0.01278 | 0.00865 | 0.02075 | 2.4440 |
| `base_semantic_supervised_0p1` | 4.2921 | 0.01226 | 0.00843 | 0.01992 | 2.4010 |
| `base_semantic_supervised_0p05` | 4.2931 | 0.01383 | 0.00833 | 0.01950 | 2.3543 |
| `current_best` | 4.3581 | 0.01437 | 0.00000 | 0.00000 | 3.5835 |

Conclusion:

- `current_best` has raw activation/category signal, but selected routes have zero category MI because BASE scheduling ignores category.
- Hard `base_semantic` makes selected routes category-dependent.
- The MI objective beats fixed supervised category-to-program targets on selected-route MI/NMI.
- Best objective candidate: `base_semantic_mi_0p5`.

### Category-Conditioned Knockout Matrix

Artifact:

`runs/benchmarks/program_specialization_knockout_full_2026_06_01/objectives.json`

Settings:

- variants: `current_best`, `base_semantic`, `base_semantic_mi_0p5`
- seeds: 11, 23, 37
- all-program category-conditioned knockouts
- 120 steps, batch size 8
- 96 train records/category, 12 eval records/category

| Variant | Eval loss | Selected NMI | Selected MI | Top knockout selectivity span | Raw NMI |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_semantic_mi_0p5` | 4.2941 | 0.01024 | 0.02392 | 0.02713 | 0.01367 |
| `base_semantic` | 4.2943 | 0.00961 | 0.02241 | 0.02546 | 0.01380 |
| `current_best` | 4.3624 | 0.00000 | 0.00000 | 0.00268 | 0.01438 |

Conclusion:

- `base_semantic_mi_0p5` produces about 10x higher category-conditioned knockout selectivity than `current_best`.
- This is functional specialization evidence: disabling a program affects categories differently.
- The exact program/category labels are not stable across seeds yet, so the valid claim is run-local functional selectivity, not "program 18 always means repair."

### Rejected Soft Routing Branch

Artifact:

`runs/benchmarks/program_specialization_soft_knockout_2026_06_01/objectives.json`

| Variant | Eval loss | Selected NMI | Knockout span | Raw NMI |
| --- | ---: | ---: | ---: | ---: |
| `base_semantic_mi_0p5` | 4.2941 | 0.010236 | 0.027125 | 0.01367 |
| `base_semantic_soft` | 4.3314 | 0.000370 | 0.002813 | 0.01632 |
| `base_semantic_soft_mi_0p5` | 4.3314 | 0.000458 | 0.002813 | 0.01579 |
| `current_best` | 4.3624 | 0.000000 | 0.002675 | 0.01438 |

Conclusion:

- Soft differentiable semantic routing does not solve specialization.
- It raises raw activation NMI but collapses selected-route category dependence and knockout selectivity back to baseline.
- Keep `base_semantic_soft` only as an ablation.

## Working Solution

Use the content-memory architecture with hard semantic routing and MI pressure:

```text
routing_type = "base_semantic"
routing_top_k = 2
routing_load_balance_weight = 0.05
category_route_weight = 0.5
category_route_objective = "mi"
```

For `kaggle/train_best_tac_agentic.py`, the corresponding flags are:

```text
--routing-type base_semantic
--routing-top-k 2
--routing-load-balance-weight 0.05
--category-route-weight 0.5
--category-route-objective mi
```

## Run 3 Targeted Routing Policy

The full forced-program matrix added a stronger checkpoint-specific solution: use the programs that were empirically useful in Run 3 and keep P0/P31 out of semantic extras.

Implemented knobs:

```text
semantic_route_allowed_programs = (8, 14, 16, 18, 22, 24)
semantic_route_suppressed_programs = (0, 31)
```

The corresponding Run 3 continuation/eval flags are:

```text
--routing-type base_semantic
--routing-top-k 3
--semantic-route-allowed-programs 8 14 16 18 22 24
--semantic-route-suppressed-programs 0 31
```

Full all-record policy eval:

- Artifact: `runs/kaggle_results/tac_targeted_routing_policy_eval/targeted_routing_policy_eval/targeted_routing_policy_eval_summary.json`
- Records: 5,441 prepared eval records
- Device: CUDA/T4
- Runtime: 317.17 seconds

| Policy | Loss | Delta vs Run 3 BASE | Token accuracy |
| --- | ---: | ---: | ---: |
| `base_semantic_k3_useful_family_suppress_p0_p31` | 0.2681 | -0.0100 | 0.9231 |
| `base_semantic_k2_useful_family` | 0.2702 | -0.0079 | 0.9223 |
| `base_semantic_k2_useful_family_suppress_p0_p31` | 0.2702 | -0.0079 | 0.9223 |
| `base_semantic_k2_suppress_p0_p31` | 0.2741 | -0.0040 | 0.9222 |
| `base_semantic_k2_all` | 0.2778 | -0.0003 | 0.9205 |
| `run3_base` | 0.2781 | 0.0000 | 0.9196 |

Category deltas for the winning policy were negative for all 13 categories. The largest gains were:

- `rag`: -0.0574
- `arc_reasoning`: -0.0569
- `coding`: -0.0501
- `failure_recovery`: -0.0447
- `filesystem`: -0.0427

Decision:

- For the trained Run 3 checkpoint, the best working routing solution is targeted hard semantic routing with top-k 3 over the useful family and P0/P31 suppression.
- For fresh random-seed training, keep the MI objective as the general solution because fixed program IDs are not proven stable across seeds.

## Promotion Decision

For pure harder memory benchmarks:

- Keep `current_best` as the control/default until a longer full training run confirms no carry regression.
- The full matrix still has `current_best` slightly ahead on aggregate carry: 0.5099 vs 0.5029 for `content_synthesis_semantic_k2`.

For specialization research and the next agentic training run:

- Use `base_semantic_mi_0p5`.
- It is the best validated solution because it improves selected-route MI and produces category-conditioned knockout selectivity.

Rejected solutions:

- pure semantic routing without the content-memory stack
- fixed supervised category-to-program targets as the primary objective
- soft semantic routing
- adding generic diversity pressure without category-conditioned routing

## Remaining Limitation

The current evidence proves functional selectivity within trained runs, not stable global program names across random seeds.

The next milestone is Run 4: train `base_semantic_mi_0p5` for 20,000 steps with
periodic specialization snapshots at 2k, 5k, 10k, and 20k. Those snapshots are
lightweight token-level attribution passes, while the final checkpoint still
runs the fuller end-of-run specialization analysis.

Run 4 launch configuration:

```bash
--routing-type base_semantic
--routing-top-k 2
--routing-load-balance-weight 0.05
--category-route-weight 0.5
--category-route-objective mi
--specialization-checkpoints 2000 5000 10000 20000
--specialization-checkpoint-max-records-per-category 16
--analyze-specialization-at-end
--specialization-max-records-per-category 64
--specialization-device cpu
--skip-end-specialization-on-time-stop
```

This fresh-training run must not use the Run 3 fixed useful-family IDs as
training constraints. The Run 3 patch is checkpoint-specific; Run 4 tests
whether the semantic objective learns useful routing under fresh program IDs.
The skip flag preserves the end-of-run analysis for true target-step completion
while avoiding repeated multi-hour specialization passes on intermediate Kaggle
time stops.

After Run 4, verify:

- selected-route MI remains above BASE
- category-conditioned knockout selectivity persists
- per-seed program/category roles either stabilize or can be aligned post hoc
- harder memory carry stays close to `current_best`

## Post-Run-4 Correction: Routing Pressure Phase Diagram

Date: 2026-06-03

Run 4 completed the intended semantic-MI diagnostic, but it did not promote the
`base_semantic_mi_0p5` training recipe.

The Run 3/Run 4 comparison is close to a controlled experiment. The base-scale
TAC backbone, identity-first attention, content-addressed read, synthesis gate,
gated residual memory adapter, novelty-gated writes, and anti-collapse losses
were effectively unchanged. The major change was routing pressure:

| Area | Run 3 | Run 4 |
| --- | --- | --- |
| Router | `base` | `base_semantic` |
| Top-k | `1` | `2` |
| Load balance | none explicit | `0.05` |
| Category objective | none | `mi` |
| Category weight | `0.0` | `0.5` |

Observed outcome:

| Run | Best/eval loss | Eval accuracy | Program-memory cosine | Selected-route signal |
| --- | ---: | ---: | ---: | ---: |
| Run 3 | `0.1645` best eval loss | about `0.93` in late metrics | about `0.56` in late metrics | BASE selected routes mostly position-driven |
| Run 4 | `6.4215` best eval loss | `0.0023` | about `0.963` | nonzero selected-route specialization |

Interpretation:

Run 4 most likely optimized cheap category-to-program assignment before useful
program computation formed. The router specialized, but program memories became
more similar and next-token capability collapsed. This weakens the broad
"identity capacity starvation" explanation because Run 3 already showed the
same core TAC stack can learn the task.

Caveat:

Run 4 late resumed logs also show optimization-health trouble under fp16
(`gradient_norm=0.0` and scaler at `0.0`), so the result should be treated as
semantic-routing pressure plus possible fp16/resume instability, not a single
cause proof.

Implemented next step:

- `build_routing_pressure_phase_variants`
- `aggregate_routing_pressure_phase_results`
- `run_routing_pressure_phase_matrix`
- `experiments/benchmark_routing_pressure_phase.py`

The new gate order is:

1. preserve next-token capability versus the BASE control,
2. keep program-memory cosine below the health ceiling,
3. only then reward selected-route MI.

Local CPU smoke artifacts:

- `runs/benchmarks/routing_pressure_phase_smoke_2026_06_03`
- `runs/benchmarks/routing_pressure_phase_longer_local_2026_06_03`

The longer local proxy compared `0.0`, `0.01`, `0.05`, `0.1`, and `0.5` MI
weights for one seed. It blocked all rows because even the BASE control had
collapsed program memory:

| Variant | Loss | Accuracy | Selected MI | Program-memory cosine |
| --- | ---: | ---: | ---: | ---: |
| `tac_base_run3_control` | `4.3797` | `0.2331` | `0.0000` | `0.9630` |
| `tac_semantic_mi_w0p01` | `4.2772` | `0.2461` | `0.0206` | `0.9520` |
| `tac_semantic_mi_w0p05` | `4.2769` | `0.2461` | `0.0207` | `0.9519` |
| `tac_semantic_mi_w0p1` | `4.2772` | `0.2448` | `0.0198` | `0.9516` |
| `tac_semantic_mi_w0p5` | `4.2744` | `0.2487` | `0.0213` | `0.9491` |

Decision:

The local CPU proxy validates the phase-diagram artifact path, but it does not
reproduce Run 3's differentiated-memory regime and therefore cannot decide the
true threshold. The next external run should execute the same phase grid at
Run-3-compatible scale and precision controls, with fp32 or healthy AMP scaler
checks, before launching another 20k semantic-MI candidate.

## Run 5B Local Correction: Capability Was Hidden by Short-Run Warmup

Date: 2026-06-03

The first 300-step local Run 5B comparisons made TAC look capability-blocked:

| Local row | Warmup | Best eval loss | Eval accuracy | Program-memory cosine |
| --- | ---: | ---: | ---: | ---: |
| Same-backbone vanilla | 15 steps | `3.4440` | `0.2240` | n/a |
| Parameter-matched vanilla | 15 steps | `2.9288` | `0.2474` | n/a |
| Program-conditioned CREB k6 TAC | 2000 steps | `5.9358` | `0.0378` | `0.2523` |

That was an unfair short-budget diagnostic. The TAC preset inherited the
20,000-step Run 5B warmup of 2,000 steps, so a 300-step local smoke never left
warmup and only reached `3e-05` learning rate by the final step. The vanilla
baseline used `steps // 20`, or 15 warmup steps, and reached the intended cosine
schedule.

After matching the short-run warmup, the same Run 5B defaults plus
program-conditioned CREB top-6 recovered capability:

| Local row | Key change | Best eval loss | Eval accuracy | Program-memory cosine |
| --- | --- | ---: | ---: | ---: |
| Identity disconnected control | No identity residual/coherence/aux/category gradients, warmup 15 | `3.5903` | `0.1745` | `0.1541` |
| Full TAC, matched beta/program-count | Program-conditioned CREB k6, warmup 15 | `3.4924` | `0.2135` | `0.1218` |
| Full TAC, Run 5B defaults | Program-conditioned CREB k6, warmup 15 | `3.4619` | `0.2070` | `0.1973` |

Interpretation:

- The local capability collapse was mostly a schedule artifact, not proof that
  semantic routing or program-conditioned memory is intrinsically broken.
- Program-conditioned candidate-memory plus CREB top-6 allocation remains the
  best memory-health fix: it keeps program-memory cosine far below the collapsed
  `0.95+` regime while preserving near-vanilla next-token capability locally.
- The remaining external question is whether this survives the true 20k Run 5B
  budget against the completed same-backbone vanilla baseline and the pending
  parameter-matched baseline.

Implemented controls:

- `--program-residual-scale` to ablate routed program-context residuals.
- `--coherence-attention-scale` to ablate identity coherence bias in attention.
- `--warmup-ratio` to derive short-diagnostic warmup from the run budget.
- run-manifest fields for `learning_rate`, `warmup_steps`, `warmup_ratio`,
  `effective_warmup_fraction`, and `min_lr_ratio`.

External baseline status:

- Same-backbone vanilla 20k completed on Kaggle with best eval loss `0.1092`
  and latest eval accuracy `0.9530`.
- Parameter-matched vanilla 20k completed on Kaggle with best eval loss
  `0.1048`, latest eval loss `0.1258`, and latest eval accuracy `0.9549`.
  This confirms the corpus and parameter-matched training budget are learnable
  outside TAC.

Prepared external TAC candidate:

```text
--preset run5b_capability
--scale base
--steps 20000
--batch-size 12
--grad-accum-steps 3
--program-memory-update-type program_conditioned
--memory-allocation-type creb
--memory-allocation-k 6
--memory-separation-weight 0.1
```

The staged Kaggle kernel is
`jeffkolo/tac-run5b-program-conditioned-creb-k6-20k`.

External TAC validation launch:

- Kernel version 1 was pushed on 2026-06-03 and entered
  `KernelWorkerStatus.RUNNING`.
- Promotion target: preserve next-token capability near the same-backbone
  vanilla baseline while keeping program-memory cosine out of the collapsed
  `0.95+` regime and producing usable route-specialization artifacts.
- Added `aggregate_external_run5b_validation` and
  `experiments/evaluate_external_run5b_validation.py` so the completed TAC
  artifacts are judged against the two vanilla summaries with explicit gates
  for completion, optimizer health, loss gap, memory cosine, MI, and knockout
  evidence.

## External Run 5B Program-Conditioned CREB Decision

Date: 2026-06-04

The external program-conditioned CREB top-6 TAC run did not finish the requested
20,000 optimizer steps before the Kaggle time limit, but it reached 11,021
steps and saw `202,345,560` training tokens. The important fair comparison point
is step 10,000: that checkpoint saw `183,600,000` tokens, exactly matching the
completed same-backbone and parameter-matched vanilla baselines.

At that fair-token checkpoint, TAC preserves language capability, keeps program
memory differentiated, and produces functional specialization evidence:

| Row | Best/eval loss | Eval accuracy | Program-memory cosine | Selected MI | Max knockout delta | Max knockout span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Same-backbone vanilla 20k | `0.1092` | `0.9530` | n/a | n/a | n/a | n/a |
| Parameter-matched vanilla 20k | `0.1048` | `0.9549` | n/a | n/a | n/a | n/a |
| TAC step 10k fair-token checkpoint | `0.1672` | `0.9414` | `0.0031` | `0.2821` | `0.3401` | `0.1868` |

Decision:

- Promote the Run 5B program-conditioned CREB top-6 configuration as the current
  solution candidate for "capability plus functional program specialization":
  `program_memory_update_type=program_conditioned`,
  `memory_allocation_type=creb`, `memory_allocation_k=6`,
  `memory_separation_weight=0.1`, `routing_type=base_semantic`,
  `routing_top_k=2`, `category_route_objective=mi`, and
  `category_route_weight=0.1`.
- Use fair-token checkpoint validation for this run. The final 20k-run summary
  still rejects only because Kaggle stopped the job before the requested target;
  the specialization and capability evidence gaps are closed at step 10k.
- Do not promote the step-6000 `best.pt` checkpoint for specialization. It has
  better eval loss (`0.1451`) but sequence-level top-program MI is `0.0`, so it
  is a capability checkpoint, not the specialization checkpoint.
- The staged `--warmup-ratio 0.05` fallback is no longer the next required
  architecture action. Keep it as a schedule-confirmation rerun only if we need
  a full 20k non-time-stopped artifact.

Key artifacts:

- `runs/benchmarks/external_run5b_program_conditioned_creb_k6_step10000_fair_token_validation_2026_06_04/RESULTS.md`
- `runs/analysis/tac_run5b_program_conditioned_creb_k6_step10000_specialization_2026_06_04/program_specialization.json`
- `runs/kaggle_results/tac_run5b_program_conditioned_creb_k6_20k_2026_06_04/best_tac_agentic_run5b_program_conditioned/specialization_checkpoints/step_010000/checkpoint.pt`

## Final Interpretation

There was no specialization in the observed BASE selected programs because BASE routing was position/schedule-driven and the analyzer looked at the wrong signal. The solution is not more generic program diversity. The solution is:

1. observe token-level routing,
2. preserve the working content-memory path,
3. add hard semantic program selection,
4. train it with a category-program MI objective,
5. validate with category-conditioned knockouts.

This is the first local evidence in this branch for functional program specialization rather than only representation differentiation.

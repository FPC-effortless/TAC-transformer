# Harder Research Matrix

Date: 2026-05-29

This closes the validation gap from `research.md`: every implemented research mechanism was run on the harder chunked-memory task set instead of only the original single-key task.

## Scope

Tasks:

- `longer_single_key`
- `multi_key`
- `delayed_query`
- `noisy_key`
- `multi_hop`

Each candidate was run for seeds `11`, `23`, and `37`, with 120 training steps and 8 evaluation batches.

Total benchmark runs: `25 variants * 5 tasks * 3 seeds = 375`.

Artifacts:

- `runs/benchmarks/harder_research_matrix_2026_05_29/per_seed_harder_research_matrix.json`
- `runs/benchmarks/harder_research_matrix_2026_05_29/aggregate_harder_research_matrix.json`
- `runs/benchmarks/harder_research_matrix_2026_05_29/RESULTS.md`

## Overall Result

| Rank | Variant | Effective | Task wins | Mean carry | Carry-reset | Gap | TPS ratio | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `base_routing` | 15/15 | 3 | 0.0677 | 0.0529 | 0.0505 | 0.5418 | Promote as new harder-task default candidate |
| 2 | `identity_compressed_attention` | 15/15 | 0 | 0.0628 | 0.0503 | 0.0456 | 0.5452 | Keep serious long-context candidate |
| 3 | `creb_match_k1` | 14/15 | 0 | 0.0612 | 0.0487 | 0.0440 | 0.5372 | Keep experimental only; dead-program rate too high |
| 4 | `separation_0p1` | 14/15 | 0 | 0.0607 | 0.0471 | 0.0435 | 0.5472 | Keep as robustness auxiliary |
| 5 | `current_best` | 14/15 | 0 | 0.0604 | 0.0466 | 0.0432 | 0.5506 | Demote from harder-task default |
| 6 | `hash_sparse_expert` | 14/15 | 0 | 0.0604 | 0.0466 | 0.0432 | 0.3611 | Reject until sparse kernels improve |
| 7 | `hierarchical_memory` | 14/15 | 0 | 0.0604 | 0.0466 | 0.0432 | 0.5151 | Keep as ablation, not default |
| 8 | `expert_choice_routing` | 13/15 | 1 | 0.0602 | 0.0469 | 0.0430 | 0.5334 | Useful on noisy-key, not overall |
| 9 | `sink_program_2` | 13/15 | 0 | 0.0602 | 0.0461 | 0.0430 | 0.5483 | Keep as ablation |
| 10 | `energy_reference` | 14/15 | 0 | 0.0599 | 0.0453 | 0.0427 | 0.5764 | Not promoted |

## Task Winners

| Task | Best carry variant | Carry | Carry-reset | Note |
| --- | --- | ---: | ---: | --- |
| `longer_single_key` | `base_routing` | 0.0677 | 0.0560 | BASE routing handles longer noisy context best. |
| `multi_key` | `state_only` | 0.0560 | 0.0378 | State-only unexpectedly wins interference-heavy recall. |
| `delayed_query` | `base_routing` | 0.0885 | 0.0755 | BASE routing clearly beats current-best here. |
| `noisy_key` | `base_routing` / `expert_choice_routing` | 0.0885 | 0.0794 / 0.0807 | Learned/balanced routing helps corrupted cues. |
| `multi_hop` | `mamba_selective_state` / `rwkv_time_mix` | 0.0651 | 0.0508 / 0.0521 | Recurrent mixers beat attention TAC on chained recall. |

## Decisions By Research Area

### Promote

`base_routing` is now the default harder-task preset. It beat the old `current_best` hash preset overall:

```text
0.0073 mean carry
+0.0063 carry-reset delta
+0.0073 TAC-baseline gap
15/15 effective runs
3/5 task wins in the original full matrix, and 4/5 in the focused CREB-load follow-up
```

It is slightly slower than `current_best` in the original run (`0.5418` vs `0.5506` TPS ratio), but the quality gain justified promotion into `best_tac_config`.

### Keep As Serious Candidates

`identity_compressed_attention` is now more credible than it looked on the easy task. It ranked second overall, with `15/15` effective runs and better carry/delta than `current_best`.

`separation_0p1` also improved under harder validation, especially delayed/noisy query. It should be treated as a robustness auxiliary, not just a diagnostic.

`mamba_selective_state` and `rwkv_time_mix` are not overall defaults, but they won the multi-hop task family. That means recurrent/state mixers should be part of the next HRM/refinement track for chained reasoning.

### Do Not Promote

`creb_match_k1` remains close to the top but still has a dead-program rate around `0.8902`. It is not clean enough for default use.

`hash_sparse_expert` ties `current_best` quality but is much slower in the current unfused implementation.

`all_features_stack` remains bad engineering: lower mean carry, very low throughput, and high dead-program rate.

`dual_stream_residual`, `reconsolidate_mlp`, `product_key_memory`, and sink programs did not justify promotion.

## Updated Best Architecture

The best tested harder-task TAC candidate is now:

```python
best_tac_config(
    ...,
    routing_type="base",
)
```

That means the working architecture direction changes from:

```text
hash-routed TAC as default
```

to:

```text
BASE-routed TAC for harder memory tasks
+ identity-compressed attention as the next long-context candidate
+ recurrent mixer branch for multi-hop reasoning
```

The current production-safe preset can stay unchanged until this is promoted into a named preset and re-run on larger budgets, but the research answer has changed: **BASE routing is the best validated harder-task architecture so far.**

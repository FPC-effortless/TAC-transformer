# Sparse Ensemble Routing Validation

Date: 2026-05-29

## Question

After BASE routing became the harder-task default, Phase 2A changed from:

```text
Does k-sparse routing beat hash k=1?
```

to:

```text
Does k-sparse routing beat BASE single-program routing on the harder tasks
where pattern-addressable memory should matter most?
```

This experiment focused on:

- `multi_key`
- `noisy_key`

## Implementation

Added `routing_type="sparse_ensemble"` with `routing_top_k`.

The route is BASE-anchored:

1. Select the BASE balanced program for each flattened token row.
2. Add the strongest content-matched programs until `routing_top_k`.
3. Trim to the route-energy budget.

This makes `routing_top_k=1` equivalent to BASE and tests whether additional engram-style pattern slots improve recall.

## Matrix

Variants:

| Variant | Notes |
| --- | --- |
| `hash_current_best` | old hash-routed control |
| `base_routing` | current harder-task default |
| `sparse_ensemble_k2` | BASE anchor + 1 matched program |
| `sparse_ensemble_k3` | BASE anchor + 2 matched programs |
| `sparse_ensemble_k4` | BASE anchor + 3 matched programs |

Tasks: `multi_key`, `noisy_key`

Seeds: `11`, `23`, `37`

Total: `5 variants x 2 tasks x 3 seeds = 30 runs`

## Overall Results

| Rank | Variant | Effective | Mean carry | Carry-reset | Gap | TPS ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `base_routing` | 6/6 | 0.0664 | 0.0547 | 0.0521 | 0.5123 |
| 2 | `sparse_ensemble_k2` | 5/6 | 0.0645 | 0.0534 | 0.0501 | 0.4817 |
| 3 | `hash_current_best` | 5/6 | 0.0632 | 0.0521 | 0.0488 | 0.5488 |
| 4 | `sparse_ensemble_k4` | 6/6 | 0.0625 | 0.0521 | 0.0482 | 0.5013 |
| 5 | `sparse_ensemble_k3` | 5/6 | 0.0618 | 0.0501 | 0.0475 | 0.4899 |

## Task-Level Notes

| Task | Winner | Carry | Note |
| --- | --- | ---: | --- |
| `multi_key` | `sparse_ensemble_k4` | 0.0456 | Slightly beats BASE, but by only `0.0013`. |
| `noisy_key` | `sparse_ensemble_k2` | 0.0898 | Slightly beats BASE by `0.0013`, but has lower shuffled-state margin. |

## Decision

Do not promote sparse ensemble routing yet.

The result is not a hard failure: sparse ensembles showed the expected local behavior, with k=4 best on multi-key and k=2 best on noisy-key. But the aggregate does not beat BASE:

- BASE has the best mean carry across the focused gate.
- BASE is the only top candidate with `6/6` effective and the best carry-shuffled margin.
- Sparse ensembles are slower in this unfused implementation.
- Larger k appears to add interference on noisy-key.

## Next Interpretation

The likely issue is that this implementation adds extra routed compute paths, but it does not yet store and retrieve explicit sparse engram patterns. It is sparse ensemble routing, not full pattern completion.

The next worthwhile Phase 2 step is therefore not "increase k"; it is Phase 2B:

```text
content-addressable pattern completion over stored sparse route patterns
```

Artifacts:

- `runs/benchmarks/sparse_ensemble_focused_2026_05_29/RESULTS.md`
- `runs/benchmarks/sparse_ensemble_focused_2026_05_29/aggregate_harder_research_matrix.json`

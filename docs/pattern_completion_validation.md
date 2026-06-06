# Pattern Completion Validation

Date: 2026-05-29

## Question

Sparse ensemble routing added multiple active program paths, but did not beat BASE routing overall. The next Phase 2B question was whether explicit content-addressable pattern completion helps:

```text
store route pattern + memory value during context
retrieve closest stored pattern during query
```

## Implementation

Added:

- `memory_read_type="pattern_completion"`
- `pattern_store_size`
- carried `engram_patterns`
- carried `engram_values`
- carried `engram_mask`
- `pattern_completion_hit` metric

The implementation stores a small rolling engram bank per identity layer. Each entry contains:

```text
route pattern: [n_programs]
memory value:  [d_model]
mask:          valid entry flag
```

During query, TAC compares the current route pattern against stored patterns and retrieves a soft nearest-neighbor memory value.

## Matrix

Variants:

| Variant | Notes |
| --- | --- |
| `base_routing` | current harder-task default |
| `sparse_ensemble_k2` | best noisy-key sparse routing control |
| `sparse_ensemble_k4` | best multi-key sparse routing control |
| `pattern_completion_k2` | k=2 sparse route plus pattern store |
| `pattern_completion_k4` | k=4 sparse route plus pattern store |

Tasks: `multi_key`, `noisy_key`

Seeds: `11`, `23`, `37`

Total: `5 variants x 2 tasks x 3 seeds = 30 runs`

## Overall Results

| Rank | Variant | Effective | Mean carry | Carry-reset | Gap | TPS ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `base_routing` | 6/6 | 0.0664 | 0.0547 | 0.0521 | 0.5447 |
| 2 | `sparse_ensemble_k2` | 5/6 | 0.0645 | 0.0534 | 0.0501 | 0.5106 |
| 3 | `pattern_completion_k2` | 5/6 | 0.0638 | 0.0527 | 0.0495 | 0.4719 |
| 4 | `sparse_ensemble_k4` | 6/6 | 0.0625 | 0.0521 | 0.0482 | 0.5058 |
| 5 | `pattern_completion_k4` | 6/6 | 0.0605 | 0.0495 | 0.0462 | 0.4838 |

## Task-Level Notes

| Task | Best Candidate | Carry | Pattern Completion Result |
| --- | --- | ---: | --- |
| `multi_key` | `sparse_ensemble_k4` | 0.0456 | `pattern_completion_k4` fell to 0.0430 |
| `noisy_key` | `sparse_ensemble_k2` | 0.0898 | `pattern_completion_k2` fell to 0.0872 |

## Decision

Do not promote pattern completion yet.

The mechanism works mechanically and changes query logits from carried engram state, but this first version does not improve the focused harder tasks:

- It is below BASE overall.
- It is below the matching sparse-ensemble control for both k=2 and k=4.
- It is slower than BASE and sparse routing.
- It does not improve carry-shuffled separation.

## Interpretation

The likely issue is that the stored pattern is still too route-centric. It stores the program route pattern, but not a supervised key/value binding or a query-aware cue vector. That means the nearest-neighbor lookup can retrieve a plausible route memory without being semantically tied to the corrupted key.

The next pattern-completion attempt should store explicit cue/value tuples:

```text
cue = projected key/query hidden state
value = memory value needed for recall
pattern = route pattern metadata
```

Then retrieval should score both cue similarity and route-pattern similarity. The current result says "route pattern alone is not enough."

Artifacts:

- `runs/benchmarks/pattern_completion_focused_2026_05_29/RESULTS.md`
- `runs/benchmarks/pattern_completion_focused_2026_05_29/aggregate_harder_research_matrix.json`

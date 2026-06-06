# CREB Load-Balancing Validation

Date: 2026-05-29

## Question

The harder research matrix showed that `creb_match_k1` was close on mean carry but had an unacceptable dead-program rate around `0.8902`. This experiment tested whether a mild running write-frequency penalty could fix that failure mode:

```text
write_score =
  alpha * (1 - stability)
+ beta  * activation
- gamma * program_age
- delta * write_frequency
```

## Implementation

Added:

- `creb_delta`
- `creb_frequency_decay`
- carried `program_write_frequency` in `IdentityState`
- `memory_allocation_write_frequency` metric

The new state is carried through normal forward calls and shuffled with the rest of the identity state during intervention probes.

## Matrix

The focused harder matrix compared:

| Variant | Notes |
| --- | --- |
| `current_best` | hash-routed control |
| `base_routing` | harder-matrix winner |
| `creb_match_k1` | old CREB k=1 control |
| `creb_load_k1_d0p5` | CREB k=1 with `creb_delta=0.5` |
| `creb_load_k1_d1p0` | CREB k=1 with `creb_delta=1.0` |
| `creb_match_k3` | old CREB k=3 control |
| `creb_load_k3_d0p5` | CREB k=3 with `creb_delta=0.5` |

Tasks:

- longer single-key
- multi-key
- delayed-query
- noisy-key
- multi-hop

Seeds: `11`, `23`, `37`

Total: `7 variants x 5 tasks x 3 seeds = 105 runs`

## Overall Results

| Rank | Variant | Effective | Task wins | Mean carry | Carry-reset | Gap | TPS ratio | Dead rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `base_routing` | 15/15 | 4 | 0.0677 | 0.0529 | 0.0505 | 0.5688 | 0.0000 |
| 2 | `creb_match_k1` | 14/15 | 1 | 0.0612 | 0.0487 | 0.0440 | 0.5333 | 0.8902 |
| 3 | `creb_load_k1_d0p5` | 14/15 | 1 | 0.0612 | 0.0487 | 0.0440 | 0.5539 | 0.8879 |
| 4 | `creb_load_k1_d1p0` | 14/15 | 1 | 0.0612 | 0.0487 | 0.0440 | 0.5269 | 0.8857 |
| 5 | `current_best` | 14/15 | 0 | 0.0604 | 0.0466 | 0.0432 | 0.5518 | 0.0000 |
| 6 | `creb_match_k3` | 14/15 | 0 | 0.0596 | 0.0458 | 0.0424 | 0.5279 | 0.7049 |
| 7 | `creb_load_k3_d0p5` | 14/15 | 0 | 0.0596 | 0.0458 | 0.0424 | 0.5481 | 0.6974 |

## Decision

Do not promote CREB load balancing.

The load penalty works mechanically, but the tested values only reduce dead-program rate by a tiny amount:

- k=1: `0.8902 -> 0.8857`
- k=3: `0.7049 -> 0.6974`

It does not improve mean carry, carry-reset delta, effective-run count, or task wins. `base_routing` remains the best validated harder-task architecture.

## Follow-Up

The useful result is negative: CREB's dead-program problem is not solved by a mild write-frequency penalty. A future CREB attempt should use a stronger allocation change, such as explicit capacity constraints, expert-choice-style write assignment, or entropy/load loss on write targets.

Artifacts:

- `runs/benchmarks/creb_load_harder_matrix_2026_05_29/RESULTS.md`
- `runs/benchmarks/creb_load_harder_matrix_2026_05_29/aggregate_harder_research_matrix.json`

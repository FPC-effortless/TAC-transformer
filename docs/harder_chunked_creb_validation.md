# Harder Chunked-Memory CREB Validation

Date: 2026-05-29

This validation was added because the Phase 1 CREB result was only tested on the original single-key chunked recall task. The harder task variants are now implemented directly in `ChunkedRecallBatcher` and exposed through `--task-variant`.

## Variants

| Variant | Purpose |
| --- | --- |
| `longer_single_key` | Same task at `seq_len=24`, adding more context/query noise. |
| `multi_key` | Multiple key-value pairs in context; query one target key. |
| `delayed_query` | Distractor tokens appear between the query key and recall marker. |
| `noisy_key` | Query uses an adjacent/corrupted key token. |
| `multi_hop` | Context contains `A -> B` and `B -> value`; query asks for `A -> value`. |

## Candidates

| Candidate | Config |
| --- | --- |
| `current_best` | Existing `best_tac_config` |
| `creb_match_k1` | `memory_allocation_type="creb"`, `memory_allocation_k=1`, `creb_alpha=0.5`, `creb_beta=2.0`, `creb_gamma=0.25` |
| `creb_match_k3` | Same CREB match bias, but write top-3 programs |

All runs used 120 train steps, seeds 11/23/37, and the same best-TAC training weights.

## Results By Task

| Task | Candidate | Effective | Carry | Carry-reset | TAC-baseline gap | Train TPS ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| delayed_query | current_best | 3/3 | 0.0794 | 0.0703 | 0.0664 | 0.5212 |
| delayed_query | creb_match_k1 | 3/3 | 0.0742 | 0.0625 | 0.0612 | 0.5692 |
| delayed_query | creb_match_k3 | 3/3 | 0.0781 | 0.0690 | 0.0651 | 0.5562 |
| longer_single_key | current_best | 3/3 | 0.0547 | 0.0430 | 0.0313 | 0.5325 |
| longer_single_key | creb_match_k1 | 3/3 | 0.0560 | 0.0469 | 0.0326 | 0.5835 |
| longer_single_key | creb_match_k3 | 3/3 | 0.0547 | 0.0430 | 0.0313 | 0.6050 |
| multi_key | current_best | 2/3 | 0.0430 | 0.0313 | 0.0286 | 0.5630 |
| multi_key | creb_match_k1 | 2/3 | 0.0469 | 0.0365 | 0.0326 | 0.5631 |
| multi_key | creb_match_k3 | 2/3 | 0.0430 | 0.0299 | 0.0286 | 0.5499 |
| noisy_key | current_best | 3/3 | 0.0833 | 0.0729 | 0.0690 | 0.5756 |
| noisy_key | creb_match_k1 | 3/3 | 0.0820 | 0.0729 | 0.0677 | 0.5441 |
| noisy_key | creb_match_k3 | 3/3 | 0.0820 | 0.0729 | 0.0677 | 0.5749 |
| multi_hop | current_best | 3/3 | 0.0417 | 0.0156 | 0.0208 | 0.5934 |
| multi_hop | creb_match_k1 | 3/3 | 0.0469 | 0.0247 | 0.0260 | 0.5731 |
| multi_hop | creb_match_k3 | 3/3 | 0.0404 | 0.0143 | 0.0195 | 0.5526 |

## Overall

| Candidate | Effective | Task wins | Mean carry | Mean carry-reset | Mean gap | Mean train TPS ratio | Dead rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_best | 14/15 | 2 | 0.0604 | 0.0466 | 0.0432 | 0.5571 | 0.0000 |
| creb_match_k1 | 14/15 | 3 | 0.0612 | 0.0487 | 0.0440 | 0.5666 | 0.8902 |
| creb_match_k3 | 14/15 | 0 | 0.0596 | 0.0458 | 0.0424 | 0.5677 | 0.7049 |

## Decision

CREB `k=1` remains the best experimental accuracy candidate, but the harder-task validation does **not** justify promoting CREB into the default architecture yet. The overall advantage over `current_best` is small:

- mean carry: `0.0612` vs `0.0604`
- mean carry-reset: `0.0487` vs `0.0466`
- task wins: `3` vs `2`

The cost is high dead-program rate (`0.8902`) and a less clean memory allocation story. CREB `k=3` is healthier but does not outperform `current_best`.

Updated recommendation:

- Default architecture: keep `current_best`.
- Experimental branch: keep `creb_match_k1` for accuracy-first follow-up.
- Next architecture work: Phase 2 pattern completion and sparse engram retrieval. Multi-key and multi-hop are now the right gates for that work.

Artifacts:

- `runs/benchmarks/harder_chunked_creb_validation_2026_05_29/aggregate_by_task.json`
- `runs/benchmarks/harder_chunked_creb_validation_2026_05_29/aggregate_overall.json`

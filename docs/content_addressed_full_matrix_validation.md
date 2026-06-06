# Content-Addressed Full Harder-Matrix Validation

Date: 2026-05-30

## Question

The focused noisy-key validation showed that hidden-state cue/value memory was the first Phase 2 mechanism to beat BASE routing. This run tested whether that win generalizes across the full harder chunked-memory suite.

## Benchmark

Command:

```powershell
python kaggle\benchmark_harder_research_matrix.py --steps 120 --batch-size 32 --eval-batches 8 --eval-batch-size 32 --seeds 11 23 37 --variants base_routing content_addressed_k1 content_addressed_k2 --output-dir runs\benchmarks\content_addressed_full_matrix_2026_05_30 --force
```

Matrix:

- 5 tasks: longer single-key, multi-key, delayed-query, noisy-key, multi-hop
- 3 variants: BASE program-memory control, content-addressed k=1, content-addressed k=2
- 3 seeds: 11, 23, 37
- 45 trained/evaluated runs

## Overall Result

| Rank | Variant | Effective | Task wins | Mean carry | Carry-reset | Carry-shuffled | Gap vs vanilla | TPS ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `content_addressed_k1` | 15/15 | 2 | 0.4349 | 0.4229 | 0.4185 | 0.4177 | 0.4823 |
| 2 | `content_addressed_k2` | 14/15 | 2 | 0.4341 | 0.4216 | 0.4198 | 0.4169 | 0.4679 |
| 3 | `base_routing` | 15/15 | 1 | 0.0677 | 0.0529 | 0.0508 | 0.0505 | 0.5236 |

## Per-Task Result

| Task | BASE carry | Content k1 carry | Content k2 carry | Winner |
| --- | ---: | ---: | ---: | --- |
| longer single-key | 0.0677 | 0.6497 | 0.6510 | `content_addressed_k2` |
| multi-key | 0.0443 | 0.7448 | 0.7409 | `content_addressed_k1` |
| delayed-query | 0.0885 | 0.6341 | 0.6393 | `content_addressed_k2` |
| noisy-key | 0.0885 | 0.1094 | 0.1081 | `content_addressed_k1` |
| multi-hop | 0.0495 | 0.0365 | 0.0312 | `base_routing` |

## Decision

Promote `content_addressed_k1` as the best current TAC architecture for direct memory recall.

Reason:

- It has the highest overall mean carry.
- It remains effective on all 15 task/seed runs.
- It massively improves carry-reset and carry-shuffled deltas, showing that the carried memory content is doing real work.
- It wins the direct recall families where cue/value memory should work: longer context, multi-key, delayed query, and noisy key.

Do not promote content-addressed memory as the complete multi-hop solution.

Multi-hop remains the boundary condition. BASE program memory still wins there, which means chained retrieval needs a different mechanism: iterative read/repair, recurrent state, planner/verifier loop, or explicit multi-step memory querying.

## TPS Finding

The focused noisy-key run showed a high `content_addressed_k2` TPS ratio, but the full matrix did not confirm that as a general speed advantage.

Full-matrix TPS:

- BASE: `0.5236`
- content-addressed k1: `0.4823`
- content-addressed k2: `0.4679`

So content-addressed memory is much more data-effective on direct recall, but still slightly slower than BASE in the full suite. The economic case is capability/data efficiency, not raw wall-clock speed.

## Architecture Update

This was the promotion decision from the first content-addressed full matrix. It has since been superseded by the automated synthesis promotion in `docs/automated_research_synthesis_promotion.md`.

At this stage, `best_tac_config(...)` used:

```text
routing_type = "base"
memory_read_type = "content_addressed"
content_store_size = 8
```

The current preset additionally uses:

```text
content_read_steps = 2
content_read_gate_type = "synthesis"
```

The benchmark harness keeps legacy non-content variants pinned to `memory_read_type="program_memory"` so future matrix names remain accurate.

Artifacts:

- `runs/benchmarks/content_addressed_full_matrix_2026_05_30/RESULTS.md`
- `runs/benchmarks/content_addressed_full_matrix_2026_05_30/aggregate_harder_research_matrix.json`

# Content-Addressed Cue/Value Memory Validation

Date: 2026-05-30

## Question

The first Phase 2B implementation stored route patterns and tried to retrieve by route-pattern similarity. That did not work because route patterns describe internal routing state, not input content.

This follow-up tested the content-addressed version:

```text
context: store (hidden_state_of_key_token, hidden_state_of_value_token)
query:   retrieve value by cosine similarity(query_key_hidden, stored_key_hidden)
```

The hypothesis was narrow and directly testable: content-addressed memory should beat BASE routing on the noisy-key task, where partial/corrupted cue retrieval matters most.

## Implementation

Added `memory_read_type="content_addressed"` with `content_store_size`.

Identity state now carries:

- `content_cues`: hidden-state cue vectors
- `content_values`: hidden-state value vectors
- `content_mask`: valid-store mask

During context processing, adjacent hidden pairs are stored as cue/value tuples. During query processing, the current hidden state retrieves the closest stored cue and injects the corresponding value through the existing gated memory residual.

This is different from `pattern_completion`:

| Read type | Address | Retrieved content |
| --- | --- | --- |
| `pattern_completion` | route/program activation pattern | stored route-derived value |
| `content_addressed` | token hidden representation | stored value hidden representation |

## Benchmark

Command:

```powershell
python kaggle\benchmark_harder_research_matrix.py --steps 120 --batch-size 32 --eval-batches 8 --eval-batch-size 32 --seeds 11 23 37 --variants base_routing sparse_ensemble_k2 pattern_completion_k2 content_addressed_k1 content_addressed_k2 --tasks noisy_key --output-dir runs\benchmarks\content_addressed_noisy_key_2026_05_30 --force
```

## Results

| Rank | Variant | Effective | Mean carry | Carry-reset | Carry-shuffled | Gap vs vanilla | TPS ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `content_addressed_k1` | 3/3 | 0.1094 | 0.1042 | 0.0885 | 0.0951 | 0.4598 |
| 2 | `content_addressed_k2` | 3/3 | 0.1081 | 0.1029 | 0.0924 | 0.0938 | 0.7344 |
| 3 | `sparse_ensemble_k2` | 3/3 | 0.0898 | 0.0807 | 0.0703 | 0.0755 | 0.4484 |
| 4 | `base_routing` | 3/3 | 0.0885 | 0.0794 | 0.0729 | 0.0742 | 0.4437 |
| 5 | `pattern_completion_k2` | 3/3 | 0.0872 | 0.0781 | 0.0664 | 0.0729 | 0.3552 |

Per-seed carry accuracy:

| Variant | Seed 11 | Seed 23 | Seed 37 |
| --- | ---: | ---: | ---: |
| `base_routing` | 0.0859 | 0.0781 | 0.1016 |
| `content_addressed_k1` | 0.1367 | 0.0977 | 0.0938 |
| `content_addressed_k2` | 0.1406 | 0.0898 | 0.0938 |

## Decision

Content-addressed cue/value memory is validated for noisy-key recall.

It should not replace BASE routing as the universal default yet, because this run only targeted noisy-key. It should become the default noisy-key/partial-cue read candidate and must be included in the next full harder-task matrix.

Architecture recommendation:

- Keep `base_routing` as the general harder-task default.
- Add `memory_read_type="content_addressed"` as a task-conditional read mode.
- Prefer `content_addressed_k1` when optimizing noisy-key carry accuracy.
- Consider `content_addressed_k2` when throughput and lower shuffled leakage matter more; it was much faster in this run while only slightly behind k=1 carry.

## Interpretation

The engram idea was not wrong; the first address space was wrong. Route patterns are not semantic addresses. Hidden cue/value tuples are.

The result also explains why sparse ensemble alone did not move much: adding more program routes does not create semantic memory. The winning change was storing the cue and value representations directly, then retrieving by content similarity.

Artifacts:

- `runs/benchmarks/content_addressed_noisy_key_2026_05_30/RESULTS.md`
- `runs/benchmarks/content_addressed_noisy_key_2026_05_30/aggregate_harder_research_matrix.json`

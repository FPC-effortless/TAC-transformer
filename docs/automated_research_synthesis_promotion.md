# Automated Research Matrix: Synthesis Promotion

Date: 2026-05-30

## Question

After the narrow multi-hop synthesis test failed, the full automated screen checked whether synthesis was useful outside multi-hop. It ran every implemented harder-matrix variant first, then confirmed the leading candidates with the full 3-seed 120-step suite.

## Stage 1 Screen

Command:

```powershell
python kaggle\benchmark_harder_research_matrix.py --seeds 11 --steps 60 --batch-size 32 --eval-batches 8 --eval-batch-size 32 --output-dir runs\benchmarks\automated_research_stage1_screen_2026_05_30 --force
```

Scope:

- 42 implemented variants
- 5 harder tasks
- 1 seed
- 60 training steps
- 210 trained/evaluated runs

Top screen results:

| Rank | Variant | Effective | Task wins | Mean carry | Carry-reset | TPS ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `content_synthesis_k2` | 4/5 | 1 | 0.2297 | 0.2078 | 0.4144 |
| 2 | `current_best` | 4/5 | 1 | 0.2258 | 0.2125 | 0.4722 |
| 3 | `content_addressed_k1` | 4/5 | 1 | 0.2258 | 0.2125 | 0.4918 |
| 4 | `content_synthesis_k1` | 5/5 | 0 | 0.2258 | 0.2055 | 0.4461 |
| 5 | `content_addressed_k2` | 4/5 | 1 | 0.2203 | 0.2055 | 0.4705 |

Stage 1 selected the content-addressed family, synthesis variants, iterative variants, confidence-gated k1, BASE, and CREB k1 for confirmation.

## Stage 2 Confirmation

Command:

```powershell
python kaggle\benchmark_harder_research_matrix.py --variants content_synthesis_k2 current_best content_addressed_k1 content_synthesis_k1 content_iterative_k2 content_iterative_k1 content_confidence_iterative_k1 base_routing creb_match_k1 --seeds 11 23 37 --steps 120 --batch-size 32 --eval-batches 8 --eval-batch-size 32 --output-dir runs\benchmarks\automated_research_stage2_confirm_2026_05_30 --force
```

Scope:

- 9 selected variants
- 5 harder tasks
- 3 seeds
- 120 training steps
- 135 trained/evaluated runs

Overall result:

| Rank | Variant | Effective | Task wins | Mean carry | Carry-reset | Carry-shuffled | Gap vs vanilla | TPS ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `content_synthesis_k2` | 15/15 | 2 | 0.4909 | 0.4742 | 0.4773 | 0.4747 | 0.4336 |
| 2 | `content_synthesis_k1` | 15/15 | 1 | 0.4901 | 0.4737 | 0.4753 | 0.4740 | 0.4691 |
| 3 | `content_addressed_k1` | 15/15 | 0 | 0.4349 | 0.4229 | 0.4185 | 0.4177 | 0.5295 |
| 4 | `content_iterative_k1` | 15/15 | 0 | 0.3565 | 0.3422 | 0.3417 | 0.3393 | 0.4622 |
| 5 | `content_iterative_k2` | 15/15 | 1 | 0.3552 | 0.3406 | 0.3391 | 0.3380 | 0.4345 |
| 6 | `content_confidence_iterative_k1` | 15/15 | 1 | 0.3320 | 0.3190 | 0.3148 | 0.3148 | 0.4779 |
| 7 | `base_routing` | 15/15 | 0 | 0.0677 | 0.0529 | 0.0508 | 0.0505 | 0.5310 |
| 8 | `creb_match_k1` | 14/15 | 0 | 0.0654 | 0.0513 | 0.0469 | 0.0482 | 0.5323 |

## Per-Task Winners

| Task | Winner | Carry | Notes |
| --- | --- | ---: | --- |
| Longer single-key | `content_synthesis_k2` | 0.7669 | k1 is close at 0.7604; both beat content k1 at 0.6497. |
| Multi-key | `content_synthesis_k2` | 0.8411 | k1 is close at 0.8320; both beat content k1 at 0.7448. |
| Delayed-query | `content_synthesis_k1` | 0.7227 | k2 is close at 0.7174; both beat content k1 at 0.6341. |
| Noisy-key | `content_confidence_iterative_k1` | 0.1263 | content k1 is second at 0.1094; synthesis regresses here. |
| Multi-hop | `content_iterative_k2` | 0.0508 | Still only a narrow win over BASE at 0.0495. |

## Decision

Promote `content_synthesis_k1` as the preferred aggregate default.

Reason:

- It improves mean carry from `0.4349` to `0.4901` over the previous content-addressed k1 preset.
- It remains effective on all 15 confirmation runs.
- It is nearly tied with `content_synthesis_k2` on mean carry (`0.4901` vs `0.4909`) while being simpler and faster (`0.4691x` vs `0.4336x` train TPS ratio).
- It keeps BASE routing and single-program routing, avoiding the extra sparse-ensemble k2 complexity.

Do not treat synthesis as the complete reasoning solution:

- It regresses noisy-key relative to confidence-gated k1.
- It still fails the multi-hop boundary relative to iterative k2 and BASE.
- The architecture is now best understood as a mode family:
  - synthesis read for clean direct recall,
  - confidence/standard content read for noisy cue retrieval,
  - iterative k2 for multi-hop experiments,
  - future verifier/halt gate to choose modes instead of hardcoding them.

Artifacts:

- `runs/benchmarks/automated_research_stage1_screen_2026_05_30/RESULTS.md`
- `runs/benchmarks/automated_research_stage1_screen_2026_05_30/aggregate_harder_research_matrix.json`
- `runs/benchmarks/automated_research_stage2_confirm_2026_05_30/RESULTS.md`
- `runs/benchmarks/automated_research_stage2_confirm_2026_05_30/aggregate_harder_research_matrix.json`

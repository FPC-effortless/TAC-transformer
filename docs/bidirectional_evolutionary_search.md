# Bidirectional Evolutionary Search Results

Date: 2026-05-31

## What Was Added

Implemented `kaggle/benchmark_bidirectional_evolution.py` as a reproducible TAC research runner.

It supports two modes:

- launch new candidate evaluations over the harder chunked-memory task matrix;
- re-score existing per-seed benchmark matrices without retraining.

The scorer treats TAC search as bidirectional:

- forward fitness: carry accuracy, carry-vs-reset delta, and TAC-vs-baseline gap;
- backward fitness: behavioral novelty, program differentiation, gate conditionality, and live program allocation;
- constraints: gate saturation, program collapse, and dead-program rate;
- outputs: survival ranking, Pareto front, MAP-Elites grid, per-seed JSON, aggregate JSON, and Markdown report.

## Local Smoke

Artifact:

- `runs/benchmarks/bidirectional_evolution_smoke_2026_05_31`

Command:

```powershell
python kaggle/benchmark_bidirectional_evolution.py --tasks multi_hop noisy_key --candidates current_best program_novelty_soft iterative_multi_hop_k2 mamba_program_memory --seeds 11 --steps 1 --batch-size 2 --eval-batches 1 --eval-batch-size 2 --output-dir runs/benchmarks/bidirectional_evolution_smoke_2026_05_31 --force --device cpu
```

Interpretation:

The smoke is only an integration proof. With one CPU training step, all carry accuracies are zero, so it should not be used as model evidence. It did catch and fix one real runner issue: candidates that disable the memory adapter now automatically zero `memory_adapter_weight`.

## Existing Matrix Reanalysis

### Attention and Identity Fusion

Input:

- `runs/benchmarks/attention_identity_fusion_full_matrix_2026_05_31/per_seed_harder_research_matrix.json`

Output:

- `runs/benchmarks/bidirectional_evolution_attention_fusion_reanalysis_2026_05_31`

Result:

- `identity_first_attention` remains the leader.
- Survival score: `2.5989`.
- Forward fitness: `1.4997`.
- Backward fitness: `1.0991`.
- Mean carry: `0.5099`.
- Effective runs: `15/15`.
- Task wins: `5/5`.

Interpretation:

The bidirectional scoring agrees with the prior promotion decision. Identity-first attention is not merely a raw carry winner; it stays on the Pareto front after novelty, differentiation, routing efficiency, and failure constraints are included.

### Synthesis and Iterative Retrieval

Input:

- `runs/benchmarks/automated_research_stage2_confirm_2026_05_30/per_seed_harder_research_matrix.json`

Output:

- `runs/benchmarks/bidirectional_evolution_synthesis_reanalysis_2026_05_31`

Result:

- `content_synthesis_k2` ranks first by survival score: `2.6294`.
- `content_synthesis_k1` ranks second: `2.6186`.
- `content_iterative_k2` remains a distinct MAP-Elites niche with lower carry but higher behavioral novelty than the single-step content paths.
- `creb_match_k1` is correctly penalized for dead programs despite high measured differentiation.

Interpretation:

This supports the current small-lab default decision: `content_synthesis_k1` is still the better practical default, while `content_synthesis_k2` is the accuracy/novelty leader when extra complexity and lower routing efficiency are acceptable. Multi-hop remains unsolved, but iterative k2 is the useful novelty seed to preserve rather than discard.

## Practical Next Use

When the current Kaggle run finishes, run the existing checkpoint specialization analyzer first:

```powershell
python kaggle/analyze_program_specialization.py --checkpoint <best.pt> --jsonl runs/prepared_corpus_agentic_hard/hard_agentic_eval.generated.jsonl --max-records-per-category 64 --output runs/analysis/kaggle_run_program_specialization/program_specialization.json --csv-output runs/analysis/kaggle_run_program_specialization/program_attribution.csv
```

Then run the harder checkpoint probe:

```powershell
python kaggle/evaluate_checkpoint_harder_matrix.py --checkpoint <best.pt> --output runs/analysis/kaggle_run_harder_matrix/checkpoint_harder_matrix.json
```

Finally, score any produced per-seed harder matrix with:

```powershell
python kaggle/benchmark_bidirectional_evolution.py --input-runs <per_seed_harder_research_matrix.json> --output-dir runs/benchmarks/bidirectional_evolution_kaggle_checkpoint
```

## Current Verdict

Bidirectional evolutionary scoring is useful immediately as a research controller and reanalysis layer. It does not yet prove open-ended program identity evolution, because current MAP-Elites coverage is still sparse and existing matrices mostly occupy low program-differentiation bins. The strongest immediate use is to preserve deceptive but promising multi-hop niches, especially iterative retrieval, while keeping the main architecture path anchored to identity-first synthesis TAC.

## Fresh Local Full Matrix

Artifact:

- `runs/benchmarks/bidirectional_evolution_full_matrix_2026_05_31`

Command:

```powershell
python kaggle/benchmark_bidirectional_evolution.py --seeds 11 23 37 --steps 120 --batch-size 32 --eval-batches 8 --eval-batch-size 32 --output-dir runs/benchmarks/bidirectional_evolution_full_matrix_2026_05_31 --force --device cpu --map-bins 5 --program-collapse-threshold 0.98
```

Completed:

- candidates: `10`;
- tasks: `5`;
- seeds: `3`;
- trained/evaluated rows: `150`;
- device: CPU.

Survival ranking:

| Rank | Candidate | Survival | Forward | Backward | Mean carry | Carry-reset | Gap | Novelty | Wins | Effective |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `content_synthesis_k2` | 2.6963 | 1.4836 | 1.2127 | 0.5042 | 0.4914 | 0.4880 | 0.1176 | 2 | 15/15 |
| 2 | `program_novelty_hard` | 2.6078 | 1.4997 | 1.1081 | 0.5099 | 0.4961 | 0.4938 | 0.0079 | 2 | 15/15 |
| 3 | `program_novelty_soft` | 2.5944 | 1.4997 | 1.0947 | 0.5099 | 0.4961 | 0.4938 | 0.0025 | 2 | 15/15 |
| 4 | `current_best` | 2.5894 | 1.4997 | 1.0897 | 0.5099 | 0.4961 | 0.4938 | 0.0000 | 2 | 15/15 |
| 5 | `content_synthesis_k1` | 2.5894 | 1.4997 | 1.0897 | 0.5099 | 0.4961 | 0.4938 | 0.0000 | 2 | 15/15 |
| 6 | `identity_first_local_w4` | 2.5849 | 1.4857 | 1.0992 | 0.5057 | 0.4919 | 0.4880 | 0.0095 | 0 | 15/15 |
| 7 | `coherence_sparse_attention` | 2.5583 | 1.4516 | 1.1068 | 0.4940 | 0.4797 | 0.4779 | 0.0225 | 0 | 15/15 |
| 8 | `iterative_multi_hop_k2` | 2.3625 | 1.1398 | 1.2227 | 0.3901 | 0.3768 | 0.3729 | 0.1260 | 1 | 15/15 |
| 9 | `confidence_iterative_k1` | 2.2933 | 1.0755 | 1.2178 | 0.3682 | 0.3563 | 0.3510 | 0.1260 | 0 | 15/15 |
| 10 | `mamba_program_memory` | 1.7903 | 0.0120 | 1.7783 | 0.0117 | 0.0003 | -0.0049 | 0.7098 | 0 | 0/15 |

Task winners by mean carry:

| Task | Winner | Carry | Carry-reset | Carry-shuffled |
| --- | --- | ---: | ---: | ---: |
| `longer_single_key` | `program_novelty_hard` | 0.7891 | 0.7760 | 0.7734 |
| `multi_key` | `content_synthesis_k2` | 0.8568 | 0.8464 | 0.8333 |
| `delayed_query` | `program_novelty_hard` | 0.7565 | 0.7474 | 0.7448 |
| `noisy_key` | `iterative_multi_hop_k2` | 0.1250 | 0.1094 | 0.1120 |
| `multi_hop` | `content_synthesis_k2` | 0.0560 | 0.0417 | 0.0378 |

Interpretation:

- `content_synthesis_k2` is the fresh full-matrix survival winner because it combines strong carry with higher behavioral novelty, and it wins `multi_key` plus the weak-but-important `multi_hop` slice.
- `current_best` / `content_synthesis_k1` remain the practical default family: same top mean carry as the novelty-weight variants, simpler routing, better routing efficiency than k2.
- `program_novelty_soft` and `program_novelty_hard` improved program differentiation without changing task-level behavior; this supports the idea that real bidirectional novelty must operate on behavioral descriptors, not just stronger separation losses.
- `iterative_multi_hop_k2` is not the multi-hop winner in this run, but it is the noisy-key winner and has the strongest useful novelty among effective candidates. Preserve it as a niche candidate for deceptive retrieval search.
- `mamba_program_memory` is behaviorally novel but ineffective: `0/15` effective and near-zero carry. Novelty alone is not enough.

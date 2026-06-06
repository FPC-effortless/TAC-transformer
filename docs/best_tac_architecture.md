# Best TAC Architecture Note

Date: 2026-05-28
Updated: 2026-05-31

This note explains the strongest TAC configuration found so far, the task it was tested on, and the limits of the evidence. The current preset is exposed as `best_tac_config(...)` in `tac_transformer/presets.py`. The original single-task winner was hash routing; the harder five-task validation on 2026-05-29 promoted BASE-style balanced routing; the first 2026-05-30 full matrix promoted content-addressed cue/value memory; the automated 2026-05-30 research funnel promoted synthesis-gated content reads as the best aggregate direct-memory path; the 2026-05-31 attention/IdentityState full matrix promoted identity-first attention.

## Summary

The best current architecture is:

```text
modern transformer backbone
+ persistent TAC identity state
+ BASE-routed linear program experts
+ novelty-gated memory writes
+ content-addressed cue/value memory
+ synthesis-gated two-step content read
+ gated residual memory adapter
+ identity-first attention K/V projection
```

This stores `(key_hidden, value_hidden)` tuples during context processing, retrieves by nearest hidden-state cue, then runs a learned synthesis over `[query, read_1, read_2, read_1 - read_2, read_1 * read_2]`. Attention K/V projections are also conditioned on each token's soft program identity. In the 2026-05-31 full attention/IdentityState matrix, this improved mean carry accuracy from the previous synthesis-gated preset `0.4904` to `0.5099` and won all five harder tasks.

It is not the all-features stack. The final matrix showed that combining every researched mechanism made the model slower and less accurate on the current task.

The main cost is throughput: the identity-first preset trains at roughly `0.42x` the tokens/sec of its parameter-matched vanilla baseline in the current harder-task harness. TAC currently shows better memory-task data efficiency, not better wall-clock efficiency.

Earlier inference profiling showed the same tradeoff for the content-addressed preset. At the default `content_store_size=8`, content-addressed k1 carried-query inference was close to BASE program memory, but one-token decode was slower: about `0.89x` BASE at seq_len 64 and `0.84x` BASE at seq_len 128 in the CPU profile. The identity-first promotion still needs fresh GPU/KV-cache profiling before serving claims.

## Task Context

The benchmark is a no-leak chunked key/value recall task. Each sample has two chunks:

```text
context chunk: [context_token, key, value, noise...]
query chunk:   [query_token, key, recall_token, noise...]
target:        predict value at the recall position
```

The value token is present in the context chunk, but it is deliberately removed from the query input. A model can only solve the query by carrying useful state from the context chunk into the query chunk.

In the final matrix, `vocab_size=64` and data tokens are IDs `4..63`, so a random guess over valid data tokens is about `1/60 = 1.67%`. The parameter-matched vanilla transformer averaged `1.17%`, which is near chance. TAC averaged `8.72%`.

That absolute accuracy is still low. This is an early, small synthetic benchmark with only 120 training steps per run. The useful signal is not that TAC is solved, but that carried TAC state helps, reset state hurts, shuffled state hurts, and vanilla is near chance under the same training budget.

## Winning Configuration

| Setting | Value | Why it is in the preset |
| --- | --- | --- |
| `norm_type` | `rmsnorm` | Modern stable normalization with fewer parameters than LayerNorm. |
| `mlp_type` | `swiglu` | Stronger modern feed-forward block than GELU in this benchmark family. |
| `position_type` | `rope` | Removes learned position-table dependency and matches current transformer practice. |
| `n_kv_heads` | `n_heads / 2` by default | Grouped-query attention reduces key/value projection size while preserving query heads. |
| `program_compute_type` | `linear_expert` | Turns identity programs into trainable compute, not just embedding bias. |
| `routing_type` | `base` | BASE-style balanced routing won the harder five-task matrix, with `15/15` effective runs and higher mean carry than the old hash preset. |
| `state_update_type` | `gated` | Learns how much new activation should update persistent stability. |
| `memory_write_type` | `novelty_gated` | Writes memory when current hidden state differs usefully from prior program memory. |
| `memory_read_type` | `content_addressed` | Stores cue/value hidden-state tuples and beat BASE across direct recall tasks in the full harder matrix. |
| `content_store_size` | `8` | Enough room for the current direct recall tasks without making the store large. |
| `content_read_steps` | `2` | Enables one chained content lookup before synthesis. |
| `content_read_gate_type` | `synthesis` | Learns to combine query, first read, second read, difference, and product features; best aggregate direct-memory result. |
| `memory_adapter_type` | `gated_residual` | Feeds memory back into hidden state through a learned gate instead of task-specific logit injection. |
| `memory_tier_type` | `flat` | Hierarchical tiers did not beat flat memory on this task. |
| `memory_lookup_type` | `none` | Product-key memory added parameters and did not improve carry accuracy. |
| `identity_attention_type` | `identity_first` | Conditioning K/V on soft program identity won the 2026-05-31 full fusion matrix: 15/15 effective, 5/5 task wins, and higher mean carry than the previous preset. |
| `residual_stream_type` | `single` | Dual streams were slower and less accurate. |
| `n_sink_programs` | `0` | Sink programs did not beat the no-sink reference. |
| `n_prediction_heads` | `1` | Multi-token prediction remained effective but reduced carry accuracy here. |

## Mechanism

For each hidden token vector `h_t`, TAC compares it to learned program embeddings `P_i` using normalized dot-product similarity:

```text
program_logit[t, i] = sqrt(d_model) * normalize(h_t) dot normalize(P_i)
activation[t, i] = sigmoid(program_logit[t, i])
```

With gated state updates, the layer maintains persistent program stability:

```text
stability_i <- gate_i * activation_i + (1 - gate_i) * previous_stability_i
```

Those stability values influence both coherence and routing. Token-program weights are:

```text
softmax(program_logits + log(stability))
```

TAC then builds a coherence matrix from the token-program weights and adds it to attention as an identity bias. Separately, the routed program path updates persistent `program_memory`.

The original single-key winner used a hash router that chooses one adaptive program per row using the current strongest stability program plus a deterministic hash:

```text
top_program = argmax(stability)
chosen_program = hash(row_id, top_program) mod n_programs
```

In the causal benchmark path, `row_id` is the flattened token row inside the current layer's `batch_size * seq_len` routing table: effectively the batch index plus sequence position after flattening, not a layer ID. This means different token rows can map to different programs even when their strongest stability program is the same.

The result is then trimmed to the energy budget. This is not a learned router. It won the original single-key slice, but the later harder-task matrix promoted BASE routing because it held up better across longer, multi-key, delayed, noisy, and multi-hop variants.

## Metrics

| Metric | Meaning |
| --- | --- |
| `carry accuracy` | Query value-token accuracy when identity state is carried from the context chunk. |
| `reset accuracy` | Same query, but identity state is reset before the query. If carry beats reset, state persistence helps. |
| `shuffled accuracy` | Same query, but identity state is shuffled across batch rows. If carry beats shuffled, the content of the state matters. |
| `baseline accuracy` | Parameter-matched vanilla transformer accuracy on the same task. |
| `used energy` | Sum of learned route costs for selected adaptive programs. It is a routing-compute proxy, not measured hardware power. |
| `active programs` | Average number of routed programs selected. |
| `active expert fraction` | Fraction of expert parameters active under sparse expert dispatch. Dense linear experts still compute densely. |
| `train TPS ratio` | TAC train tokens/sec divided by vanilla train tokens/sec. Below 1.0 means TAC is slower wall-clock. |
| `effective` | Count of seeds where all benchmark checks passed: carry beats reset, carry beats shuffled, and TAC carry accuracy matches or beats the parameter-matched vanilla baseline. |

## Final Matrix Excerpt

The final matrix ran 10 candidates across seeds `11, 23, 37`, with 120 training steps per run and a parameter-matched vanilla transformer baseline for each candidate. Each seed evaluated 8 batches of 32 query targets, so the three-seed aggregate contains 768 scored value predictions.

| Candidate | Effective | Carry acc | Reset acc | Shuffled acc | Baseline acc | TAC-baseline gap | Used energy | Active programs | Train TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hash_current_best | 3/3 | 0.0872 | 0.0208 | 0.0156 | 0.0117 | 0.0755 | 1.0183 | 1.0000 | 0.5341 |
| hash_sparse_expert | 3/3 | 0.0872 | 0.0208 | 0.0156 | 0.0117 | 0.0755 | 1.0183 | 1.0000 | 0.3690 |
| hash_hierarchical | 3/3 | 0.0859 | 0.0195 | 0.0156 | 0.0117 | 0.0742 | 1.0194 | 1.0000 | 0.5430 |
| energy_reference | 3/3 | 0.0833 | 0.0273 | 0.0117 | 0.0117 | 0.0716 | 3.4771 | 4.0000 | 0.5026 |
| hash_identity_compressed | 3/3 | 0.0833 | 0.0260 | 0.0104 | 0.0117 | 0.0716 | 0.9795 | 1.0000 | 0.5944 |
| all_features_stack | 3/3 | 0.0612 | 0.0208 | 0.0195 | 0.0182 | 0.0430 | 1.0208 | 3.0000 | 0.2826 |

`hash_current_best` and `hash_sparse_expert` tied on recall quality in the original single-key matrix, and the dense hash path was preferred there because it achieved the same carry accuracy at about 45% higher training throughput (`0.5341x` vanilla train TPS vs. `0.3690x`). The harder five-task matrix supersedes this router choice and promotes BASE routing as the current preset. The sparse variant is still important: it activated only `0.0625` of expert parameters, but the current unfused sparse dispatch does not convert that proxy into speed.

The `all_features_stack` baseline differs because parameter matching is done per TAC candidate. The stacked TAC variant has more TAC parameters (`396,450` in the small matrix), so its matched vanilla baseline is wider (`d_model=112`, `408,240` parameters) than the baseline used for the simpler `330,530`-parameter candidates (`d_model=104`, `340,808` parameters). Its stronger baseline makes the stacked candidate look slightly better on baseline accuracy, but the TAC side still underperforms the simpler preset on carry accuracy and throughput.

Full artifacts:

- `runs/benchmarks/final_research_matrix_2026_05_28/RESULTS.md`
- `runs/benchmarks/final_research_matrix_2026_05_28/aggregate_final_research_matrix.json`
- `runs/benchmarks/final_research_matrix_2026_05_28/per_seed_final_research_matrix.json`

## Content-Addressed Full-Matrix Result

The corrected Phase 2B memory implementation stores cue/value hidden-state tuples instead of route patterns. On the full five-task harder matrix:

| Candidate | Effective | Mean carry | Carry-reset | Carry-shuffled | TAC-baseline gap | Train TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `content_addressed_k1` | 15/15 | 0.4349 | 0.4229 | 0.4185 | 0.4177 | 0.4823 |
| `content_addressed_k2` | 14/15 | 0.4341 | 0.4216 | 0.4198 | 0.4169 | 0.4679 |
| `base_routing` | 15/15 | 0.0677 | 0.0529 | 0.0508 | 0.0505 | 0.5236 |

Decision: `content_addressed_k1` is the best current direct-memory TAC architecture and is now the preset. It wins longer single-key, multi-key, delayed-query, and noisy-key by large margins. BASE program-memory still wins multi-hop, so multi-hop remains the next architecture target.

Artifact: `runs/benchmarks/content_addressed_full_matrix_2026_05_30/RESULTS.md`

## Automated Synthesis Promotion

The automated research funnel then re-ran all implemented variants as a broad screen and confirmed the top candidates. This changed the aggregate default.

| Variant | Effective | Task wins | Mean carry | Carry-reset | Carry-shuffled | TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `content_synthesis_k2` | 15/15 | 2 | 0.4909 | 0.4742 | 0.4773 | 0.4336 |
| `content_synthesis_k1` | 15/15 | 1 | 0.4901 | 0.4737 | 0.4753 | 0.4691 |
| `content_addressed_k1` | 15/15 | 0 | 0.4349 | 0.4229 | 0.4185 | 0.5295 |
| `base_routing` | 15/15 | 0 | 0.0677 | 0.0529 | 0.0508 | 0.5310 |

Decision: promote `content_synthesis_k1`.

`content_synthesis_k2` has a tiny mean-carry edge, but `content_synthesis_k1` is nearly tied, simpler, faster, and avoids sparse-ensemble k2 routing. The preset therefore uses BASE routing plus synthesis-gated two-step content reads.

Task-specific caveat:

- Noisy-key still prefers `content_confidence_iterative_k1`.
- Multi-hop still prefers `content_iterative_k2`, and only narrowly beats BASE.
- Synthesis is the best aggregate direct-memory default, not the final reasoning loop.

Artifact: `runs/benchmarks/automated_research_stage2_confirm_2026_05_30/RESULTS.md`

## Multi-Hop Follow-Up

The first iterative retrieval test added `content_read_steps=2`, where the first retrieved value can be used as a second lookup query. It narrowly improved multi-hop:

| Variant | Multi-hop carry | Carry-reset |
| --- | ---: | ---: |
| `content_iterative_k2` | 0.0508 | 0.0417 |
| `base_routing` | 0.0495 | 0.0234 |
| `content_addressed_k1` | 0.0365 | 0.0221 |

But it is not the default because full-matrix mean carry drops from `0.4349` for content k1 to `0.3552` for iterative k2.

Artifact: `runs/benchmarks/content_iterative_full_matrix_2026_05_30/RESULTS.md`

A follow-up confidence-gated two-step read used the content store's own cosine scores to continue only when lookup 2 was more confident than lookup 1. It did not beat the learned k2 path or BASE:

| Variant | Multi-hop carry | Carry-reset | TPS ratio |
| --- | ---: | ---: | ---: |
| `content_iterative_k2` | 0.0508 | 0.0417 | 0.4150 |
| `base_routing` | 0.0495 | 0.0234 | 0.5254 |
| `content_confidence_iterative_k1` | 0.0482 | 0.0313 | 0.4775 |
| `content_confidence_iterative_k2` | 0.0482 | 0.0339 | 0.4507 |

Decision: confidence gating remains an ablation. Multi-hop now needs a stronger verifier or supervised halt/continue head, not another fixed cosine rule.

Artifact: `runs/benchmarks/conditional_iterative_focused_2026_05_30/RESULTS.md`

The Phase 1A synthesis gate also failed its go/no-go. It learned a representation from `[query, read_1, read_2, read_1 - read_2, read_1 * read_2]`, but did not beat BASE or learned iterative k2:

| Variant | Multi-hop carry | Carry-reset | TPS ratio |
| --- | ---: | ---: | ---: |
| `content_iterative_k2` | 0.0508 | 0.0417 | 0.4253 |
| `base_routing` | 0.0495 | 0.0234 | 0.5435 |
| `content_synthesis_k1` | 0.0417 | 0.0247 | 0.4548 |
| `content_synthesis_k2` | 0.0391 | 0.0234 | 0.4336 |

Decision: synthesis remains an ablation. The next multi-hop mechanism should be a retrieval graph or supervised verifier/halt loop.

Artifact: `runs/benchmarks/synthesis_iterative_focused_2026_05_30/RESULTS.md`

## Inference Cost

At the promoted default store size:

| Seq len | Query vs BASE | Decode vs BASE |
| ---: | ---: | ---: |
| 16 | 1.1061 | 0.8800 |
| 64 | 0.9701 | 0.8907 |
| 128 | 0.7749 | 0.8373 |

Decision: keep content-addressed k1 as the default because the capability gain is large, but do not claim raw serving efficiency yet. `content_addressed_k2` is not promoted because it is slower on decode and was 14/15 effective in the full matrix.

Artifact: `runs/benchmarks/inference_profile_2026_05_30/RESULTS.md`

## Statistical Status

The result is promising but not final statistical evidence.

For `hash_current_best` across three seeds:

| Metric | Mean | Seed sd | Approx. 95% CI half-width |
| --- | ---: | ---: | ---: |
| carry accuracy | 0.0872 | 0.0137 | 0.0341 |
| reset accuracy | 0.0208 | 0.0126 | 0.0312 |
| shuffled accuracy | 0.0156 | 0.0078 | 0.0194 |
| baseline accuracy | 0.0117 | 0.0068 | 0.0168 |
| TAC-baseline gap | 0.0755 | 0.0148 | 0.0367 |
| carry-reset delta | 0.0664 | 0.0179 | 0.0445 |

The seed-level intervals are wide because `n=3`. Even so, the approximate lower bound for the TAC-baseline gap is still positive (`0.0755 - 0.0367 = 0.0388`), and the sign of the effect held across all three seeds. This strengthens the early result, but it should not be treated as final statistical evidence. The next validation should increase seed count, increase evaluation batches, and sweep sequence/chunk lengths.

## Routing Justification

Soft and learned-ish alternatives were tested:

| Routing | Carry acc | Carry-reset delta | Used energy | Active programs |
| --- | ---: | ---: | ---: | ---: |
| energy | 0.0833 | 0.0560 | 3.4771 | 4.0000 |
| expert_choice | 0.0833 | 0.0573 | 2.6756 | 2.8190 |
| base | 0.0833 | 0.0599 | 1.2747 | 1.0000 |
| hash | 0.0872 | 0.0664 | 1.0183 | 1.0000 |

Hash routing won this original single-key slice. It is deterministic and cheap, so it cannot learn a semantic router on its own. The practical conclusion is narrower: on the original benchmark, TAC did not need four active programs, and a stable one-program route preserved or improved recall. On the harder task family, BASE routing became the better default because it improved mean carry and effective-run consistency.

## What Is Not Proven Yet

This benchmark compares TAC variants to a parameter-matched vanilla transformer and to the implemented recurrent mixer ablations, but not yet to production-grade external-memory systems. It also does not prove real hardware energy efficiency. The current dense winning path is slower than vanilla by wall-clock throughput, and the sparse path needs fused/grouped kernels before route sparsity becomes actual speed.

The defensible claim today is:

```text
On the harder no-leak chunked recall benchmark family, with 120 training steps
and three seeds, the BASE-routed TAC preset uses carried identity state in a
way that beats reset, shuffled state, and a parameter-matched vanilla
transformer more consistently than the old hash preset.

On noisy-key partial-cue recall, content-addressed cue/value memory improves
carry accuracy over BASE routing in a focused 3-seed validation.

On the full five-task harder matrix, content-addressed cue/value memory improves
mean carry accuracy from 0.0677 to 0.4349, but BASE program memory still wins
the multi-hop slice.
```

The next claim to test is:

```text
Does the same advantage hold with more seeds, larger budgets, real agentic
traces, and stronger recurrent/hybrid persistent-memory baselines?
```

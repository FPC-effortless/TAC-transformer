# Attention and IdentityState Fusion Experiments

Date: 2026-05-31

Question: should IdentityState stay parallel to attention, shorten attention, replace KV cache behavior, sparsify attention, or shape K/V projections directly?

## Implemented Options

- Option 1, identity-gated sparse attention: `identity_attention_type="coherence_sparse"` masks token attention to same dominant-program edges. The repair variant `identity_attention_type="coherence_sparse_compressed"` also keeps carried program-memory slots as global compressed K/V bridges.
- Option 2, compressed IdentityState K/V: existing `identity_attention_type="compressed_memory"` was retested on top of the current synthesis-gated content-memory preset as `compressed_memory_best`.
- Option 3, identity-first attention: `identity_attention_type="identity_first"` adds an identity-aware K/V projection over `[hidden, soft_program_identity]`.
- Option 4, attention shortening: `attention_window_size=<N>` masks token attention to a local causal window while leaving identity-memory paths intact.

The local-window, compressed-memory, and sparse-attention switches remain opt-in. After the full 120-step validation below, `identity_attention_type="identity_first"` is promoted into `best_tac_config(...)`.

## Hypotheses And Fix Attempts

- Local attention should force IdentityState to carry long-range information. Result: it was behaviorally safe and slightly better than current best in the one-seed 60-step screen, but it does not provide real CPU speedup because the implementation still uses dense masked attention. A real efficiency claim needs a block-sparse or sliding-window kernel.
- Hard sparse attention should reduce attention redundancy. Result: the hard same-program mask hurt multi-hop. Hypothesis: the identity graph is not differentiated enough to cut cross-program bridges early in training. Fix attempted: sparse token graph plus compressed memory slots; this repaired some short-screen multi-hop behavior but was unstable on noisy-key across seeds.
- Compressed memory should be the cleanest medium-term fusion. Result: it improved mean carry over current best in the 3-seed confirmation and matched current best on effective-run count.
- Identity-first attention should be the most expressive long-term fusion. Result: it produced the highest aggregate mean carry in the 3-seed confirmation, especially on multi-key, but failed more effectiveness gates and did not solve multi-hop reliably.

## Focused Screen

Artifact: `runs/benchmarks/attention_identity_fusion_screen_2026_05_31`

Settings: seed 11, 30 steps, tasks `noisy_key` and `multi_hop`.

- Current best, local windows, compressed memory, and sparse+compressed+local tied on mean carry at `0.0391`.
- Hard sparse attention dropped to `0.0312` mean carry.
- Identity-first improved noisy-key carry to `0.0625` but failed multi-hop effectiveness because reset beat carry.

## Confirmation

Artifact: `runs/benchmarks/attention_identity_fusion_confirm_2026_05_31`

Settings: seeds 11/23/37, 60 steps, all five harder chunked-memory tasks.

| Variant | Effective | Mean carry | Carry-reset | Carry-shuffled | Gap | TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `identity_first_attention` | 12/15 | 0.1896 | 0.1740 | 0.1688 | 0.1656 | 0.3775 |
| `coherence_sparse_compressed_local_w4` | 11/15 | 0.1875 | 0.1646 | 0.1677 | 0.1635 | 0.3715 |
| `compressed_memory_best` | 13/15 | 0.1854 | 0.1594 | 0.1656 | 0.1615 | 0.3646 |
| `current_best` | 13/15 | 0.1729 | 0.1521 | 0.1521 | 0.1490 | 0.3801 |

## Full Matrix

Artifact: `runs/benchmarks/attention_identity_fusion_full_matrix_2026_05_31`

Settings: seeds 11/23/37, 120 steps, all five harder chunked-memory tasks, full fusion variant set.

| Rank | Variant | Effective | Task wins | Mean carry | Carry-reset | Gap | TPS ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `identity_first_attention` | 15/15 | 5 | 0.5099 | 0.4961 | 0.4938 | 0.4248 |
| 2 | `identity_first_local_w4` | 15/15 | 0 | 0.5057 | 0.4919 | 0.4880 | 0.4454 |
| 3 | `coherence_sparse_attention` | 15/15 | 0 | 0.4940 | 0.4797 | 0.4779 | 0.4702 |
| 4 | `local_attention_w8` | 15/15 | 0 | 0.4911 | 0.4753 | 0.4750 | 0.4766 |
| 5 | `local_attention_w4` | 15/15 | 0 | 0.4911 | 0.4745 | 0.4734 | 0.4447 |
| 6 | `current_best` | 15/15 | 0 | 0.4904 | 0.4740 | 0.4742 | 0.4431 |
| 11 | `compressed_memory_best` | 15/15 | 0 | 0.4724 | 0.4555 | 0.4562 | 0.4380 |

By-task result: `identity_first_attention` won raw carry on all five tasks: longer single key, multi-key, delayed query, noisy key, and multi-hop.

## Decision

Promote `identity_first_attention`. The full matrix overturned the earlier 60-step signal: identity-first attention was not just the highest mean-carry variant, it was 15/15 effective and won every task. The preset now uses `identity_attention_type="identity_first"` with full attention, not the local-window branch.

Do not promote compressed memory as the general default. It looked promising in the 60-step confirmation, but at 120 steps it finished last in the full matrix and underperformed current best on aggregate carry.

Sparse and local attention remain experimental. Coherence sparse attention beat current best by mean carry, and local windows were safe, but neither matched identity-first attention. A real efficiency claim still needs a block-sparse or sliding-window kernel; the current implementation masks dense logits.

Next research branch: test identity-first attention with learned/top-k sparse bridge edges and GPU profiling, because the quality winner is still slower than several alternatives on this CPU harness.

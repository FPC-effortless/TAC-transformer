# Hybrid Mixer Experiments

Date: 2026-05-28

Question: are we trapped in the current TAC transformer shape, or do Jamba/BlackMamba-style hybrid sequence mixers improve the architecture for agentic use?

## Research Basis

Jamba combines Transformer attention, Mamba-style state-space sequence mixing, and MoE capacity. BlackMamba explores the related idea of combining state-space modeling with sparse expert capacity. For TAC, the practical hypothesis is:

```text
TAC identity state
+ attention for precise in-context binding
+ state-space-style mixer for cheap sequential dynamics
+ sparse program experts for conditional compute
```

This is a different way to apply the research than putting planner/tool/reflection heads inside the model.

## Implementation

Added `sequence_mixer_type` to `TACConfig`:

```text
attention    existing TAC path
state        causal recurrent/state mixer replaces attention
hybrid       attention + causal state mixer in every block
alternating  attention on even layers, state mixer on odd layers
```

The state mixer is a small trainable causal proxy, not a full fused Mamba kernel:

```text
gate, value = Linear(x)
value = causal depthwise_conv(value)
s_t = decay * s_{t-1} + (1 - decay) * value_t
y_t = Linear(s_t * silu(gate_t))
```

This is enough to test the architectural direction before adding a specialized dependency or CUDA kernel.

## Results

Longer 120-step check, seeds 11/23/37, same chunked no-leak carry/reset/shuffle benchmark:

| Variant | Effective | Carry | Reset | Shuffled | Baseline | Gap | TPS Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hash_attention_best | 3/3 | 0.0872 | 0.0208 | 0.0156 | 0.0117 | 0.0755 | 0.6156 |
| jamba_alternating | 3/3 | 0.0612 | 0.0143 | 0.0130 | 0.0117 | 0.0495 | 0.5968 |
| jamba_hybrid | 3/3 | 0.0625 | 0.0104 | 0.0156 | 0.0169 | 0.0456 | 0.6867 |
| blackmamba_sparse_hybrid | 3/3 | 0.0625 | 0.0104 | 0.0156 | 0.0169 | 0.0456 | 0.4777 |

Shorter 80-step exploratory matrix also tested `state_only` and `blackmamba_sparse_state`; both were weaker than attention and hybrid variants.

Artifacts:

- `runs/benchmarks/hybrid_mixer_matrix_2026_05_28/RESULTS.md`
- `runs/benchmarks/hybrid_mixer_longcheck_2026_05_28/RESULTS.md`

## Recurrent Mixer Follow-Up

After the first hybrid matrix, TAC also tested stronger pure-PyTorch recurrent/state alternatives:

```text
selective_state  Mamba-inspired input-selective recurrent mixer
rwkv             RWKV-inspired time-mix recurrent weighted value path
xlstm            xLSTM-inspired exponential-gated recurrent mixer
```

Longer 120-step check, seeds 11/23/37:

| Variant | Effective | Carry | Reset | Shuffled | Baseline | Gap | TPS Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hash_attention_best | 3/3 | 0.0872 | 0.0208 | 0.0156 | 0.0117 | 0.0755 | 0.6271 |
| mamba_selective_state | 3/3 | 0.0599 | 0.0169 | 0.0130 | 0.0091 | 0.0508 | 0.6256 |
| rwkv_time_mix | 2/3 | 0.0534 | 0.0169 | 0.0195 | 0.0130 | 0.0404 | 0.5434 |
| xlstm_gated | 3/3 | 0.0495 | 0.0182 | 0.0143 | 0.0195 | 0.0299 | 0.6389 |
| rwkv_sparse_expert | 2/3 | 0.0534 | 0.0169 | 0.0195 | 0.0130 | 0.0404 | 0.4311 |

Artifact:

- `runs/benchmarks/recurrent_mixer_longcheck_2026_05_28/RESULTS.md`

## Decision

Current TAC attention core remains the best default.

The best architecture today is still:

```text
best_tac_config
= attention + hash-routed identity programs
+ novelty-gated writes
+ program-memory readout
+ gated residual memory adapter
```

But the search space is now wider:

- `jamba_hybrid` is worth keeping as an efficiency-oriented ablation because it remained effective on 3/3 seeds and had the best TAC-vs-baseline TPS ratio in the long check.
- `mamba_selective_state` is the strongest tested recurrent/state-only direction, but it still loses to attention TAC on carry accuracy and does not improve throughput enough to justify replacing attention.
- `jamba_alternating` is simpler and effective, but lower accuracy and no throughput advantage over the current best in the long check.
- `blackmamba_sparse_hybrid` is not worth promoting yet because it tied hybrid accuracy while reducing throughput.
- RWKV/xLSTM-style mixers are useful baselines, but neither is currently a better TAC core on this memory-binding task.
- `state_only` is not good enough for TAC's current memory-binding task; attention still matters for precise key/value recall.

For agentic use, this means TAC should not be tied to a pure transformer box, but attention is still the best tested binding mechanism. The next serious hybrid test should use a fused Mamba/xLSTM/RWKV package where the local platform supports it and run longer data-efficiency curves.

## Sources

- Jamba: https://arxiv.org/abs/2403.19887
- Mamba: https://arxiv.org/abs/2312.00752
- BlackMamba: https://arxiv.org/abs/2402.01771
- RWKV: https://arxiv.org/abs/2305.13048
- xLSTM: https://arxiv.org/abs/2405.04517

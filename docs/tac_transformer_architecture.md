# TAC-Transformer Architecture

This repository now has two layers of implementation:

- `src/lib/identityField.js`: deterministic browser lab core.
- `tac_transformer/`: trainable PyTorch architecture.

## Trainable Block

Each `TACTransformerBlock` keeps the standard transformer path and adds an executable identity path:

1. Normalize hidden states.
2. Run `IdentityFieldLayer`.
3. Compute pairwise token coherence from token-to-program activations.
4. Inject coherence into self-attention logits:

   ```text
   attention = softmax(QK^T / sqrt(d_head) + beta * C)
   ```

5. Route program embeddings under an energy budget.
6. Add a routed program context vector back into the residual stream.
7. Run the MLP path.
8. Return updated persistent identity state.

The default backbone remains the original `LayerNorm` plus GELU MLP path so earlier parameter counts and benchmark results stay comparable. For modern-backbone experiments, `TACConfig` also supports:

- `norm_type="rmsnorm"`
- `mlp_type="swiglu"`
- `position_type="rope"`
- `n_kv_heads=<fewer than n_heads>` for grouped-query attention
- `program_compute_type="linear_expert"` for routed per-program linear compute instead of embedding-only program context
- `program_compute_type="sparse_linear_expert"` for the same expert bank with active-expert compute metrics based on the energy route
- `routing_type="expert_choice"`, `"base"`, or `"hash"` for sparse-routing ablations beyond the default TAC energy router
- `n_sink_programs=<N>` for StreamingLLM-style identity sink programs that are always visible without spending adaptive route energy
- `state_update_type="gated"` for learned stability and memory update gates instead of fixed `state_decay`
- `memory_write_type="novelty_gated"` for Titans-inspired trainable memory writes that gate candidate program-memory updates against previous memory
- `memory_tier_type="hierarchical"` for HMT/RMT-style recent, stable, and archival program-memory tiers
- `memory_lookup_type="product_key"` for a sparse Product-Key-style memory table read through top-k key lookup
- `identity_attention_type="compressed_memory"` for attention to read carried identity `program_memory` as compressed K/V slots
- `identity_attention_type="coherence_sparse"` for token attention masked to same dominant-program edges
- `identity_attention_type="coherence_sparse_compressed"` for sparse token attention with carried program-memory bridge slots
- `identity_attention_type="identity_first"` for K/V projections shaped by each token's soft program identity
- `attention_window_size=<N>` for local causal token attention while leaving identity-memory paths available
- `residual_stream_type="dual_stream"` for ResiDual/mHC-style separate gated content and identity residual streams
- `n_prediction_heads=<N>` for multi-token prediction heads that add future-token auxiliary losses
- `memory_read_type="program_memory"` for direct supervised reads from identity program memory
- `memory_adapter_type="residual"` for model-native linear memory-vector injection into query hidden state
- `memory_adapter_type="gated_residual"` for a higher-capacity gated adapter that blends a memory MLP update into the query hidden state

Use the same options for TAC and vanilla baselines when running comparisons.

## Identity State

`IdentityState` contains:

- `stability`: per-program persistence score, shaped `[batch, n_programs]`.
- `program_memory`: per-program memory vectors, shaped `[batch, n_programs, d_model]`.

Passing `identity_states` from one forward call into the next gives the model temporal continuity beyond the current token window.

## Auxiliary Losses

The model returns `output.aux.losses`:

- `coherence`: encourages stable coherence structure.
- `program_reuse`: encourages reusable program activation.
- `energy`: exposes routed energy use as a trainable pressure.

These can be combined with next-token loss:

```python
output = model(input_ids, labels=labels, identity_states=state)
loss = (
    output.loss
    + 0.05 * output.aux.losses["coherence"]
    + 0.05 * output.aux.losses["program_reuse"]
    + 0.01 * output.aux.losses["energy"]
)
loss.backward()
```

## Smoke Example

```python
import torch
from tac_transformer import TACConfig, TACTransformerLM

config = TACConfig(vocab_size=128, d_model=64, n_heads=4, n_layers=2)
model = TACTransformerLM(config)

tokens = torch.randint(0, config.vocab_size, (2, 16))
output = model(tokens)

print(output.logits.shape)
print(output.aux.coherence.shape)
print(output.aux.used_energy)
```

# TAC-Transformer Legacy Baseline (Chunked-Recall Preset)

This document freezes the **validated chunked-recall baseline** as of 2026-05-31.
It describes the architecture and training configuration that proved `carry > reset / shuffled / vanilla`
on chunked-recall tasks. All future TAC-SCM work should treat this as the "before" state.

---

## What This Is

`best_tac_config` / `best_chunked_recall_tac_config` produce the architecture validated by the
harder 2026-05-31 evaluation matrix. The preset name `BEST_TAC_ARCHITECTURE` refers specifically
to **this chunked-recall architecture**, not to any forthcoming TAC-SCM design.

This is **not** the TAC-SCM (Structure-Conditioned Memory) architecture. TAC-SCM introduces
concept volumes, two-level structure routing, slot-conditioned program bottlenecks, and a
structure-to-behavior bridge — none of which are present here.

---

## Architecture Dict (`BEST_TAC_ARCHITECTURE`)

```python
BEST_TAC_ARCHITECTURE: dict[str, Any] = {
    "norm_type": "rmsnorm",
    "mlp_type": "swiglu",
    "position_type": "rope",
    "program_compute_type": "linear_expert",
    "routing_type": "base",
    "state_update_type": "gated",
    "memory_write_type": "novelty_gated",
    "memory_tier_type": "flat",
    "memory_lookup_type": "none",
    "memory_read_type": "content_addressed",
    "content_store_size": 8,
    "content_read_steps": 2,
    "content_read_gate_type": "synthesis",
    "memory_adapter_type": "gated_residual",
    "identity_attention_type": "identity_first",
    "residual_stream_type": "single",
    "sequence_mixer_type": "attention",
    "state_mixer_kernel_size": 4,
    "n_sink_programs": 0,
    "n_prediction_heads": 1,
    "multi_token_loss_weight": 0.0,
    "memory_separation_weight": 0.01,
    "content_cue_separation_weight": 0.005,
    "content_gate_entropy_weight": 0.005,
    "content_reconsolidate": True,
    "content_reconsolidate_rate": 0.1,
    "detach_identity_state": False,
}
```

### Key architectural choices

| Field | Value | Purpose |
|---|---|---|
| `norm_type` | `rmsnorm` | Modern normalization; more stable than LayerNorm at scale |
| `mlp_type` | `swiglu` | Gated activation matching LLaMA/Mistral backbones |
| `position_type` | `rope` | Rotary positional encoding |
| `program_compute_type` | `linear_expert` | Per-program routed linear compute instead of embedding-only context |
| `routing_type` | `base` | Standard TAC energy router (no semantic routing pressure) |
| `state_update_type` | `gated` | Learned stability and memory-update gates |
| `memory_write_type` | `novelty_gated` | Titans-inspired trainable write gate against prior memory |
| `memory_tier_type` | `flat` | Single-tier program memory (no hierarchical tiers) |
| `memory_lookup_type` | `none` | No Product-Key sparse lookup |
| `memory_read_type` | `content_addressed` | Soft content-addressed read from program memory |
| `content_store_size` | `8` | Number of content-addressed memory slots |
| `content_read_steps` | `2` | Iterative read refinement steps |
| `content_read_gate_type` | `synthesis` | Synthesis gate for content read |
| `memory_adapter_type` | `gated_residual` | Gated MLP adapter blending memory into query hidden state |
| `identity_attention_type` | `identity_first` | K/V projections shaped by each token's soft program identity |
| `residual_stream_type` | `single` | Standard single residual stream |
| `n_sink_programs` | `0` | No StreamingLLM-style always-visible sink programs |
| `memory_separation_weight` | `0.01` | Auxiliary loss pushing program memories apart |
| `content_reconsolidate` | `True` | Memory reconsolidation after content reads |
| `content_reconsolidate_rate` | `0.1` | Reconsolidation learning rate |

---

## Training Dict (`BEST_TAC_CHUNKED_MEMORY_TRAINING`)

```python
BEST_TAC_CHUNKED_MEMORY_TRAINING: dict[str, float] = {
    "value_loss_weight": 3.0,
    "memory_read_loss_weight": 3.0,
    "memory_injection_weight": 0.0,
    "memory_adapter_weight": 6.0,
}
```

These weights are passed as training kwargs alongside the standard next-token loss.
`memory_adapter_weight` is elevated to 6.0 to strongly supervise the gated residual
memory adapter during chunked-recall training. `memory_injection_weight` is 0 (disabled).

---

## Validated Claims

The following results were confirmed by the harder 2026-05-31 evaluation matrix:

- **Carry > Reset**: the TAC model with persistent identity state (`carry` condition) outperforms
  the same model with identity state reset between chunks (`reset` condition) on chunked-recall tasks.
- **Carry > Shuffled**: the `carry` condition outperforms the shuffled-identity ablation, ruling out
  the hypothesis that any non-zero identity state improves recall regardless of continuity.
- **Carry > Vanilla**: the `carry` condition outperforms a vanilla transformer baseline with no
  identity-state mechanism.

These results hold under the architecture and training weights above.

---

## What Is NOT Yet Proven

The following claims are **not established** by this baseline and must not be assumed:

- **Hardware efficiency or throughput** at production scale — no wall-clock, memory-bandwidth,
  or inference-latency comparisons have been validated.
- **Production memory comparisons** — no head-to-head against production-grade external memory
  systems (e.g., retrieval-augmented generation, differentiable neural dictionaries).
- **Structure-to-behavior decoding** — whether the program routing learned here produces
  interpretable, steerable, or compositionally reusable structure is not demonstrated.
- **TAC-SCM generalization** — this preset uses `routing_type="base"` with no semantic routing
  pressure. The TAC-SCM architecture (concept volumes, two-level routing, slot-conditioned
  bottleneck) is a future development, not a variant of this preset.
- **Long-horizon advantage beyond chunked recall** — results on sliding-window or unbounded-context
  tasks are not covered by this validation.

---

## How to Instantiate

```python
from tac_transformer import best_chunked_recall_tac_config, best_chunked_memory_training_kwargs

# Build model config (legacy chunked-recall baseline)
config = best_chunked_recall_tac_config(vocab_size=32000)

# Build training kwargs
train_kwargs = best_chunked_memory_training_kwargs()
```

`best_chunked_recall_tac_config` is a named alias for `best_tac_config`. Both functions are
identical and produce the same `TACConfig`. The alias exists solely to make the legacy-baseline
role explicit in code that needs to distinguish this preset from forthcoming TAC-SCM presets.

---

## Relation to Other Presets

All other presets in `presets.py` build on or depart from `BEST_TAC_ARCHITECTURE`:

| Preset | Relation to baseline |
|---|---|
| `RUN5_CAPABILITY_ARCHITECTURE` | Adds `routing_type="base_semantic"`, top-k routing, load-balance weight, more programs |
| `MEMORY_ADVANTAGE_ARCHITECTURE` | Adds CREB allocation, program-memory graph coalition context, larger program count |
| `CPU_RESEARCH_TAC_ARCHITECTURE` | Reduces size for CPU-only local research runs |
| `KAGGLE_FAST_TAC_ARCHITECTURE` | Adds windowed attention and top-k query read for Kaggle speed |
| `RUN5B_BEST_CAPABILITY_FAST_ARCHITECTURE` | Combines MEMORY_ADVANTAGE with `cue_match` gate and windowed attention |

The baseline itself (`BEST_TAC_ARCHITECTURE`) is **not modified** by any of the above; they all
spread it via `{**BEST_TAC_ARCHITECTURE, ...}` and add their own fields.

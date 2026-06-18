# TAC-SCM v0.2 Architecture Lane

TAC-SCM v0.2 is an opt-in research lane inside TAC-transformer.  It keeps the
normal TACTransformerLM shell: tokens enter the transformer/TAC backbone, hidden
states are adapted by structure modules, and the standard LM head remains the
language-output path.

This lane is separate from `best_tac_config`.  The legacy preset remains the
best chunked-recall memory baseline.  TAC-SCM v0.2 is exposed through
`tac_scm_v02_config` and composable structure modules.

## Core Modules

- `structure_types.py` and `structure_memory.py`: explicit reusable structure
  objects plus persistent structure memory.
- `concept_volumes.py` and `structure_routing.py`: concept-family discovery and
  two-level family-to-specialist routing.
- `structure_slots.py`: slot-conditioned program bottleneck that forces hidden
  states through reusable structure slots.
- `structure_bridge.py`: REAL003 structure-to-behavior bridge variants that map
  structure reads back into hidden state, not directly into logits.
- `structure_lifecycle.py`: NSF-style scoring for preserve/decay/retire
  decisions from usage, success, transfer, and robustness signals.
- `procedural_memory.py` and `repair_controller.py`: external verifier-guided
  repair memory and retry control, kept outside the base language model.

## Benchmark Gates

- `kaggle/benchmark_real003_structure_to_behavior.py`: smoke gate for linear,
  MLP, gated residual, and oracle structure-to-behavior bridges.
- `kaggle/benchmark_structure_compression_roi.py`: required 10x and 20x
  compression ROI gates, with 50x marked experimental.
- `kaggle/benchmark_procedural_repair_memory.py`: external procedural repair
  memory smoke gate.

## Boundary

This architecture lane does not promote planner/orchestration, world/reward
heads, direct logit injection, product-key memory, hierarchical memory, or the
all-features stack into the default model.  Those remain ablations or external
experiments until validated against carry/reset/shuffle, transfer, robustness,
compression, and behavior-decoding gates.

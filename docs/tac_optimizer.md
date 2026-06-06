# TAC Optimizer

Date: 2026-06-02

TAC now uses a shared optimizer factory for training and research entrypoints:

- `TACOptimizerConfig`
- `tac_optimizer_param_groups`
- `build_tac_optimizer`

The optimizer returned by `build_tac_optimizer` is standard `torch.optim.AdamW`,
so existing checkpointing, AMP, DDP, profiler, and resume paths keep working.
The TAC-specific behavior is in the parameter grouping:

- `core`: token embeddings, attention, MLPs, norms, and non-TAC backbone params
- `identity`: program embeddings, program experts, program updates, and program
  projections
- `router`: authority routing, stability gates, routing costs, and other
  routing/gating surfaces
- `memory`: identity-aware attention memory K/V, content reads, memory lookup,
  reconsolidation, memory stores, and memory adapters
- `head`: LM heads, multi-token heads, agent action heads, world/reward/
  reflection/planner heads, and orchestration heads

Biases, norm-like tensors, and one-dimensional parameters are placed in
`*_no_decay` groups. Matrix-like parameters are placed in `*_decay` groups.
Each group carries:

- `tac_group`
- `tac_category`
- `tac_lr_mult`
- `base_lr`
- `tac_param_names`

This lets schedulers preserve TAC learning-rate multipliers instead of flattening
all groups to a single LR.

Current integration coverage:

- local synthetic training
- local language-model training
- chunked-memory training
- capability sanity/pathfinder research
- agentic controller training
- recurrent agentic baseline training
- Kaggle Run trainer and resume checkpoints
- program-specialization objective benchmark
- Kaggle memory profiler
- long-context efficiency timing experiment

Defaults are conservative: all LR multipliers are `1.0` unless the caller
explicitly tunes them. This preserves comparability with previous research while
making TAC-specific optimizer tuning available for future sweeps.

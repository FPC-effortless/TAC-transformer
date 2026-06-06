# TAC Next-Stage Research Contract

## Phase A Freeze

- Reference: TAC-Control-v1
- Checkpoint step: 10000
- Tokens seen: 183600000
- Decision: freeze_ready
- Route MI: 0.282102
- Program-memory cosine: 0.00248082
- content_addressed hit: 0.17204

### Resolved Gaps

- content_addressed_store: resolved
- identity_first_path: resolved
- fair_token_checkpoint: resolved
- specialization_signal: resolved

### Open Gaps

- multi-hop: Does BASE routing still dominate multi-hop behavior under the frozen Run 5B configuration?
- long-context: Does program specialization improve retrieval and synthesis beyond the short-context training regime?
- seed-stability: Are learned program identities stable across independent seeds?
- decode-economics: Does any capability gain justify the observed TAC decode penalty?

## Phase B Replication

- Seeds: 11, 23, 37
- Success criteria: route MI >= 0.15, program-memory cosine <= 0.25
- Frozen config includes content_addressed memory reads and identity_first attention.

### Commands

- Seed 11: `python kaggle/train_best_tac_agentic.py --preset run5b_capability --steps 20000 --seed 11 --identity-attention-type identity_first --memory-read-type content_addressed --content-read-steps 2 --content-read-gate-type synthesis --program-memory-update-type program_conditioned --memory-allocation-type creb --memory-allocation-k 6 --memory-separation-weight 0.1 --routing-type base_semantic --routing-top-k 2 --category-route-objective mi --category-route-weight 0.1 --precision fp32 --output-dir runs/phase_b_tac_control_v1/tac_control_v1_seed_11`
- Seed 23: `python kaggle/train_best_tac_agentic.py --preset run5b_capability --steps 20000 --seed 23 --identity-attention-type identity_first --memory-read-type content_addressed --content-read-steps 2 --content-read-gate-type synthesis --program-memory-update-type program_conditioned --memory-allocation-type creb --memory-allocation-k 6 --memory-separation-weight 0.1 --routing-type base_semantic --routing-top-k 2 --category-route-objective mi --category-route-weight 0.1 --precision fp32 --output-dir runs/phase_b_tac_control_v1/tac_control_v1_seed_23`
- Seed 37: `python kaggle/train_best_tac_agentic.py --preset run5b_capability --steps 20000 --seed 37 --identity-attention-type identity_first --memory-read-type content_addressed --content-read-steps 2 --content-read-gate-type synthesis --program-memory-update-type program_conditioned --memory-allocation-type creb --memory-allocation-k 6 --memory-separation-weight 0.1 --routing-type base_semantic --routing-top-k 2 --category-route-objective mi --category-route-weight 0.1 --precision fp32 --output-dir runs/phase_b_tac_control_v1/tac_control_v1_seed_37`

## Phase C Identity Stability

- Decision gate: Stable program identities across at least two passing Phase B seeds
- Blocked by: Phase B complete passing seed evidence
- Alignment components: memory_vector, selected_route_distribution, knockout_profile
- Minimum seeds: 2
- Minimum alignment similarity: 0.8
- Report role permutations rather than assuming program ids are stable.

## Phase D Benchmark Protocol

- Decision gate: TAC > parameter-matched vanilla
- Tasks: multi_hop_chain_retrieval, long_context_retrieval_4096, episodic_fact_update, tool_selection, delayed_goal_binding
- Controls: parameter_matched_vanilla, tac_shuffled_state, tac_base_routing_ablation, loss_matched_run5b_bestpt
- Report every capability score beside wall-clock, tokens/s, and cost-normalized deltas.

# HRM + Engram Phase 1 Results

Date: 2026-05-29

Baseline throughout: `best_tac_config` / hash-routed, novelty-gated, gated-residual TAC.

Reference baseline:

| Candidate | Carry | Reset | Shuffled | Carry-reset | TAC-baseline gap | Train TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_no_reconsolidation | 0.0872 | 0.0208 | 0.0156 | 0.0664 | 0.0755 | 0.5467 |

## Phase 1A: Orthogonality Loss

Implemented:

- `memory_separation_weight`
- `program_memory_cosine`
- CLI support in chunked, best-TAC, and efficiency benchmarks

Result:

| Weight | Carry | Carry-reset | Program-memory cosine | Train TPS ratio | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.0 | 0.0872 | 0.0664 | 0.9258 | 0.5467 | Baseline |
| 0.001 | 0.0859 | 0.0651 | 0.9256 | 0.5998 | Safe, not better |
| 0.01 | 0.0859 | 0.0651 | 0.9250 | 0.5585 | Safe, not better |
| 0.1 | 0.0872 | 0.0651 | 0.9178 | 0.5010 | Lowers cosine, not better |

Decision: keep as an opt-in diagnostic/loss. It reduces memory-vector similarity at high weight, but it does not improve recall enough to become default.

Artifacts:

- `runs/benchmarks/engram_separation_2026_05_29/aggregate_engram_separation.json`

## Phase 1B: Reconsolidation On Read

Implemented:

- `memory_reconsolidate`
- `reconsolidate_gate_type = linear | mlp`
- `memory_reconsolidation_gate`

Result:

| Candidate | Carry | Reset | Shuffled | Carry-reset | Gate avg | Train TPS ratio | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_no_reconsolidation | 0.0872 | 0.0208 | 0.0156 | 0.0664 | 0.0000 | 0.5467 | Baseline |
| reconsolidate_linear | 0.0651 | 0.0195 | 0.0156 | 0.0456 | 0.0319 | 0.4788 | Reject |
| reconsolidate_mlp | 0.0794 | 0.0169 | 0.0156 | 0.0625 | 0.0316 | 0.5124 | Keep opt-in |

Decision: do not promote reconsolidation yet. MLP reconsolidation passes the carry-reset gate but loses carry accuracy against the baseline. Linear reconsolidation fails.

Artifacts:

- `runs/benchmarks/engram_reconsolidation_2026_05_29/aggregate_engram_reconsolidation.json`

## Phase 1C: CREB-Like Allocation

Implemented:

- `memory_allocation_type = stability | creb`
- `memory_allocation_k`
- `creb_alpha`, `creb_beta`, `creb_gamma`
- `program_age` in `IdentityState`
- `memory_allocation_dead_rate`, `memory_allocation_age`, `memory_allocation_load_std`

Initial sweep:

| Candidate | Carry | Reset | Shuffled | Carry-reset | Dead rate | Train TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_stability_allocation | 0.0872 | 0.0208 | 0.0156 | 0.0664 | 0.0000 | 0.5467 |
| creb_available | 0.0768 | 0.0221 | 0.0156 | 0.0547 | 0.9086 | 0.5184 |
| creb_default | 0.0807 | 0.0234 | 0.0130 | 0.0573 | 0.9142 | 0.4822 |
| creb_match | 0.0924 | 0.0208 | 0.0143 | 0.0716 | 0.8906 | 0.4807 |

Top-k refinement of the winning match-biased setting:

| Candidate | Carry | Reset | Shuffled | Carry-reset | Dead rate | Train TPS ratio | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| creb_match_k1 | 0.0924 | 0.0208 | 0.0143 | 0.0716 | 0.8906 | 0.4807 | Best accuracy |
| creb_match_k2 | 0.0872 | 0.0208 | 0.0156 | 0.0664 | 0.7949 | 0.5545 | Ties baseline |
| creb_match_k3 | 0.0898 | 0.0195 | 0.0156 | 0.0703 | 0.7056 | 0.5786 | Best balanced candidate |
| creb_match_k4 | 0.0833 | 0.0208 | 0.0156 | 0.0625 | 0.6208 | 0.5457 | Reject |

Decision: CREB allocation is worth carrying forward. For raw accuracy, `creb_match_k1` is the current leader. For a small-lab efficiency tradeoff, `creb_match_k3` is the better balanced candidate because it slightly beats baseline carry and delta while improving train TPS ratio in this sweep and using more programs.

Artifacts:

- `runs/benchmarks/engram_creb_allocation_2026_05_29/aggregate_engram_creb_allocation.json`
- `runs/benchmarks/engram_creb_allocation_topk_2026_05_29/aggregate_engram_creb_topk.json`

## Current Architecture Decision

Production default remains `best_tac_config`. The CREB result was tested on harder variants after this Phase 1 sweep; it did not win strongly enough to promote.

Experimental best candidates:

- Accuracy-first experimental branch: `best_tac_config + memory_allocation_type="creb", memory_allocation_k=1, creb_alpha=0.5, creb_beta=2.0, creb_gamma=0.25`
- Balanced CREB candidate: not promoted; harder validation shows `k=3` is healthier than `k=1` but does not beat `current_best` overall.

Next phase:

Move to sparse ensemble routing and pattern-completion retrieval. The harder validation in `docs/harder_chunked_creb_validation.md` shows allocation alone is not enough for multi-key and multi-hop robustness.

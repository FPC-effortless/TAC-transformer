# PNDS Research Results Ledger

Retrospective mapping of existing TAC Transformer experiments into PNDS-URP v0.1. Original experiment claims remain bounded by their source documents.

Evidence: E0 idea; E1 implemented; E2 smoke; E3 controlled; E4 reproduced; E5 cross-condition; E6 cross-domain.

## PNDS-TAC-PERSIST-001 — Parameter-matched TAC vs vanilla
- Status: PARTIALLY_SUPPORTED
- Source: main @ f4eeac021f871a0eb67da315407ceeed9b18763d
- Date: 2026-05-26
- Baseline/control: parameter-matched vanilla Transformer; seeds 11,23,37.
- Result: TAC accuracy 0.2936 vs vanilla 0.3747; loss 3.3159 vs 3.0875; PPL 27.5898 vs 21.9247; train TPS 9544.7 vs 12724.6.
- Observed: TAC did not beat the stateless matched baseline on this ordinary synthetic task.
- PNDS relation: establishes that persistence must be tested on state-dependent tasks, not assumed to improve generic prediction.
- Evidence: E4. Decision: PARTIALLY_SUPPORTED.
- Next: carry/reset/shuffle tests where current input alone cannot determine the answer.

## PNDS-TAC-ADDRESS-001 — Sparse ensemble routing
- Status: INCONCLUSIVE
- Date: 2026-05-29
- Controls: BASE routing, hash routing, sparse k=2/3/4; seeds 11,23,37.
- Result: BASE mean carry 0.0664 and 6/6 effective; sparse k2 0.0645, k3 0.0618, k4 0.0625.
- Observed: extra routed programs did not improve aggregate recall and were slower.
- PNDS relation: negative evidence that routing breadth alone provides addressability.
- Evidence: E5. Decision: INCONCLUSIVE.
- Next: explicit content-addressed state.

## PNDS-TAC-ADDRESS-002 — Content-addressed cue/value memory
- Status: SUPPORTED
- Date: 2026-05-30
- Controls: BASE routing, sparse ensemble, pattern completion; seeds 11,23,37.
- Result: noisy-key mean carry: BASE 0.0885, content k1 0.1094, content k2 0.1081. Carry/reset: 0.0794/0.1042 for BASE/k1. Carry/shuffled: 0.0729/0.0885.
- Observed: hidden-state cue/value memory improved noisy-key recall.
- PNDS relation: direct support for Addressability and the relevance-routing stage R_t.
- Evidence: E4. Decision: SUPPORTED.
- Next: full harder-memory matrix.

## PNDS-TAC-ADDRESS-003 — Content-addressed full harder-memory matrix
- Status: SUPPORTED
- Date: 2026-05-30
- Conditions: 5 tasks x 3 variants x 3 seeds = 45 runs.
- Result: content k1 mean carry 0.4349, 15/15 effective, 2 task wins; content k2 0.4341, 14/15; BASE 0.0677, 15/15. Content memory won longer-single-key, multi-key, delayed-query and noisy-key; BASE retained multi-hop.
- PNDS relation: strongest TAC evidence for persistent addressable state; multi-hop remains the boundary between retrieval and compositional execution.
- Evidence: E5. Decision: SUPPORTED.
- Next: iterative read/repair/planning with explicit multi-hop controls.

## PNDS-TAC-PERSIST-002 — Content-memory causal audit
- Status: SUPPORTED
- Date: 2026-06-01
- Task: multi-key. Controls: randomized query, reset state, shuffled state.
- Result: normal carry 0.2539; randomized-query carry 0.0625; randomized-query reset 0.0117; randomized-query shuffled 0.0117.
- Observed: removing query identity sharply reduced recall; shuffled/reset state also failed.
- PNDS relation: supports causal state-conditioned decision formation and leakage resistance.
- Limitation: local audit; single-key audit was correctly rejected as a decisive leakage test.
- Evidence: E4. Decision: SUPPORTED.
- Next: repeat on promoted stack and harder variants.

## PNDS-TAC-EXEC-001 — TAC-SCM REAL004 causal structure-to-behavior
- Status: SUPPORTED
- Date: 2026-06-18
- Controls: vanilla, legacy TAC, no-structure, no-bridge, reset, shuffled, wrong-slot, oracle bridge; 10 seeds.
- Result: full TAC-SCM 0.734375 vs vanilla 0.340625 and legacy TAC 0.275. Carry/reset +0.34375; carry/shuffled +0.40000; correct-slot knockout +0.34375; wrong-slot knockout 0.0; structure read hit 1.0.
- PNDS relation: supports Executability: persistent structural state can be bridged into behavior.
- Limitation: synthetic controlled benchmark; bridge comparison was seed-sensitive in the smoke run.
- Evidence: E5. Decision: SUPPORTED.
- Next: harder and realistic transfer.

## PNDS-TAC-EXEC-002 — TAC-SCM REAL005 bridge stability/generalization
- Status: SUPPORTED
- Date: 2026-06-19
- Conditions: seeds 0-9, multiple widths, steps, train sizes and bridge variants.
- Result: best learned accuracy 0.824103; vanilla gap +0.462891; legacy gap +0.447193; carry/reset +0.507697; carry/shuffled +0.531120; transfer gain +0.500463; multi-hop retention 0.805671.
- PNDS relation: strengthens the C_t to A_t execution bridge across harder structural conditions.
- Evidence: E5. Decision: SUPPORTED.
- Next: realistic transfer and blind executable recovery.

## PNDS-TAC-EXEC-003 — TAC-SCM REAL006 realistic structure transfer
- Status: SUPPORTED
- Date: 2026-06-19
- Controls: vanilla, legacy TAC, retrieval-only, no-structure, no-slot, no-bridge, reset, shuffled, wrong-slot, oracle.
- Result: full-sweep accuracy 0.898958. Coding repair 0.9490; long-document compression 0.7604; multi-session memory 0.9281; research workflow transfer 0.9583. 10x/20x compression ROI passed; 50x remained experimental.
- PNDS relation: strongest TAC evidence spanning persistence, addressability and executable structure transfer in one controlled workload.
- Limitation: deterministic controlled workload; not open-ended generalization.
- Evidence: E5. Decision: SUPPORTED.
- Next: add explicit outcome verification and held-out executable recovery.

## PNDS-TAC-COMP-001 — Identity-first attention fusion
- Status: SUPPORTED
- Date: 2026-05-31
- Conditions: 5 harder-memory tasks; seeds 11,23,37; 120 steps.
- Result: identity-first attention 15/15 effective, 5/5 task wins, mean carry 0.5099 vs previous best 0.4904; TPS ratio 0.4248.
- PNDS relation: supports a shared substrate in which persistent state conditions computation itself rather than merely sitting beside attention.
- Limitation: no hardware efficiency claim; CPU throughput was lower.
- Evidence: E5. Decision: SUPPORTED.
- Next: GPU/KV-cache profiling and sparse bridge variants.

## PNDS-TAC-EXEC-004 — TAC-SIE EXP009/EXP009B preserve-retrieve-execute scaffold
- Status: BLOCKED
- Source: main; scaffold merged from integration/tac-sie-clean.
- Observed: minimal preserve -> retrieve -> decode -> execute substrate exists, but robust arbitrary binding is not yet validated.
- PNDS relation: direct precursor to the executable action stage, but not evidence of robust binding.
- Evidence: E2. Decision: BLOCKED.
- Next: EXP009C with correct, shuffled, wrong-slot, unseen-symbol and corrupted-binding controls.

## PNDS-TAC-ACT-001 — Agentic objectives and memory-conditioned action
- Status: INCONCLUSIVE
- Date: 2026-05-28
- Controls: policy-only, memory-policy, planner, recurrent, modular, orchestration, world/reward/reflection and all-agentic stack.
- Observed: naive world/reward/reflection heads remained near chance. Memory-policy was the strongest candidate, but shuffled-state accuracy remained too high; later budget/action-space sweeps failed the required carry-vs-reset/shuffle gates.
- PNDS relation: negative evidence against calling an action head agency. PNDS agency requires state-conditioned decisions whose causal dependence survives intervention.
- Evidence: E3. Decision: INCONCLUSIVE.
- Next: validate memory-conditioned action causality before adding agentic layers.

## TAC-to-PNDS synthesis

| Capability | Evidence | Status |
|---|---|---|
| Persistence | causal carry/reset/shuffle audits; REAL004-006 | bounded support |
| Addressability | content-addressed memory | strong bounded support |
| Executability | REAL004-006; TAC-SIE scaffold | bounded support; arbitrary binding unresolved |
| Verification | causal audits and benchmark controls | partial; learned verifier loop unresolved |

The unresolved transition is persistent state + relevance routing -> verified decision -> outcome-conditioned state update.

## PNDS-SETUP-001
- Status: SUPPORTED
- Purpose: universal protocol installation.

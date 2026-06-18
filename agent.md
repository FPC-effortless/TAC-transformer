# Agent Handoff

## Current State

The project is a TAC-Transformer research lab with:

- deterministic browser identity-field prototype in `src/`;
- trainable PyTorch TAC architecture in `tac_transformer/`;
- local benchmark suites in `experiments/`;
- PRD/progress/research tracking in `prd.json`, `progress.txt`, and `research.md`.

The latest completed research tickets are TAC-195, TAC-196, and TAC-197.
TAC-194 added the `run5b_best_capability_fast` Kaggle launch preset and
repaired the DDP aux-cadence unused-parameter failure. Kaggle kernel version 2
was pushed and source-pulled, but completed external outputs are still pending.

## Active Research Split

- TAC-195 completed: controlled multi-hop reasoning-vs-recall benchmark shows
  direct recall parity at chain length 1 and controlled multi-hop graph
  composition advantage for carried identity state.
- TAC-196 completed: external runtime search loop preserves direct lookup and
  improves controlled multi-hop from greedy 0.0000 to search 1.0000 without
  adding planner heads to `TACTransformerLM`.
- TAC-197 completed as `external_pending`: Run 5B best-capability v2 source
  passes, but completed Kaggle output artifacts are still missing.

## Verification Caveat

Verification on 2026-06-06:

- Focused TAC-195/TAC-196/TAC-197 tests passed.
- `npm test` passed with 3 Node tests plus 398 Python tests.
- `npm run lint` passed.
- `npm run build` passed.
- The generated bundle contains the new TAC-195/TAC-196/TAC-197 files.

## Next Useful Slice

Next useful slice: monitor the TAC-194/TAC-197 Kaggle run for completed output
artifacts. If outputs arrive, run the external validation evaluator against the
same-backbone and parameter-matched vanilla baselines. If outputs remain
missing, continue local research by making TAC-195/TAC-196 less controlled and
more live-model dependent.

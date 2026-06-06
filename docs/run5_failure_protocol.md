# Run 5 Failure Protocol

Date: 2026-06-03

Purpose:

This document defines when Run 5 or Run 5B should be considered recovered,
failed, or inconclusive, and how to compare TAC against a vanilla transformer
before attributing a capability miss to the TAC architecture.

## Gate 1 Success Criteria

Run 5/5B passes Gate 1 when all required artifacts exist and the best checkpoint
meets the minimum capability bar on the same hard-agentic evaluation corpus:

- `final_summary.json`, `metrics.jsonl`, `run_manifest.json`, `best.pt`, and
  `last.pt` are present.
- Training completed the target step count or stopped only because of the
  wall-clock guard with a resumable `last.pt`.
- Optimization health is passed, or no configured optimizer-health failure is
  reported.
- Best eval loss improves over initial eval loss and does not plateau at random
  baseline behavior.
- Hard-agentic eval accuracy is non-zero after the smoke phase and trends upward
  over checkpoints.
- Program specialization is not the only evidence of progress; next-token loss
  or task score must also improve.

Minimum local/Kaggle comparison bar:

- TAC must match or beat the same-data vanilla baseline on best eval loss, or
  close within `2%` relative loss while showing a compensating capability signal
  that vanilla lacks.
- If both TAC and vanilla fail to improve over initial loss, the result is a
  data/training-pipeline failure, not a TAC architecture failure.
- If vanilla improves materially and TAC does not under the same token budget,
  mark TAC Gate 1 failed and inspect identity-state, routing, and memory-update
  telemetry before another full run.

## Required Vanilla Comparisons

Run the vanilla baseline with the same corpus files, tokenizer/vocab size,
sequence length, batch schedule, optimizer family, eval cadence, and wall-clock
budget.

Two comparison modes are required:

- `same_backbone`: same `d_model`, `n_layers`, `n_heads`, and sequence length as
  the TAC preset. This isolates the identity-field architectural change.
- `parameter_matched`: widened vanilla baseline chosen by
  `parameter_matched_baseline_config`. This checks whether TAC is losing to a
  similarly sized plain transformer.

Use `kaggle/train_vanilla_baseline.py`:

```bash
python kaggle/train_vanilla_baseline.py \
  --preset run5b_capability \
  --scale base \
  --baseline-mode parameter_matched \
  --steps 20000 \
  --warmup-steps 500 \
  --batch-size 12 \
  --grad-accum-steps 3 \
  --eval-every 500 \
  --eval-batches 8 \
  --checkpoint-every 250 \
  --output-dir /kaggle/working/vanilla_run5b_parameter_matched \
  --device auto \
  --precision fp16 \
  --max-seconds 30600 \
  --stop-buffer-seconds 1200
```

Repeat with `--baseline-mode same_backbone`.

## Failure Classification

Use these labels in `research.md` and any Kaggle pull notes:

- `recovered`: TAC meets Gate 1 and no vanilla fallback is needed for blame.
- `tac_underperforms_vanilla`: vanilla improves under the same data/token budget
  while TAC does not.
- `pipeline_failure`: both TAC and vanilla fail to improve, or both show broken
  data/optimizer behavior.
- `insufficient_runtime`: run stops before enough optimizer steps to compare
  trends and has a valid resume checkpoint.
- `inconclusive`: artifacts are missing or eval settings differ enough to block
  a fair comparison.

## Decision Protocol

1. Verify Run 5/5B artifacts and optimizer-health status.
2. Record best eval loss, latest eval loss, accuracy, tokens seen, wall-clock
   stop reason, and checkpoint paths.
3. Run `same_backbone` vanilla with identical data and training budget.
4. Run `parameter_matched` vanilla with identical data and training budget.
5. Compare best eval loss, accuracy, tokens/sec, and completed steps.
6. Declare TAC failed only if vanilla improves materially under a fair match and
   TAC does not.
7. If TAC fails, use the Track A/B diagnostics before launching another full
   Kaggle run:
   - A-side: decode policy, sparse writes, identity-state maintenance cost.
   - B-side: routed-is-best, program-memory cosine, dead-program fraction, and
     write-frequency entropy.

## Minimum Artifacts To Archive

- TAC `run_manifest.json`, `metrics.jsonl`, `final_summary.json`
- TAC `best.pt` and `last.pt`
- Vanilla same-backbone `run_manifest.json`, `metrics.jsonl`, `final_summary.json`
- Vanilla parameter-matched `run_manifest.json`, `metrics.jsonl`, `final_summary.json`
- Any specialization reports used to justify a capability claim

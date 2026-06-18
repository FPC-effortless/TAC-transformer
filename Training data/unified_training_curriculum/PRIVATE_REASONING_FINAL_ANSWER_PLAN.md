# Private Reasoning Final-Answer Plan

Use this path when reasoning traces should improve final answers without teaching the model to print trace text.

## Principle

The model-visible training fields are only:

- `prompt`
- `answer`

Reasoning traces are stored separately as private metadata. They are not included in `prompt`, `answer`, or `text`.

This means the current TAC trainer learns final-answer behavior. It does not yet consume private traces as a true latent scratchpad objective.

## Build Dataset

```bash
python scripts/build_private_reasoning_final_answer_dataset.py \
  --output-dir runs/private_reasoning_final_answer_2026_06_07 \
  --max-total-bytes 176
```

Current generated counts:

- Train: `20,863`
- Eval: `1,090`
- Main task type: `sudoku_next_move`
- Secondary task type: `trace_inversion_final_answer`

The prompt+answer byte limit is `176` to match the completed Run5B context.

## Validate

Validation artifact:

- `runs/benchmarks/private_reasoning_final_answer_validation_2026_06_07/RESULTS.md`
- `runs/benchmarks/private_reasoning_final_answer_validation_2026_06_07/validation_summary.json`

Validation passed:

- no train/eval prompt or prompt-answer overlap
- no visible think markers in prompt/answer
- no strict red-team reject-pattern rows
- no duplicate IDs
- no empty prompts or answers
- every row fits `176` bytes
- private reasoning exists in metadata files

## Train

```bash
python kaggle/train_best_tac_agentic.py \
  --train-jsonl runs/private_reasoning_final_answer_2026_06_07/train.completions.jsonl \
  --eval-jsonl runs/private_reasoning_final_answer_2026_06_07/eval.completions.jsonl \
  --resume runs/kaggle_outputs/run5b_best_capability_fast_v3_completed_jeffkolo_20260607/run5b_best_capability_fast/best.pt \
  --scale base \
  --seq-len 176 \
  --steps 2000 \
  --device auto \
  --supervision-mode answer_only \
  --prompt-field prompt \
  --completion-field answer \
  --precision fp32 \
  --min-healthy-gradient-norm 1e-12 \
  --fail-on-unhealthy-optimization \
  --routing-type base_semantic \
  --routing-top-k 2 \
  --category-route-weight 0.3 \
  --category-route-objective selected_mi \
  --learning-rate 5e-6 \
  --output-dir runs/tac_private_reasoning_final_answer_repair
```

## Recommended Sequence

1. Run this private-reasoning final-answer phase from the completed Run5B checkpoint.
2. Evaluate exact final-answer accuracy on held-out reasoning rows.
3. Then run the red-team-clean ATS repair mix:
   `runs/ats_transfer_10k_anchor_mix_redteam_clean_2026_06_07`.

Do not train ordinary SFT directly on `unified_reasoning_traces.jsonl`; that teaches visible trace style.

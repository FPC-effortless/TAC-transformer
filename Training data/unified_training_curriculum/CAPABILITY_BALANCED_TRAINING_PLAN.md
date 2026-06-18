# Capability-Balanced Training Plan

This dataset path is for teaching the model more than ATS exact answers.

It combines four visible objectives:

- `assistant_qna`: concise English assistant answers
- `agentic_next_action`: compact task-state to next-tool/action behavior
- `private_reasoning_final_answer`: final-answer reasoning tasks without visible trace text
- `ats_transfer`: exact-answer transfer pressure

## Datasets

### Recommended General-Capability Mix

Use:

- `runs/capability_balanced_clean_seq512_2026_06_07/train.completions.jsonl`
- `runs/capability_balanced_clean_seq512_2026_06_07/eval.completions.jsonl`
- `runs/capability_balanced_clean_seq512_2026_06_07/sampling_weights.json`

Counts:

- Train: `25,212`
- Eval: `3,549`
- Train streams:
  - assistant Q&A: `8,000`
  - private-reasoning final answer: `12,000`
  - ATS exact answer: `5,000`
  - compact agentic next-action: `212`

This is the better dataset for fluent English and assistant behavior because `512` bytes admits thousands of clean assistant examples. A one-step CPU smoke from the completed Run5B checkpoint passed at `seq_len=512`.

### External Local-Machine Mix

Use this when the local clean mix is not enough for assistant fluency,
language modelling, reasoning breadth, or tool-use exposure:

- `runs/capability_balanced_external_seq512_2026_06_07/train.completions.jsonl`
- `runs/capability_balanced_external_seq512_2026_06_07/eval.completions.jsonl`
- `runs/capability_balanced_external_seq512_2026_06_07/sampling_weights.json`

Counts:

- Train: `89,645`
- Eval: `6,549`
- Train streams:
  - assistant Q&A: `30,998`
  - English LM continuation: `23,000`
  - private-reasoning final answer: `19,237`
  - agentic/tool next-action: `11,410`
  - ATS exact answer: `5,000`

External sources:

- `HuggingFaceH4/ultrachat_200k`
- `Open-Orca/SlimOrca-Dedup`
- `HuggingFaceFW/fineweb_edu_100BT-shuffled`
- `HuggingFaceTB/cosmopedia`
- `openai/gsm8k`
- `glaiveai/glaive-function-calling-v2`

Validation artifact:

- `runs/benchmarks/external_capability_dataset_validation_2026_06_07/RESULTS.md`
- `runs/benchmarks/external_capability_dataset_validation_2026_06_07/validation_summary.json`

Validation passed with no strict red-team rows, no answer role markers, no
visible think/reasoning tags, no empty prompt/answer rows, no rows over `512`
bytes, and no train/eval prompt or prompt-answer overlap. The artifact is
about `99 MB` across train/eval/manifest files and was built without full
dataset downloads by using the Hugging Face Dataset Viewer rows API.

### Compatibility Mix

Use only when the run must stay at `seq_len=176`:

- `runs/capability_balanced_clean_seq176_2026_06_07/train.completions.jsonl`
- `runs/capability_balanced_clean_seq176_2026_06_07/eval.completions.jsonl`
- `runs/capability_balanced_clean_seq176_2026_06_07/sampling_weights.json`

Counts:

- Train: `17,271`
- Eval: `3,012`
- Assistant Q&A train rows: only `59`

This is not sufficient by itself for fluent assistant behavior. It mainly preserves exact-answer and compact reasoning behavior within the original short context.

## Validation

Validation artifact:

- `runs/benchmarks/capability_balanced_dataset_validation_2026_06_07/RESULTS.md`
- `runs/benchmarks/capability_balanced_dataset_validation_2026_06_07/validation_summary.json`

Both `seq176` and `seq512` pass:

- all four capability streams present
- no train/eval prompt overlap
- no train/eval prompt-answer overlap
- no strict red-team risk rows
- no answer role markers
- no empty prompt/answer rows
- no rows over the target byte limit

Smoke artifact:

- `runs/benchmarks/capability_seq512_resume_smoke_step_2026_06_07`

The smoke run loaded the completed Run5B checkpoint with `--preset run5b_best_capability_fast --seq-len 512`, ran one optimizer step, and optimizer health passed.

## Recommended Training Command

For the external local-machine mix:

```bash
python kaggle/train_best_tac_agentic.py \
  --preset run5b_best_capability_fast \
  --train-jsonl runs/capability_balanced_external_seq512_2026_06_07/train.completions.jsonl \
  --eval-jsonl runs/capability_balanced_external_seq512_2026_06_07/eval.completions.jsonl \
  --sampling-weights-json runs/capability_balanced_external_seq512_2026_06_07/sampling_weights.json \
  --resume runs/kaggle_outputs/run5b_best_capability_fast_v3_completed_jeffkolo_20260607/run5b_best_capability_fast/best.pt \
  --scale base \
  --seq-len 512 \
  --steps 24000 \
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
  --output-dir runs/tac_capability_external_seq512_repair
```

For the smaller local-only mix:

```bash
python kaggle/train_best_tac_agentic.py \
  --preset run5b_best_capability_fast \
  --train-jsonl runs/capability_balanced_clean_seq512_2026_06_07/train.completions.jsonl \
  --eval-jsonl runs/capability_balanced_clean_seq512_2026_06_07/eval.completions.jsonl \
  --sampling-weights-json runs/capability_balanced_clean_seq512_2026_06_07/sampling_weights.json \
  --resume runs/kaggle_outputs/run5b_best_capability_fast_v3_completed_jeffkolo_20260607/run5b_best_capability_fast/best.pt \
  --scale base \
  --seq-len 512 \
  --steps 24000 \
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
  --output-dir runs/tac_capability_balanced_seq512_repair
```

## Remaining Limitation

The agentic rows are compact and repetitive after dedupe: they teach the basic loop of testing, inspecting, patching, and retesting, but they are not yet a broad agentic benchmark. For stronger agentic ability, add more diverse tool-use tasks with held-out goals and success checks.

# ATS Transfer Weighted Fine-Tune Plan

Use this phase after broad full-corpus continuation training.

## 1. Stage A 10k ATS Curriculum

`examples_per_domain=2500` gives 10,000 train rows because the suite has two train domains and two task IDs.

```bash
python experiments/stage_ats_transfer_corpus.py \
  --output-dir runs/ats_transfer_10k_2026_06_07 \
  --seed 37 \
  --examples-per-domain 2500
```

## 2. Prepare Anchor Completion Rows

This export is answer-only clean: prompt fields keep role markers, but answer fields contain only the assistant target text. That matters for TAC because the repair objective is exact answer generation, not prompt reconstruction or chat-template prediction.

```bash
python scripts/prepare_unified_curriculum_for_tac.py \
  --profile full \
  --exclude-multimodal \
  --splits train validation \
  --write-completions \
  --output-dir "Training data/unified_training_curriculum/tac_completions_full_text_only_clean"
```

## 3. Build ATS + Anchor Mix

This keeps all ATS rows, samples 10% anchor rows from the main completion corpus, and writes sampler weights instead of physically duplicating ATS 20-40x.

```bash
python scripts/build_ats_anchor_mix.py \
  --ats-train-jsonl runs/ats_transfer_10k_2026_06_07/train.prepared.jsonl \
  --ats-eval-jsonl runs/ats_transfer_10k_2026_06_07/eval.prepared.jsonl \
  --anchor-train-jsonl "Training data/unified_training_curriculum/tac_completions_full_text_only_clean/train.completions.jsonl" \
  --output-dir runs/ats_transfer_10k_anchor_mix_2026_06_07 \
  --anchor-fraction-of-ats 0.10 \
  --ats-weight 30 \
  --anchor-weight 1 \
  --seed 20260607
```

## 4. Fine-Tune From The Broad Checkpoint

Prefer the red-team-clean anchor mix for the next run. It uses the same ATS rows and weights, but replaces broad anchor rows that matched strict PII/safety/prompt-injection reject patterns.

```bash
python kaggle/train_best_tac_agentic.py \
  --train-jsonl runs/ats_transfer_10k_anchor_mix_redteam_clean_2026_06_07/train.completions.jsonl \
  --eval-jsonl runs/ats_transfer_10k_anchor_mix_redteam_clean_2026_06_07/eval.completions.jsonl \
  --sampling-weights-json runs/ats_transfer_10k_anchor_mix_redteam_clean_2026_06_07/sampling_weights.json \
  --resume runs/kaggle_outputs/run5b_best_capability_fast_v3_completed_jeffkolo_20260607/run5b_best_capability_fast/best.pt \
  --supervision-mode answer_only \
  --prompt-field prompt \
  --completion-field answer \
  --category-route-weight 0.5 \
  --category-route-objective selected_mi \
  --routing-type base_semantic \
  --routing-top-k 2 \
  --learning-rate 1e-5 \
  --scale base \
  --seq-len 192 \
  --steps 4000 \
  --device auto \
  --precision fp32 \
  --output-dir runs/tac_ats_transfer_10k_weighted_redteam_clean_repair
```

## Why This Plan

- Fixes the previous objective problem where preference chosen answers could include assistant role markers in the target.
- Keeps full broad-corpus signal during phase 1.
- Uses answer-only masking for ATS phase 2 so prompts are not trained as targets.
- Uses selected-MI category-route pressure for the routing behavior under test.
- Gives ATS high effective sampling exposure without physically repeating records.
- Keeps a small anchor sample to reduce catastrophic forgetting.

## Validation Artifact

Current corrected artifacts were validated in:

- `runs/benchmarks/dataset_objective_correction_2026_06_07/RESULTS.md`
- `runs/benchmarks/dataset_objective_correction_2026_06_07/dataset_correction_summary.json`
- `runs/benchmarks/unified_curriculum_redteam_2026_06_07/REDTEAM_REPORT.md`
- `runs/benchmarks/unified_curriculum_redteam_2026_06_07/redteam_clean_mix_validation.json`

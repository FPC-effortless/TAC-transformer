# 20M From-Scratch Pretraining Dataset

Use this dataset when training an approximately 20M-parameter TAC model from
scratch. This is full language-model pretraining data, not answer-only
instruction repair data.

## Artifact

- `runs/pretrain_20m_from_scratch_seq512_2026_06_08/train.jsonl`
- `runs/pretrain_20m_from_scratch_seq512_2026_06_08/eval.jsonl`
- `runs/pretrain_20m_from_scratch_seq512_2026_06_08/tokenized/train/manifest.json`
- `runs/pretrain_20m_from_scratch_seq512_2026_06_08/tokenized/valid/manifest.json`
- `runs/pretrain_20m_from_scratch_seq512_2026_06_08/manifest.json`

## Size

- Train records: `104,632`
- Train byte tokens: `190,107,567`
- Eval records: `4,755`
- Eval byte tokens: `4,477,028`
- Sequence length target: `512`

For a 20M-parameter model, this gives about `9.5` training tokens per
parameter. A compute-optimal target would be closer to `400M` tokens, but this
corpus is a practical local-machine build with the current disk budget.

## Mix

Train streams:

- `pretrain_english`: `39,117` records from FineWeb-Edu 100BT shuffled
- `pretrain_textbook`: `16,117` records from Cosmopedia web samples
- `pretrain_seed`: `49,398` records from the validated local assistant,
  reasoning, tool-use, continuation, and ATS corpus

The seed stream is included so the model is not only exposed to raw prose. It
does not replace later instruction fine-tuning.

## Validation

Validation artifact:

- `runs/benchmarks/pretrain_20m_dataset_validation_2026_06_08/RESULTS.md`
- `runs/benchmarks/pretrain_20m_dataset_validation_2026_06_08/validation_summary.json`

Validation passed:

- no empty text rows
- no duplicate train/eval text overlap
- no strict red-team reject rows
- tokenized file sizes match manifest token counts
- `TokenizedMemmapBatcher` sampled train/eval batches at shape `[2, 512]`

## Recommended 20M Config

The closest checked TAC config is:

- `d_model=224`
- `n_layers=8`
- `n_heads=8`
- `n_programs=24`
- parameter count: `20,013,416`

## Training Command

```bash
python kaggle/train_best_tac_agentic.py \
  --preset run5b_best_capability_fast \
  --train-tokenized-manifest runs/pretrain_20m_from_scratch_seq512_2026_06_08/tokenized/train/manifest.json \
  --eval-tokenized-manifest runs/pretrain_20m_from_scratch_seq512_2026_06_08/tokenized/valid/manifest.json \
  --scale base \
  --d-model 224 \
  --n-heads 8 \
  --n-layers 8 \
  --n-programs 24 \
  --seq-len 512 \
  --steps 30000 \
  --device auto \
  --supervision-mode full_lm \
  --precision fp32 \
  --min-healthy-gradient-norm 1e-12 \
  --fail-on-unhealthy-optimization \
  --routing-type base_semantic \
  --routing-top-k 2 \
  --category-route-weight 0.1 \
  --category-route-objective selected_mi \
  --learning-rate 3e-4 \
  --output-dir runs/tac_20m_from_scratch_pretrain_seq512
```

After this pretraining phase, run instruction/answer-only fine-tuning on:

- `runs/capability_balanced_external_seq512_2026_06_07`


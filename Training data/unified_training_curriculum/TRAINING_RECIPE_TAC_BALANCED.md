# TAC Balanced Training Recipe

Use this balanced train view for TAC experiments where source dominance would distort identity-transfer results.

Balanced train files:

- `splits_balanced/train/unified_cpt.jsonl`
- `splits_balanced/train/unified_sft_messages.jsonl`
- `splits_balanced/train/unified_reasoning_traces.jsonl`
- `splits_balanced/train/unified_preference_pairs.jsonl`

Validation files remain:

- `splits/validation/unified_cpt.jsonl`
- `splits/validation/unified_sft_messages.jsonl`
- `splits/validation/unified_reasoning_traces.jsonl`
- `splits/validation/unified_preference_pairs.jsonl`

Per-source cap: `35000` training records across all train streams.
Total balanced train records: `150902`.
Max dataset share: `0.232`.

The full unbalanced unified files are retained for inspection and alternate sampling strategies.

## Prepare TAC Text JSONL

The current TAC trainers consume JSONL rows with a `text` field. Build those prepared files from the balanced curriculum with:

```bash
python scripts/prepare_unified_curriculum_for_tac.py \
  --profile balanced \
  --exclude-multimodal \
  --write-prepared \
  --output-dir "Training data/unified_training_curriculum/tac_prepared_balanced_text_only"
```

This writes:

- `Training data/unified_training_curriculum/tac_prepared_balanced_text_only/train.prepared.jsonl`
- `Training data/unified_training_curriculum/tac_prepared_balanced_text_only/eval.prepared.jsonl`
- `Training data/unified_training_curriculum/tac_prepared_balanced_text_only/manifest.json`

Preference pairs are serialized as prompt plus chosen response only for full-LM training. Keep `unified_preference_pairs.jsonl` for a separate DPO/preference objective if you add one.

Structured multimodal rows are separated into:

- `multimodal_dataset/unified_sft_messages.jsonl`
- `multimodal_dataset/splits/train/unified_sft_messages.jsonl`
- `multimodal_dataset/splits/validation/unified_sft_messages.jsonl`

The current TAC trainer is byte-text only, so use `--exclude-multimodal` until a real multimodal encoder path exists.

## Train TAC

Recommended local smoke run:

```bash
python kaggle/train_best_tac_agentic.py \
  --train-jsonl "Training data/unified_training_curriculum/tac_prepared_balanced_text_only/train.prepared.jsonl" \
  --eval-jsonl "Training data/unified_training_curriculum/tac_prepared_balanced_text_only/eval.prepared.jsonl" \
  --preset cpu_research_tac \
  --scale smoke \
  --steps 100 \
  --eval-every 50 \
  --device auto \
  --output-dir runs/tac_unified_balanced_smoke
```

Recommended larger run:

```bash
python kaggle/train_best_tac_agentic.py \
  --train-jsonl "Training data/unified_training_curriculum/tac_prepared_balanced_text_only/train.prepared.jsonl" \
  --eval-jsonl "Training data/unified_training_curriculum/tac_prepared_balanced_text_only/eval.prepared.jsonl" \
  --preset run5b_best_capability_fast \
  --scale base \
  --steps 10000 \
  --eval-every 500 \
  --checkpoint-every 500 \
  --device auto \
  --precision auto \
  --output-dir runs/tac_unified_balanced_run
```

Token count reports:

- `TOKEN_COUNT_TAC_BALANCED.md`
- `TOKEN_COUNT_TAC_FULL.md`
- `TOKEN_COUNT_TAC_BALANCED_TEXT_ONLY.md`
- `TOKEN_COUNT_TAC_FULL_TEXT_ONLY.md`

Balance and BPE notes:

- `DATASET_BALANCE_AND_BPE_NOTES.md`
- `FULL_CORPUS_SOURCE_WEIGHTS.json`
- `FULL_CORPUS_WEIGHTED_SAMPLING_PLAN.md`
- `ATS_TRANSFER_WEIGHTED_FINETUNE_PLAN.md`

For TAC transfer experiments, use the balanced view. For production-style continuation training, prefer the full text-only view with `--sampling-weights-json FULL_CORPUS_SOURCE_WEIGHTS.json` so the dominant source is not discarded.

## BPE/Subword Tokenized Training

The model can train on BPE-style IDs if the JSONL is already tokenized, for example with an `input_ids` field, and `--vocab-size` matches that tokenizer.

```bash
python scripts/prepare_tac_tokenized_corpus.py \
  --train-jsonl path/to/train.bpe.jsonl \
  --valid-jsonl path/to/eval.bpe.jsonl \
  --output-dir tokenized_bpe \
  --vocab-size 32000 \
  --tokens-field input_ids \
  --eos-token-id 2
```

```bash
python kaggle/train_best_tac_agentic.py \
  --train-tokenized-manifest tokenized_bpe/train/manifest.json \
  --eval-tokenized-manifest tokenized_bpe/valid/manifest.json \
  --vocab-size 32000 \
  --preset run5b_best_capability_fast \
  --scale base \
  --steps 10000 \
  --device auto \
  --output-dir runs/tac_bpe_run
```

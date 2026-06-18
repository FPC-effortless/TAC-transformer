# Dataset Balance And BPE Notes

## Why The Full View Is Unbalanced

The full text-only curriculum keeps every available record. It is dominated by one source:

| Source | TAC byte tokens | Share |
| --- | ---: | ---: |
| `prepared_corpus_agentic_hard` | 1,056,953,453 | 75.2% |
| `distillation_datasets_70k` | 133,852,304 | 9.5% |
| `enriched_transcripts_llm_dataset` | 111,838,166 | 8.0% |

To match `prepared_corpus_agentic_hard` by only increasing the two smaller sources, `distillation_datasets_70k` would need about 7.9x more exposure and `enriched_transcripts_llm_dataset` would need about 9.5x more exposure. That is possible through oversampling or synthetic expansion, but it would make the full corpus much larger and more repetitive.

## Recommended Balance

Use the balanced text-only train view for TAC transfer experiments. It already raises the two sources substantially while capping the dominant source:

| Source | Balanced train TAC byte tokens | Share |
| --- | ---: | ---: |
| `enriched_transcripts_llm_dataset` | 106,225,909 | 28.4% |
| `prepared_corpus_agentic_hard` | 87,188,397 | 23.3% |
| `distillation_datasets_70k` | 82,846,384 | 22.2% |

This is more useful than the full view for transfer testing because no single source controls most updates. It is still not perfectly token-balanced because the cap is record-based and transcripts have longer records.

## Recommended Full-Corpus Weighted Sampling

For production-style continuation training, keep the full text-only train split and use source-aware sampling weights instead of the capped balanced view.

Weights are stored in:

- `FULL_CORPUS_SOURCE_WEIGHTS.json`

Current weights:

- `prepared_corpus_agentic_hard`: implicit `1.0`
- `distillation_datasets_70k`: `3.0`
- `enriched_transcripts_llm_dataset`: `4.0`

This changes the effective train-token exposure to:

| Source | Effective share |
| --- | ---: |
| `prepared_corpus_agentic_hard` | 52.7% |
| `enriched_transcripts_llm_dataset` | 22.3% |
| `distillation_datasets_70k` | 19.9% |

Prepare full text-only JSONL without writing the duplicate `all` file:

```bash
python scripts/prepare_unified_curriculum_for_tac.py \
  --profile full \
  --exclude-multimodal \
  --splits train validation \
  --write-prepared \
  --output-dir "Training data/unified_training_curriculum/tac_prepared_full_text_only"
```

Train with weighted full-LM and selected-MI routing:

```bash
python kaggle/train_best_tac_agentic.py \
  --train-jsonl "Training data/unified_training_curriculum/tac_prepared_full_text_only/train.prepared.jsonl" \
  --eval-jsonl "Training data/unified_training_curriculum/tac_prepared_full_text_only/eval.prepared.jsonl" \
  --sampling-weights-json "Training data/unified_training_curriculum/FULL_CORPUS_SOURCE_WEIGHTS.json" \
  --category-route-weight 0.5 \
  --category-route-objective selected_mi \
  --preset run5b_best_capability_fast \
  --scale base \
  --steps 20000 \
  --device auto \
  --output-dir runs/tac_full_weighted_jsonl
```

For answer-only masking, export only completion-capable SFT/preference rows:

```bash
python scripts/prepare_unified_curriculum_for_tac.py \
  --profile full \
  --exclude-multimodal \
  --splits train validation \
  --write-completions \
  --output-dir "Training data/unified_training_curriculum/tac_completions_full_text_only_clean"
```

The clean completion export keeps role markers in `prompt` only. The `answer` field is assistant target text only, including preference `chosen` rows.

Then train answer-only with the same source weights:

```bash
python kaggle/train_best_tac_agentic.py \
  --train-jsonl "Training data/unified_training_curriculum/tac_completions_full_text_only_clean/train.completions.jsonl" \
  --eval-jsonl "Training data/unified_training_curriculum/tac_completions_full_text_only_clean/eval.completions.jsonl" \
  --supervision-mode answer_only \
  --prompt-field prompt \
  --completion-field answer \
  --sampling-weights-json "Training data/unified_training_curriculum/FULL_CORPUS_SOURCE_WEIGHTS.json" \
  --category-route-weight 0.5 \
  --category-route-objective selected_mi \
  --preset run5b_best_capability_fast \
  --scale base \
  --steps 5000 \
  --device auto \
  --output-dir runs/tac_full_weighted_answer_only
```

## BPE-Style Training

The TAC model can train on BPE/subword IDs if:

- the dataset is pretokenized into integer IDs, for example an `input_ids` JSON field,
- `--vocab-size` matches the tokenizer ID range,
- training uses the tokenized memmap path.

Build tokenized memmaps from pretokenized JSONL:

```bash
python scripts/prepare_tac_tokenized_corpus.py \
  --train-jsonl path/to/train.bpe.jsonl \
  --valid-jsonl path/to/eval.bpe.jsonl \
  --output-dir tokenized_bpe \
  --vocab-size 32000 \
  --tokens-field input_ids \
  --eos-token-id 2
```

Train from those manifests:

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

The current BPE path supports full-LM next-token training. Answer-only masking and category-route loss remain JSONL-text paths unless separate pretokenized prompt/completion masks are added.

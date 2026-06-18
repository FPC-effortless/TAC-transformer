# Multimodal Dataset

This dataset contains the structured multimodal-style SFT rows separated from the unified text TAC curriculum.

Files:

- `unified_sft_messages.jsonl`
- `splits/train/unified_sft_messages.jsonl`
- `splits/validation/unified_sft_messages.jsonl`
- `splits_balanced/train/unified_sft_messages.jsonl`

Current count: 17 records total, all in train, all sourced from `llm-distillation-spec`.

Use this dataset only with a trainer that has a real multimodal encoder or adapter path. The current TAC byte-text trainer should use the text-only prepared view generated with `--exclude-multimodal`.

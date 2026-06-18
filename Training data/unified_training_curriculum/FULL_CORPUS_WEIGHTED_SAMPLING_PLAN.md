# Full Corpus Weighted Sampling Plan

This plan keeps the full text-only train split and changes sampling exposure instead of capping records.

Weights:
- `*`: `1.0`
- `distillation_datasets_70k`: `3.0`
- `enriched_transcripts_llm_dataset`: `4.0`

Raw train TAC byte tokens: `1333810741`
Effective train TAC byte tokens after weighting: `1905062064`

| Dataset | Raw tokens | Weight | Effective tokens | Effective share |
| --- | ---: | ---: | ---: | ---: |
| `prepared_corpus_agentic_hard` | 1004069347 | 1.00 | 1004069347 | 52.7% |
| `enriched_transcripts_llm_dataset` | 106225909 | 4.00 | 424903636 | 22.3% |
| `distillation_datasets_70k` | 126286798 | 3.00 | 378860394 | 19.9% |
| `angrygiraffe__claude-opus-4.6-4.7-reasoning-8.7k` | 44397825 | 1.00 | 44397825 | 2.3% |
| `Jackrong__Claude-opus-4.6-TraceInversion-9000x` | 23872207 | 1.00 | 23872207 | 1.3% |
| `distillation_datasets` | 14025151 | 1.00 | 14025151 | 0.7% |
| `enriched_sudoku_dataset` | 6189720 | 1.00 | 6189720 | 0.3% |
| `llm-distillation-spec` | 3377343 | 1.00 | 3377343 | 0.2% |
| `enriched_new_folder_llm_dataset` | 3224409 | 1.00 | 3224409 | 0.2% |
| `openai__gsm8k` | 1165123 | 1.00 | 1165123 | 0.1% |
| `SupraLabs__SupraThink-Dataset-500x` | 924739 | 1.00 | 924739 | 0.0% |
| `ReasonCore__open-spatial-reasoning` | 52170 | 1.00 | 52170 | 0.0% |

Use this with `--sampling-weights-json FULL_CORPUS_SOURCE_WEIGHTS.json` on the full text-only prepared JSONL.

# Dataset Preparation

This repo can now prepare both agent JSONL records and procedural JSON-array records for TAC training.

## Agent Planner Dataset

Use `master_500k` as the main split:

```bash
python -m tac_transformer.data \
  --input "C:\Users\warit\OneDrive\Documents\transformer fusion experiment\kaggle_dataset\data\master_500k\master_train.jsonl" \
  --output runs\prepared_master_500k_train.jsonl \
  --duplicate-cap 3
```

The preparer:

- drops rows missing `final_answer`,
- caps exact duplicate `(prompt, final_answer, target_plan)` triples,
- serializes each row into a `text` field,
- leaves the source dataset untouched.

Avoid the smaller `master/` split except for debugging; it has the same schema but much less data.

## USEF Procedural Dataset

The USEF generated data at `C:\Users\warit\OneDrive\Documents\My Programs\usef\artifacts\generated_data` is JSON-array based. The same preparer streams array records and caps exact serialized duplicates:

```bash
python -m tac_transformer.data \
  --input "C:\Users\warit\OneDrive\Documents\My Programs\usef\artifacts\generated_data\l2_verification_trajectories.json" \
  --output runs\prepared_l2_verification.jsonl \
  --duplicate-cap 3
```

Best USEF files for TAC identity-field training:

- `arc_curriculum_dataset.json`
- `l0_primitive_transformations.json`
- `l1_compositional_procedures.json`
- `l2_verification_trajectories.json`
- `l3_procedural_memory.json`
- `l4_long_horizon_workflows.json`
- `l7_open_ended_growth.json`

Use lower weight or stricter duplicate caps for highly repetitive files:

- `advanced_tool_use_dataset.json`
- `algorithmic_dataset.json`
- `clrs_benchmark_dataset.json`
- `l6_cross_domain_transfer.json`
- `math_dataset.json`

## Training From Prepared JSONL

Prepared rows contain a `text` field. Train on them with:

```bash
python kaggle/train_tac_synthetic.py \
  --dataset-jsonl runs\prepared_master_500k_train.jsonl \
  --device auto \
  --steps 1000
```

The text path uses a byte-level batcher, so no external tokenizer is required for first experiments.

## Build The Recommended Corpus

Use the bundled corpus builder to prepare `master_500k` plus the recommended USEF procedural files:

```bash
python kaggle/prepare_tac_corpus.py \
  --agent-data-root "C:\Users\warit\OneDrive\Documents\transformer fusion experiment\kaggle_dataset\data" \
  --usef-root "C:\Users\warit\OneDrive\Documents\My Programs\usef\artifacts\generated_data" \
  --output-dir runs\prepared_corpus \
  --duplicate-cap 3
```

This writes:

- `runs\prepared_corpus\train.prepared.jsonl`
- `runs\prepared_corpus\eval.prepared.jsonl`
- `runs\prepared_corpus\manifest.json`

The builder sanitizes secret-like strings such as fake API keys, tokens, and passwords.

## Extend To A 1B-Token Knowledge-Work Curriculum

After building `runs\prepared_corpus`, extend it to an effective 1B-token curriculum:

```bash
python kaggle/extend_corpus_to_1b.py \
  --base-dir runs\prepared_corpus \
  --output-dir runs\prepared_corpus_1b \
  --target-tokens 1000000000
```

This keeps the prepared procedural/RAG/agent corpus, then appends generated knowledge-work records covering:

- multi-hop RAG with citations and distractor rejection,
- bounded agentic tool-use and recovery,
- synthesis for business/research workflows,
- coding and test repair,
- spreadsheet analysis,
- evidence-grounded research briefs.

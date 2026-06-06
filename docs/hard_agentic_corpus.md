# Hard Agentic Corpus

Date: 2026-05-29

This corpus replaces the repetitive `prepared_corpus_1b` expansion for serious TAC agentic training.

## Location

Folder:

```text
runs/prepared_corpus_agentic_hard
```

Upload zip:

```text
runs/prepared_corpus_agentic_hard_upload.zip
```

The upload zip contains:

```text
train.prepared.jsonl
eval.prepared.jsonl
manifest.json
```

## Build Command

```bash
python kaggle/build_hard_agentic_corpus.py \
  --base-dir runs/prepared_corpus \
  --output-dir runs/prepared_corpus_agentic_hard \
  --template-cap 3 \
  --exact-cap 1 \
  --hard-train-records 120000 \
  --hard-eval-records 5000
```

## Manifest Summary

| Split | Records | Approx Tokens |
| --- | ---: | ---: |
| train | 428,671 | 263,030,010 |
| eval | 5,441 | 1,274,261 |

Train parts:

| Part | Records | Approx Tokens |
| --- | ---: | ---: |
| deduped base train | 308,671 | 236,626,157 |
| hard generated train | 120,000 | 26,404,372 |

## Audit

First 50k train records:

| Metric | Value |
| --- | ---: |
| exact unique rate | 1.0000 |
| exact duplicate texts | 0 |
| normalized template unique rate | 0.9074 |

Hard generated train:

| Metric | Value |
| --- | ---: |
| records | 120,000 |
| exact unique rate | 1.0000 |
| exact duplicate texts | 0 |
| normalized template unique rate | 0.8009 |

Hard generated train domain balance:

| Domain | Records |
| --- | ---: |
| stale_memory_rejection | 20,000 |
| tool_choice | 20,000 |
| repair_after_failure | 20,000 |
| memory_counterfactual | 20,000 |
| verification_planning | 20,000 |
| argument_schema | 20,000 |

Eval:

| Metric | Value |
| --- | ---: |
| records | 5,441 |
| exact unique rate | 1.0000 |
| exact duplicate texts | 0 |
| normalized template unique rate | 0.8734 |

## Kaggle

If uploaded as `tac-hard-agentic-corpus`, use:

```text
/kaggle/input/tac-hard-agentic-corpus/train.prepared.jsonl
/kaggle/input/tac-hard-agentic-corpus/eval.prepared.jsonl
```

If Kaggle nests the folder, find the paths with:

```python
!find /kaggle/input -name "train.prepared.jsonl" -o -name "eval.prepared.jsonl"
```

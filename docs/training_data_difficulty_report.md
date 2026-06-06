# Training Data Difficulty Report

Date: 2026-05-29

Question: why did the Kaggle training loss collapse so quickly, and how hard is the prepared agentic corpus?

## Bottom Line

The current `prepared_corpus_1b` is too easy for serious agentic training.

It is useful for teaching the model:

- XML-ish record format;
- common tool/action vocabulary;
- simple plan/result/final-answer patterns;
- byte-level syntax regularities.

It is not strong enough to prove agentic reasoning, tool use, memory use, or long-horizon behavior. The fast loss collapse is expected.

## Corpus Shape

From `runs/prepared_corpus_1b/manifest.json`:

| Part | Approx Tokens | Notes |
| --- | ---: | --- |
| base prepared train | 336,264,445 | Mixed prepared corpus |
| generated extension | 663,735,663 | Synthetic knowledge-work generator |
| total train | 1,000,000,108 | Mostly generated or templated data |
| eval | 174,381 | Only 441 records |

File sizes:

| File | Size |
| --- | ---: |
| `train.prepared.jsonl` | 4.68 GB |
| `knowledge_work_generated.prepared.jsonl` | 3.20 GB |
| `eval.prepared.jsonl` | 0.81 MB |

The generated extension is about 90% of records by count and dominates random-line sampling.

## Difficulty Evidence

### Generated Data Is Highly Repetitive

First 50,000 generated records:

| Metric | Value |
| --- | ---: |
| exact unique rate | 0.0240 |
| exact duplicate texts | 48,800 / 50,000 |
| normalized template unique rate | 0.0176 |
| average chars per record | 651.5 |

Cross-file sample, every 1000th generated record across all 4,075,762 generated records:

| Metric | Value |
| --- | ---: |
| sample size | 4,076 |
| exact unique rate | 0.0226 |
| normalized unique rate | 0.0069 |
| top exact repeat count | 361 |
| top normalized repeat count | 367 |

So the generated extension is not just templated; many records have exactly repeated text.

### Eval Is Too Small

`eval.prepared.jsonl` has only 441 records. It has no exact duplicate texts, but it is still small and schema-similar:

| Metric | Value |
| --- | ---: |
| records | 441 |
| exact unique rate | 1.0 |
| normalized template unique rate | 0.8005 |
| unique prompt rate | 0.7868 |
| tag char fraction | 0.1412 |

This eval set can detect total failure, but it is too small and too close to the training format to validate commercial agentic ability.

### The Training Logs Match The Data

Kaggle train loss dropped very quickly:

```text
step 50:  next_token_loss 4.93
step 250: next_token_loss 1.92
step 400: next_token_loss 0.26
step 550: next_token_loss 0.067
```

`exp(0.067) ~= 1.07`, which means the next byte/token is nearly deterministic for the sampled training windows. That is consistent with repeated templates and near-duplicate generated records.

## Why This Happened

`extend_corpus_to_1b.py` used `generate_knowledge_work_records()` to expand the corpus to 1B approximate tokens. The generator has limited combinatorics:

- 6 domains;
- 10 topics;
- small finite choices for tools, failures, recoveries, metrics, bugs, audiences, claims, columns;
- repeated fixed section tags and answer shapes;
- many generated records do not include the unique index inside the actual `text` field.

That creates a large corpus by byte count, but not by behavioral diversity.

## Decision

Do not treat the current 1B corpus as a hard agentic training set.

It is acceptable as:

- format pretraining;
- a short warmup;
- smoke testing the TAC training stack;
- teaching basic tool-plan syntax.

It is not acceptable as the main evidence that TAC learned agentic reasoning.

## Recommended Fix

Build a harder `prepared_corpus_agentic_hard` before spending full Kaggle budget:

1. Deduplicate by exact `text` hash.
2. Deduplicate again by normalized template hash.
3. Cap each normalized template family.
4. Increase real task traces and reduce synthetic template expansion.
5. Generate tasks where the answer depends on nontrivial retrieved content, failed tool results, repair decisions, and hidden constraints.
6. Add held-out eval splits by tool, domain, template family, and goal type.
7. Add negative/counterfactual records:
   - wrong tool result;
   - stale memory;
   - irrelevant retrieval;
   - conflicting instruction;
   - failed verification;
   - bad plan that must be rejected.
8. Track evals beyond next-token loss:
   - tool choice accuracy;
   - argument exact match;
   - repair action accuracy;
   - memory carry/reset/shuffle;
   - final answer correctness;
   - verification pass rate.

## Immediate Kaggle Recommendation

The current run can be stopped after a checkpoint if the goal is serious agentic training. Continuing for 9 hours will mostly overfit easy templates.

Use the checkpoint as a format-warmup model, then train the next run on a deduplicated and harder agentic corpus.

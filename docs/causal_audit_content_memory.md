# Content Memory Causal Audit

Date: 2026-06-01

## Question

Can TAC content-addressed memory answer recall probes when the query key is removed or randomized?

This matters because the promoted content store is trained with context/query splits. The read path should depend on carried context state plus the query cue, not on future information hidden in the query sequence.

## Implementation

Added:

`kaggle/audit_content_memory_causality.py`

The audit trains the current best content-addressed TAC stack on chunked recall, then evaluates:

- normal carry
- randomized-query carry
- randomized-query reset
- randomized-query shuffled state

The randomized-query batcher preserves the query and recall marker tokens by default but replaces the actual query content/key tokens with random data tokens that exclude the target value.

The default audit task is `multi_key`, not `single_key`. A single-key context has only one meaningful value, so a model can sometimes answer from carried state without using a query key. `multi_key` is the correct leakage test because the query key selects among several values.

## Local Result

Artifact:

`runs/analysis/content_memory_causal_audit_multikey_2026_06_01/audit.json`

Settings:

- task: `multi_key`
- steps: 60
- batch size: 16
- eval batches: 16
- eval batch size: 16
- device: CPU

| Probe | Value accuracy |
| --- | ---: |
| normal carry | 0.2539 |
| randomized-query carry | 0.0625 |
| randomized-query reset | 0.0117 |
| randomized-query shuffled | 0.0117 |

Verdict:

`pass`

The randomized-query carry accuracy falls below the 0.10 leakage threshold and far below normal carry. This local audit does not show causal leakage through the query representation on the multi-key task.

## Caveat

A single-key smoke audit produced randomized-query carry accuracy above the threshold. That is not treated as a leakage verdict because the single-key context contains only one value; the model can exploit carried state without needing a discriminative query key. Future causal audits should use multi-key or harder variants where query identity is necessary.

## Next Check

Repeat this audit on the full promoted `base_semantic_mi_0p5` training run once that checkpoint exists. The current result validates the content-memory path locally, not every future routing objective.

# Iterative Content Retrieval Validation

Date: 2026-05-30

## Question

Content-addressed k1 solves direct key/value recall but loses multi-hop. This tested whether a minimal two-step lookup can improve chained recall:

```text
read_1 = lookup(query_hidden)
read_2 = lookup(read_1)
read = learned_gate(query_hidden, read_1, confidence) * read_1
     + (1 - gate) * read_2
```

The gate is initialized toward the first read so direct recall is not immediately overwritten by an untrained second lookup.

## Implementation

Added:

- `content_read_steps`
- iterative content read path when `content_read_steps > 1`
- `content_iterative_k1` and `content_iterative_k2` harder-matrix variants
- CLI support for `--content-read-steps`

## Focused Multi-Hop Result

| Variant | Effective | Multi-hop carry | Carry-reset | Carry-shuffled | TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `content_iterative_k2` | 3/3 | 0.0508 | 0.0417 | 0.0313 | 0.4686 |
| `base_routing` | 3/3 | 0.0495 | 0.0234 | 0.0286 | 0.5242 |
| `content_iterative_k1` | 3/3 | 0.0482 | 0.0365 | 0.0313 | 0.5003 |
| `content_addressed_k1` | 3/3 | 0.0365 | 0.0221 | 0.0169 | 0.5197 |

`content_iterative_k2` narrowly beats BASE on carry and improves carry-reset delta. This confirms that chained lookup is the right direction, but the gain is small.

## Full Matrix Result

| Variant | Effective | Task wins | Mean carry | Carry-reset | Carry-shuffled | TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `content_addressed_k1` | 15/15 | 4 | 0.4349 | 0.4229 | 0.4185 | 0.4840 |
| `content_iterative_k2` | 15/15 | 1 | 0.3552 | 0.3406 | 0.3391 | 0.4630 |
| `base_routing` | 15/15 | 0 | 0.0677 | 0.0529 | 0.0508 | 0.5413 |

Per-task carry:

| Task | BASE | Content k1 | Iterative k2 |
| --- | ---: | ---: | ---: |
| longer single-key | 0.0677 | 0.6497 | 0.5495 |
| multi-key | 0.0443 | 0.7448 | 0.5638 |
| delayed-query | 0.0885 | 0.6341 | 0.5052 |
| noisy-key | 0.0885 | 0.1094 | 0.1068 |
| multi-hop | 0.0495 | 0.0365 | 0.0508 |

## Decision

Do not promote iterative retrieval as the default.

It solves the narrow multi-hop target by a small margin, but it gives up too much direct-recall accuracy. The default remains `content_addressed_k1`.

What this proves:

- Multi-hop needs additional retrieval steps.
- Always-on second lookup is too blunt.
- The next version should be conditional: halt/continue, verifier-gated, or query-type-gated retrieval.

The better next mechanism is Phase 4A/4B style:

```text
try one lookup
verify whether answer is sufficient
continue only when verifier/halt gate says the query is chained
```

Artifacts:

- `runs/benchmarks/content_iterative_multihop_2026_05_30/RESULTS.md`
- `runs/benchmarks/content_iterative_full_matrix_2026_05_30/RESULTS.md`

## Conditional Retrieval Follow-Up

The next attempted fix was confidence-gated two-step retrieval:

```text
read_1 = lookup(query_hidden)
read_2 = lookup(read_1)
continue = sigmoid(4 * (confidence(read_2) - confidence(read_1)))
read = (1 - continue) * read_1 + continue * read_2
```

This is cheaper than a learned halt/verifier because it uses the content store's own cosine match score as the continue signal.

Focused multi-hop result:

| Variant | Effective | Multi-hop carry | Carry-reset | Carry-shuffled | TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `content_iterative_k2` | 3/3 | 0.0508 | 0.0417 | 0.0313 | 0.4150 |
| `base_routing` | 3/3 | 0.0495 | 0.0234 | 0.0286 | 0.5254 |
| `content_confidence_iterative_k1` | 3/3 | 0.0482 | 0.0313 | 0.0273 | 0.4775 |
| `content_confidence_iterative_k2` | 3/3 | 0.0482 | 0.0339 | 0.0260 | 0.4507 |
| `content_addressed_k1` | 3/3 | 0.0365 | 0.0221 | 0.0169 | 0.4934 |

Decision: do not promote confidence-gated retrieval.

It confirms that a second lookup can recover some multi-hop signal over single-pass content memory, but cosine confidence alone is not a strong enough verifier. The learned two-step k2 path still has the best multi-hop carry, and the global default remains single-step `content_addressed_k1` because it dominates direct recall.

Artifact:

- `runs/benchmarks/conditional_iterative_focused_2026_05_30/RESULTS.md`

## Synthesis-Gated Retrieval Follow-Up

The Phase 1A roadmap proposed a learned synthesis step after the second retrieval:

```text
read_1 = lookup(query_hidden)
read_2 = lookup(read_1)
synthesis_input = [query_hidden, read_1, read_2, read_1 - read_2, read_1 * read_2]
synthesized = linear(synthesis_input)
read = gated_blend(read_1, synthesized)
```

Implementation details:

- `content_read_gate_type="synthesis"`
- `content_read_synthesis`: learned projection from `5 * d_model` to `d_model`
- `content_read_synthesis_gate`: learned scalar gate for how much synthesized representation to use
- `content_synthesis_gate`: logged metric
- synthesis initialized conservatively so the path starts close to the first retrieved value

Focused multi-hop result:

| Variant | Effective | Multi-hop carry | Carry-reset | Carry-shuffled | TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `content_iterative_k2` | 3/3 | 0.0508 | 0.0417 | 0.0313 | 0.4253 |
| `base_routing` | 3/3 | 0.0495 | 0.0234 | 0.0286 | 0.5435 |
| `content_synthesis_k1` | 3/3 | 0.0417 | 0.0247 | 0.0273 | 0.4548 |
| `content_synthesis_k2` | 3/3 | 0.0391 | 0.0234 | 0.0299 | 0.4336 |
| `content_addressed_k1` | 3/3 | 0.0365 | 0.0221 | 0.0169 | 0.4472 |

Decision: Phase 1A synthesis gate fails the go/no-go.

It improves over single-pass content memory but does not beat BASE (`0.0495`) or learned iterative k2 (`0.0508`) on multi-hop. It is therefore an ablation, not a promoted reasoning path. The result argues that simple feature synthesis over two retrieved vectors is not enough; multi-hop likely needs either a retrieval graph or a supervised verifier/halt loop.

Artifact:

- `runs/benchmarks/synthesis_iterative_focused_2026_05_30/RESULTS.md`

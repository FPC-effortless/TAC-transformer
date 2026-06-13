# TAC v0.2 Stage-4 Roadmap

TAC v0.1 is a Stage-3 research asset: public, documented, reproducible, and
externally replicated on a bounded validation pack.

TAC v0.2 should answer the next hard question:

> Do TAC mechanisms survive scaling?

## Stage-4 Goal

Demonstrate that the TAC mechanisms validated in bounded benchmarks remain
useful in a significantly larger learned system.

The goal is not to claim frontier-model superiority. The goal is to test
survival of mechanism under scale.

## Candidate Milestones

### TAC-v0.2-A: 100M+ Parameter TAC Model

Train a materially larger TAC model with:

- persistent IdentityState
- routed program modules
- repair/control heads
- state-compression diagnostics
- transformer baseline with matched training budget

Required outputs:

- loss curves
- state-carry versus reset comparisons
- program knockout tests
- routing selectivity
- throughput and memory cost

### TAC-v0.2-B: Real Repository Repair Corpus

Move from bounded simulated repair chains to a small real repository repair
corpus.

Dataset requirements:

- real failing tests
- multi-file bugs
- incomplete or misleading local tests
- known patches for evaluation
- held-out repositories

Metrics:

- first-pass root-cause selection
- patch success
- regression avoidance
- chain completion
- state continuity
- token/context cost

### TAC-v0.2-C: Multi-Session Software Workflow

Evaluate whether TAC can maintain project state across sessions.

Task pattern:

```text
Session 1: inspect repository
Session 2: diagnose failure
Session 3: implement patch
Session 4: run tests and repair
Session 5: document and continue
```

Controls:

- transformer with full context
- transformer with retrieval
- TAC carried state
- TAC reset state

### TAC-v0.2-D: External Benchmark Comparisons

Candidate external comparisons:

- software repair benchmarks
- long-context memory tasks
- agent workflow tasks
- ARC-style adaptation tasks if scoped carefully

Do not use external benchmarks to claim broad AGI. Use them to test whether TAC
state/control mechanisms transfer beyond local synthetic benchmarks.

## Decision Gates

TAC v0.2 should validate only if:

- carried TAC state beats reset state
- program knockout still harms performance
- compression benefit survives larger training
- repair/control metrics remain positive outside bounded toy tasks
- gains are not explained by retrieval-only or prompt-only controls

## Stop Conditions

Stop or redesign if:

- routing collapses at scale
- state carry no longer beats reset
- program knockouts stop mattering
- repair gains disappear on real repositories
- context compression only works in synthetic tasks

## Recommended Next Action

Do not start TAC v0.2 until TAC v0.1 has been presented to external researchers
and at least a few serious critiques have been collected.


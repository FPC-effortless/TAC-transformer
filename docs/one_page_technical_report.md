# TAC-Transformer One-Page Technical Report

## Problem

Transformers are strong sequence models, but standard transformer inference does not expose a first-class persistent computational state for reusable task structures. TAC investigates whether adding persistent identity state, routed programs, and structure memory can improve controlled long-horizon recall, compression, and bounded agentic repair workflows.

## Architecture

At a high level:

```text
input tokens
  -> token / positional representation
  -> transformer-style attention and MLP path
  -> TAC identity field and program routing path
  -> optional memory / structure read-write path
  -> language, behavior, repair, or structure-conditioned output
```

Core mechanisms:

- persistent `IdentityState` carried across chunks or sessions;
- program routing under energy / capacity constraints;
- coherence-modulated attention and program-conditioned computation;
- content-addressed and program-conditioned memory reads;
- structure slots, structure bridge, lifecycle scoring, and procedural memory in TAC-SCM.

## Benchmark logic

The repository emphasizes causal controls, not only raw accuracy.

Typical interventions:

- carry state normally;
- reset state;
- shuffle state;
- remove structure slots;
- remove structure bridge;
- knock out correct vs wrong slots;
- compare against vanilla transformer and legacy TAC baselines.

A mechanism is considered more credible when normal carry beats reset/shuffle and correct-slot interventions matter more than wrong-slot interventions.

## Current validated evidence

- TAC identity/state mechanisms are implemented and testable.
- Carry/reset/shuffle probes exist and are used in evaluation.
- Bounded compression and repair-control benchmarks show useful controlled behavior.
- TAC-SCM REAL004/005/006 validate controlled structure-to-behavior use and bridge stability.
- REAL011 improves the benchmark design for future executable-structure recovery tests.

## Current limitations

TAC does not yet establish:

- transformer or LLM superiority;
- open-ended coding/math/planning superiority;
- reliable autonomous agents;
- open-ended structure discovery;
- large-scale pretraining survival;
- wall-clock efficiency advantage.

## Next experiment

The highest-value next step is not another architecture feature. It is a larger, cleaner reproduction suite with 10-30 seeds, parameter-matched baselines, larger eval batches, fixed artifacts, runtime profiling, and one public result table.

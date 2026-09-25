# TAC-OSM v0.1

## Status

Implementation scaffold for the unified persistent structured-computation model. It is not yet evidence of a new capability.

## Model hypothesis

> A model can improve long-horizon transfer by learning to retain compact, task-relevant computational structure and selectively execute that structure when useful.

## Implemented mechanisms

1. Explicit persistent structural state.
2. Bounded structural write/compression.
3. Query-conditioned top-k structure retrieval.
4. Conditional reusable computation bank.
5. Typed decision output and verifier surface.

## Explicit non-goals

- arbitrary binding;
- faithful executable-structure recovery;
- open-ended agency;
- active inference;
- local credit assignment;
- large-scale language modeling;
- LLM replacement;
- REAL017 verifier/repair claims.

## State contract

`PersistentStructureState` contains `slots [B,K,D]` and `validity [B,K]`. The caller owns the state, making carry/reset/shuffle interventions explicit.

## Training principle

The compressor is not an ordinary reconstruction autoencoder. It should be trained by future utility: compressed state must preserve downstream task performance while minimizing persistent memory and active computation.

## First scientific test

Use a parameter-matched 2x2x2 ablation over persistent state, compression, and conditional computation.

Measure current-task accuracy, held-out structure combinations, long-horizon degradation, carry/reset/shuffle sensitivity, perturbation recovery, structured execution accuracy, memory footprint, active compute, and capability per parameter/FLOP.

Do not promote a result without multiple seeds, matched baselines, leakage checks, and reproducible artifacts.

## Veritas boundary

TAC-OSM owns the computational mechanism. Veritas owns operational worlds, episodes, outcomes, verification, and qualification. The first commercial target is verified next-action prediction under persistent state.

# TAC-SIE Research Lane

TAC-SIE is the engine-decomposition lane for the broader structure-centric intelligence program.

Its purpose is not to replace TAC-SCM. Its purpose is to test whether the structure thesis can be decomposed into clean, role-separated modules that can be validated independently.

## Current status

A clean TAC-SIE preserve/retrieve/execute substrate has been merged into `main` through PR #6:

- Branch used for clean integration: `integration/tac-sie-clean`
- PR: #6, `Add clean TAC-SIE preserve-retrieve-execute lane`
- Status: merged as a minimal scaffold, not promoted as robust binding evidence

The original historical branch remains preserved for auditability:

- Historical branch: `research/tac-sie-mvp002-exp009`
- Historical PR: #4, closed as preservation-only

This distinction matters. TAC-SIE is now present on `main`, but its current EXP009/EXP009B evidence is still provisional. It should not be described as a complete structure-intelligence engine until EXP009C-style binding controls pass.

## Why TAC-SIE matters

Earlier TAC-style systems risked overloading `IdentityState` with too many responsibilities:

- discovery;
- memory;
- retrieval;
- execution;
- repair;
- evolution.

TAC-SIE makes the architecture more scientific by forcing explicit module boundaries.

## Target decomposition

| Function | TAC-SIE responsibility |
|---|---|
| Preservation | Keep reusable structures or bindings in state/memory |
| Retrieval | Retrieve the relevant stored key/value or structure |
| Binding | Bind arbitrary symbols to recovered parameters |
| Execution | Apply recovered parameters through an executor |
| Refinement | Retry or repair through verifier feedback |
| Evolution | Score, retain, retire, or update structures over time |

## Current MVP shape

The merged TAC-SIE substrate contains a minimal preserve-retrieve-execute engine:

```text
rule id + offset id
  -> write key/value into IdentityState memory
  -> retrieve value from memory using rule query
  -> decode offset
  -> project retrieved value into executor offset vector
  -> execute with a separate executor
```

EXP009 and EXP009B are scaffold tests for this substrate. They are not final arbitrary-binding validation.

## Relationship to TAC-SCM

TAC-SCM asks:

> Can a structure-centric model preserve, route, reuse, transfer, compress, bridge, and recover structure?

TAC-SIE asks:

> Can those capabilities be decomposed into clean engine-level modules that actually work?

The shared frontier is:

```text
binding + recovery + execution
```

## Promotion gate for TAC-SIE claims

TAC-SIE should move from scaffold evidence to promoted evidence only after a focused validation gate passes:

1. import tests for `tac_sie`;
2. unit tests for memory write/read correctness;
3. executor pretraining/reuse tests;
4. EXP009C arbitrary-symbol binding test;
5. correct vs shuffled binding control;
6. wrong-slot binding control;
7. unseen symbol binding control;
8. corrupted-binding execution break;
9. documentation of what passed, what failed, and what remains unvalidated.

## Next experiment: EXP009C

Question:

> Can the engine robustly bind arbitrary new symbols to retrieved parameters?

Required controls:

- correct binding;
- shuffled binding;
- wrong-slot binding;
- unseen symbol binding;
- retrieved-parameter execution;
- corrupted-structure execution break.

Pass condition:

TAC-SIE should recover and execute the correct binding significantly better than shuffled/wrong-slot controls, and behavior should break when the relevant binding is corrupted.

## Public interpretation

TAC-SIE is currently best described as:

> a minimal merged engine-decomposition scaffold for testing the substrate behind structure preservation, retrieval, binding, execution, refinement, and evolution.

It is not yet a complete structure intelligence engine, and it should not be cited as robust arbitrary-binding evidence until EXP009C passes.

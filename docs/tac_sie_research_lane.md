# TAC-SIE Research Lane

TAC-SIE is the engine-decomposition lane for the broader structure-centric intelligence program.

Its purpose is not to replace TAC-SCM. Its purpose is to test whether the structure thesis can be decomposed into clean, role-separated modules that can be validated independently.

## Current status

TAC-SIE currently exists as a preserved research branch and draft PR:

- Branch: `research/tac-sie-mvp002-exp009`
- PR: #4, `Preserve TAC-SIE MVP002 EXP009 research lane`
- Status: preserved for auditability; not merged into `main`

This is intentional. TAC-SIE is less mature than TAC-SCM and should not be merged into `main` until it passes a focused validation gate.

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

The preserved TAC-SIE branch contains a minimal preserve-retrieve-execute engine:

```text
rule id + offset id
  -> write key/value into IdentityState memory
  -> retrieve value from memory using rule query
  -> decode offset
  -> project retrieved value into executor offset vector
  -> execute with a separate executor
```

The branch also contains EXP009 retrieved-rule-transfer experiments.

## Relationship to TAC-SCM

TAC-SCM asks:

> Can a structure-centric model preserve, route, reuse, transfer, compress, bridge, and recover structure?

TAC-SIE asks:

> Can those capabilities be decomposed into clean engine-level modules that actually work?

The shared frontier is:

```text
binding + recovery + execution
```

## Merge gate for TAC-SIE

TAC-SIE should move from preserved research lane to `main` only after a focused PR passes:

1. a minimal import test for `tac_sie`;
2. unit tests for memory write/read correctness;
3. executor pretraining/reuse tests;
4. EXP009C arbitrary-symbol binding test;
5. documentation of what passed, what failed, and what remains unvalidated.

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

> a preserved experimental engine-decomposition lane for testing the minimal substrate behind structure preservation, retrieval, binding, execution, refinement, and evolution.

It is not yet a complete structure intelligence engine.

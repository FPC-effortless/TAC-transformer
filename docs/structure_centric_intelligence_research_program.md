# Structure-Centric Intelligence Research Program

This repository is organized around a single research thesis:

> Intelligence is structure acquisition and structure use.

More specifically, the working thesis is that useful intelligence requires systems that can discover, compress, preserve, retrieve, bind, execute, compose, refine, and eventually evolve reusable computational structures.

This document is the reviewer-facing map for the TAC research program after the TAC-SCM and TAC-SIE lanes were clarified.

## Program hierarchy

```text
Theory: Structure-Centric Intelligence
  ↓
Model science: TAC-SCM
  ↓
Engine decomposition: TAC-SIE
  ↓
Validation: REAL / EXP benchmarks
```

## Lane responsibilities

| Lane | Research role | Current maturity |
|---|---|---|
| TAC core | Tests persistent identity state, program routing, memory carry, compression, and bounded repair-control mechanisms | Mature relative to other lanes, but still bounded |
| TAC-SCM | Tests whether structure-centric models can preserve, route, reuse, transfer, compress, bridge, and recover structures | More mature structure lane; REAL004/005/006/011 now live on `main` |
| TAC-SIE | Tests whether the structure thesis decomposes into clean engine-level modules: preserve, retrieve, bind, execute, refine, evolve | Less mature; preserved as draft PR #4 and not yet merged into `main` |
| REAL / EXP | Controlled validation layer for individual claims and failure modes | Active benchmark layer |

## Capability status

### Strongly validated or supported in bounded settings

- Structure preservation.
- Structure routing and retrieval.
- Structure reuse.
- Controlled transfer.
- Structure memory.
- Structure compression.
- Controlled structure-to-behavior use.
- Procedural memory and bounded verifier-guided repair control.

### Partially validated

- Latent structure organization.
- Representation signal for hidden structure.
- Structure bridge effects.
- Compiler/executor behavioral lift.
- Minimal preserve-retrieve-execute substrate.

### Not yet validated

- Robust arbitrary binding.
- Faithful executable recovery.
- Accurate structure compiler.
- Causally controlled execution from recovered hidden structure.
- Composition of learned structures.
- Refinement integrated into the trainable model architecture.
- Evolution of structure libraries.
- Open-ended structure creation.
- Self-improving structure memory.

## Key research insight

The most important current insight is:

> Structure representation and structure use are different capabilities.

The current architecture should therefore be evaluated as a pipeline:

```text
discover → compile → preserve → retrieve → bind → execute → refine
```

not as a single-step path:

```text
encode → answer
```

This is why TAC-SIE matters: it prevents `IdentityState` from becoming a vague catch-all mechanism and forces role separation.

## Role separation target

| Function | Responsible module or lane |
|---|---|
| Discovery | JEPA-style latent prediction / structure-slot competition / representation learning |
| Compilation | Structure compiler or bridge-to-executor path |
| Preservation | IdentityState and structure memory |
| Retrieval | Structure memory / procedural memory / key-value memory |
| Binding | Query-key/value binding system |
| Execution | Structure executor / procedural executor / verified action loop |
| Refinement | Verifier-guided repair loop |
| Evolution | Lifecycle scoring and retention/retirement rules |

## Roadmap position

| Stage | Capability | Status |
|---|---|---|
| 1 | Structure preservation | Validated |
| 2 | Structure routing / retrieval | Validated |
| 3 | Structure reuse / transfer | Validated in controlled settings |
| 4 | Structure compression | Validated / partial |
| 5 | Controlled structure use | Validated / partial |
| 6 | Latent structure organization | Partially validated |
| 7 | Direct structure-to-behavior bridge | Not sufficient alone |
| 8 | Compiler/executor behavioral lift | Partially validated |
| 9 | Faithful compiler | Current frontier |
| 10 | Robust arbitrary binding | Current frontier |
| 11 | Executable recovery | Not yet validated |
| 12 | Composition | Not yet validated |
| 13 | Refinement and evolution | Not yet validated as integrated architecture |

## Next decisive tests

The next two decisive tests should target the shared frontier between TAC-SCM and TAC-SIE:

### TAC-SIE EXP009C

Question:

> Can the engine robustly bind arbitrary new symbols to retrieved parameters?

This tests the minimal engine substrate.

### TAC-SCM REAL012-A

Question:

> Can TAC-SCM recover family and parameter structure faithfully enough to execute?

This tests the model-science lane.

Together, these tests decide whether the program moves from:

> structure signals exist

into:

> executable structures can be recovered, bound, and used.

## Conservative public claim

The current repo supports this claim:

> TAC-style systems can preserve, route, reuse, transfer, compress, and partially organize reusable computational structures under controlled conditions.

The current frontier claim is:

> The next missing capability is faithful executable recovery: converting latent or stored structure into compiled objects that causally control behavior.

# TAC Technical Report

## Claim

TAC is an experimental persistent-state and structure-centric architecture for long-horizon AI agents. The validated evidence currently supports bounded mechanisms for memory, compression, control, repair, causal fix selection, structure routing, and causal structure-to-behavior use.

The repo does not claim that TAC beats transformers or replaces LLMs.

## Research Program Map

```text
Theory: Structure-Centric Intelligence
  ↓
Model science: TAC-SCM
  ↓
Engine decomposition: TAC-SIE
  ↓
Validation: REAL / EXP benchmarks
```

The central research thesis is:

> Intelligence is structure acquisition and structure use.

See `docs/structure_centric_intelligence_research_program.md` for the full program hierarchy and capability status.

## Lane Summary

| Lane | Role | Repo status |
|---|---|---|
| TAC core | Persistent identity state, program routing, memory, compression, repair-control substrate | Implemented on `main` |
| TAC-SCM | Structure-centric model-science validation: structure slots, routing, bridge, lifecycle, memory, REAL benchmarks | Implemented on `main` |
| TAC-SIE | Engine-level decomposition: preserve, retrieve, bind, execute, refine, evolve | Preserved as PR #4; not merged yet |
| REAL / EXP | Controlled benchmark validation layer | Active |

## Evidence Summary

TAC-235 and TAC-236 establish causal program dependence and reproduction across the core benchmark matrix.

TAC-242, TAC-246, and TAC-247 support reusable algorithmic specialization under transfer and break-control tests.

TAC-245, TAC-248, TAC-249, TAC-251, TAC-252, and TAC-262 support context compression as an agentic enabler, with repeated evidence around the 20x boundary.

TAC-267 validates a repair-grounded control loop: verification failure, failure localization, responsible program selection, targeted activation, and re-verification.

TAC-268 through TAC-270 validate bounded repair execution, including multi-file sandbox repair without full-file known-good restoration.

TAC-271 identifies a failure frontier: first-pass causal fix choice under ambiguity.

TAC-272 directly targets that frontier with a causal-fix scoring step and validates first-pass disambiguation under bounded injected ambiguity.

TAC-273 exposes the next frontier: interacting multi-bug repair-chain completion. It passes root-cause set, regression avoidance, average repair-step budget, and state-continuity gates, but misses chain completion.

TAC-274 targets the TAC-273 failure with dependency-graph planning, patch-order prediction, and interaction tracking. It validates interaction-aware repair planning by improving chain completion while maintaining regression avoidance.

TAC-SCM REAL004 validates causal structure-to-behavior use in a controlled benchmark: carried structure improves behavior, reset/shuffle interventions hurt, and correct-slot knockout matters more than wrong-slot knockout.

TAC-SCM REAL005 validates bridge stability and harder structure generalization, promoting a linear structure bridge as the default candidate in the TAC-SCM v0.2 lane.

TAC-SCM REAL006 tests real or realistic structure transfer workloads: coding repair, long-document compression/recall, multi-session assistant memory, and research-workflow transfer.

TAC-SCM REAL011 redesigns the executable-structure benchmark to reduce benchmark flaws and make future hidden-structure recovery tests more scientifically meaningful.

## Main Research Insight

The key current insight is:

> Structure representation and structure use are different capabilities.

Therefore the architecture should be evaluated as a pipeline:

```text
discover → compile → preserve → retrieve → bind → execute → refine
```

not as a direct shortcut:

```text
encode → answer
```

## Current Frontier

The active frontier is whether TAC can handle:

- simultaneous independent bugs;
- longer live-repository repair chains;
- incomplete and deceptive tests outside bounded injected patterns;
- ambiguous root causes in real project environments;
- robust arbitrary binding;
- autonomous open-ended structure discovery;
- faithful executable recovery on valid benchmarks;
- scale with a genuinely capable pretrained model.

## Next Decisive Tests

The next decisive tests should target the shared TAC-SCM/TAC-SIE frontier:

- `TAC-SIE EXP009C`: robust arbitrary-symbol binding to retrieved parameters.
- `TAC-SCM REAL012-A`: faithful family/parameter recovery and execution on the redesigned executable-structure benchmark.

## Non-Claims

This report does not claim TAC beats transformers. It does not claim open-ended autonomous software engineering. It does not claim large-scale foundation-model validation.

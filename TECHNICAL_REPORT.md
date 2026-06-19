# TAC Technical Report

## Claim

TAC is an experimental persistent-state and structure-centric architecture for long-horizon AI agents. The validated evidence currently supports bounded mechanisms for memory, compression, control, repair, causal fix selection, structure routing, and causal structure-to-behavior use.

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

## Current Frontier

The active frontier is whether TAC can handle:

- simultaneous independent bugs;
- longer live-repository repair chains;
- incomplete and deceptive tests outside bounded injected patterns;
- ambiguous root causes in real project environments;
- autonomous open-ended structure discovery;
- executable structure recovery on valid benchmarks;
- scale with a genuinely capable pretrained model.

## Non-Claims

This report does not claim TAC beats transformers. It does not claim open-ended autonomous software engineering. It does not claim large-scale foundation-model validation.

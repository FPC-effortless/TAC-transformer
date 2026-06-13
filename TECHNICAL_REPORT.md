# TAC v0.1 Technical Report

## Claim

TAC is an experimental persistent-state architecture for long-horizon AI agents, with validated mechanisms for memory, compression, control, repair, and causal fix selection in bounded benchmarks.

## Evidence Summary

TAC-235 and TAC-236 establish causal program dependence and reproduction across the core benchmark matrix.

TAC-242, TAC-246, and TAC-247 support reusable algorithmic specialization under transfer and break-control tests.

TAC-245, TAC-248, TAC-249, TAC-251, TAC-252, and TAC-262 support context compression as an agentic enabler, with repeated evidence around the 20x boundary.

TAC-267 validates a repair-grounded control loop: verification failure, failure localization, responsible program selection, targeted activation, and re-verification.

TAC-268 through TAC-270 validate bounded repair execution, including multi-file sandbox repair without full-file known-good restoration.

TAC-271 identifies a failure frontier: first-pass causal fix choice under ambiguity.

TAC-272 directly targets that frontier with a causal-fix scoring step and validates first-pass disambiguation under bounded injected ambiguity.

TAC-273 then exposes the next frontier: interacting multi-bug repair-chain completion. It passes root-cause set, regression avoidance, average repair-step budget, and state-continuity gates, but misses chain completion.

TAC-274 directly targets that TAC-273 failure with dependency-graph planning, patch-order prediction, and interaction tracking. It validates interaction-aware repair planning by improving chain completion from 0.6335 to 0.7306 while maintaining 0.9557 regression avoidance.

## Current Frontier

The next hard questions are no longer whether persistent state, routing, repair, compression, single-fix ambiguity resolution, or bounded interaction-aware repair planning can matter in controlled settings. The active frontier is whether TAC can handle:

- simultaneous independent bugs
- longer live-repository repair chains
- incomplete and deceptive tests outside bounded injected patterns
- ambiguous root causes in real project environments
- scale with a genuinely capable pretrained model

## Non-Claims

This report does not claim TAC beats transformers. It does not claim open-ended autonomous software engineering. It does not claim large-scale foundation-model validation.

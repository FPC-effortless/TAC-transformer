# TAC v0.1 Technical Report

## Claim

TAC is an experimental persistent-state architecture for long-horizon AI agents, with validated mechanisms for memory, compression, control, repair, and causal fix selection in bounded benchmarks.

## Evidence Summary

TAC-235 and TAC-236 establish causal program dependence and reproduction across the core benchmark matrix.

TAC-242, TAC-246, and TAC-247 support reusable algorithmic specialization under transfer and break-control tests.

TAC-245, TAC-248, TAC-249, TAC-251, TAC-252, and TAC-262 support context compression as an agentic enabler, with repeated evidence around the 20x boundary.

TAC-267 validates a repair-grounded control loop: verification failure, failure localization, responsible program selection, targeted activation, and re-verification.

TAC-268 through TAC-270 validate bounded repair execution, including multi-file sandbox repair without full-file known-good restoration.

TAC-271 identifies the current failure frontier: first-pass causal fix choice under ambiguity.

TAC-272 directly targets that frontier with a causal-fix scoring step and validates first-pass disambiguation under bounded injected ambiguity.

## Current Frontier

The next hard questions are no longer whether persistent state, routing, repair, or compression can matter in bounded settings. The active frontier is whether TAC can handle:

- simultaneous independent bugs
- long repair chains
- incomplete and deceptive tests
- ambiguous root causes
- scale with a genuinely capable pretrained model

## Non-Claims

This report does not claim TAC beats transformers. It does not claim open-ended autonomous software engineering. It does not claim large-scale foundation-model validation.


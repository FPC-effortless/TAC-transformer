# USEF Authority Reporting Transfer

TAC now carries the useful part of the USEF multi-domain curriculum work as a
small, model-independent reporting module:

- `AuthorityEvent` records the source of a decision, whether it was accepted,
  whether it was correct, and whether the source domain differs from the target
  domain.
- `AuthorityReport` counts trusted authority accuracy, false trusted authority,
  rejected guesses, proposals, and cross-domain authority violations.
- `VerifierCase` turns executable or externally checkable tasks into authority
  events by comparing `expected` and `observed` outputs.
- `CurriculumReport` writes schema-versioned `manifest.json`, `cases.jsonl`,
  and `authority_events.jsonl` artifacts.

The design intentionally keeps this outside the model and trainer internals.
Training scripts, checkpoint evaluators, Kaggle diagnostics, and future
multi-domain curriculum builders can all emit the same artifact contract without
coupling to a specific experiment.

## Why This Matters

The main USEF lesson worth preserving is that identity/memory should not merely
be scored by whether the final answer is right. It should also be scored by
whether the model trusted the right kind of evidence for the right domain.

For TAC experiments this gives us direct gates for:

- false authority: trusted memory or execution was accepted but wrong;
- contamination: trusted evidence from one domain was accepted in another;
- verifier performance: executable or externally checkable tasks produce
  durable positive and negative evidence;
- negative results: rejected guesses and wrong proposals stay visible instead
  of disappearing into aggregate accuracy.

Current schema names:

- `tac_transformer.authority_report.v1`
- `tac_transformer.curriculum_report.v1`

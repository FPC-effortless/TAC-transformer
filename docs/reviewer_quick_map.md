# Reviewer Quick Map

This document tells external reviewers what to read first and how to interpret the repository.

## Read first

1. `README.md` — overview, claim boundary, quickstart.
2. `LIMITATIONS.md` — what TAC does not prove.
3. `RESULTS_SUMMARY.md` — compact evidence map.
4. `REPRODUCIBILITY.md` — commands and expected interpretation.
5. `TECHNICAL_REPORT.md` — one-page evidence narrative.

## Core implementation

- `tac_transformer/model.py` — main TAC model implementation.
- `tac_transformer/training.py` — training helpers and parameter counting.
- `tac_transformer/evaluation.py` — carry/reset/shuffle evaluation logic.
- `tac_transformer/serving.py` — checkpoint serving/generation utilities.

## Stable facades

- `tac_transformer/core/` — stable import facade for model/config/state classes.
- `tac_transformer/memory/` — stable import facade for memory modules.
- `tac_transformer/routing/` — stable import facade for routing-related modules.

These facades are documentation and import boundaries. They do not move or delete the older research modules.

## Experimental lanes

- `experiments/` — research scripts and benchmark harnesses.
- `kaggle/` — Kaggle-compatible benchmark/training scripts.
- `docs/` — research notes, runbooks, diagrams, reports.
- `tests_py/` — Python tests.

## Branch policy

- `main` should be reviewer-stable.
- large research lanes should land through draft PRs first.
- diverged branches should be cherry-picked or split, not blindly merged.

## Current preserved PRs

- PR #2: TAC v0.1 public release assets.
- PR #3: TAC v0.2 development lane.
- PR #4: TAC-SIE MVP002 EXP009 research lane.

## Claim discipline

Safe claim:

> TAC is an experimental persistent-state and structure-centric transformer research architecture with bounded evidence for memory, compression, repair control, causal fix selection, and controlled structure-to-behavior use.

Unsafe claim:

> TAC is already a proven LLM replacement.

# TAC-transformer Agent Instructions

This repository contains TAC research code and controlled experiments. Treat benchmark integrity as a first-class correctness property.

## Before changing experiments

- Read the relevant experiment, environment, model, tests, and benchmark documentation before editing.
- If a benchmark result could support a research claim, audit for leakage, shortcut routes, capacity confounds, oracle access, train/eval contamination, and invalid controls before running it.
- Do not report an experiment as evidence for a mechanism that the harness does not isolate.
- Prefer a small falsification test before a large multi-seed run.

## TAC-OSM benchmark rule

For TAC-OSM controlled experiments, read:
- `.agents/rules/benchmark-integrity.md`
- `.agents/skills/benchmark-design/SKILL.md`
- `docs/tac_osm_v04.md`

The v0.4 benchmark must use temporal state carry, matched parameter counts, deterministic/expected oracle evaluation, forecast-derived primary decisions, held-out evaluation conditions, and state/action intervention probes.

## Verification

At minimum, run the focused TAC-OSM tests after benchmark changes. Run the broader test suite when shared package interfaces change.

Record exact commands, seeds, configuration, commit SHA, and whether any control or audit was skipped.

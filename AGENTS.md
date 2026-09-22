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

## Execution plane (Termux)

This repository is edited on Android/Termux. **Code locally; execute remotely.**

`requirements.txt` declares `torch>=2.0`, which is never installed on this
device: it has no GPU, no CUDA, and insufficient RAM. Every run of the TAC test
suite is a GitHub Actions task, not a local one.

The authoritative execution boundary is `~/AGENTS.md` (global Termux resource
policy). Where this file and `~/AGENTS.md` conflict on *local execution*,
`~/AGENTS.md` wins; where they conflict on scientific integrity or benchmark
safety, this file and `.agents/rules/benchmark-integrity.md` win.

Locally allowed: reading, editing, writing tests, `python -m py_compile`,
import probes on already-installed modules, `git`, and `gh` orchestration.
Never locally: installing anything, building anything, or running the torch
suite.

Dispatch lanes (`.github/workflows/`): `fast-check` (lint + focused unit tests),
`experiment` (TAC/OSM experiments), `adversarial` (leakage/benchmark
falsification), `full-validation` (complete suite + artifacts). Only the Runner
role dispatches the expensive lanes; check `gh run list` first.

If a local run is impossible, report in the form `~/AGENTS.md` specifies
(BLOCKED / affected files / affected tests / possible resolution / continue
with). A skipped control or unrun suite must always be recorded as such, never
silently omitted.

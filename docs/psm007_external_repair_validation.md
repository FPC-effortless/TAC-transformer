# PSM-007 External Repair Validation

Date: 2026-06-16

Status: evaluator implemented; real external rows pending.

## Purpose

PSM-007 tests whether TAC's repair advantage survives outside TAC-authored
benchmarks.

This is a credibility gate. It does not redesign TAC, retune TAC, or change
metrics after seeing results.

## Required Sources

- real GitHub bugs
- SWE-bench-lite
- human-written repair tasks

## Required Controls

- frozen TAC
- matched transformer
- reset-memory TAC

## Frozen Constraints

Every row must record:

- `no_redesign = true`
- `no_retuning = true`
- `no_metric_changes = true`

Rows that violate these constraints block the gate instead of producing a
positive result.

## Row Format

```json
{
  "task_id": "example-task-id",
  "source": "swe_bench_lite",
  "control": "frozen_tac",
  "primary_score": 1.0,
  "resolved": 1.0,
  "constraints": {
    "no_redesign": true,
    "no_retuning": true,
    "no_metric_changes": true
  }
}
```

Valid `source` values:

- `real_github_bugs`
- `swe_bench_lite`
- `human_written_repair_tasks`

Valid `control` values:

- `frozen_tac`
- `matched_transformer`
- `reset_memory_tac`

## Decision Rule

PSM-007 validates only if:

- all three required sources are present
- all three required controls are present for every source
- frozen TAC beats matched transformer on every source
- frozen TAC beats reset-memory TAC on every source
- all rows preserve the frozen constraints

If evidence is missing, the gate is `blocked`. If evidence is complete but TAC
does not beat controls on every source, the gate is `not_validated`.

## Command

```powershell
python experiments\benchmark_psm007_external_repair_validation.py `
  --results-path runs\benchmarks\psm007_external_rows.json `
  --output-dir runs\benchmarks\psm007_external_repair_validation
```

The evaluator writes:

- `psm007_external_repair_validation.json`
- `PSM007_RESULTS.md`

## Boundary

The evaluator consumes external repair results. It does not generate the
external tasks, alter TAC, alter baselines, or change metrics.

# ID001 Identity Carry Validation

Date: 2026-06-16

Status: evaluator implemented; trained-checkpoint rows pending.

## Purpose

ID001 tests whether structure and procedure behavior improves when memory is
carried by persistent computational identities.

This is an architecture gate. It should answer whether `IdentityState` and
`IdentityField` add value beyond retrieval-only or reset-state behavior.

## Required Effects

- structure memory
- procedural memory
- identity carry

## Required Controls

- carried identity
- reset identity
- shuffled identity
- identity knockout

## Row Format

```json
{
  "task_id": "example-task-id",
  "effect": "procedural_memory",
  "control": "carried_identity",
  "primary_score": 0.78,
  "mechanisms": ["IdentityState", "IdentityField"]
}
```

Valid `effect` values:

- `structure_memory`
- `procedural_memory`
- `identity_carry`

Valid `control` values:

- `carried_identity`
- `reset_identity`
- `shuffled_identity`
- `identity_knockout`

## Decision Rule

ID001 validates only if every required effect has all required controls and:

- carried identity beats reset identity
- carried identity beats shuffled identity
- identity knockout scores lower than carried identity

If required evidence is missing, the gate is `blocked`. If evidence is complete
but any effect fails one of the comparisons, the gate is `not_validated`.

## Command

```powershell
python experiments\benchmark_id001_identity_carry_validation.py `
  --results-path runs\benchmarks\id001_identity_rows.json `
  --output-dir runs\benchmarks\id001_identity_carry_validation
```

The evaluator writes:

- `id001_identity_carry_validation.json`
- `ID001_RESULTS.md`

## Boundary

The evaluator consumes identity-carry result rows. It does not by itself prove a
trained checkpoint result unless the rows come from trained checkpoint probes.

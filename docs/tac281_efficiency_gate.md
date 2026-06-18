# TAC-281 Efficiency Gate

Date: 2026-06-16

Status: gate wrapper implemented; full three-variant decision pending.

## Purpose

TAC-281 tests whether TAC can keep its persistent-computation mechanism while
reducing the language-model loss and speed penalties observed in TAC-280.

This is an efficiency gate. It does not prove 112M scaling by itself. It only
decides whether a 112M pilot is justified.

## Required Variants

- `late_bottleneck`
- `small_adapter`
- `auxiliary_mechanism`

## Input Artifact

The gate consumes the existing TAC-281 summarizer output:

- `tac281_variant_decision.json`

That artifact is produced by:

```powershell
python scripts\summarize_tac281_variants.py `
  --transformer-summary <transformer_50m_summary.json> `
  --variant late_bottleneck=<summary.json>=<retest.json> `
  --variant small_adapter=<summary.json>=<retest.json> `
  --variant auxiliary_mechanism=<summary.json>=<retest.json> `
  --output-dir runs\benchmarks\tac281_variant_decision
```

## Decision Rule

TAC-281 is `blocked` until all three required variants are accounted for.

Once all variants are present, TAC-281 validates only if at least one variant is
`scale_ready`, meaning it satisfies the underlying TAC-281 checks:

- mechanism wins at least 3 of 4 families
- carry advantage remains positive
- bottleneck knockout delta remains positive
- LM loss gap shrinks by at least 30%
- speed penalty is reduced

If all three variants complete and none are `scale_ready`, the gate is
`not_validated` and 112M remains blocked.

## Command

```powershell
python experiments\benchmark_tac281_efficiency_gate.py `
  --decision-path runs\benchmarks\tac281_variant_decision\tac281_variant_decision.json `
  --output-dir runs\benchmarks\tac281_efficiency_gate
```

The evaluator writes:

- `tac281_efficiency_gate.json`
- `TAC281_GATE.md`

## Boundary

This gate wraps completed TAC-281 summaries. It does not train models, retest
checkpoints, or claim that TAC has survived 112M scale.

# REAL017 Audit Note

REAL017 should currently be treated as an audit candidate, not a validated milestone.

## Current status

- Original branch artifact: `feature/tac-scm-real003`
- Original file: `kaggle/benchmark_tac_scm_real017.py`
- Original classification: do-not-cite until audited
- Reason: the committed implementation gives the verifier access to `corruption_type` and gives the repair path access to `gold_slot`.

This does not invalidate the TAC-SCM/TAC-SIE program. It means REAL017's perfect metrics are explained by benchmark design and should not be used as evidence for verifier-guided repair until redesigned.

## Code-level audit scaffold

This branch adds a replacement scaffold:

- `kaggle/benchmark_tac_scm_real017_audit.py`
- `tests_py/test_tac_scm_real017_audit.py`

The audit scaffold is intentionally stricter than the original REAL017 path:

- `verify_slot(slot)` receives only the candidate slot.
- `repair_slot_blind(slot)` receives only the candidate slot.
- Oracle repair is separated into an explicit `oracle_repair` variant.
- The benchmark reports leakage guardrails in its JSON output.
- Tests inspect the public verifier/repair signatures and fail if `corruption_type`, `gold`, `gold_slot`, or `example` are accepted by blind APIs.

This scaffold is still provisional. It is an audit harness, not a final repair milestone.

## Known leakage paths in original REAL017

The original committed branch artifact constructs cases with:

```python
{
    "example": example,
    "corruption_type": corruption_type,
    "corrupted_slot": corrupted,
    "gold_slot": make_gold_slot(example),
}
```

The verifier path receives `corruption_type` directly. The repair path receives `gold_slot` directly. Therefore perfect detection, corruption-type accuracy, repair accuracy, and zero oracle gap are expected and are not evidence of learned or inferred repair.

## Required redesign before promotion

Before REAL017 can be cited as evidence, the implementation must satisfy these constraints:

1. `verify(...)` cannot receive `corruption_type`.
2. `repair_slot(...)` cannot receive `gold_slot` except in an explicitly named oracle variant.
3. Non-oracle repair must infer or recover structure from available context, corrupted slot fields, and learned/verifiable constraints.
4. Evaluation must include corruption types not present during repair-tuning.
5. Corruption generation must include randomized and adversarial perturbations, not only deterministic template shifts.
6. Clean-but-suspicious decoys must be included to test overrepair.
7. Corrupted-but-surface-normal decoys must be included to test verifier depth.
8. Metadata-stripped evaluation must produce the same public metrics.
9. A real seen-pair shortcut baseline must be implemented, not hard-coded to zero.
10. Tests must fail if any non-oracle path receives gold slot or corruption type.

## Required controls

- corruption-label permutation;
- hidden corruption type at evaluation;
- metadata stripping;
- random repair;
- wrong repair;
- no-op repair;
- no-store/no-lifecycle controls if state is involved;
- independent corruption generator;
- heldout family/parameter/binding combinations;
- external reviewer reproduction.

## Allowed language

Use:

> REAL017 is a scaffold for verifier-guided bound-structure refinement, currently downgraded to audit-candidate status because its perfect internal metrics are explained by verifier/repair access to benchmark metadata and gold slots. A replacement blind audit scaffold has been added, but it remains provisional.

Do not use:

> REAL017 validates verifier-guided structure repair.

## Promotion gate

REAL017 can be promoted only after a redesigned `REAL017-AUDIT` or replacement benchmark passes with:

- verifier blind to corruption labels;
- repair blind to gold slots;
- unseen corruption types;
- hard clean decoys;
- hard corrupted decoys;
- at least 10 seeds;
- fixed JSON artifacts;
- complete pass/fail gate;
- independent reproduction attempt.

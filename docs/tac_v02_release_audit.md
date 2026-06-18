# TAC v0.1 Release Audit For v0.2 Start

Status: complete enough to freeze as the stable v0.1 reference branch.

Branch policy:

- `main`: stable integration branch.
- `release/tac-v0.1`: frozen v0.1 public reference, no new benchmark work.
- `develop/tac-v0.2`: scaling work toward the 112M survival test.

Audit checklist:

| Item | Status | Reference |
|---|---|---|
| README | pass | `README.md` |
| Technical report | pass | `TECHNICAL_REPORT.md` |
| Benchmark artifacts | pass | `runs/benchmarks/benchmark_summary_tac235_tac272.md` and Kaggle validation output |
| Diagrams | partial | Architecture docs exist; a publication-grade overview diagram should be added before broad external release |
| Replication instructions | pass | `REPRODUCIBILITY.md` |

Reproduction command:

```bash
python experiments/kaggle_validate_tac_core.py --benchmarks tac251,tac252,tac267,tac270,tac272 --seeds 5 --cases 50 --output runs/kaggle_validation/tac_core_validation.json
```

Boundary:

TAC v0.1 validates bounded mechanisms. It does not answer whether those
mechanisms survive at 112M parameters on real language/code data. That is the
sole v0.2 stage-gate.


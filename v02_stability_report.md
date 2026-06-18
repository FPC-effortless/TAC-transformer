# TAC v0.2 Stability Report

- status: not_started
- metrics: runs\v02_scaling_dry_run\metrics_v02.json
- divergence: pending real 112M training
- routing collapse: pending real 112M TAC metrics
- state collapse: pending real 112M TAC metrics

Decision rule:

- continue scaling only if TAC finishes the matched-token run without divergence, routing collapse, or state collapse.
- stop scaling if the matched transformer wins while TAC loses persistent-state, repair, and compression gates.

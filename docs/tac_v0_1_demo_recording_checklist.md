# TAC v0.1 Demo Recording Checklist

Goal: record and publish a 3-5 minute demo that makes TAC v0.1 legible without
adding new benchmark claims.

## Recording Flow

Use `docs/tac_v0_1_demo_script.md` as the narration source.

1. Show the README and public claim.
2. Show the architecture diagram.
3. Show the Kaggle validation result and command.
4. Show the benchmark summary.
5. Show TAC-273 as the frontier failure.
6. Show TAC-274 as the targeted recovery.
7. Show limitations.
8. Close on the TAC v0.2 scaling question.

## Screen Targets

Open these files before recording:

- `README.md`
- `docs/tac_v0_1_architecture_diagram.md`
- `REPRODUCIBILITY.md`
- `runs/benchmarks/benchmark_summary_tac235_tac272.md`
- `LIMITATIONS.md`
- `docs/tac_v0_2_stage4_roadmap.md`

## Required Spoken Boundaries

Say these explicitly:

- TAC v0.1 does not claim to beat transformers.
- TAC v0.1 does not claim open-ended autonomy.
- Kaggle replication validates the bounded core validation pack.
- TAC-273 is a negative frontier result.
- TAC-274 is a targeted recovery result in a bounded benchmark.
- TAC v0.2 asks whether mechanisms survive scale.

## Minimal Recording Checklist

- Audio is clear.
- Cursor movement is slow enough to follow.
- The architecture diagram is visible for at least 20 seconds.
- The Kaggle PASS result is visible.
- TAC-273 failure and TAC-274 recovery are both shown.
- Limitations are shown before future work.
- Video title avoids overclaiming.

Recommended title:

```text
TAC v0.1: Persistent-State Agent Architecture, Reproducible Benchmarks, and Scaling Question
```

Recommended description:

```text
TAC v0.1 is a public research package for an experimental persistent-state
architecture for long-horizon AI agents. It includes bounded benchmark evidence,
limitations, reproducibility docs, Kaggle replication, a failure case, and a
targeted recovery benchmark. It does not claim to beat transformers or solve
open-ended autonomy.
```

## Publish Targets

- GitHub README link.
- YouTube unlisted first, then public after review.
- Optional: short post linking the GitHub branch, Kaggle kernel, and diagram.


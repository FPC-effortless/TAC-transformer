# TAC Run 5 Pathfinder Research

Date: 2026-06-02

## Why TAC-090 Was Not Enough

The first capability gate was useful but too narrow. It compared only vanilla,
base TAC, and one low-weight semantic TAC candidate. That was enough to show
that low-weight semantic routing did not immediately reproduce Run 4's collapse,
but not enough to choose the right implementation path.

Missing dimensions:

- identity/program parameter share
- semantic-route weight sweep
- authority-gated routing comparison
- selected-route and activation-level category MI
- rejection of candidates above the 50% identity-share gate

## Pathfinder Harness

Implemented:

- `build_run5_pathfinder_variants`
- `run_run5_pathfinder_matrix`
- `aggregate_run5_pathfinder_results`
- `category_program_mi_bits_from_probs`
- `experiments/benchmark_run5_pathfinder.py`

The pathfinder ranks candidates using language loss, loss improvement, accuracy,
selected-route MI, activation MI, throughput, and identity-share penalty.

## Invalid First Pathfinder Artifact

Artifact:

`runs/benchmarks/run5_pathfinder_local_2026_06_02`

This run is invalid for identity-share conclusions. The variant grid generated
different `n_programs` settings, but the runner ignored candidate-specific
program counts and used the CLI default for every TAC row. A regression test now
covers this failure:

`test_pathfinder_runner_uses_candidate_program_count`

## Corrected Pathfinder Artifact

Artifact:

`runs/benchmarks/run5_pathfinder_local_2026_06_02_fixed`

Command:

```bash
python experiments/benchmark_run5_pathfinder.py \
  --output-dir runs/benchmarks/run5_pathfinder_local_2026_06_02_fixed \
  --program-counts 8 12 16 24 \
  --semantic-weights 0.0 0.01 0.05 0.1 0.2 \
  --include-authority \
  --seeds 11 23 \
  --train-records 64 \
  --eval-records 24 \
  --steps 30 \
  --seq-len 64 \
  --batch-size 4 \
  --eval-batches 3 \
  --eval-batch-size 4 \
  --d-model 48 \
  --n-heads 4 \
  --n-layers 2 \
  --default-n-programs 8 \
  --device cpu \
  --torch-threads 4
```

Top corrected candidates:

| Rank | Variant | Final loss | Accuracy | Selected MI | Activation MI | Identity share | TPS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `tac_semantic_w0p1_p12` | `5.4395` | `0.1439` | `0.0199` | `0.0029` | `0.360` | `1831.1` |
| 2 | `tac_semantic_w0p05_p12` | `5.4393` | `0.1439` | `0.0202` | `0.0028` | `0.360` | `1736.3` |
| 3 | `tac_semantic_w0p01_p12` | `5.4392` | `0.1439` | `0.0207` | `0.0028` | `0.360` | `1619.0` |
| 4 | `tac_semantic_w0p2_p12` | `5.4397` | `0.1439` | `0.0203` | `0.0030` | `0.360` | `1618.8` |
| 5 | `tac_semantic_w0p2_p16` | `5.5148` | `0.1497` | `0.0318` | `0.0021` | `0.409` | `1644.7` |

Rejected:

- `tac_authority_p24`: identity share `0.507`, above the 50% gate.

Identity-share sanity:

| Base variant | Programs | Identity share |
| --- | ---: | ---: |
| `tac_base_p8` | 8 | `0.304` |
| `tac_base_p12` | 12 | `0.360` |
| `tac_base_p16` | 16 | `0.409` |
| `tac_base_p24` | 24 | `0.486` |

## Recommendation

The path is no longer simply "low-weight semantic routing." The better
recommendation is:

- keep the promoted TAC memory stack
- use `base_semantic` routing
- use `n_programs=12`
- train with `category_route_weight=0.1`
- use `category_route_objective="mi"`
- use `routing_top_k=2`
- keep `routing_load_balance_weight=0.05`
- keep the 2k-step warmup

Implemented:

- `run5_capability_config`
- `run5_capability_training_kwargs`
- `python kaggle/train_best_tac_agentic.py --preset run5_capability`

Important caveat:

The top p12 semantic weights are close. The choice of `0.1` is based on the
current pathfinder score and throughput, not a decisive scientific separation
from `0.05` or `0.01`. The next serious run should log periodic specialization
and forced-program evidence so the route signal is proven functional, not merely
selected-route dependent.

# TAC Capability Sanity Gate

Date: 2026-06-02

Run 4 proved that category-conditioned semantic routing can be induced, but it
also failed next-token capability. The next architecture-improvement loop is
therefore gated on a clean capability sanity matrix before any Run 5 launch.

## Implemented Gate

Command:

```bash
python experiments/benchmark_capability_sanity.py
```

The gate compares:

- `vanilla_10m_proxy`
- `tac_base_proxy`
- `tac_semantic_low_weight`

The default local command is intentionally small so it can run on CPU and write
artifacts. Larger evidence runs should increase seeds, steps, model width, and
record counts.

Output files:

- `capability_sanity_matrix.json`
- `RESULTS.md`

## Decision Rules

The gate returns:

- `blocked` if the vanilla proxy does not learn the corpus.
- `blocked` if base TAC does not learn after vanilla does.
- `blocked` if low-weight semantic routing materially regresses base TAC.
- `pass` only when vanilla, base TAC, and low-weight semantic TAC all improve
  loss under the same corpus and training settings.

## Smoke Result

Artifact:

`runs/benchmarks/capability_sanity_smoke_2026_06_02`

The smoke run used only two CPU training steps. It correctly exercised the
artifact path and returned `blocked`, because even the vanilla proxy did not
learn under that tiny budget. This is not scientific evidence about TAC
capability; it proves the automated gate is wired.

Next evidence run should use at least:

```bash
python experiments/benchmark_capability_sanity.py \
  --output-dir runs/benchmarks/capability_sanity_local_2026_06_02_fuller \
  --seeds 11 23 37 \
  --train-records 64 \
  --eval-records 24 \
  --steps 120 \
  --seq-len 64 \
  --batch-size 4 \
  --eval-batches 4 \
  --eval-batch-size 4 \
  --d-model 64 \
  --n-heads 4 \
  --n-layers 2 \
  --n-programs 8
```

Kaggle-scale Run 5 remains blocked until a meaningful gate passes.

## Fuller Local Gate

Artifact:

`runs/benchmarks/capability_sanity_local_2026_06_02_fuller`

Command:

```bash
python experiments/benchmark_capability_sanity.py \
  --output-dir runs/benchmarks/capability_sanity_local_2026_06_02_fuller \
  --seeds 11 23 37 \
  --train-records 64 \
  --eval-records 24 \
  --steps 40 \
  --seq-len 64 \
  --batch-size 4 \
  --eval-batches 3 \
  --eval-batch-size 4 \
  --d-model 48 \
  --n-heads 4 \
  --n-layers 2 \
  --n-programs 8 \
  --device cpu \
  --torch-threads 4
```

Result:

| Variant | Loss improvement | Final loss | Accuracy | Perplexity | TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vanilla_10m_proxy` | `1.6584` | `4.7523` | `0.2131` | `117.54` | `5164.34` |
| `tac_base_proxy` | `1.0659` | `5.3365` | `0.1753` | `208.78` | `1532.77` |
| `tac_semantic_low_weight` | `1.1352` | `5.2704` | `0.1862` | `195.53` | `1649.72` |

Verdict:

`pass`

The low-weight semantic objective did not reproduce Run 4's capability collapse
in the local sanity regime. It slightly beat base TAC on final loss and accuracy,
while vanilla remained the stronger pure language-model baseline.

## Superseded Run 5 Candidate

This section has been superseded by the broader pathfinder in
`docs/run5_pathfinder_research.md`. The pathfinder keeps the same p12
architecture family but updates the training objective weight from `0.05` to
`0.1`.

Implemented preset:

```python
from tac_transformer import run5_capability_config
```

Trainer selection:

```bash
python kaggle/train_best_tac_agentic.py --preset run5_capability
```

The candidate keeps the promoted TAC memory architecture but changes the Run 4
routing setup:

- `routing_type="base_semantic"`
- `routing_top_k=2`
- `routing_load_balance_weight=0.05`
- `category_route_weight=0.1`
- `category_route_objective="mi"`
- `warmup_steps=2000`
- `n_programs=12`

At the base training width, `n_programs=12` keeps identity-field parameters
under the roadmap gate of 50% of total parameters.

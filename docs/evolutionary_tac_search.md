# Evolutionary TAC Search

Date: 2026-06-03

Purpose:

Apply the Sakana-style lesson to TAC development: promote architecture
mutations only when they survive capability, memory-health, route-utility,
identity-share, and throughput gates together.

## Implemented Surface

- `aggregate_evolutionary_search_results` in `tac_transformer.capability`
- `format_evolutionary_search_markdown` for audit reports
- `experiments/select_evolutionary_tac_candidate.py` for artifact-based ranking

The selector accepts JSON artifacts shaped like:

- `per_seed` rows from local benchmark matrices
- `ranked` rows from pathfinder reports
- `aggregate` dictionaries from existing TAC reports
- explicit `candidate_rows`

## Gate Logic

A TAC mutation is rejected if it violates any available gate:

- identity share above `0.50`
- loss improvement below `0.05`
- final loss more than `2%` worse than the best vanilla row when a vanilla row
  is present
- program-memory cosine above `0.85` when memory-health evidence is present
- dead-program fraction above `0.20` when write/allocation evidence is present
- routed-is-best fraction below `0.20` when route-utility evidence is present

Missing memory or route evidence does not automatically reject a candidate, but
it also does not earn the candidate a health bonus. Serious promotion should
therefore combine pathfinder, write/allocation, route-utility, and vanilla
baseline artifacts.

## Smoke Artifact

Command:

```powershell
python experiments/select_evolutionary_tac_candidate.py --input runs/benchmarks/run5_pathfinder_local_2026_06_02_fixed/run5_pathfinder_matrix.json --output-dir runs/benchmarks/evolutionary_tac_search_smoke_2026_06_03
```

Result:

The selector produced a valid report and blocked promotion. Every TAC candidate
in the Run 5 pathfinder artifact exceeded the strict `2%` vanilla-loss gap, and
`tac_authority_p24` also exceeded the identity-share gate. This is a smoke test
of the artifact workflow, not a new scientific result.

## Next Use

For the next real candidate decision, feed the selector a combined evidence pool
that includes:

- Run 5/5B TAC rows
- same-backbone vanilla rows
- parameter-matched vanilla rows
- program-memory write/allocation diagnostic rows
- route-and-reconstruct or B1 route-utility rows

The selector should promote only a candidate that can be defended across all of
those evidence streams.

## Program-Conditioned Smoke

Date: 2026-06-03

Added a pathfinder mutation family for the Track B candidate:

```text
tac_program_conditioned_creb_k6_w<semantic_weight>_p<n_programs>
```

This maps to the diagnostic evidence alias:

```text
program_conditioned_creb_k6_task_memsep
```

The mutation uses:

- `program_memory_update_type="program_conditioned"`
- `memory_allocation_type="creb"`
- `memory_allocation_k=6`
- `memory_separation_weight=0.1`
- `routing_type="base_semantic"`
- `category_route_objective="mi"`

Compact capability smoke:

```powershell
python experiments/benchmark_run5_pathfinder.py --output-dir runs/benchmarks/run5_pathfinder_program_conditioned_smoke_2026_06_03 --program-counts 12 --semantic-weights 0.1 --include-memory-mutations --variant-names vanilla_10m_proxy tac_semantic_w0p1_p12 tac_program_conditioned_creb_k6_w0p1_p12 --seeds 11 --train-records 32 --eval-records 12 --steps 12 --seq-len 48 --batch-size 4 --eval-batches 2 --eval-batch-size 4 --d-model 48 --n-heads 4 --n-layers 2 --default-n-programs 12 --device cpu --torch-threads 4
```

Combined selector smoke:

```powershell
python experiments/select_evolutionary_tac_candidate.py --input runs/benchmarks/run5_pathfinder_program_conditioned_smoke_2026_06_03/run5_pathfinder_matrix.json runs/benchmarks/program_conditioned_memory_budget_sweep_local_2026_06_03/program_memory_write_diagnostic.json runs/benchmarks/program_conditioned_b1_balance_local_2026_06_03/program_contrastive_refinement.json --output-dir runs/benchmarks/evolutionary_tac_combined_program_conditioned_smoke_2026_06_03
```

Result:

The pathfinder-only ranking still slightly preferred `tac_semantic_w0p1_p12` on
final loss. The combined evolutionary selector promoted
`program_conditioned_creb_k6_task_memsep` for longer validation because it
combined acceptable compact capability with memory-health and route-utility
evidence:

- final loss: `5.9668`
- loss improvement: `0.3829`
- accuracy: `0.1120`
- selected MI: `0.0296`
- program-memory cosine: `0.2061`
- dead-program fraction: `0.0833`
- routed-is-best fraction: `0.3359`
- identity share: `0.384`

Decision:

Promote only to longer validation. This is not a final architecture win because
the run is one seed, short CPU budget, and still needs same-backbone plus
parameter-matched vanilla comparisons on the Run 5/5B data path.

## Local Vanilla-Gated Validation

Date: 2026-06-03

After the combined smoke promoted `program_conditioned_creb_k6_task_memsep` only
to longer validation, the next failure-protocol step was to run local vanilla
comparisons and a matching TAC run on the same prepared hard-agentic corpus.

Artifacts:

- `runs/benchmarks/vanilla_program_conditioned_same_backbone_local_2026_06_03`
- `runs/benchmarks/vanilla_program_conditioned_parameter_matched_local_2026_06_03`
- `runs/benchmarks/tac_program_conditioned_run5b_smoke_local_2026_06_03`

Commands used:

```powershell
python kaggle/train_vanilla_baseline.py --output-dir runs/benchmarks/vanilla_program_conditioned_same_backbone_local_2026_06_03 --scale smoke --baseline-mode same_backbone --steps 300 --batch-size 4 --grad-accum-steps 1 --eval-every 100 --eval-batches 3 --eval-batch-size 4 --checkpoint-every 100 --log-every 100 --device cpu --precision fp32 --max-seconds 1200 --stop-buffer-seconds 0
python kaggle/train_vanilla_baseline.py --output-dir runs/benchmarks/vanilla_program_conditioned_parameter_matched_local_2026_06_03 --scale smoke --baseline-mode parameter_matched --steps 300 --batch-size 4 --grad-accum-steps 1 --eval-every 100 --eval-batches 3 --eval-batch-size 4 --checkpoint-every 100 --log-every 100 --device cpu --precision fp32 --max-seconds 1200 --stop-buffer-seconds 0
python kaggle/train_best_tac_agentic.py --preset run5b_capability --scale smoke --output-dir runs/benchmarks/tac_program_conditioned_run5b_smoke_local_2026_06_03 --program-memory-update-type program_conditioned --memory-allocation-type creb --memory-allocation-k 6 --memory-separation-weight 0.1 --steps 300 --batch-size 4 --grad-accum-steps 1 --eval-every 100 --eval-batches 3 --eval-batch-size 4 --checkpoint-every 100 --log-every 100 --device cpu --precision fp32 --max-seconds 1200 --stop-buffer-seconds 0
```

Results:

| Row | Steps | Best eval loss | Eval accuracy | Train TPS | Program-memory cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vanilla same-backbone | 300 | `3.4440` | `0.2240` | `2385.9` | n/a |
| Vanilla parameter-matched | 300 | `2.9288` | `0.2474` | `2334.3` | n/a |
| TAC program-conditioned CREB k6 | 300 | `5.9358` | `0.0378` | `418.2` | `0.2523` |

Decision:

The mutation remains memory-healthy but capability-blocked. It should not be
promoted as a Run 5B architecture win. The useful result is narrower:
program-conditioned CREB top-6 fixes memory collapse locally, but the current
TAC training path still underperforms both fair vanilla baselines on the same
300-step local smoke budget.

Next research direction:

Stop spending the next cycle on more memory diversification alone. The blocker
is now language capability/optimization under the identity path. The next
mutation should target TAC capability recovery directly: reduce auxiliary
pressure early, delay or anneal identity-memory writes, or add a vanilla-to-TAC
staged curriculum before enabling the full identity stack.

## Auxiliary Pressure Ablation

Date: 2026-06-03

The first capability-recovery probe tested whether TAC's weighted auxiliary
losses were suppressing next-token learning. The trainer now exposes:

- `--aux-loss-scale`
- `--aux-loss-warmup-steps`

The ablation ran the same program-conditioned CREB top-6 TAC configuration with
all weighted auxiliary pressure disabled:

```powershell
python kaggle/train_best_tac_agentic.py --preset run5b_capability --scale smoke --output-dir runs/benchmarks/tac_program_conditioned_aux0_run5b_smoke_local_2026_06_03 --program-memory-update-type program_conditioned --memory-allocation-type creb --memory-allocation-k 6 --memory-separation-weight 0.1 --aux-loss-scale 0.0 --steps 300 --batch-size 4 --grad-accum-steps 1 --eval-every 100 --eval-batches 3 --eval-batch-size 4 --checkpoint-every 100 --log-every 100 --device cpu --precision fp32 --max-seconds 1200 --stop-buffer-seconds 0
```

Results:

| Row | Steps | Aux scale | Best eval loss | Eval accuracy | Train TPS | Program-memory cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TAC program-conditioned CREB k6 | 300 | `1.0` | `5.9358` | `0.0378` | `418.2` | `0.2523` |
| TAC program-conditioned CREB k6 aux-off | 300 | `0.0` | `5.9332` | `0.0391` | `608.3` | `0.2583` |
| Vanilla same-backbone | 300 | n/a | `3.4440` | `0.2240` | `2385.9` | n/a |
| Vanilla parameter-matched | 300 | n/a | `2.9288` | `0.2474` | `2334.3` | n/a |

Decision:

Disabling auxiliary pressure does not recover TAC next-token capability. The
best eval loss and accuracy are effectively unchanged from the aux-on TAC run
and remain far behind both local vanilla baselines. This exonerates weighted
auxiliary losses as the primary blocker for this mutation.

Next research direction:

Move from objective-pressure ablation to identity-path and training-schedule
ablation. The next useful probe should test whether TAC can learn the language
task when identity memory writes, memory adapters, or identity-first behavior
are delayed, disabled, or annealed after a vanilla-style warmup.

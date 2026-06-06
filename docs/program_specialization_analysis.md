# Program Specialization Analysis

Date: 2026-05-31

## Decision

Fuse this with Run 3 rather than treating it as a separate afterthought. The identity-first Kaggle command should enable `--analyze-specialization-at-end`, so the same run trains, selects `best.pt`, and writes the functional-specialization matrix before any new architecture launch.

## Tool

`kaggle/analyze_program_specialization.py` loads a checkpoint and labeled hard-agentic JSONL records, then reports:

- per-record top-k program attribution;
- per-token program attribution across meaningful sequence positions;
- record-level and token-level mutual information between program use and trace category;
- per-category program utilization matrices with token activation probabilities, selected-route frequencies, routing entropy, and top-program counts;
- one-program-at-a-time knockout loss deltas.
- selectivity summaries for both token utilization and category-conditioned knockouts.

The token-level path is the primary observability signal for BASE routing. Record-level dominant-program fields are retained for compatibility, but they can be dominated by fixed final-position scheduling artifacts.

Example:

```bash
python kaggle/train_best_tac_agentic.py \
  --scale base \
  --steps 20000 \
  --output-dir /kaggle/working/best_tac_agentic_identity_first_run3 \
  --device auto \
  --precision fp16 \
  --analyze-specialization-at-end \
  --specialization-max-records-per-category 64 \
  --specialization-device cpu
```

Manual rerun:

```bash
python kaggle/analyze_program_specialization.py \
  --checkpoint /path/to/best.pt \
  --jsonl runs/prepared_corpus_agentic_hard/hard_agentic_eval.generated.jsonl \
  --max-records-per-category 64 \
  --output runs/analysis/program_specialization/report.json \
  --csv-output runs/analysis/program_specialization/attribution.csv
```

## Local Smoke Result

Artifact:

```text
runs/analysis/program_specialization_identity_first_preflight_2026_05_31
```

Checkpoint:

```text
runs/preflights/identity_first_run3_preflight_2026_05_31/best.pt
```

This checkpoint is only a 20-step local preflight, so it is not a valid concept-formation result. It verifies the analysis pipeline.

Result:

- Categories sampled: `argument_schema`, `memory_counterfactual`, `repair_after_failure`, `stale_memory_rejection`, `tool_choice`, `verification_planning`.
- Records sampled: 8 per category, 48 total.
- Dominant program: all sampled records routed to program 15.
- Mutual information: `0.0` bits.
- Program entropy: `0.0` bits.
- Largest smoke-check knockout deltas were tiny: program 9 `+0.0035`, program 10 `+0.0015`, program 0 `+0.0012` loss.

Interpretation:

- The preflight checkpoint has no functional-specialization signal, which is expected after 20 CPU smoke steps.
- The analysis should be rerun on the trained Kaggle `best.pt` checkpoint that produced the low program-memory cosine.

## Fused Run 3 Gate

The trainer path now supports:

```bash
--analyze-specialization-at-end \
--specialization-max-records-per-category 64
```

When enabled, it analyzes `best.pt` after training and writes:

```text
<output-dir>/specialization/program_specialization.json
<output-dir>/specialization/program_attribution.csv
```

The summary is also embedded in `<output-dir>/final_summary.json` under `specialization_analysis`. A local two-step fused smoke at `runs/preflights/fused_run3_specialization_smoke_2026_05_31` verified the full trainer-to-analysis pipeline.

## 2026-06-01 Observability Update

The model aux output now exposes:

```text
token_program_activations      [batch, sequence, programs]
token_selected_program_mask    [batch, sequence, programs]
```

`program_specialization.json` now includes `token_mutual_information`, `token_raw_activation_mutual_information`, and `specialization_metrics`. Each category row under `activation_histogram.by_category` includes `token_top_program_counts`, `token_raw_top_program_counts`, `mean_token_activation_probabilities`, and `mean_token_selected_frequencies`.

This specifically fixes the Run 3 interpretability gap where final-token attribution under BASE routing could report a dominant program even when raw token activations were high-entropy.

A tiny Run 3 smoke with one record per category is saved at:

```text
runs/analysis/token_specialization_run3_smoke_2026_06_01
```

It confirms the final-token artifact mechanically: record-level program entropy stays `0`, but selected token routes have the full `5.0` bits of program entropy expected from BASE scheduling over 32 programs.

## Success Gate For The Kaggle Checkpoint

Minimum evidence for functional specialization:

- non-zero token-level program entropy across categories;
- token-level program-category MI clearly above 0;
- at least 2-3 programs with category-specific knockout deltas;
- examples where ablating a program hurts one category more than unrelated categories.

If these fail on the trained checkpoint, the next work should investigate why the measured program-memory separation is not becoming functional specialization before launching another architecture run.

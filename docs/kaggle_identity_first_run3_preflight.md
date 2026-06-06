# Kaggle Identity-First Run 3 Preflight

Date: 2026-05-31

## Decision

Run 3 should start fresh with the promoted identity-first TAC preset:

```text
identity_attention_type = "identity_first"
memory_separation_weight = 0.01
content_cue_separation_weight = 0.005
content_gate_entropy_weight = 0.005
content_reconsolidate = true
precision = "fp16" on Kaggle T4
```

`best_tac_config(...)` already contains these model-side settings. The training loop logs `aux_loss_separation`, `weighted_aux_loss_separation`, `metric_program_memory_cosine`, and the explicit `metric_program_ortho` alias for the same orthogonality pressure.

Run 3 should also enable the fused specialization gate. That makes the same Kaggle job train identity-first TAC and then analyze `best.pt` for per-input program attribution, program-category mutual information, and one-program knockout deltas before another architecture run is considered.

## Local Preflight

Artifact:

```text
runs/preflights/identity_first_run3_preflight_2026_05_31
```

Command:

```powershell
python kaggle/train_best_tac_agentic.py --scale smoke --steps 20 --batch-size 2 --grad-accum-steps 1 --eval-every 10 --eval-batches 2 --checkpoint-every 10 --train-jsonl runs/prepared_corpus_agentic_hard/train.prepared.jsonl --eval-jsonl runs/prepared_corpus_agentic_hard/eval.prepared.jsonl --output-dir runs/preflights/identity_first_run3_preflight_2026_05_31 --device cpu --precision fp32 --max-seconds 3600 --stop-buffer-seconds 0
```

Result:

- Completed 20/20 steps.
- Wrote `last.pt`, `best.pt`, `metrics.jsonl`, `run_manifest.json`, and `final_summary.json`.
- Manifest confirmed `identity_attention_type="identity_first"`.
- Eval step 20: loss `6.3283`, `program_memory_cosine=0.9303`, `metric_program_ortho=0.9233`, `content_synthesis_gate=0.2694`, `content_gate_entropy=0.8406`, `content_cue_cosine=0.1305`.
- CPU preflight used `fp32`; Kaggle should use `fp16`.

## Fused Specialization Smoke

Artifact:

```text
runs/preflights/fused_run3_specialization_smoke_2026_05_31
```

This was a tiny two-step CPU plumbing run with `--analyze-specialization-at-end`, one record per hard-agentic category, and knockout programs `0` and `1`.

Result:

- Completed 2/2 steps.
- Wrote `best.pt`, `last.pt`, `metrics.jsonl`, `run_manifest.json`, and `final_summary.json`.
- Wrote `specialization/program_specialization.json` and `specialization/program_attribution.csv`.
- `final_summary.json` contains `specialization_analysis` with 6 sampled records across all six hard-agentic categories.
- Smoke MI was `0.0` bits, as expected for a two-step plumbing run.

## Run 3 Command

Start from a fresh output directory:

```bash
torchrun --standalone --nproc_per_node=2 kaggle/train_best_tac_agentic.py \
  --scale base \
  --steps 20000 \
  --warmup-steps 500 \
  --batch-size 12 \
  --grad-accum-steps 3 \
  --eval-every 500 \
  --eval-batches 8 \
  --checkpoint-every 250 \
  --output-dir /kaggle/working/best_tac_agentic_identity_first_run3 \
  --device auto \
  --precision fp16 \
  --max-seconds 30600 \
  --stop-buffer-seconds 1200 \
  --analyze-specialization-at-end \
  --specialization-max-records-per-category 64 \
  --specialization-device cpu \
  --auto-resume
```

Do not attach old collapsed or synthesis-only outputs when starting the fresh concept-formation run.

## Watch Metrics

```text
program_memory_cosine              primary concept differentiation signal
metric_program_ortho               explicit orthogonality penalty proxy
memory_allocation_load_std         program write specialization
content_synthesis_gate             should avoid saturation near 1.0
content_gate_entropy               gate should remain conditional
content_cue_cosine                 cue-store collapse signal
used_energy                        routing efficiency
gradient_norm / grad_scaler_scale  fp16 stability
```

## Post-Run Tools

With `--analyze-specialization-at-end`, the trainer writes:

```text
/kaggle/working/best_tac_agentic_identity_first_run3/specialization/program_specialization.json
/kaggle/working/best_tac_agentic_identity_first_run3/specialization/program_attribution.csv
```

The compact summary is also embedded in `final_summary.json` under `specialization_analysis`.

To rerun the same analysis manually, or to increase `--max-records-per-category` after the job finishes:

```bash
python kaggle/analyze_program_specialization.py \
  --checkpoint /kaggle/working/best_tac_agentic_identity_first_run3/best.pt \
  --jsonl /kaggle/input/tac-hard-agentic-corpus/hard_agentic_eval.generated.jsonl \
  --max-records-per-category 64 \
  --output /kaggle/working/best_tac_agentic_identity_first_run3/program_specialization.json \
  --csv-output /kaggle/working/best_tac_agentic_identity_first_run3/program_attribution.csv
```

Inspect memory directly:

```bash
python kaggle/inspect_identity_memory.py \
  --checkpoint /kaggle/working/best_tac_agentic_identity_first_run3/best.pt \
  --prompt "Use calculator, verify result, then answer." \
  --max-slots 8 \
  --top-k 5 \
  --output /kaggle/working/best_tac_agentic_identity_first_run3/memory_inspection.json
```

Run checkpoint-only harder probes:

```bash
python kaggle/evaluate_checkpoint_harder_matrix.py \
  --checkpoint /kaggle/working/best_tac_agentic_identity_first_run3/best.pt \
  --seeds 11 23 37 \
  --eval-batches 8 \
  --eval-batch-size 32 \
  --output /kaggle/working/best_tac_agentic_identity_first_run3/checkpoint_harder_matrix.json
```

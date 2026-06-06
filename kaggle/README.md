# Kaggle Training

## Best TAC Agentic Training

The current best architecture is exposed through:

```bash
python kaggle/train_best_tac_agentic.py
```

It uses `best_tac_config`:

- RMSNorm
- SwiGLU
- RoPE
- grouped-query attention
- BASE-routed identity programs
- novelty-gated memory writes
- content-addressed cue/value memory readout
- `content_store_size=8`
- synthesis-gated two-step content readout
- gated residual memory adapter
- identity-first attention K/V projection

The script is designed for Kaggle's 9-hour style runtime constraint:

- default wall budget is 8.5 hours,
- default stop buffer is 20 minutes,
- checkpoints are written during training,
- `last.pt` is always resumable,
- `best.pt` is updated when eval loss improves,
- random JSONL windows reset identity state by default to avoid cross-record memory contamination.
- metrics include all scalar auxiliary losses, all scalar IdentityState metrics, gradient norm, CUDA memory, token/sequence counters, and the headline TAC diagnostics (`content_addressed_hit`, `content_synthesis_gate`, `content_gate_entropy`, `content_cue_cosine`, `content_reconsolidation_gate`, `program_memory_cosine`, `metric_program_ortho`).

By default, each JSONL training window is split into context/query halves. The first half writes identity state; the second half reads that state. This is required for content-addressed TAC to train its memory path while still resetting state between random JSONL windows. Use `--no-chunked-state-within-batch` only for a pure unchunked next-token baseline.

Recommended Kaggle command:

```bash
python kaggle/train_best_tac_agentic.py \
  --scale base \
  --steps 20000 \
  --warmup-steps 500 \
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

To use both Kaggle T4 GPUs, launch with `torchrun` instead of plain `python`:

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

The `--batch-size` value is per GPU under `torchrun`. The faster profile keeps the original effective batch size but uses larger microbatches:

```text
12 batch * 3 grad_accum * 2 GPUs = 72 sequences
```

If Kaggle runs out of memory, use `--batch-size 8 --grad-accum-steps 4`. The old conservative profile was `--batch-size 6 --grad-accum-steps 6`.

The content-addressed store is fixed-size, not proportional to the JSONL record length. At the default `base` scale and `content_store_size=8`, the estimated persistent identity state is about `2.63 MiB` per device and the content store itself is about `0.38 MiB`; activations and optimizer states dominate memory. To profile CUDA peak allocation on Kaggle before a long run:

```bash
python kaggle/profile_kaggle_memory.py --scales base --precision fp16 --device cuda --run-forward
```

For a quick notebook smoke run:

```bash
python kaggle/train_best_tac_agentic.py \
  --scale smoke \
  --steps 20 \
  --output-dir /kaggle/working/best_tac_agentic_identity_first_run3_smoke \
  --device auto \
  --auto-resume
```

Build a code-only upload bundle:

```bash
python kaggle/make_agentic_training_bundle.py
```

Upload:

```text
runs/kaggle_agentic_training_bundle/best-tac-agentic-training-bundle.zip
```

as a Kaggle Dataset, and upload `runs/prepared_corpus_agentic_hard_upload.zip`
as the separate hard training corpus dataset.

Resume:

```bash
python kaggle/train_best_tac_agentic.py \
  --scale base \
  --resume /kaggle/input/previous-best-tac-agentic-output/last.pt \
  --output-dir /kaggle/working/best_tac_agentic_identity_first_run3 \
  --device auto
```

Automatic resume:

- Same Kaggle session: keep the same `--output-dir` and pass `--auto-resume`; the trainer loads `/kaggle/working/best_tac_agentic_identity_first_run3/last.pt` when it exists.
- New Kaggle session: save the prior notebook output, attach that output as an input dataset, and pass `--auto-resume`; the trainer searches attached inputs for `last.pt` first, then `best.pt`.
- Explicit resume still works with `--resume /kaggle/input/previous-output/last.pt`.
- Start this identity-first run fresh; do not attach old collapsed or synthesis-only outputs unless you explicitly intend to fine-tune them.

The useful output files are:

```text
/kaggle/working/best_tac_agentic_identity_first_run3/last.pt
/kaggle/working/best_tac_agentic_identity_first_run3/best.pt
/kaggle/working/best_tac_agentic_identity_first_run3/metrics.jsonl
/kaggle/working/best_tac_agentic_identity_first_run3/run_manifest.json
/kaggle/working/best_tac_agentic_identity_first_run3/final_summary.json
/kaggle/working/best_tac_agentic_identity_first_run3/specialization/program_specialization.json
/kaggle/working/best_tac_agentic_identity_first_run3/specialization/program_attribution.csv
```

Key diagnostics to watch in `metrics.jsonl`:

```text
program_memory_cosine              lower means identity programs are separating
metric_program_ortho               explicit orthogonality penalty proxy
content_synthesis_gate             should not saturate near 1.0 for the whole run
content_gate_entropy               higher means the synthesis gate is still conditional
content_addressed_hit              retrieval confidence / hit proxy
content_cue_cosine                 lower means content cues are less collapsed
content_reconsolidation_gate       confirms read-time cue refresh is active
metric_memory_allocation_dead_rate detects unused programs
metric_routing_load_std            detects routing imbalance
aux_loss_*                         raw objective components
weighted_aux_loss_*                actual contribution to the training loss
gradient_norm / grad_scaler_scale  clipping pressure and fp16 scaler health
cuda_*_mib                         GPU memory pressure
tokens_seen / sequences_seen       data exposure accounting
epoch_equivalent                   approximate JSONL record pass count
```

## Hard Agentic Corpus

The harder replacement corpus is packaged at:

```text
runs/prepared_corpus_agentic_hard_upload.zip
```

Upload it to Kaggle as `tac-hard-agentic-corpus`, then train with:

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
  --output-dir /kaggle/working/best_tac_agentic_run4_semantic_selected_mi \
  --device auto \
  --precision fp32 \
  --min-healthy-gradient-norm 1e-12 \
  --fail-on-unhealthy-optimization \
  --max-seconds 30600 \
  --stop-buffer-seconds 1200 \
  --routing-type base_semantic \
  --routing-top-k 2 \
  --routing-load-balance-weight 0.05 \
  --category-route-weight 0.5 \
  --category-route-objective selected_mi \
  --specialization-checkpoints 2000 5000 10000 20000 \
  --specialization-checkpoint-max-records-per-category 16 \
  --analyze-specialization-at-end \
  --specialization-max-records-per-category 64 \
  --specialization-device cpu \
  --skip-end-specialization-on-time-stop
```

This Run 4 command is a fresh selected semantic-MI training run. Do not attach the Run 3
checkpoint dataset unless intentionally resuming or analyzing Run 3; the fixed
useful-family policy `P8/P14/P16/P18/P22/P24` is a Run 3 checkpoint patch, not
the fresh-training objective.
The final specialization pass still runs when the target step is reached; the
skip flag only avoids spending hours on end-of-run analysis after an intermediate
Kaggle time stop.

If TAC training throughput is much worse than the vanilla LLM job, use the
opt-in fast profile for the next run. It keeps semantic selected-MI TAC
training, but turns on top-k content reads and local causal attention:

```bash
python experiments/benchmark_kaggle_tac_training_speed_profile.py \
  --device cuda \
  --seq-len 176 \
  --batch-size 4 \
  --iters 5 \
  --output-dir /kaggle/working/kaggle_tac_training_speed_profile

torchrun --standalone --nproc_per_node=2 kaggle/train_best_tac_agentic.py \
  --preset kaggle_fast_tac \
  --scale base \
  --seq-len 176 \
  --steps 5000 \
  --batch-size 12 \
  --grad-accum-steps 3 \
  --eval-every 1000 \
  --eval-batches 4 \
  --checkpoint-every 500 \
  --output-dir /kaggle/working/best_tac_agentic_fast_selected_mi \
  --device auto \
  --max-seconds 30600 \
  --stop-buffer-seconds 1200 \
  --specialization-checkpoint-max-records-per-category 8 \
  --skip-end-specialization-on-time-stop
```

For the best current Run 5B capability launch, use the integrated preset. It
combines the TAC-188 memory-advantage stack, the TAC-169 cue-chain readout,
Run 5B fp32/fail-fast optimizer health, selected-route MI, and the implemented
training-speed cadence:

```bash
torchrun --standalone --nproc_per_node=2 kaggle/train_best_tac_agentic.py \
  --preset run5b_best_capability_fast \
  --scale base \
  --seq-len 176 \
  --steps 20000 \
  --batch-size 12 \
  --grad-accum-steps 3 \
  --eval-every 1000 \
  --eval-batches 4 \
  --checkpoint-every 500 \
  --aux-loss-cadence 4 \
  --output-dir /kaggle/working/run5b_best_capability_fast \
  --device auto \
  --max-seconds 30600 \
  --stop-buffer-seconds 1200 \
  --specialization-checkpoints 2000 5000 10000 20000 \
  --specialization-checkpoint-max-records-per-category 16 \
  --analyze-specialization-at-end \
  --specialization-max-records-per-category 64 \
  --specialization-device cpu \
  --skip-end-specialization-on-time-stop
```

If Kaggle nests the folder, find the exact path:

```python
!find /kaggle/input -name "train.prepared.jsonl" -o -name "eval.prepared.jsonl"
```

The trainer searches `/kaggle/input` recursively, so explicit `--train-jsonl`
and `--eval-jsonl` paths are only needed when multiple prepared corpora are attached.

## Post-Run Checkpoint Tools

When `--analyze-specialization-at-end` is enabled, the trainer automatically writes the functional-specialization matrix from `best.pt` to `specialization/program_specialization.json`, writes per-input attribution to `specialization/program_attribution.csv`, and records a summary under `specialization_analysis` in `final_summary.json`.

To rerun the same analysis manually, or to increase the sampled records after the Kaggle job finishes:

```bash
python kaggle/analyze_program_specialization.py \
  --checkpoint /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/best.pt \
  --jsonl /kaggle/input/tac-hard-agentic-corpus/hard_agentic_eval.generated.jsonl \
  --max-records-per-category 64 \
  --output /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/program_specialization.json \
  --csv-output /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/program_attribution.csv
```

After the run completes, keep `best.pt` and inspect it directly:

```bash
python kaggle/inspect_identity_memory.py \
  --checkpoint /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/best.pt \
  --prompt "Use calculator, verify result, then answer." \
  --max-slots 8 \
  --top-k 5 \
  --output /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/memory_inspection.json
```

Run checkpoint-only harder probes without retraining:

```bash
python kaggle/evaluate_checkpoint_harder_matrix.py \
  --checkpoint /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/best.pt \
  --seeds 11 23 37 \
  --eval-batches 8 \
  --eval-batch-size 32 \
  --output /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/checkpoint_harder_matrix.json
```

## Legacy Synthetic Training

Upload this repository as a Kaggle dataset or add it to a Kaggle notebook, then run from the project root:

```bash
python kaggle/train_tac_synthetic.py --device auto --steps 1000
```

The script:

- uses CUDA automatically when Kaggle exposes a GPU,
- trains `TACTransformerLM` on a synthetic executable-pattern next-token task,
- logs parameter counts and training metrics,
- saves a checkpoint to `/kaggle/working/tac_transformer.pt`.

The default model is `150,002,688` trainable parameters. It uses `batch-size=1` by default to stay realistic on a 16 GB Kaggle GPU.

For a smaller smoke run:

```bash
python kaggle/train_tac_synthetic.py --steps 20 --batch-size 8 --d-model 64 --n-layers 1 --n-heads 4 --n-programs 16 --vocab-size 128 --seq-len 33
```

To compare against plain attention, set identity coherence off:

```bash
python kaggle/train_tac_synthetic.py --beta 0
```

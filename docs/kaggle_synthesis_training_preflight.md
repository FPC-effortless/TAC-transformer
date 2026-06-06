# Kaggle Synthesis Training Preflight

Date: 2026-05-30

## Purpose

Verify that the promoted synthesis-gated TAC preset trains through the Kaggle agentic trainer on the real hard corpus before launching the full dual-T4 Kaggle run.

This local machine is CPU-only, so this is a short functional preflight, not the full Kaggle training run.

## Local Preflight Command

```powershell
python kaggle\train_best_tac_agentic.py --scale smoke --train-jsonl runs\prepared_corpus_agentic_hard\train.prepared.jsonl --eval-jsonl runs\prepared_corpus_agentic_hard\eval.prepared.jsonl --output-dir runs\kaggle_preflight_synthesis_2026_05_30 --device cpu --precision fp32 --steps 20 --batch-size 4 --grad-accum-steps 1 --eval-every 10 --checkpoint-every 10 --log-every 5 --eval-batches 2 --eval-batch-size 4 --max-seconds 600 --stop-buffer-seconds 30
```

## Result

The preflight completed all 20 steps and wrote both checkpoints:

- `runs/kaggle_preflight_synthesis_2026_05_30/last.pt`
- `runs/kaggle_preflight_synthesis_2026_05_30/best.pt`

The run manifest confirms the promoted architecture:

```text
memory_read_type = "content_addressed"
content_store_size = 8
content_read_steps = 2
content_read_gate_type = "synthesis"
routing_type = "base"
```

Final local preflight metrics:

| Metric | Value |
| --- | ---: |
| completed steps | 20 |
| best eval loss | 6.4155 |
| final eval loss | 6.4446 |
| final eval accuracy | 0.0117 |
| train `content_addressed_hit` | 0.2806 |
| train `content_synthesis_gate` | 0.2695 |
| train `program_memory_cosine` | 0.9335 |
| eval `content_addressed_hit` | 0.1915 |
| eval `content_synthesis_gate` | 0.2694 |
| eval `program_memory_cosine` | 0.9328 |

## Kaggle Launch Command

Use the full hard corpus and dual T4 GPUs:

```bash
torchrun --standalone --nproc_per_node=2 kaggle/train_best_tac_agentic.py \
  --scale base \
  --train-jsonl /kaggle/input/tac-1b-agentic-corpus/train.prepared.jsonl \
  --eval-jsonl /kaggle/input/tac-1b-agentic-corpus/eval.prepared.jsonl \
  --output-dir /kaggle/working/best_tac_agentic \
  --device auto \
  --precision auto \
  --max-seconds 30600 \
  --stop-buffer-seconds 1200
```

Expected training metrics now include:

- `content_addressed_hit`
- `content_synthesis_gate`
- `program_memory_cosine`

## Decision

The trainer is ready for the full Kaggle run with the promoted synthesis-gated architecture. The local run proves config wiring, corpus IO, chunked context/query state carry, metric logging, eval, and checkpoint writing all work.

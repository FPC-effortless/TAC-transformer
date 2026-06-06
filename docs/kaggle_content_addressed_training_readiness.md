# Kaggle Content-Addressed Training Readiness

Date: 2026-05-30

## Decision

Kaggle training is unblocked for the current best direct-memory TAC architecture.

`kaggle/train_best_tac_agentic.py` already builds its model through `best_tac_config(...)`, so after the 2026-05-30 preset promotion it now uses:

```text
routing_type = "base"
memory_read_type = "content_addressed"
content_store_size = 8
identity_attention_type = "identity_first"
memory_adapter_type = "gated_residual"
```

## Memory Profile

Command:

```powershell
python kaggle\profile_kaggle_memory.py --scales smoke small base large --precision fp16 --content-store-size 8 --output runs\benchmarks\kaggle_memory_profile_2026_05_30.json
```

The content store is not the OOM risk. It is fixed-size and small.

| Scale | Seq len | Batch/GPU | Params | Identity state | Content store |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 64 | 8 | 0.39M | 0.13 MiB | 0.03 MiB |
| small | 256 | 8 | 9.77M | 1.55 MiB | 0.28 MiB |
| base | 256 | 6 | 26.91M | 2.63 MiB | 0.38 MiB |
| large | 384 | 2 | 98.52M | 2.35 MiB | 0.23 MiB |

The large persistent memory contributors are still model activations, attention, optimizer states, and gradients. The content-addressed store itself is tiny at `content_store_size=8`.

## Logging Update

The Kaggle trainer now logs:

- `content_addressed_hit`
- `program_memory_cosine`

Evaluation now resets identity state between random JSONL windows by default. It only carries state across eval batches when `--carry-state-across-batches` is explicitly set.

## Training Path Update

The trainer now splits each JSONL window into context/query halves by default:

```text
first half  -> context forward, writes IdentityState
second half -> query forward, reads carried IdentityState
```

This matters for content-addressed TAC. A single unchunked language-model pass writes the content store only after the pass, so the read path would not be trained. Chunked windows reset between random records while still exercising memory inside each record.

Use `--no-chunked-state-within-batch` only for a pure next-token baseline that intentionally disables this within-window state training.

## Recommended Command

```bash
torchrun --standalone --nproc_per_node=2 kaggle/train_best_tac_agentic.py \
  --scale base \
  --train-jsonl /kaggle/input/datasets/jeffwilliamsr/tac-hard-agentic-corpus/train.prepared.jsonl \
  --eval-jsonl /kaggle/input/datasets/jeffwilliamsr/tac-hard-agentic-corpus/eval.prepared.jsonl \
  --output-dir /kaggle/working/best_tac_agentic_hard \
  --device auto \
  --precision auto \
  --max-seconds 30600 \
  --stop-buffer-seconds 1200
```

## Remaining Caveat

This memory profile was analytical on CPU because CUDA is not available in the local environment. On Kaggle, run:

```bash
python kaggle/profile_kaggle_memory.py --scales base --precision fp16 --device cuda --run-forward
```

before the long training run if you want an actual CUDA peak allocation measurement.

Artifact:

- `runs/benchmarks/kaggle_memory_profile_2026_05_30.json`

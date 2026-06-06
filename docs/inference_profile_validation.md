# TAC Inference Profile Validation

Date: 2026-05-30

## Question

Content-addressed memory is now the best direct-memory TAC architecture, but training TPS is not serving cost. This profile measures inference-only cost before treating the new preset as a Kaggle/serving default.

Important implementation detail: the current content store is fixed-size, controlled by `content_store_size`. It does not grow unbounded with sequence length unless that config is increased.

## Benchmark

Command:

```powershell
python kaggle\benchmark_inference.py --seq-lens 16 64 128 --content-store-sizes 4 8 16 32 --variants vanilla_matched base_program_memory content_addressed_k1 content_addressed_k2 --batch-size 16 --decode-steps 32 --warmup 2 --iters 5 --output-dir runs\benchmarks\inference_profile_2026_05_30
```

Measured paths:

- `prefill`: one full forward pass with no carried state
- `carried_query`: one full query forward pass using identity state from a context prefill
- `decode`: repeated one-token forward passes using carried identity state

This repo does not yet implement a transformer KV cache, so the decode profile measures the current one-token TAC path, not an optimized production decoder.

## Default Store-Size Result

For the promoted default, compare `content_addressed_k1` with `content_store_size=8` against BASE program-memory routing.

| Seq len | Variant | Query vs BASE | Decode vs BASE | Query vs vanilla | Decode vs vanilla |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | `base_program_memory` | 1.0000 | 1.0000 | 0.2232 | 0.2581 |
| 16 | `content_addressed_k1` | 1.1061 | 0.8800 | 0.2469 | 0.2271 |
| 16 | `content_addressed_k2` | 0.9062 | 0.6071 | 0.2023 | 0.1567 |
| 64 | `base_program_memory` | 1.0000 | 1.0000 | 0.6608 | 0.3174 |
| 64 | `content_addressed_k1` | 0.9701 | 0.8907 | 0.6411 | 0.2827 |
| 64 | `content_addressed_k2` | 0.6246 | 0.6610 | 0.4128 | 0.2098 |
| 128 | `base_program_memory` | 1.0000 | 1.0000 | 0.5318 | 0.2087 |
| 128 | `content_addressed_k1` | 0.7749 | 0.8373 | 0.4121 | 0.1747 |
| 128 | `content_addressed_k2` | 0.7838 | 0.5055 | 0.4169 | 0.1055 |

## Interpretation

Content-addressed k1 is acceptable for the Kaggle training/default architecture, but not free:

- Carried-query inference is close to BASE at the default store size: slightly faster at seq 16, about equal at seq 64, and slower at seq 128.
- One-token decode is consistently slower than BASE at the default store size: about `0.84x-0.89x` BASE for k1 at seq 64/128.
- k2 is not worth promoting for serving: it was not more accurate overall and is slower on decode.
- Vanilla remains much faster than TAC in this CPU profile, especially on decode. The TAC argument remains capability/data efficiency, not raw serving speed.

Store-size scaling was noisy on CPU and not strictly monotonic, but increasing beyond `content_store_size=8` did not reveal a clear immediate win. Keep the default at 8 until a capacity sweep shows accuracy needs more.

## Decision

Keep `content_addressed_k1` as the research/default architecture because the capability jump is large enough to justify the measured inference overhead.

Serving caveat:

- Use `content_addressed_k1`, not k2.
- Keep `content_store_size=8`.
- Treat production serving cost as unresolved until a GPU profile with batching and a real KV cache exists.

Artifacts:

- `runs/benchmarks/inference_profile_2026_05_30/RESULTS.md`
- `runs/benchmarks/inference_profile_2026_05_30/inference_profile.json`

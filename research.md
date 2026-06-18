# TAC-Transformer Research Notes

Date: 2026-05-27

## Current Architecture Baseline

This prototype intentionally started as a deterministic, inspectable simulation before adding a trainable PyTorch path. The trainable TAC block keeps the standard transformer path intact while adding an Identity Field Layer that:

- computes token-to-program activations from learnable program embeddings,
- turns those activations into a pairwise token coherence matrix,
- injects that coherence into attention logits with beta scaling,
- updates persistent program stability across calls,
- applies hard energy-budget routing for executable program selection,
- returns auxiliary losses for coherence, program reuse, and energy.

The latest benchmark notes show the important constraint: TAC is trainable and causally correct, but the parameter-matched vanilla transformer currently wins on the synthetic benchmark. That means the next architecture work should prove a capability advantage, not merely add parameters.

## DeepSeek Papers: Relevant Architecture Lessons

### DeepSeekMoE and DeepSeek-V2

DeepSeekMoE proposes two mechanisms that map directly onto TAC's identity-field hypothesis:

- Fine-grained expert segmentation: split coarse experts into more smaller experts and activate a flexible combination.
- Shared experts: reserve fixed shared experts for common knowledge so routed experts can specialize instead of redundantly learning basics.

DeepSeek-V2 combines DeepSeekMoE with Multi-head Latent Attention (MLA), reporting large practical efficiency gains: lower training cost, much smaller KV cache, and higher generation throughput. For TAC, the main takeaway is that routing must select actual compute paths, not only add a selected embedding vector to the residual stream. The current `program_embeddings` are closer to routing labels than executable experts.

Recommended TAC change:

- Replace or augment `program_embeddings` with lightweight routed program FFNs.
- Keep one shared program path always active.
- Route only a few fine-grained program experts per token or chunk.
- Compare against both same-backbone and parameter-matched dense baselines.

### DeepSeek-V3

DeepSeek-V3 keeps MLA and DeepSeekMoE, adds an auxiliary-loss-free load-balancing strategy, and uses a multi-token prediction training objective. TAC currently uses explicit auxiliary losses for coherence, reuse, and energy. Those are useful for early inspection, but they may be over-steering optimization and hurting next-token quality.

Recommended TAC change:

- Add a multi-token prediction head during training, especially for procedural and algorithmic datasets.
- Replace fixed auxiliary loss weights with a phased schedule: warm up identity losses, then decay them.
- Add router/load statistics and move toward bias-based or objective-free balancing if auxiliary losses fight language modeling.

### DeepSeek-R1 and DeepSeekMath

DeepSeek-R1 emphasizes verifiable reward training that elicits self-reflection, verification, and strategy adaptation without relying only on supervised traces. DeepSeekMath introduced GRPO for reasoning with lower memory pressure than PPO.

Recommended TAC change:

- Do not judge TAC only on next-token loss.
- Add verifiable sequence tasks where the final answer can be checked.
- Reward state reuse only when it improves final correctness across chunks.
- Track whether identity state helps verification or only creates attractive-looking internal structure.

### DeepSeek-Coder and DeepSeek-Coder-V2

DeepSeek-Coder emphasizes project-level code data and fill-in-the-blank training with a longer window. DeepSeek-Coder-V2 continues from DeepSeek-V2 with a large code/math continuation corpus and extends context length. For TAC, this suggests the identity field should be evaluated on structured procedural corpora, repository-style dependencies, infilling, and code/math traces rather than only tiny synthetic token patterns.

Recommended TAC change:

- Add infilling and repository/procedure-style examples to the prepared corpus.
- Evaluate whether identity memory tracks definitions, repeated procedures, and long-range dependencies.
- Keep separate code/math/procedure benchmark splits so routing specialization can be inspected by domain.

### DeepSeek-V3.2, DeepSeek-V4, and Sparse Long-Context Attention

DeepSeek-V3.2 introduces DeepSeek Sparse Attention for long-context efficiency while preserving model quality. V3.2-Exp describes a two-part long-context direction: an indexer identifies useful context, then attention runs over the selected subset.

DeepSeek-V4 pushes this further. The official Hugging Face model card presents DeepSeek-V4-Pro and DeepSeek-V4-Flash as MoE models with one-million-token context, using a hybrid attention architecture with Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA). Hugging Face's Transformers implementation notes describe CSA as a low-compression pool plus an indexer that gathers top blocks, while HCA uses a much higher-compression pool over long context. NVIDIA's Megatron Bridge docs also identify V4 features that matter for TAC: hybrid dense/compressed attention, DSA indexing, mHC hyper-connections, hash-routed MoE bootstrap layers, refined multi-token prediction, YaRN RoPE, expert-bias load balancing, and grouped low-rank output projection.

Recommended TAC change:

- Turn coherence into a sparse attention indexer instead of a dense additive bias everywhere.
- Split identity memory into at least two pools: a lightly compressed recent/global pool and a heavily compressed older pool.
- Select top-k coherent memory tokens/programs, then attend only to those plus a local sliding window.
- Add a small deterministic or hash-routed bootstrap mode for early layers before learned routing takes over.
- Experiment with multiple residual/state streams for TAC memory before full mHC complexity.
- Add grouped low-rank output projection if attention output cost becomes the bottleneck.
- Report compute saved, selected-token recall, and quality against dense attention.

### DeepSeek-OCR

DeepSeek-OCR is less directly relevant to this text-only model, but its core idea is useful: compress historical context into a lower-token representation and accept a measurable reconstruction/forgetting tradeoff.

Recommended TAC change:

- Treat identity memory as lossy compression with tests.
- Add reconstruction probes: can `program_memory` recover repeated latent rules or needed prior facts?
- Make forgetting explicit: prune stale program memory and measure when pruning hurts.

## Other Open Research: Relevant Architecture Lessons

### Long Memory

Transformer-XL shows that segment-level recurrence and relative positions address fixed-window context fragmentation. Compressive Transformer extends this with compressed older memories. TAC already has persistent `IdentityState`, so the next step should make it compete in this territory.

Recommended TAC change:

- Train/evaluate on chunked sequences where the target depends on earlier chunks.
- Add recurrence-aware positional handling so carried state does not ignore order.
- Add a compressed memory bank separate from short-term per-program stability.

### Selective Sequence State

Mamba shows that input-dependent state updates can selectively propagate or forget information with linear sequence scaling. TAC's `state_decay` is currently a fixed scalar, which is too blunt.

Recommended TAC change:

- Replace fixed `state_decay` with a learned input-conditioned gate per program.
- Add forget/write gates for `program_memory`.
- Measure whether gates specialize by program and sequence regime.

### Sparse Compute

Switch Transformer confirms that sparse activation can increase parameter count while keeping compute roughly constant, but also warns about routing instability and load balancing. This fits TAC's energy-budget routing.

Recommended TAC change:

- Promote selected programs into sparse FFN experts.
- Add top-1/top-2 routing modes and load metrics.
- Keep a dense fallback/shared expert so routing errors are not catastrophic early in training.

### Attention Efficiency

GQA reduces autoregressive inference cost by reducing KV heads while keeping quality close to multi-head attention. RoPE and ALiBi address position handling and length extrapolation better than learned absolute embeddings.

Recommended TAC change:

- Replace learned absolute `position_embedding` with RoPE or ALiBi.
- Add GQA or MLA-lite before scaling context length.
- Keep dense attention available as an ablation baseline.

### Backbone Modernization

RMSNorm is a cheaper alternative to LayerNorm with comparable quality in many settings. GLU variants such as SwiGLU improve transformer feed-forward layers over plain GELU in the original GLU-variant paper.

Recommended TAC change:

- Add config switches for RMSNorm and SwiGLU.
- Default smoke tests can stay tiny, but benchmark configs should test the modernized backbone before attributing gains/losses to the identity field.

## Prioritized Implementation Roadmap

1. Modernize the baseline backbone first.
   - Add RoPE or ALiBi.
   - Add RMSNorm.
   - Add SwiGLU.
   - Add GQA or an MLA-lite KV compression mode.
   - Reason: current learned positions, LayerNorm, GELU, and full MHA make the baseline less comparable to current open models.

2. Convert identity programs into real sparse compute.
   - Add shared program expert plus routed fine-grained experts.
   - Make routing differentiable enough to train, with hard routing only for eval or straight-through experiments.
   - Track per-expert load, entropy, and selected-token counts.
   - Reason: DeepSeekMoE's lesson is compute specialization, not only embedding selection.

3. Replace fixed state decay with learned memory gates.
   - Add write, forget, and read gates per program.
   - Train on chunked tasks that require remembering across calls.
   - Reason: current persistent state is useful but too rigid.

4. Turn coherence into sparse retrieval/indexing.
   - Use coherence to select memory/context candidates.
   - Add recent/local, lightly compressed global, and heavily compressed long-memory pools.
   - Attend over local window plus selected global tokens.
   - Compare compute and quality against dense coherence bias.
   - Reason: DeepSeek-V3.2/V4's sparse attention lesson fits TAC better than dense additive coherence.

5. Add multi-token and verifiable objectives.
   - Add auxiliary future-token heads during training.
   - Add algorithmic/verifiable tasks with correctness rewards or offline scoring.
   - Decay current identity losses after warmup.
   - Reason: DeepSeek-V3/R1 suggest stronger training signals than next-token plus static auxiliary weights.

6. Build ablation discipline into benchmarks.
   - Variants: vanilla modern baseline, +state only, +coherence only, +program experts only, +sparse attention, full TAC.
   - Required metrics: loss, perplexity, accuracy, tokens/sec, active parameters/token, routed load balance, memory carry benefit.
   - Reason: TAC needs to prove when identity state helps and when it is decorative overhead.

## Harder Validation Update: 2026-05-29

The implemented research mechanisms were validated beyond the original single-key chunked recall task.

Validation matrix:

- `25` variants
- `5` harder tasks: longer single-key, multi-key, delayed-query, noisy-key, multi-hop
- `3` seeds
- `375` benchmark runs total

Key result:

- `base_routing` is now the best harder-task candidate overall.
- `identity_compressed_attention` is the strongest non-routing long-context/memory candidate.
- `mamba_selective_state` and `rwkv_time_mix` are the strongest multi-hop/chained-recall candidates.
- `creb_match_k1` is close on mean carry but still has too high a dead-program rate for default promotion.
- `all_features_stack` remains a bad default.

Artifacts:

- `docs/harder_research_matrix.md`
- `runs/benchmarks/harder_research_matrix_2026_05_29/RESULTS.md`
- `runs/benchmarks/harder_research_matrix_2026_05_29/aggregate_harder_research_matrix.json`

## CREB Load-Balancing Update: 2026-05-29

The proposed CREB dead-program fix was implemented and validated on the harder task suite.

Implemented:

- carried `program_write_frequency` state
- `creb_delta` write-frequency penalty
- `creb_frequency_decay`
- `memory_allocation_write_frequency` metric

Validation matrix:

- `7` variants: current best, BASE routing, old CREB k=1/k=3, and three CREB load-penalty variants
- `5` harder tasks
- `3` seeds
- `105` benchmark runs total

Result:

- `base_routing` still wins overall with mean carry `0.0677`, carry-reset delta `0.0529`, `15/15` effective runs, and `0.0000` dead-program rate.
- `creb_load_k1_d1p0` keeps the same mean carry as old `creb_match_k1` (`0.0612`) but only reduces dead-program rate from `0.8902` to `0.8857`.
- `creb_load_k3_d0p5` keeps the same mean carry as old `creb_match_k3` (`0.0596`) but only reduces dead-program rate from `0.7049` to `0.6974`.

Decision:

- Do not promote CREB load balancing.
- CREB remains an experimental branch.
- The default harder-task architecture remains `base_routing`.
- CREB's issue appears structural, not merely under-tuned: the availability term and match term compete because both are functions of activation/stability, so write-target and read-target selection are not separated.
- If CREB is revisited, prefer structural allocation changes such as hard write lockout or threshold round-robin over stronger penalties on the same formula.

Artifacts:

- `docs/creb_load_balancing_validation.md`
- `runs/benchmarks/creb_load_harder_matrix_2026_05_29/RESULTS.md`
- `runs/benchmarks/creb_load_harder_matrix_2026_05_29/aggregate_harder_research_matrix.json`

## Sparse Ensemble Routing Update: 2026-05-29

Phase 2A was updated after BASE routing became the harder-task default. The question is now whether k-sparse ensemble routing beats single-program BASE routing on the tasks where pattern-addressable memory should help most.

Implemented:

- `routing_type="sparse_ensemble"`
- `routing_top_k`
- BASE-anchored routing where `k=1` is equivalent to BASE and `k>1` adds content-matched programs before energy trimming

Focused validation:

- variants: hash control, BASE, sparse ensemble k=2/k=3/k=4
- tasks: multi-key and noisy-key
- seeds: `11`, `23`, `37`
- `30` benchmark runs total

Result:

- `base_routing` still wins overall: mean carry `0.0664`, carry-reset delta `0.0547`, `6/6` effective, TPS ratio `0.5123`.
- `sparse_ensemble_k2` slightly wins noisy-key carry (`0.0898` vs BASE `0.0885`) but loses overall and has only `5/6` effective.
- `sparse_ensemble_k4` slightly wins multi-key carry (`0.0456` vs BASE `0.0443`) but loses noisy-key and overall.

Decision:

- Do not promote sparse ensemble routing yet.
- Keep BASE as the default.
- Move Phase 2 priority toward content-addressable pattern completion, because extra routed programs alone do not create reliable sparse engram retrieval.

Artifacts:

- `docs/sparse_ensemble_routing_validation.md`
- `runs/benchmarks/sparse_ensemble_focused_2026_05_29/RESULTS.md`
- `runs/benchmarks/sparse_ensemble_focused_2026_05_29/aggregate_harder_research_matrix.json`

## Pattern Completion Update: 2026-05-29

Phase 2B content-addressable pattern completion was implemented and tested.

Implemented:

- `memory_read_type="pattern_completion"`
- `pattern_store_size`
- carried `engram_patterns`, `engram_values`, and `engram_mask`
- route-pattern nearest-neighbor retrieval during query
- `pattern_completion_hit` metric

Focused validation:

- variants: BASE, sparse ensemble k=2/k=4, pattern completion k=2/k=4
- tasks: multi-key and noisy-key
- seeds: `11`, `23`, `37`
- `30` benchmark runs total

Result:

- `base_routing` still wins overall: mean carry `0.0664`, carry-reset delta `0.0547`, `6/6` effective, TPS ratio `0.5447`.
- `pattern_completion_k2` reached mean carry `0.0638`, below sparse k=2 (`0.0645`) and BASE (`0.0664`).
- `pattern_completion_k4` reached mean carry `0.0605`, below sparse k=4 (`0.0625`).

Decision:

- Do not promote route-pattern completion.
- The mechanism works, but route pattern alone is not enough for semantic partial-cue retrieval.
- The next version should store explicit cue/value tuples, not only route patterns.

Artifacts:

- `docs/pattern_completion_validation.md`
- `runs/benchmarks/pattern_completion_focused_2026_05_29/RESULTS.md`
- `runs/benchmarks/pattern_completion_focused_2026_05_29/aggregate_harder_research_matrix.json`

## Content-Addressed Cue/Value Memory Validation: 2026-05-30

The corrected Phase 2B content-addressable memory was implemented and tested.

Implementation:

- `memory_read_type="content_addressed"`
- `content_store_size`
- carried `IdentityState.content_cues`
- carried `IdentityState.content_values`
- carried `IdentityState.content_mask`
- `content_addressed_hit` metric

This stores hidden cue/value tuples rather than route patterns:

```text
context: store (hidden_state_of_key_token, hidden_state_of_value_token)
query:   retrieve value by cosine_similarity(query_key_hidden, stored_key_hidden)
```

Focused noisy-key validation:

- Task: `noisy_key`
- Variants: `base_routing`, `sparse_ensemble_k2`, `pattern_completion_k2`, `content_addressed_k1`, `content_addressed_k2`
- Seeds: 11, 23, 37
- Steps: 120
- Runs: 15 total

Result:

- `content_addressed_k1` wins noisy-key carry: `0.1094` vs BASE `0.0885`.
- `content_addressed_k1` wins carry-reset delta: `0.1042` vs BASE `0.0794`.
- `content_addressed_k2` is very close on carry: `0.1081`, with stronger carry-shuffled delta `0.0924` and higher measured TPS ratio `0.7344`.
- Route-pattern `pattern_completion_k2` remains below BASE: `0.0872`.
- Sparse ensemble alone remains nearly flat: `0.0898`.

Decision:

- Content-addressed cue/value memory is validated for noisy-key/partial-cue recall.
- Do not make it the universal default yet; the full harder-task suite still uses `base_routing` as the general default.
- Promote it as the next full-matrix candidate and as the best validated read mode for corrupted or partial key retrieval.
- Use `content_addressed_k1` when optimizing carry accuracy.
- Use `content_addressed_k2` as the throughput/shuffle-gap candidate.

Artifact:

- `docs/content_addressed_memory_validation.md`
- `runs/benchmarks/content_addressed_noisy_key_2026_05_30/RESULTS.md`
- `runs/benchmarks/content_addressed_noisy_key_2026_05_30/aggregate_harder_research_matrix.json`

## Content-Addressed Full Matrix Validation: 2026-05-30

The focused noisy-key win was expanded to the full five-task harder matrix.

Setup:

- Variants: `base_routing`, `content_addressed_k1`, `content_addressed_k2`
- Tasks: longer single-key, multi-key, delayed-query, noisy-key, multi-hop
- Seeds: 11, 23, 37
- Runs: 45 total

Overall result:

- `content_addressed_k1`: mean carry `0.4349`, carry-reset delta `0.4229`, carry-shuffled delta `0.4185`, effective `15/15`, TPS ratio `0.4823`.
- `content_addressed_k2`: mean carry `0.4341`, carry-reset delta `0.4216`, carry-shuffled delta `0.4198`, effective `14/15`, TPS ratio `0.4679`.
- `base_routing`: mean carry `0.0677`, carry-reset delta `0.0529`, carry-shuffled delta `0.0508`, effective `15/15`, TPS ratio `0.5236`.

Per-task result:

- Longer single-key: content-addressed wins, about `0.65` carry vs BASE `0.0677`.
- Multi-key: content-addressed wins, about `0.74` carry vs BASE `0.0443`.
- Delayed-query: content-addressed wins, about `0.64` carry vs BASE `0.0885`.
- Noisy-key: content-addressed wins, `0.1094` vs BASE `0.0885`.
- Multi-hop: BASE wins, `0.0495` vs content k1 `0.0365` and k2 `0.0312`.

Decision:

- Promote `content_addressed_k1` as the best current TAC architecture for direct memory recall.
- Keep BASE/program-memory as the current better multi-hop control.
- The next multi-hop work should add iterative retrieval or verifier-guided repair rather than more single-pass cue/value lookup.
- The noisy-key TPS spike for k2 did not generalize; full-matrix TPS is slightly slower than BASE, so the content-addressed win is a data/capability efficiency result, not raw wall-clock efficiency.

Artifacts:

- `docs/content_addressed_full_matrix_validation.md`
- `runs/benchmarks/content_addressed_full_matrix_2026_05_30/RESULTS.md`
- `runs/benchmarks/content_addressed_full_matrix_2026_05_30/aggregate_harder_research_matrix.json`

## Inference Profile Validation: 2026-05-30

The content-addressed architecture was profiled separately from training TPS.

Implementation:

- Added `kaggle/benchmark_inference.py`.
- Profiles `prefill`, `carried_query`, and one-token `decode`.
- Compares `vanilla_matched`, `base_program_memory`, `content_addressed_k1`, and `content_addressed_k2`.
- Sweeps sequence lengths `16`, `64`, `128` and content store sizes `4`, `8`, `16`, `32`.

Important correction:

- The current content store is fixed-size by `content_store_size`; it does not grow unbounded with sequence length.
- The default promoted store size is `8`.

Default store-size result for `content_addressed_k1`:

- seq 16: carried-query `1.1061x` BASE, decode `0.8800x` BASE.
- seq 64: carried-query `0.9701x` BASE, decode `0.8907x` BASE.
- seq 128: carried-query `0.7749x` BASE, decode `0.8373x` BASE.

Decision:

- Keep `content_addressed_k1` as the default because the direct-memory accuracy jump is much larger than the measured inference overhead.
- Do not promote `content_addressed_k2`: it is close in accuracy but slower on decode and not fully effective in the full matrix.
- Keep `content_store_size=8`.
- Production serving still needs GPU profiling and a real KV-cache path; this CPU profile is enough to avoid blocking the Kaggle training default.

Artifacts:

- `docs/inference_profile_validation.md`
- `runs/benchmarks/inference_profile_2026_05_30/RESULTS.md`
- `runs/benchmarks/inference_profile_2026_05_30/inference_profile.json`

## Kaggle Content-Addressed Training Readiness: 2026-05-30

The Kaggle agentic trainer now inherits the promoted `content_addressed_k1` preset through `best_tac_config(...)`.

Training config:

- `routing_type="base"`
- `memory_read_type="content_addressed"`
- `content_store_size=8`
- `memory_adapter_type="gated_residual"`

Memory profile:

- Added `kaggle/profile_kaggle_memory.py`.
- Ran smoke/small/base/large analytical fp16 profile locally.
- At `scale=base`, `seq_len=256`, `batch_size=6`, `n_layers=8`, `d_model=256`, `n_programs=32`, `content_store_size=8`:
  - model parameters: `26.91M`
  - estimated persistent identity state: `2.63 MiB`
  - estimated content store: `0.38 MiB`
  - estimated parameters + gradients + AdamW states: `307.95 MiB`

Decision:

- Content store memory does not block Kaggle training.
- The OOM risk is ordinary activation/attention/optimizer memory, not the content-addressed store.
- The long Kaggle run can proceed with `scale=base`; run `profile_kaggle_memory.py --device cuda --run-forward` on Kaggle first if an actual CUDA peak allocation is desired.
- Training and eval metrics now include `content_addressed_hit` and `program_memory_cosine`.
- Eval resets identity state between random JSONL windows by default, matching the training default.
- The trainer splits each JSONL window into context/query halves by default so content-addressed memory is actually read during training. Without this, random unchunked LM windows would write the store only after the forward pass and the memory read path would not train.

Artifacts:

- `docs/kaggle_content_addressed_training_readiness.md`
- `runs/benchmarks/kaggle_memory_profile_2026_05_30.json`

## Iterative Retrieval Validation: 2026-05-30

A minimal two-step content-addressed retrieval path was implemented and tested for multi-hop.

Implementation:

- `content_read_steps`
- learned first-read vs second-read blend gate
- `content_iterative_k1` and `content_iterative_k2` harder-matrix variants
- CLI support for `--content-read-steps`

Focused multi-hop result:

- `content_iterative_k2`: carry `0.0508`, carry-reset `0.0417`, effective `3/3`.
- `base_routing`: carry `0.0495`, carry-reset `0.0234`, effective `3/3`.
- `content_addressed_k1`: carry `0.0365`, carry-reset `0.0221`, effective `3/3`.

Full five-task result:

- `content_addressed_k1`: mean carry `0.4349`, task wins `4`, effective `15/15`.
- `content_iterative_k2`: mean carry `0.3552`, task wins `1`, effective `15/15`.
- `base_routing`: mean carry `0.0677`, task wins `0`, effective `15/15`.

Decision:

- Do not promote iterative retrieval as the default.
- Keep `content_addressed_k1` as the default direct-memory architecture.
- Iterative k2 validates the direction for multi-hop, but always-on second lookup regresses direct recall too much.
- The next multi-hop mechanism should be conditional retrieval: halt/continue, verifier-gated, or task/query-gated refinement.

Artifacts:

- `docs/iterative_retrieval_validation.md`
- `runs/benchmarks/content_iterative_multihop_2026_05_30/RESULTS.md`
- `runs/benchmarks/content_iterative_full_matrix_2026_05_30/RESULTS.md`

## Source Index

- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- DeepSeek-V2: https://arxiv.org/abs/2405.04434
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- DeepSeekMath: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DeepSeek-Coder: https://arxiv.org/abs/2401.14196
- DeepSeek-Coder-V2: https://arxiv.org/abs/2406.11931
- DeepSeek-V3.2: https://arxiv.org/abs/2512.02556
- DeepSeek-V3.2-Exp repository: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- DeepSeek-V4-Pro model card and technical report: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- DeepSeek-V4 Transformers implementation notes: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deepseek_v4.md
- DeepSeek-V4 Megatron Bridge notes: https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/deepseek/deepseek-v4.html
- DeepSeek-OCR: https://arxiv.org/abs/2510.18234
- Transformer-XL: https://arxiv.org/abs/1901.02860
- Compressive Transformer: https://arxiv.org/abs/1911.05507
- Mamba: https://arxiv.org/abs/2312.00752
- Switch Transformer: https://arxiv.org/abs/2101.03961
- GQA: https://arxiv.org/abs/2305.13245
- RoPE/RoFormer: https://arxiv.org/abs/2104.09864
- ALiBi: https://arxiv.org/abs/2108.12409
- RMSNorm: https://arxiv.org/abs/1910.07467
- GLU variants: https://arxiv.org/abs/2002.05202
- Multi-token prediction: https://arxiv.org/abs/2404.19737

## Open Research Addendum: TAC-Relevant Leads

Date: 2026-05-28

The most useful non-DeepSeek research for TAC falls into four groups: persistent memory, long-context compression, sparse routing, and residual/state stability. These are relevant because TAC's advantage should come from identity state that can be carried, read, reset, shuffled, and trained directly.

### Persistent and Test-Time Memory

Titans proposes a neural long-term memory module that can learn to memorize context at test time, treating attention as short-term memory and a neural memory module as persistent long-term memory. This maps directly to TAC's `IdentityState`, but suggests that the state update itself should be trainable and surprise/novelty-aware rather than only a fixed decay or simple gate.

Recommended TAC change:

- Add a novelty/write gate for `program_memory`.
- Train memory updates to store surprising or future-useful information.
- Compare carried, reset, and shuffled state on tasks where the answer only exists in prior chunks.

Implementation status:

- Added `memory_write_type="novelty_gated"` as the first Titans-inspired slice.
- The gate sees candidate program memory and previous program memory, then modulates the normal memory write gate.
- This is intentionally smaller than the full Titans neural-memory learner; it tests whether TAC benefits from learned write selectivity before adding a separate memory optimizer.

Memorizing Transformers add an approximate kNN lookup over past internal representations at inference time. TAC can use the same idea with program memory as the lookup table rather than raw token hidden states.

Recommended TAC change:

- Add optional kNN-style reads over stored program-memory keys.
- Store `(program_key, memory_value, stability, age)` tuples.
- Let coherence choose between local attention, program-memory read, and external memory read.

HMT and RMT-style memory transformers show that flat recurrent memory is weaker than hierarchical or segment-level memory. TAC currently has per-program memory only; it should probably split state into recent, stable, and archival identity memory.

Recommended TAC change:

- Replace one flat `program_memory` tensor with memory tiers:
  - recent working program state,
  - stable consolidated program memory,
  - compressed/archive memory.
- Add promotion/pruning rules and test whether consolidation helps across long chunks.

Implementation status:

- Added `memory_tier_type="hierarchical"` as the first HMT/RMT-style slice.
- The hierarchical state carries recent, stable, and archival program-memory tiers, then reads from a weighted mix of the tiers for program context and direct memory readout.

### Long-Context Compression and Streaming

Infini-attention combines local attention with compressive memory for unbounded context. This is a strong fit for TAC because identity state can be the compressive memory substrate.

Recommended TAC change:

- Add a local attention window plus a compressed identity-memory branch.
- Use TAC coherence as a learned indexer over compressed entries.
- Keep dense attention as an ablation, not the target long-context path.

Implementation status:

- Added `identity_attention_type="compressed_memory"` as a first compressive-memory branch.
- Carried TAC `program_memory` is projected into compressed key/value slots and concatenated with normal causal token keys.
- Initial 3-seed benchmark passed effectiveness but did not beat the prior best gated residual adapter, so the next version should add selection/gating over compressed slots instead of exposing all program memory equally.

StreamingLLM shows that preserving attention sinks plus a recent window stabilizes long streaming inference. TAC could reserve identity sink programs that always remain visible to the attention/router.

Recommended TAC change:

- Add one or more always-visible identity sink programs.
- Compare against normal learned programs and random sink programs.

Implementation status:

- Added `n_sink_programs` as a StreamingLLM-style identity sink ablation.
- Sink programs are always selected by the router but excluded from adaptive route-energy accounting.
- Initial 3-seed benchmark showed 1 and 2 sink programs stayed effective, but did not beat the no-sink novelty-gated TAC reference on carry accuracy or throughput.

LongNet and Ring Attention are less TAC-specific, but useful if context length itself becomes the bottleneck. LongNet's dilated attention is attractive as a drop-in sparse attention baseline; Ring Attention is more about distributed exact attention and is probably later-stage infrastructure.

Recommended TAC change:

- Try dilated attention before distributed ring attention.
- Use ring/blockwise attention only when training contexts exceed one device's practical memory.

### Sparse Routing and Memory Layers

Expert Choice, BASE Layers, Switch, and Hash Layers all attack sparse routing stability from different angles. TAC already has an energy budget, so its router can learn from these without copying them wholesale.

Recommended TAC change:

- Compare token-choice routing against expert-choice routing.
- Add hash or deterministic bootstrap routing for early layers.
- Track expert load, specialization, dead experts, and identity-state dependence.

Implementation status:

- Added `routing_type="expert_choice"`, `"base"`, and `"hash"` alongside the default TAC energy router.
- Initial 3-seed benchmark showed hash routing gave the best routing result in this slice: 0.0872 carry accuracy versus 0.0833 for the energy router, while reducing routed-energy proxy from 3.4771 to 1.0183 and active programs from 4 to 1.

Memory Layers at Scale and Product-Key Memory suggest another route: use sparse lookup memory inside the model rather than only FFN experts. This may fit TAC even better than conventional MoE because TAC programs are partly memory-bearing.

Recommended TAC change:

- Test a product-key memory layer as a replacement for, or companion to, routed program experts.
- Let program stability bias which memory slots are eligible.

Implementation status:

- Added `memory_lookup_type="product_key"` as a sparse memory-table ablation.
- The first implementation adds a trainable top-k product-key lookup to TAC's identity program context, separate from carried `program_memory`.

### Residual and State Stability

DeepNet, ReZero, and ResiDual are practical references for stabilizing deeper transformer stacks. Full mHC is interesting but complex; TAC should first test simpler multi-stream or gated residual designs.

Recommended TAC change:

- Add a gated residual scale to the program path, initialized small.
- Split hidden flow into content and identity streams before attempting full hyper-connections.
- Use DeepNorm-style scaling if TAC grows much deeper.

Implementation status:

- Added `residual_stream_type="dual_stream"` as the first ResiDual/mHC-style slice.
- Dual-stream mode separately gates content attention updates and identity/program updates before the MLP residual.

### Added Source Index

- Titans: https://arxiv.org/abs/2501.00663
- Memorizing Transformers: https://arxiv.org/abs/2203.08913
- Recurrent Memory Transformer: https://arxiv.org/abs/2207.06881
- Scaling Transformer to 1M tokens and beyond with RMT: https://arxiv.org/abs/2304.11062
- HMT: https://arxiv.org/abs/2405.06067
- Infini-attention: https://arxiv.org/abs/2404.07143
- StreamingLLM / Attention Sinks: https://arxiv.org/abs/2309.17453
- LongNet: https://arxiv.org/abs/2307.02486
- Ring Attention: https://arxiv.org/abs/2310.01889
- Expert Choice Routing: https://arxiv.org/abs/2202.09368
- BASE Layers: https://arxiv.org/abs/2103.16716
- Hash Layers: https://arxiv.org/abs/2106.04426
- Memory Layers at Scale: https://arxiv.org/abs/2412.09764
- Product-Key Memory: https://arxiv.org/abs/1907.05242
- DeepNet / DeepNorm: https://arxiv.org/abs/2203.00555
- ReZero: https://arxiv.org/abs/2003.04887
- ResiDual: https://arxiv.org/abs/2304.14802

### Multi-Token and Verifiable Objectives

Multi-token prediction can make hidden states more predictive of future structure instead of only the immediate next token. TAC needs this because identity state should carry program coherence across more than one step.

Recommended TAC change:

- Add optional future-token prediction heads.
- Train them as auxiliary losses rather than changing default next-token behavior.
- Compare whether future-token heads improve chunked carry accuracy or state sensitivity.

Implementation status:

- Added `n_prediction_heads` and `multi_token_loss_weight`.
- Extra prediction heads expose `aux.losses["multi_token"]` and are trained by the existing trainer when the weight is nonzero.

## Agentic TAC Architecture Research Addendum

Date: 2026-05-28

Goal: determine the best way to apply the requested agentic layers to TAC for a commercially viable small-lab agent platform.

External research surveyed:

- ReAct: interleaves reasoning traces and environment/tool actions.
- Toolformer: trains when/how to call APIs through self-supervised tool-call insertion.
- ToolLLM / ToolBench: uses large tool-use instruction/solution-path datasets over real APIs.
- Reflexion: stores verbal feedback from failures as memory for future trials.
- Voyager: combines curriculum, executable skill library, and iterative environment-feedback repair.
- Tree of Thoughts and LATS: use search/tree expansion and value estimates at inference/runtime.
- MemGPT, MemoryBank, Generative Agents, Titans: point toward tiered memory, write filtering, reflection, consolidation, and test-time memory.
- Mamba/RWKV/xLSTM-style recurrent models: important efficiency baselines for agentic memory tasks.
- AgentBench, WebArena, tau-bench: realistic evaluation should measure interactive success, not only token loss.

Research decision:

- Keep `best_tac_config` as the model core.
- Do not put the full agent loop inside the base model.
- Keep planning, tool execution, reflection, memory consolidation, and multi-agent orchestration in the runtime platform.
- Continue model-side work only on memory-aware tool/action adapters and state supervision.
- Train from real or simulated tool execution traces rather than tiny class-label action tasks.

Recommended architecture:

```text
TAC base model
+ optional memory-conditioned tool/action adapter
+ external ReAct-style runtime
+ Memory OS with working / episodic / semantic / procedural stores
+ verifier and repair loop
+ adaptive planner/search only when uncertainty or risk justifies cost
+ sparse multi-agent orchestration only for high-value tasks
```

Recommended next implementation:

- Build `TACAgentRuntime` around the model.
- Add `ToolTraceBatcher` with tool schemas, arguments, execution results, repair traces, and memory writes.
- Train losses for tool choice, argument schema, result integration, memory read/write, verifier score, and state contrast.
- Promote an adapter only if carry beats reset, shuffled, vanilla, and recurrent baselines at acceptable throughput.

Full recommendation written to `docs/agentic_architecture_research_recommendation.md`.

## Hybrid Sequence Mixer Addendum

Date: 2026-05-28

Prompt from user: do not stay boxed into the current transformer path; test other ways to apply the research, including Jamba and BlackMamba-style designs.

Research interpretation:

- Jamba suggests interleaving attention with state-space sequence mixing and sparse capacity.
- BlackMamba suggests combining state-space sequence modeling with MoE/sparse expert capacity.
- For TAC, the testable version is not a wholesale rewrite; it is a sequence-mixer ablation around the identity-state core.

Implemented:

- `sequence_mixer_type="attention"`: current TAC path.
- `sequence_mixer_type="state"`: causal recurrent/state mixer replaces attention.
- `sequence_mixer_type="hybrid"`: attention plus causal state mixer in every block.
- `sequence_mixer_type="alternating"`: attention on even layers and state mixer on odd layers.
- BlackMamba-style sparse variants by combining state/hybrid mixers with `program_compute_type="sparse_linear_expert"`.

Benchmark decision:

- Current attention TAC remains best on the 120-step long check: carry 0.0872, TAC-baseline gap 0.0755, effective 3/3.
- `jamba_hybrid` stayed effective 3/3 and had a better TAC-vs-baseline TPS ratio, but lower carry accuracy: 0.0625.
- `blackmamba_sparse_hybrid` tied hybrid accuracy but reduced throughput, so sparse dispatch is still not worth promoting in this implementation.
- `state_only` and sparse-state were weaker in the shorter exploratory matrix.

Conclusion:

- TAC is not tied to a pure transformer box anymore; the repo now has a testable hybrid sequence-mixer axis.
- The best current production/default architecture remains attention-based TAC.
- The next serious non-transformer test should replace the proxy state mixer with a real Mamba/xLSTM/RWKV implementation and run longer data-efficiency curves.

Full result written to `docs/hybrid_mixer_experiments.md`.

### Recurrent Mixer Follow-Up

Implemented and tested additional sequence mixers:

- `sequence_mixer_type="selective_state"`: Mamba-inspired input-selective recurrent mixer.
- `sequence_mixer_type="rwkv"`: RWKV-inspired time-mix recurrent weighted value mixer.
- `sequence_mixer_type="xlstm"`: xLSTM-inspired exponential-gated recurrent mixer.
- `rwkv_sparse_expert`: BlackMamba-style sparse expert capacity around RWKV-style state mixing.

120-step recurrent matrix result:

- `hash_attention_best`: carry 0.0872, gap 0.0755, effective 3/3.
- `mamba_selective_state`: carry 0.0599, gap 0.0508, effective 3/3.
- `rwkv_time_mix`: carry 0.0534, gap 0.0404, effective 2/3.
- `xlstm_gated`: carry 0.0495, gap 0.0299, effective 3/3.
- `rwkv_sparse_expert`: carry 0.0534, gap 0.0404, effective 2/3.

Decision:

- Attention TAC remains the best core.
- Selective-state is the best recurrent/state alternative so far, but not strong enough to replace attention.
- Sparse experts again did not provide practical throughput gains in this unfused PyTorch implementation.
- Keep recurrent mixers as ablations and future long-context candidates, not the default.

Artifact: `runs/benchmarks/recurrent_mixer_longcheck_2026_05_28/RESULTS.md`.

## Kaggle Agentic Training Addendum

Date: 2026-05-28

Goal: train the current best TAC architecture for agentic use under Kaggle's 9-hour-style training limit.

Decision:

- Use the current `best_tac_config` attention-based TAC core, not the experimental recurrent/hybrid variants.
- Train on prepared agentic/procedural JSONL traces as byte-level language modeling.
- Keep planning/tool/reflection orchestration outside the base model for now; train the model on traces that include prompts, plans, tool results, verification, procedural memory, and final answers.
- Reset identity state between random JSONL windows by default to avoid cross-record memory contamination.
- Save resumable checkpoints often and stop before Kaggle's wall-time cutoff.

Implementation:

- Added `kaggle/train_best_tac_agentic.py`.
- Added `kaggle/make_agentic_training_bundle.py`.
- Updated `kaggle/README.md` with the exact Kaggle command, resume command, and output artifacts.
- Added `torchrun` / DistributedDataParallel support for dual-T4 Kaggle notebooks.

Default Kaggle behavior:

- `--scale base` uses a practical 9-hour-oriented TAC size.
- `--max-seconds 30600` gives an 8.5-hour run window.
- `--stop-buffer-seconds 1200` leaves 20 minutes for final checkpoint writes.
- Outputs `last.pt`, `best.pt`, `metrics.jsonl`, `run_manifest.json`, and `final_summary.json`.
- Use `torchrun --standalone --nproc_per_node=2` to train on both T4 GPUs.

## Hard Agentic Corpus Build

Date: 2026-05-29

Built `runs/prepared_corpus_agentic_hard` and upload artifact `runs/prepared_corpus_agentic_hard_upload.zip`.

Implementation:

- `dedupe_prepared_jsonl()` and `normalize_template_text()` in `tac_transformer/data.py`;
- `tac_transformer/hard_agentic_data.py` for harder traces;
- `kaggle/build_hard_agentic_corpus.py` as the reproducible builder.

Final artifact:

- train records: 428,671;
- train approx tokens: 263,030,010;
- eval records: 5,441;
- eval approx tokens: 1,274,261.

Audit:

- first 50k train exact unique rate: 1.0000;
- first 50k train normalized template unique rate: 0.9074;
- hard generated train exact unique rate: 1.0000;
- hard generated train normalized template unique rate: 0.8009;
- eval exact unique rate: 1.0000;
- eval normalized template unique rate: 0.8734.

The hard generated train split is balanced at 20,000 records each for tool choice, repair after failure, memory counterfactual, verification planning, argument schema repair, and stale-memory rejection.

## Content-Addressed Memory and Iterative Retrieval Addendum

Date: 2026-05-30

Implemented and validated the corrected content-addressed memory path:

- `memory_read_type="content_addressed"` stores `(cue_hidden, value_hidden)` pairs instead of route patterns.
- `content_store_size` controls the fixed carried store size.
- `content_addressed_hit` logs cue-match strength.
- The earlier content-addressed default used `content_read_steps=1`; the current promoted default is synthesis-gated two-step content retrieval (`content_read_steps=2`, `content_read_gate_type="synthesis"`).

Full harder-task result:

- `content_addressed_k1`: mean carry 0.4349, effective 15/15, train TPS ratio 0.4823.
- `content_addressed_k2`: mean carry 0.4341, effective 14/15, train TPS ratio 0.4679.
- `base_routing`: mean carry 0.0677, effective 15/15, train TPS ratio 0.5236.

Decision:

- Promote single-step content-addressed k1 for direct recall.
- Do not promote k2 because it is not more accurate and loses one effective run.
- BASE/program-memory remains the stronger path for the multi-hop slice.

Iterative retrieval was tested for multi-hop:

- Learned two-step k2: carry 0.0508, carry-reset 0.0417, TPS ratio 0.4150.
- BASE routing: carry 0.0495, carry-reset 0.0234, TPS ratio 0.5254.
- Single-step content k1: carry 0.0365, carry-reset 0.0221, TPS ratio 0.4934.

The learned two-step path confirms that chained lookup is useful, but the full matrix drops from 0.4349 mean carry for single-step content k1 to 0.3552 for iterative k2, so it is not the default.

Confidence-gated iterative retrieval was also tested:

- `content_confidence_iterative_k1`: multi-hop carry 0.0482, carry-reset 0.0313.
- `content_confidence_iterative_k2`: multi-hop carry 0.0482, carry-reset 0.0339.

Decision:

- Keep confidence gating as an ablation only.
- Cosine lookup confidence is not a strong enough halt/continue verifier.
- The next multi-hop mechanism should be a supervised verifier or halt head trained from correctness traces, not another fixed confidence rule.

Phase 1A synthesis-gated retrieval was tested next:

- `content_read_gate_type="synthesis"` uses `[query, read_1, read_2, read_1 - read_2, read_1 * read_2]`.
- `content_synthesis_k1`: multi-hop carry 0.0417, carry-reset 0.0247, TPS ratio 0.4548.
- `content_synthesis_k2`: multi-hop carry 0.0391, carry-reset 0.0234, TPS ratio 0.4336.
- Control `content_iterative_k2`: multi-hop carry 0.0508.
- Control `base_routing`: multi-hop carry 0.0495.

Decision:

- Phase 1A synthesis fails the go/no-go because it does not beat BASE.
- Keep synthesis as an ablation.
- The failure suggests that simply combining two retrieved vectors is not enough; the next reasoning attempt should test graph traversal or supervised verification, not another fixed two-vector blend.

Artifacts:

- `docs/content_addressed_full_matrix_validation.md`
- `docs/iterative_retrieval_validation.md`
- `runs/benchmarks/content_addressed_full_matrix_2026_05_30/RESULTS.md`
- `runs/benchmarks/content_iterative_full_matrix_2026_05_30/RESULTS.md`
- `runs/benchmarks/conditional_iterative_focused_2026_05_30/RESULTS.md`
- `runs/benchmarks/synthesis_iterative_focused_2026_05_30/RESULTS.md`

## Automated Research Funnel and Synthesis Promotion

Date: 2026-05-30

The full automated research mode ran in two stages.

Stage 1 screened every implemented harder-matrix variant:

- 42 variants;
- 5 harder tasks;
- seed 11;
- 60 training steps;
- 210 trained/evaluated runs.

Top screen result:

- `content_synthesis_k2`: mean carry 0.2297;
- `current_best` / `content_addressed_k1`: mean carry 0.2258;
- `content_synthesis_k1`: mean carry 0.2258 with 5/5 effective runs.

Stage 2 confirmed the selected candidates:

- 9 variants;
- 5 harder tasks;
- seeds 11, 23, 37;
- 120 training steps;
- 135 trained/evaluated runs.

Confirmed ranking:

| Variant | Effective | Mean carry | Carry-reset | Carry-shuffled | TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `content_synthesis_k2` | 15/15 | 0.4909 | 0.4742 | 0.4773 | 0.4336 |
| `content_synthesis_k1` | 15/15 | 0.4901 | 0.4737 | 0.4753 | 0.4691 |
| `content_addressed_k1` | 15/15 | 0.4349 | 0.4229 | 0.4185 | 0.5295 |
| `content_iterative_k1` | 15/15 | 0.3565 | 0.3422 | 0.3417 | 0.4622 |
| `content_iterative_k2` | 15/15 | 0.3552 | 0.3406 | 0.3391 | 0.4345 |
| `base_routing` | 15/15 | 0.0677 | 0.0529 | 0.0508 | 0.5310 |

Decision:

- Promote `content_synthesis_k1` into `best_tac_config`.
- `content_synthesis_k2` is the numerical mean-carry leader but is only 0.0008 higher while slower and more complex.
- `content_synthesis_k1` is the better default for a small lab: simpler BASE k=1 routing, 15/15 effective, nearly identical accuracy, better throughput.

Task-specific read modes:

- Clean direct recall: synthesis-gated content read.
- Noisy-key partial cue: confidence-gated content k1 is strongest.
- Multi-hop: iterative k2 is the narrow best, but still not a solved reasoning loop.

Artifacts:

- `docs/automated_research_synthesis_promotion.md`
- `runs/benchmarks/automated_research_stage1_screen_2026_05_30/RESULTS.md`
- `runs/benchmarks/automated_research_stage2_confirm_2026_05_30/RESULTS.md`

## Kaggle Synthesis Training Preflight

Date: 2026-05-30

The local machine is CPU-only, so the full dual-T4 Kaggle run cannot execute here. A local real-corpus preflight was run against `runs/prepared_corpus_agentic_hard` to verify that the promoted synthesis-gated preset trains, evaluates, logs metrics, and writes checkpoints.

Preflight:

- scale: `smoke`;
- train file: `runs/prepared_corpus_agentic_hard/train.prepared.jsonl`;
- eval file: `runs/prepared_corpus_agentic_hard/eval.prepared.jsonl`;
- steps: 20;
- completed: yes;
- checkpoints written: `last.pt` and `best.pt`.

Final preflight metrics:

- best eval loss: 6.4155;
- final eval loss: 6.4446;
- train `content_addressed_hit`: 0.2806;
- train `content_synthesis_gate`: 0.2695;
- train `program_memory_cosine`: 0.9335;
- eval `content_addressed_hit`: 0.1915;
- eval `content_synthesis_gate`: 0.2694;
- eval `program_memory_cosine`: 0.9328.

Decision:

- Kaggle trainer is ready for the full dual-T4 launch.
- Training and eval now log `content_synthesis_gate`, which is required for the promoted synthesis architecture.

Artifact:

- `docs/kaggle_synthesis_training_preflight.md`

## Attention and IdentityState Fusion Experiments

Date: 2026-05-31

Implemented opt-in fusion controls for the user's attention/IdentityState architecture question:

- `attention_window_size` for local causal attention pressure.
- `identity_attention_type="coherence_sparse"` for hard same-program sparse attention.
- `identity_attention_type="coherence_sparse_compressed"` for sparse token attention plus carried program-memory K/V bridge slots.
- `identity_attention_type="identity_first"` for K/V projections conditioned on soft token program identity.

Focused screen:

- Artifact: `runs/benchmarks/attention_identity_fusion_screen_2026_05_31`.
- Settings: seed 11, 30 steps, `noisy_key` and `multi_hop`.
- Hard sparse attention hurt multi-hop.
- Sparse+compressed+local repaired the short-screen multi-hop drop.
- Identity-first improved noisy-key but failed multi-hop effectiveness.

Confirmation:

- Artifact: `runs/benchmarks/attention_identity_fusion_confirm_2026_05_31`.
- Settings: seeds 11/23/37, 60 steps, all five harder chunked-memory tasks.
- `identity_first_attention`: mean carry 0.1896, 12/15 effective.
- `coherence_sparse_compressed_local_w4`: mean carry 0.1875, 11/15 effective.
- `compressed_memory_best`: mean carry 0.1854, 13/15 effective.
- `current_best`: mean carry 0.1729, 13/15 effective.

Full matrix:

- Artifact: `runs/benchmarks/attention_identity_fusion_full_matrix_2026_05_31`.
- Settings: seeds 11/23/37, 120 steps, all five harder chunked-memory tasks, full fusion variant set.
- `identity_first_attention`: mean carry 0.5099, 15/15 effective, 5/5 task wins, carry-reset 0.4961, gap 0.4938, TPS ratio 0.4248.
- `identity_first_local_w4`: mean carry 0.5057, 15/15 effective.
- `coherence_sparse_attention`: mean carry 0.4940, 15/15 effective.
- `local_attention_w8`: mean carry 0.4911, 15/15 effective.
- `local_attention_w4`: mean carry 0.4911, 15/15 effective.
- `current_best`: mean carry 0.4904, 15/15 effective.
- `compressed_memory_best`: mean carry 0.4724, 15/15 effective.

Decision:

- Promote identity-first attention into `best_tac_config`, because the full 120-step matrix validated the earlier quality signal and removed the effectiveness concern.
- Do not promote compressed memory as the general default. It won the smaller near-term screen but regressed to last place in the full matrix.
- Sparse identity attention should stay experimental. Coherence sparse attention beat the prior default, but not identity-first; future fixes should test learned/top-k bridge edges rather than hard same-program masking.
- Local attention is behaviorally safe but not yet an efficiency win because the implementation still uses dense masked logits.

Documentation:

- `docs/attention_identity_fusion_experiments.md`

## Identity-First Run 3 Preflight And Inspection Tools

Date: 2026-05-31

Implemented the handoff tooling for the next Kaggle concept-formation run:

- Added explicit `program_ortho` metric alias for the existing program-memory separation loss, so metrics now include both `aux_loss_separation`/`weighted_aux_loss_separation` and `metric_program_ortho`.
- Added `kaggle/inspect_identity_memory.py` to load a checkpoint, run a prompt, and report top programs plus decoded content-store cue/value projections.
- Added `kaggle/evaluate_checkpoint_harder_matrix.py` to run carry/reset/shuffled harder-task probes against a completed checkpoint without retraining.
- Updated the Kaggle bundle to include the new tools and the identity-first Run 3 command.

Local preflight:

- Artifact: `runs/preflights/identity_first_run3_preflight_2026_05_31`.
- Command: smoke scale, 20 steps, hard agentic train/eval JSONL, CPU fp32.
- Completed 20/20 steps and wrote `last.pt`, `best.pt`, `metrics.jsonl`, `run_manifest.json`, and `final_summary.json`.
- Manifest confirmed `identity_attention_type="identity_first"`, `memory_separation_weight=0.01`, `content_cue_separation_weight=0.005`, `content_gate_entropy_weight=0.005`, and `content_reconsolidate=True`.
- Eval step 20: loss 6.3283, `program_memory_cosine=0.9303`, `metric_program_ortho=0.9233`, `content_synthesis_gate=0.2694`, `content_gate_entropy=0.8406`, `content_cue_cosine=0.1305`.

Post-run recommendation:

- Start fresh in `/kaggle/working/best_tac_agentic_identity_first_run3`.
- Use dual T4 `torchrun`, `--precision fp16`, `--steps 20000`, `--warmup-steps 500`, `--batch-size 12`, `--grad-accum-steps 3`.
- Enable `--analyze-specialization-at-end` so Run 3 writes the functional-specialization matrix from `best.pt` before the session ends.
- Save `best.pt`; then run checkpoint inspection and harder-matrix evaluation tools.

Documentation:

- `docs/kaggle_identity_first_run3_preflight.md`

## Program Specialization Analysis

Date: 2026-05-31

Decision:

- Pause additional architecture launches until existing trained checkpoints are analyzed for functional specialization.
- Differentiation is not enough; the next evidence gate is whether dominant program IDs correlate with trace categories and whether program knockout causes category-specific loss deltas.

Implementation:

- Added `kaggle/analyze_program_specialization.py`.
- The tool records per-input top-k program attribution, computes mutual information between dominant program and hard-agentic trace category, builds per-category activation histograms, and runs one-program-at-a-time knockout ablations.

Local smoke analysis:

- Artifact: `runs/analysis/program_specialization_identity_first_preflight_2026_05_31`.
- Checkpoint: `runs/preflights/identity_first_run3_preflight_2026_05_31/best.pt`.
- Data: `runs/prepared_corpus_agentic_hard/hard_agentic_eval.generated.jsonl`.
- Sample: 8 records per category, 48 total.
- Result: all sampled records routed to program 15; program-category MI = 0.0 bits; program entropy = 0.0 bits.
- Largest smoke-check knockout deltas were tiny: program 9 +0.0035 loss, program 10 +0.0015 loss, program 0 +0.0012 loss.

Interpretation:

- The local checkpoint is a 20-step smoke preflight, so this is a pipeline validation, not a concept-formation verdict.
- The decisive analysis should be rerun on the trained Kaggle `best.pt` checkpoint that produced low `program_memory_cosine`.

Run 3 fusion update:

- The Kaggle trainer now accepts `--analyze-specialization-at-end`.
- When enabled, training writes `specialization/program_specialization.json`, `specialization/program_attribution.csv`, and a compact `specialization_analysis` section in `final_summary.json`.
- The fused Run 3 command should use `--specialization-max-records-per-category 64` and omit `--specialization-knockout-programs` so every program is knocked out one at a time.

Documentation:

- `docs/program_specialization_analysis.md`

## Run 3 Forced Program Routing Diagnostic

Date: 2026-06-01

Question:

- Does the Run 3 program-31 dominance mean program 31 contains uniquely useful content, or is the specialization report collapsing a richer routing pattern into a misleading final-token attribution?

Implementation:

- Added `kaggle/evaluate_forced_programs.py`.
- Added a regression test covering checkpoint load, natural evaluation, and forced-program comparison in `tests_py/test_tac_transformer.py`.
- The evaluator loads a trained checkpoint, evaluates natural routing, then monkey-patches each identity layer's route selection so every token is forced through one requested program.

Local Run 3 sample:

- Artifact: `runs/analysis/forced_program_run3_sample/forced_programs_max1.json`.
- Attribution diagnostic: `runs/analysis/forced_program_run3_sample/routing_collapse_max1_no_interventions.json`.
- Checkpoint: `runs/kaggle_results/tac_identity_first_run3_fused/best_tac_agentic_identity_first_run3/best.pt`.
- Data: `runs/prepared_corpus_agentic_hard/eval.prepared.jsonl`.
- Sample: 1 record per category, 13 records total.
- Natural routing loss: `0.7141`.
- Natural top-program report: program `31` for `13/13` records.
- Best forced program: program `18`, loss `0.7014`.
- Worst forced program: program `31`, loss `0.8929`.
- Forced-program loss range: `0.1915`.
- Forced programs beating natural routing on this sample: `5/32`.
- Raw activation distribution is high-entropy rather than collapsed: mean entropy `4.9891` bits out of a 5-bit maximum, mean top probability `0.0394`.
- BASE full-sequence schedule is exactly balanced for a 256-token window: `8` positions per program across all 32 programs.
- Training `metric_routing_load_std` is constant at `0.1767767`; this is consistent with single-program final-token masking, not an observed learned collapse trajectory.

Critical code inspection:

- Current `routing_type="base"` is not a learned per-program bias router.
- In the causal path, route selection is computed for flattened token rows and then only the final token's route is exposed as `selected_program_mask`.
- `_base_route` selects the primary adaptive program with `row_id % adaptive_programs`; with 32 programs and 256-token windows, the final token naturally maps to program `31`.
- Therefore, the all-program-31 specialization report is likely a final-token attribution artifact, not proof that all tokens were routed through program 31.

Interpretation:

- The original "learned bias collapsed into program 31" hypothesis is not supported by the current code path.
- The forced-program result still matters: programs are behaviorally non-identical under intervention, and program 31 is not the best forced path on the sampled eval set.
- This shifts the immediate priority from adding a Switch-style load loss to fixing the routing diagnostics first: log token-level route distributions, token-level soft routing probabilities, and category-conditioned program use before changing the training objective.
- A full forced-program pass over the entire hard eval set should be run on GPU or with a smaller diagnostic batch plan; the local CPU sample already shows enough signal to reject the simplest "program 31 owns the computation" interpretation.

## Bidirectional Evolutionary Search Full Matrix

Date: 2026-06-01

Implemented and locally validated a bidirectional evolutionary research controller for TAC.

Implementation:

- Added `kaggle/benchmark_bidirectional_evolution.py`.
- Added focused tests in `tests_py/test_bidirectional_evolution.py`.
- The runner can either launch a new harder-task matrix or re-score an existing per-seed matrix.
- Scoring combines forward task fitness with backward behavioral novelty:
  - forward fitness: carry accuracy, carry-vs-reset delta, and TAC-vs-baseline gap;
  - backward fitness: behavioral novelty, program differentiation, gate conditionality, and live program allocation;
  - constraints: gate saturation, program collapse, and dead-program rate;
  - outputs: survival ranking, Pareto front, MAP-Elites grid, per-seed JSON, aggregate JSON, and Markdown report.

Fresh local full matrix:

- Artifact: `runs/benchmarks/bidirectional_evolution_full_matrix_2026_05_31`.
- Device: local CPU; Kaggle training was not touched.
- Candidates: `10`.
- Tasks: `5`.
- Seeds: `11`, `23`, `37`.
- Total trained/evaluated rows: `150`.
- Settings: `120` steps, batch size `32`, `8` eval batches.

Survival ranking:

| Rank | Candidate | Survival | Forward | Backward | Mean carry | Carry-reset | Gap | Novelty | Wins | Effective |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `content_synthesis_k2` | 2.6963 | 1.4836 | 1.2127 | 0.5042 | 0.4914 | 0.4880 | 0.1176 | 2 | 15/15 |
| 2 | `program_novelty_hard` | 2.6078 | 1.4997 | 1.1081 | 0.5099 | 0.4961 | 0.4938 | 0.0079 | 2 | 15/15 |
| 3 | `program_novelty_soft` | 2.5944 | 1.4997 | 1.0947 | 0.5099 | 0.4961 | 0.4938 | 0.0025 | 2 | 15/15 |
| 4 | `current_best` | 2.5894 | 1.4997 | 1.0897 | 0.5099 | 0.4961 | 0.4938 | 0.0000 | 2 | 15/15 |
| 5 | `content_synthesis_k1` | 2.5894 | 1.4997 | 1.0897 | 0.5099 | 0.4961 | 0.4938 | 0.0000 | 2 | 15/15 |
| 6 | `identity_first_local_w4` | 2.5849 | 1.4857 | 1.0992 | 0.5057 | 0.4919 | 0.4880 | 0.0095 | 0 | 15/15 |
| 7 | `coherence_sparse_attention` | 2.5583 | 1.4516 | 1.1068 | 0.4940 | 0.4797 | 0.4779 | 0.0225 | 0 | 15/15 |
| 8 | `iterative_multi_hop_k2` | 2.3625 | 1.1398 | 1.2227 | 0.3901 | 0.3768 | 0.3729 | 0.1260 | 1 | 15/15 |
| 9 | `confidence_iterative_k1` | 2.2933 | 1.0755 | 1.2178 | 0.3682 | 0.3563 | 0.3510 | 0.1260 | 0 | 15/15 |
| 10 | `mamba_program_memory` | 1.7903 | 0.0120 | 1.7783 | 0.0117 | 0.0003 | -0.0049 | 0.7098 | 0 | 0/15 |

Task winners:

- `longer_single_key`: `program_novelty_hard`, carry `0.7891`.
- `multi_key`: `content_synthesis_k2`, carry `0.8568`.
- `delayed_query`: `program_novelty_hard`, carry `0.7565`.
- `noisy_key`: `iterative_multi_hop_k2`, carry `0.1250`.
- `multi_hop`: `content_synthesis_k2`, carry `0.0560`.

Decision:

- `content_synthesis_k2` is the fresh full-matrix survival winner because it combines strong carry with higher behavioral novelty, and it wins both `multi_key` and the weak-but-important `multi_hop` slice.
- Keep `current_best` / `content_synthesis_k1` as the practical default family: same top mean carry as the novelty-weight variants, simpler routing, and better routing efficiency than k2.
- Do not treat `program_novelty_soft` or `program_novelty_hard` as real behavioral novelty mechanisms yet. They improve program differentiation metrics without changing task-level behavior enough; real bidirectional novelty needs to score behavioral descriptors directly.
- Preserve `iterative_multi_hop_k2` as a niche candidate for deceptive retrieval search. It does not win multi-hop in this run, but it wins noisy-key and has the strongest useful novelty among effective candidates.
- Reject `mamba_program_memory` for this branch: it is behaviorally novel but ineffective, with `0/15` effective runs and near-zero carry.

Documentation:

- `docs/bidirectional_evolutionary_search.md`
## Automated Program-Specialization Routing Research

Date: 2026-06-01

Problem:

- Run 3 learned useful, behaviorally non-identical program compute, but BASE routing assigns selected programs by token position rather than task semantics.
- Token-level observability confirms selected routes are balanced by schedule, not category-conditioned.

Candidate solution family:

1. `base_semantic`: preserve the BASE scheduled program as a stable compute anchor, then add one or more activation-conditioned programs selected from the token's raw program activations.
2. `base_semantic` + routing load-balance loss: discourage the learned semantic slot from collapsing to a single high-activation program.
3. Existing `sparse_ensemble_k2`: BASE anchor plus stability-selected extra program, used as the closest already-implemented control.
4. Existing stronger-memory candidates (`content_synthesis_k1/k2`): preserve as quality controls so specialization pressure does not accidentally damage recall.

Decision criteria:

- First screen: harder chunked-memory carry/reset/shuffle must not regress badly against `current_best`.
- Specialization screen: token-level activation/route entropy and category-conditioned knockout selectivity should improve on real hard-agentic samples.
- Promotion requires both behavioral quality and functional specialization; a route that only improves utilization without category-conditioned knockout selectivity is not enough.

Full automated continuation:

- Ran the full harder routing matrix requested after the initial stage-1 screen: 7 variants x 5 tasks x 3 seeds at 120 steps. `current_best` still wins aggregate carry at `0.5099`; `content_synthesis_semantic_k2` remains close at `0.5029`; route-only semantic variants collapse to about `0.05` carry and are rejected.
- Ran a multi-seed category objective matrix over hard-agentic labels. `base_semantic_mi_0p5` is the strongest selected-route specialization candidate: eval loss `4.2899`, selected-route NMI `0.00947`, selected-route MI `0.02209` bits. `current_best` has selected-route MI `0` because BASE selected routes are schedule-driven.
- Ran the full category-conditioned knockout matrix. `base_semantic_mi_0p5` has top knockout selectivity span `0.02713` versus `0.00268` for `current_best`, about a 10x functional-selectivity improvement.
- Tested and rejected `base_semantic_soft` and `base_semantic_soft_mi_0p5`: both keep knockout selectivity near baseline (`0.002813`) and selected-route NMI near zero despite higher raw activation NMI.

Working solution:

```text
routing_type = "base_semantic"
routing_top_k = 2
routing_load_balance_weight = 0.05
category_route_weight = 0.5
category_route_objective = "mi"
```

Decision:

- Keep `current_best` as the pure harder-memory control/default until a longer full training run proves no carry regression.
- Use `base_semantic_mi_0p5` for specialization research and the next agentic training run because it is the only tested branch with selected-route category MI plus category-conditioned knockout selectivity.
- The current evidence proves run-local functional selectivity. It does not yet prove stable global meanings for fixed program IDs across random seeds.

Documentation:

- `docs/program_specialization_solution_research.md`

## TAC Next Research Plan P0 Start

Date: 2026-06-01

Implemented the first P0 diagnostics from the next sprint plan.

Token-level routing telemetry:

- `kaggle/analyze_program_specialization.py` now supports `--all-records`, `--no-knockouts`, and `--token-csv-output`.
- Token CSV rows include token position, raw top program/probability, raw activation entropy, selected top program/probability, route entropy, active selected program count, and active selected program IDs.
- The JSON report now includes `category_route_histogram` in addition to the existing activation histogram and MI reports.
- Smoke artifact: `runs/analysis/token_telemetry_run3_smoke_2026_06_01`.
- Run 3 smoke result still confirms BASE selected routes are schedule-balanced rather than category-conditioned: selected-route token MI `0`, raw token MI `0.001684`, activation sparsity `0.109441`.

Forced-program matrix tooling:

- `kaggle/evaluate_forced_programs.py` now reports per-program `loss_delta_vs_natural`, category-conditioned best/worst program rankings, forced loss variance, and optional CSV output.
- Smoke artifact: `runs/analysis/forced_program_delta_smoke_2026_06_01`.
- Smoke result on 6 category-balanced records: natural loss `0.1844`, best forced program `16` loss `0.171559`, worst forced program `31` loss `0.268417`, forced loss range `0.096858`, and `22/32` forced programs beat natural routing.
- Kaggle GPU launch for the full 5,441-record forced matrix was blocked by the account's weekly GPU quota.
- A CPU full-matrix fallback was attempted at `runs/analysis/forced_program_full_eval_cpu_2026_06_01`, but batch size 64 caused a native PyTorch crash and batch size 32 did not complete even the natural all-record pass after `698.6` seconds. The full all-record forced matrix should be rerun when GPU quota is available.
- GPU-ready kernel package prepared at `runs/kaggle_forced_program_full_eval_2026_06_01`; `kaggle kernels push` currently fails with `Maximum weekly GPU quota of 30.00 hours reached`.

Causal content-memory audit:

- Added `kaggle/audit_content_memory_causality.py`.
- The correct default audit task is `multi_key`, because `single_key` can be answered from a one-value context without using a query key.
- Multi-key local audit artifact: `runs/analysis/content_memory_causal_audit_multikey_2026_06_01/audit.json`.
- Result: normal carry `0.2539`, randomized-query carry `0.0625`, randomized-query reset `0.0117`, randomized-query shuffled `0.0117`; verdict `pass` under the 0.10 leakage threshold.

Documentation:

- `docs/causal_audit_content_memory.md`

## Fused P0 Research Gate Update

Date: 2026-06-01

Read the fused research-state note and updated the active sequencing.

Corrections:

- The full 5,441-record forced-program matrix is the hard gate before launching a long semantic-routing run. `base_semantic_mi_0p5` remains the best current specialization candidate, but it should not consume a 20k-step GPU run until the full forced-program matrix confirms broad non-identical program function.
- Add a mandatory modern-backbone attribution check before interpreting the next Kaggle result as TAC-specific: parameter-matched RoPE + RMSNorm + SwiGLU, no TAC modifications, harder matrix at 120 steps. If this closes more than 30% of the current carry gap, prior promotion claims need to be revisited.
- Move GPU decode/inference profiling into P0. If promoted TAC decode latency is more than 3x vanilla, the commercial framing should shift before more long training runs.
- Multi-hop work is gated on category-conditioned forced-program evidence. If a program shows specific multi-hop advantage, fix routing first; if category effects are random, treat multi-hop as a likely scale blocker at the current 26.91M-parameter size.

Immediate order:

1. Retry/monitor the Kaggle forced-program full matrix with refreshed auth.
2. If GPU quota still blocks execution, continue with unblocked P0 tooling and local smoke validation for inference profiling and modern-backbone attribution.
3. Do not start the semantic-routing 20k run until the forced-program full matrix produces the go/no-go result.

Backbone attribution audit result:

- Added `kaggle/audit_modern_backbone_attribution.py`.
- Full audit artifact: `runs/benchmarks/modern_backbone_attribution_2026_06_01`.
- Compared parameter-matched vanilla backbones over all five harder chunked-memory tasks, seeds `11/23/37`, `120` steps.
- Legacy vanilla mean carry: `0.01745`.
- Modern RoPE + RMSNorm + SwiGLU vanilla mean carry: `0.01563`.
- Current TAC reference mean carry: `0.50990`.
- Backbone gap closure fraction: `-0.00370`, far below the `0.30` risk threshold.
- Verdict: `pass`. The modern backbone alone does not explain the TAC carry result in this benchmark family.

Full forced-program matrix result:

- Artifact: `runs/kaggle_results/tac_forced_program_full_eval_5441`.
- True full prepared eval: `5,441` records from `eval.prepared.jsonl`.
- Runtime: `1778.0s` on CUDA/T4, batch size `64`.
- Natural loss: `0.27810`.
- Best forced program: `24`, loss `0.27240`, delta vs natural `-0.00571`.
- Worst forced program: `31`, loss `0.36404`, delta vs natural `+0.08594`.
- Forced loss range: `0.09165`.
- Forced programs better than natural: `11/32`.
- Strict global gate (`>=10` programs with `|loss_delta| > 0.01`) did not pass: only `6/32` programs exceeded that threshold globally, all on the harmful side (`P0`, `P1`, `P2`, `P4`, `P30`, `P31`).
- Category-conditioned specialization is much stronger than the global average:
  - `arc_reasoning`: best `P18`, range `0.26239`.
  - `coding`: best `P24`, range `0.27259`.
  - `failure_recovery`: best `P24`, range `0.33700`.
  - `filesystem`: best `P18`, range `0.24011`.
  - `rag`: best `P22`, range `0.27480`.
  - `testing`: best `P18`, range `0.29562`.
  - `memory_counterfactual`: best `P8`, range `0.07680`.
  - `repair_after_failure`: best `P14`, range `0.08055`.
  - `stale_memory_rejection`, `tool_choice`, and `verification_planning`: best `P16`.
- Interpretation: Run 3 does contain functional program differentiation, but it is concentrated and category-conditioned rather than globally broad. `P31` is not a useful dominant program; it is the worst program overall and worst in most categories. `P0` is also strongly harmful on several prepared-eval categories.

Routing decision after the full matrix:

- Do not launch a long 20k semantic-routing run under the original strict gate as if all checks passed.
- The next routing objective should be targeted: train `base_semantic_mi_0p5` or a derivative to select among the empirically useful program family (`P8`, `P14`, `P16`, `P18`, `P22`, `P24`) and suppress harmful programs (`P0`, `P31`) under category-conditioned evidence.
- The full matrix strengthens the case for semantic routing as an observability/routing solution, but it also says specialization is not yet broad enough to claim stable global program meanings.

GPU inference profile result:

- Artifact: `runs/kaggle_results/tac_gpu_inference_profile`.
- Current promoted stack mean decode throughput vs parameter-matched vanilla: `0.12263`, about `8.15x` slower.
- Current promoted stack mean carried-query throughput vs vanilla: `0.09618`, about `10.40x` slower.
- Current promoted stack mean decode vs BASE program-memory TAC: `0.69654`.
- Verdict under the fused commercial framing gate: `fail` for commercial viability (`>3x` vanilla decode tax). Treat TAC as a research demonstrator until decode/KV-cache/state-update overhead is optimized.

## Targeted Routing Policy Full Eval

Date: 2026-06-01

Implemented the post-forced-matrix routing solution as actual configurable policy knobs.

Implementation:

- Added `semantic_route_allowed_programs` and `semantic_route_suppressed_programs` to `TACConfig`.
- `base_semantic` and `base_semantic_soft` now mask semantic extra-program candidates through those filters.
- `kaggle/train_best_tac_agentic.py` accepts:
  - `--semantic-route-allowed-programs`
  - `--semantic-route-suppressed-programs`
- Added focused tests for targeted semantic routing and invalid filter rejection.

Run 3 policy screen:

- Local smoke artifact: `runs/analysis/targeted_semantic_policy_smoke_2026_06_01/policy_screen.json`.
- The useful-family policy beat original BASE on a 2-record/category smoke sample: `0.69238` loss vs `0.70984`.
- Unconstrained semantic extras regressed on the smoke sample, matching the forced-matrix warning that the router must prefer the useful family rather than all programs.

Full all-record GPU eval:

- Kaggle kernel: `mathewlincoln/tac-targeted-routing-policy-eval`.
- Artifact: `runs/kaggle_results/tac_targeted_routing_policy_eval/targeted_routing_policy_eval`.
- Records: `5,441`.
- Runtime: `317.17s` on CUDA/T4, batch size `64`.
- Baseline Run 3 BASE policy: loss `0.2781038760`, token accuracy `0.9195675772`.
- Best policy: `base_semantic_k3_useful_family_suppress_p0_p31`, loss `0.2680741841`, delta vs BASE `-0.0100296920`, token accuracy `0.9230782485`.
- Useful-family top-k 2 also helped: loss `0.2702030335`, delta `-0.0079008426`.
- Unconstrained semantic top-k 2 barely helped: loss `0.2777651412`, delta `-0.0003387348`.

Winning policy category deltas vs Run 3 BASE:

- `arc_reasoning`: `-0.05695`
- `argument_schema`: `-0.00344`
- `coding`: `-0.05007`
- `failure_recovery`: `-0.04470`
- `filesystem`: `-0.04271`
- `memory_counterfactual`: `-0.00607`
- `multi_step_agent`: `-0.00408`
- `rag`: `-0.05737`
- `repair_after_failure`: `-0.01020`
- `stale_memory_rejection`: `-0.00399`
- `testing`: `-0.01102`
- `tool_choice`: `-0.00632`
- `verification_planning`: `-0.00894`

Decision:

- This is the first working Run 3 routing solution validated on the true full prepared eval.
- For the existing Run 3 checkpoint, use hard `base_semantic` top-k 3 restricted to useful programs `P8/P14/P16/P18/P22/P24`, with `P0/P31` suppressed from semantic extras.
- For fresh training, keep `base_semantic_mi_0p5` as the seed-stable general solution until cross-seed program-role alignment is proven.

## Run 4 Semantic-MI Launch Prep

Date: 2026-06-01

The forced-program full matrix and targeted-policy full eval close the launch
gate for a fresh semantic-routing run. The next training run should not hardcode
Run 3 useful program IDs; it should train the general objective that produced
the best multi-seed specialization signal:

- `--routing-type base_semantic`
- `--routing-top-k 2`
- `--routing-load-balance-weight 0.05`
- `--category-route-weight 0.5`
- `--category-route-objective mi`

Added periodic specialization checkpoint support to the Kaggle trainer:

- `--specialization-checkpoints 2000 5000 10000 20000`
- `--specialization-checkpoint-max-records-per-category 16`
- optional `--specialization-checkpoint-run-knockouts` for heavier checkpoint
  probes

Run 4 should still use `--analyze-specialization-at-end` with
`--specialization-max-records-per-category 64` on CPU so the final checkpoint
gets the fuller token-level attribution and knockout report.

The backbone-modernization audit and GPU decode profile are already closed:

- backbone attribution passed; TAC carry advantage is not explained by the
  non-modern baseline
- GPU decode profile failed commercial viability for now, with the promoted TAC
  path about 8x to 10x slower than vanilla decode depending on carried-query
  setup

## Run 4 Semantic-MI Partial Result and Resume

Date: 2026-06-02

Kernel version 2 completed successfully but stopped for Kaggle wall-clock time
before the 20,000-step target.

Pulled artifact:

- `runs/kaggle_results/tac_run4_semantic_mi_20k_latest/best_tac_agentic_run4_semantic_mi`

Partial training state:

- completed steps: `8,221 / 20,000`
- `stopped_for_time`: `true`
- best eval loss: `6.421461343765259`
- latest elapsed seconds: `29400.572753777997`
- latest tokens seen: `150,937,560`
- latest sequences seen: `591,912`
- latest `program_memory_cosine`: `0.9630166292190552`
- route objective remained `base_semantic`, top-k 2, category MI weight 0.5

Specialization checkpoints:

- step 2,000 artifact:
  `specialization_checkpoints/step_002000`
- step 5,000 artifact:
  `specialization_checkpoints/step_005000`
- both sampled 96 records, reported `mi_bits=0.8085222685473687`,
  `normalized_mi=0.3127791092991895`, and
  `program_entropy_bits=3.274849878784676`

The time-stopped run also executed the expensive end specialization on `best.pt`
from checkpoint step 7,500:

- sampled records: `384`
- categories observed: `argument_schema`, `memory_counterfactual`,
  `repair_after_failure`, `stale_memory_rejection`, `tool_choice`,
  `verification_planning`
- `mi_bits=0.5330981670335928`
- `normalized_mi=0.20623052244853393`
- `program_entropy_bits=3.2813949309672394`
- runtime: `5794.346525305995` seconds

Dominant partial-run programs by sampled category:

- `argument_schema`: P0
- `memory_counterfactual`: P1
- `repair_after_failure`: P2
- `stale_memory_rejection`: P0
- `tool_choice`: P1
- `verification_planning`: P5

Decision:

- Treat the v2 result as promising partial evidence that semantic-MI routing is
  creating category/program dependence, not as the final Run 4 conclusion.
- Because end specialization consumed about 1.6 hours on an intermediate
  time-stop, future resume legs use `--skip-end-specialization-on-time-stop`.
  Periodic checkpoints remain enabled, and the final end specialization still
  runs if the 20,000-step target is reached.
- Resume from `last.pt` rather than restarting; the output directory was
  packaged as private dataset
  `mathewlincoln/tac-run4-semantic-mi-resume-8221-2026-06-02`.
- Kernel `mathewlincoln/tac-run-4-semantic-mi-20k` version 3 was pushed with
  the resume dataset attached and is RUNNING.

## TAC Long-Context Efficiency Direction

Date: 2026-06-02

Wrote `docs/tac_agentic_long_context_efficiency_research.md` to fuse the
current TAC efficiency work with a usable context-window plan for agentic and
knowledge work.

Decision:

- Do not simply raise `max_seq_len` from 256 to 4096/8192 with dense attention.
  The current 256 length is byte-level, and dense attention would scale
  quadratically.
- Treat long-context TAC as a working-memory architecture:
  local dense attention for the active workspace, TAC identity/program state
  for persistent computation, compressed segment memory for prior context, and
  retrieval-backed memory for knowledge.
- First implementation target should be a measured 1024-token path:
  tokenized/memmap batcher, RoPE scaling controls, local/global attention,
  identity global slots, and segment carry benchmarks.
- The long-term target is 4096 usable tokens for agentic work and 8192-16384
  usable tokens plus retrieval for knowledge work.

Success gates:

- 1024 TAC carry beats 256 truncation, reset, and shuffled-state controls.
- 4096-token agentic TAC improves long tool/repair/verification tasks over
  truncation.
- Decode throughput reaches at least `0.25x` vanilla for the first usable
  long-context prototype, then `0.5x` vanilla after KV/state-cache work.
- Functional program specialization survives long-context training.

## TAC Long-Context Local Automation Pass

Date: 2026-06-02

Ran the first local-only implementation and measurement pass for
`docs/tac_agentic_long_context_efficiency_research.md`. No Kaggle API commands
were run.

Implemented:

- tokenized/memmap file contract and batcher;
- RoPE scaling controls for TAC and vanilla attention;
- local long-context decomposition benchmark;
- local benchmark plumbing for RoPE/local-attention variants.

Artifacts:

- `runs/benchmarks/long_context_efficiency_local_2026_06_02/RESULTS.md`
- `runs/benchmarks/rope_scaling_local_2026_06_02`

Result:

- Byte-token memmap is functional and faster than online JSONL byte batching,
  but it only reached `2.38x` at seq_len 256 and `1.39x` at seq_len 1024, so
  the strict `>=5x` tokenized-batcher gate is not passed.
- Local attention masking reduces useful attention-edge proxy to `0.125` at
  seq_len 1024/window 128, but wall-clock speed does not improve because the
  current code still materializes dense attention logits.
- Tiny delayed-query segment carry remained positive: carry `0.0625` versus
  reset/shuffled/baseline `0.0` for both dense identity-first and local
  identity-first.
- RoPE `none`, `linear`, and `yarn-lite` controls all ran, but the 8-step CPU
  smoke was inconclusive and should not be used to pick a scaling mode.

Decision:

- Next work should prioritize true subword-token memmap, sampler overhead
  removal, and real sliding-window/block-sparse attention kernels before any
  4096-token training claim.

## TAC Long-Context Local Solution Candidate

Date: 2026-06-02

Continued the long-context work after the first local pass only identified
blockers. Implemented and tested three concrete fixes:

- compact causal sliding-window attention materializing `B * H * L * W` logits
  instead of dense `B * H * L * L` logits;
- optimized memmap sampling with preallocated NumPy arrays and CPU zero-copy
  `torch.from_numpy(...)`;
- serving-style inference controls:
  `collect_auxiliary=False` and one-token decode with
  `update_content_memory=False`.

Artifacts:

- `runs/benchmarks/long_context_efficiency_solution_attempt_2026_06_02/RESULTS.md`
- `runs/benchmarks/long_context_large_seq_inference_2026_06_02/RESULTS.md`
- `runs/benchmarks/long_context_decode_gate_fast_decode_2026_06_02/RESULTS.md`

Gate results:

- Optimized memmap batching passes the `>=5x` gate:
  - seq_len 256: `22.26x` faster than online JSONL byte batching.
  - seq_len 1024: `55.78x` faster.
- Compact local attention passes the memory-shape gate and becomes useful at
  4096-token inference:
  - `current_best_local` prefill `15,741.53 tok/s` vs vanilla `11,369.70 tok/s`.
  - `current_best_local` carried query `15,026.80 tok/s` vs vanilla
    `11,420.75 tok/s`.
- First local decode gate passes narrowly:
  - `current_best_local` decode `0.2519x` vanilla at seq_len 4096 with
    auxiliary diagnostics disabled and content-memory writes gated during
    one-token decode.

Working local candidate:

```text
4096-token TAC
+ optimized tokenized/memmap input path
+ RoPE scaling controls
+ compact causal local attention, window=128
+ content-addressed memory reads
+ collect_auxiliary=False for serving
+ update_content_memory=False during one-token decode
```

Decision:

- This is the first local solution candidate that clears the initial 4096
  inference efficiency gates.
- It is not yet a full long-context training/capability proof. Next validation
  must run multi-seed long segment-carry and long agentic trace tasks, then GPU
  serving profiling with the same inference switches.

## TAC Long-Context Full Matrix Check

Date: 2026-06-02

Ran the requested local full five-task matrix for the long-context solution
candidate without touching the active Kaggle job.

Artifact:

- `runs/benchmarks/long_context_solution_matrix_256_solution_k1_full_2026_06_02/RESULTS.md`

Matrix:

- seq_len `256`, compact local window `128`, five chunked-memory task families,
  seeds `11/23/37`, `120` optimizer steps, CPU batch size `4`.
- Tested three variants:
  - `solution_local_w128_k1`: exact fast-decode solution shortcut
    (`content_read_steps=1`, learned gate).
  - `solution_local_w128_synthesis`: local window with the stronger synthesis
    read gate.
  - `dense_current_best_synthesis`: dense current-best control.

Result:

- The exact fast-decode variant ranked first by mean carry, but it did not pass
  the strict all-effective capability gate:
  - `solution_local_w128_k1`: `12/15` effective, mean carry `0.18125`, carry
    minus reset `0.16875`, TAC-baseline gap `0.16458`.
  - `solution_local_w128_synthesis`: `10/15` effective, mean carry `0.16667`.
  - `dense_current_best_synthesis`: `10/15` effective, mean carry `0.16667`.
- Local attention is not the source of the capability failure:
  `solution_local_w128_synthesis` matched dense current-best carry exactly at
  this budget while improving train TPS ratio from `0.3937` to `0.7449` and
  query TPS ratio from `0.4217` to `1.0139`.
- The failure remains concentrated in `noisy_key` and `multi_hop`.

Decision:

- The local-window efficiency mechanism is validated at seq_len `256`, but the
  fast k1 serving shortcut is only a partial solution.
- Do not claim the long-context TAC capability gate is solved yet. The next
  candidate needs a multi-hop/noisy-key repair, likely semantic global tokens,
  retrieval-graph traversal, or a task/verifier-gated heavier read path.

## TAC Run 4 Semantic-MI Resume State

Date: 2026-06-02

Run 4 is still an active Kaggle workflow. Kernel
`mathewlincoln/tac-run-4-semantic-mi-20k` reached its second Kaggle time stop at
16,800/20,000 optimizer steps and was relaunched as kernel version 4 from the
16,800-step checkpoint.

Artifacts:

- Latest pulled output:
  `runs/kaggle_results/tac_run4_semantic_mi_20k_latest/best_tac_agentic_run4_semantic_mi`
- Resume dataset:
  `mathewlincoln/tac-run4-semantic-mi-resume-16800-2026-06-02`
- Kernel package:
  `runs/kaggle_run4_semantic_mi_2026_06_01`

Second leg result:

- Completed steps: `16800`
- Target steps: `20000`
- Stopped for time: `true`
- Latest leg elapsed seconds: `29400.539021799`
- Best eval loss: `6.421461343765259`
- Latest train loss: `6.4464826583862305`
- Tokens seen: `157510440`
- Tokens/sec: `5357.399736216198`
- Program memory cosine: `0.9640434384346008`
- Category route loss: `-0.008442441932857037`
- Weighted category route loss: `-0.004221220966428518`

Specialization state:

- End specialization was intentionally skipped because the run stopped for time
  before the 20k target.
- Step 10k checkpoint artifact exists and reports:
  - MI: `0.8085222685473687` bits
  - normalized MI: `0.3127791092991895`
  - program entropy: `3.274849878784676` bits

Remaining work:

- Train the final `3200` optimizer steps from 16,800 to 20,000.
- Write the 20k periodic specialization checkpoint.
- Run the final full specialization analysis on the completed checkpoint.
- Pull outputs, inspect `final_summary.json`, specialization artifacts, and log.
- Decide whether Run 4 validates semantic-MI routing as a training solution or
  only as an intermediate partial result.

### Interim Interpretation at 16,800 Steps

The 16,800-step state should be treated as a routing-objective diagnostic
partial success, not as a capability checkpoint.

Likely regression cause:

- `category_route_weight=0.5` is too high for the 30.6M-parameter model. It is
  acting like a co-primary objective beside next-token loss rather than a weak
  auxiliary regularizer.
- The parameter budget is skewed: about 20.2M / 30.6M parameters are in the
  identity/program field, leaving the byte-level language-modeling backbone
  comparatively small.
- The observed pattern matches this failure mode: eval loss remains near
  `6.42`, perplexity near `615-621`, and eval token accuracy near chance, while
  specialization MI reaches `0.8085` bits. The model appears to have learned
  category-conditioned routing better than it learned the language/task
  distribution.

What still survives:

- `base_semantic_mi_0p5` can produce visible category-conditioned routing.
- The objective is stable enough to train without divergence or obvious dead
  programs.
- It does not yet show that semantic routing improves capability.

Run 4 completion policy:

- Keep the active Run 4 job running to 20k because the checkpoint and final
  specialization/knockout artifacts are still diagnostically useful.
- After completion, explicitly run the harder-task carry/reset/shuffle
  checkpoint eval and a forced-program matrix on Run 4 best.pt, but expect
  weaker deltas than Run 3 because the base loss is near random.

Run 5 gate:

- Do not launch Run 5 until the baseline capability question is answered.
- First run a clean sanity baseline on the same corpus and comparable scale:
  no semantic MI objective, using BASE routing or no TAC.
- If that baseline also stays near perplexity `600+`, the issue is
  corpus/architecture scale mismatch.
- If the baseline learns normally, the current `category_route_weight=0.5` is
  confirmed as the main regression source.
- Candidate Run 5 direction: reduce identity/program parameter share toward
  roughly `40-50%`, use `category_route_weight≈0.05`, and schedule semantic MI
  as a weak delayed/decayed auxiliary objective rather than a co-primary loss.

## TAC Run 4 Final Result

Date: 2026-06-02

Kernel `mathewlincoln/tac-run-4-semantic-mi-20k` completed successfully after
the final resume from step 16,800.

Artifacts:

- Final pulled output:
  `runs/kaggle_results/tac_run4_semantic_mi_20k_latest/best_tac_agentic_run4_semantic_mi`
- Final summary:
  `runs/kaggle_results/tac_run4_semantic_mi_20k_latest/best_tac_agentic_run4_semantic_mi/final_summary.json`
- Final specialization:
  `runs/kaggle_results/tac_run4_semantic_mi_20k_latest/best_tac_agentic_run4_semantic_mi/specialization/program_specialization.json`
- 20k periodic specialization checkpoint:
  `runs/kaggle_results/tac_run4_semantic_mi_20k_latest/best_tac_agentic_run4_semantic_mi/specialization_checkpoints/step_020000/program_specialization.json`
- Log:
  `runs/kaggle_results/tac_run4_semantic_mi_20k_latest/tac-run-4-semantic-mi-20k.log`

Completion:

- Completed steps: `20000 / 20000`
- Stopped for time: `false`
- Wrapper return code: `0`
- Successful training tokens: `367200000`
- Final leg elapsed seconds: `14999.266624156999`

Final capability metrics:

- Best eval loss: `6.421461343765259`
- Last eval loss: `6.429963290691376`
- Last eval accuracy: `0.0022786458333333335`
- Last eval perplexity: `620.1510620117188`
- Final train loss: `6.486166000366211`
- Final next-token loss: `6.401911576588948`

Final routing/memory metrics:

- Program memory cosine: `0.9628605643908182`
- Content-addressed hit: `0.18461100260416666`
- Category route loss: `-0.006596214758853118`
- Weighted category route loss: `-0.003298107379426559`

Final specialization:

- Final full analysis records: `384`
- MI: `0.5330981670335928` bits
- Normalized MI: `0.20623052244853393`
- Program entropy: `3.2813949309672394` bits
- Token MI: `0.027051542952014687` bits
- Raw-token MI: `0.036279674725884906` bits
- Top knockout delta: P2 at `+0.003616290787855784`

Dominant program by category in the final full analysis:

- `argument_schema`: P0, `14 / 64`
- `memory_counterfactual`: P1, `14 / 64`
- `repair_after_failure`: P2, `15 / 64`
- `stale_memory_rejection`: P0, `25 / 64`
- `tool_choice`: P1, `12 / 64`
- `verification_planning`: P5, `18 / 64`

Verdict:

- Run 4 validates that the semantic-MI objective can produce observable
  category-conditioned routing.
- Run 4 does not validate semantic-MI routing as a capability-improving
  training solution at `category_route_weight=0.5`.
- Capability is near random for the 512-token vocabulary, and the program
  memories remain highly similar.
- The next training launch must be gated by a clean LM/base-routing sanity
  baseline and a lower, scheduled category-routing objective.

## USEF Transfer: Authority Reporting Contract

The useful USEF pattern to preserve is not the older model code; it is the
explicit separation between trusted authority, proposals, guesses, verifier
evidence, and cross-domain contamination.

Implemented local contract:

- `tac_transformer.authority.AuthorityEvent` records accepted/rejected
  authority claims with `domain`, `source_domain`, `authority_mode`,
  correctness, optional program ID, confidence, and metadata.
- `AuthorityReport` computes trusted accuracy, false trusted authority,
  rejected-event counts, proposal/guess counts, and cross-domain trusted-source
  violations.
- `VerifierCase` provides a simple expected-vs-observed executable-task bridge.
- `CurriculumReport` writes schema-versioned manifests plus JSONL case and
  authority-event artifacts.

Use this for future Run 4/Run 5 evidence gates when evaluating whether TAC is
learning domain-appropriate memory/execution trust rather than merely routing to
a category-conditioned program ID.

## USEF Transfer: Authority-Gated TAC Routing

The strongest architecture-level idea from USEF is the
`AuthorityGatedHyperRouter`: route selection should be separated from authority
mode, verifier need, and halt/escalation signals.

Local TAC implementation:

- Added opt-in `routing_type="authority_gated"` to `IdentityFieldLayer`.
- The route derives six USEF-style authority features from current program
  activations and persistent program stability: exact-memory evidence,
  proposal availability, calibrated confidence, contamination risk,
  verifier confidence, and memory distance.
- A small learned router predicts program scores, five authority modes
  (`exact_memory`, `proposal_verified`, `calibrated_fast_path`,
  `fresh_repair`, `system2_verify`), verifier-required flags, and halt
  probability.
- Program selection remains top-k and energy-budgeted, so existing TAC compute
  controls and sink-program behavior still apply.
- TAC auxiliary output now exposes authority logits/probabilities, authority
  indices, verifier-required flags, halt probability, token-level authority
  probabilities, and authority metrics for benchmark comparisons.

This is intentionally not promoted into `best_tac_config`. It should be treated
as the next benchmark candidate against `base`, targeted `base_semantic`, and
Run 4's semantic-MI routing objective.

2026-06-02 follow-up architecture review fixes:

- Added straight-through routing for authority-gated program selection. The
  forward pass still uses hard top-k, energy-budgeted routing, but gradients now
  flow through a masked soft route surrogate so the authority program-score head
  can learn from normal LM loss.
- Made BASE-semantic extra routing trainable through the same straight-through
  path, so `routing_load_balance` is no longer just a logged scalar when a
  learned semantic extra-choice surface is active.
- Added optional supervised losses for authority mode, verifier-required
  probability, and halt probability. This keeps the heuristic authority features
  useful as a bootstrap while giving verifier/curriculum artifacts a direct path
  to ground the authority semantics.
- Exported stable authority mode indices and added `authority_gated` to the
  harder research and specialization benchmark matrices, so the candidate can be
  tested by the existing promotion/rejection workflow.

## TAC-090 Run 5 Capability Sanity Gate

Date: 2026-06-02

The post-Run-4 research plan from `tac_unified_research_plan.html` makes
capability restoration the immediate gate. Run 4 reached visible routing
specialization but remained near-random as a byte-level language model, so the
next architecture-improvement step is not another semantic-routing launch. It
is a same-corpus sanity matrix.

Implemented local harness:

- `tac_transformer.capability.run_capability_sanity_matrix`
- `experiments/benchmark_capability_sanity.py`
- artifact writer for `capability_sanity_matrix.json` and `RESULTS.md`

Variants:

- `vanilla_10m_proxy`
- `tac_base_proxy`
- `tac_semantic_low_weight`

Gate logic:

- If vanilla does not learn, block architecture conclusions and fix data/scale.
- If vanilla learns but base TAC does not, investigate TAC optimization.
- If base TAC learns but low-weight semantic routing regresses capability,
  lower, delay, or reschedule the category-routing objective.
- Only pass Run 5 when all three variants improve loss under the same settings.

Smoke artifact:

- `runs/benchmarks/capability_sanity_smoke_2026_06_02`

Smoke verdict:

- `blocked`
- Reason: vanilla did not learn under the deliberately tiny two-step CPU smoke.

Interpretation:

The smoke is an automation validation, not scientific evidence. It proves the
gate and artifacts are wired. A meaningful local gate or Kaggle gate must use a
larger budget before Run 5 is allowed to launch.

Fuller local gate artifact:

- `runs/benchmarks/capability_sanity_local_2026_06_02_fuller`

Settings:

- seeds `11/23/37`
- 64 train records / 24 eval records
- 40 optimizer steps
- seq_len 64
- d_model 48, 2 layers, 8 TAC programs for local proxies

Result:

- `vanilla_10m_proxy`: loss improvement `1.6584`, final loss `4.7523`,
  accuracy `0.2131`, perplexity `117.54`
- `tac_base_proxy`: loss improvement `1.0659`, final loss `5.3365`,
  accuracy `0.1753`, perplexity `208.78`
- `tac_semantic_low_weight`: loss improvement `1.1352`, final loss `5.2704`,
  accuracy `0.1862`, perplexity `195.53`

Gate verdict:

- `pass`

Interpretation:

Low-weight semantic MI did not reproduce the Run 4 capability collapse in the
local sanity regime. It slightly improved over base TAC on loss and accuracy,
while vanilla remained a stronger pure language-model baseline. This justifies a
named Run 5 candidate, not a full promotion.

Implemented Run 5 candidate:

- `run5_capability_config`
- `run5_capability_training_kwargs`
- `python kaggle/train_best_tac_agentic.py --preset run5_capability`

Candidate architecture/training deltas:

- `routing_type="base_semantic"`
- `routing_top_k=2`
- `routing_load_balance_weight=0.05`
- `category_route_weight=0.05`
- `category_route_objective="mi"`
- `warmup_steps=2000`
- `n_programs=12`

The `n_programs=12` setting keeps the identity-field parameter share under 50%
at the base training width, satisfying the post-Run-4 roadmap gate.

## TAC-091 Run 5 Pathfinder Sweep

Date: 2026-06-02

The TAC-090 sanity gate was not thorough enough to determine the right path.
It only compared three candidates and did not jointly measure parameter share,
specialization, routing alternatives, and semantic weight. TAC-091 adds the
broader pathfinder required before trusting a Run 5 implementation.

Implemented:

- `build_run5_pathfinder_variants`
- `run_run5_pathfinder_matrix`
- `aggregate_run5_pathfinder_results`
- `category_program_mi_bits_from_probs`
- `experiments/benchmark_run5_pathfinder.py`

First pathfinder artifact:

- `runs/benchmarks/run5_pathfinder_local_2026_06_02`

This artifact is invalid for identity-share conclusions. The grid generated
different `n_programs` settings, but the runner used the CLI default for every
candidate. The bug is now covered by:

- `test_pathfinder_runner_uses_candidate_program_count`

Corrected pathfinder artifact:

- `runs/benchmarks/run5_pathfinder_local_2026_06_02_fixed`

Matrix:

- program counts: `8/12/16/24`
- semantic weights: `0.0/0.01/0.05/0.1/0.2`
- authority-gated ablations included
- vanilla 10M/30M proxies included
- seeds: `11/23`
- local budget: 30 steps, seq_len 64, d_model 48, 2 layers

Corrected top candidates:

- `tac_semantic_w0p1_p12`: final loss `5.4395`, accuracy `0.1439`,
  selected MI `0.0199`, activation MI `0.0029`, identity share `0.360`,
  train TPS `1831.1`
- `tac_semantic_w0p05_p12`: final loss `5.4393`, accuracy `0.1439`,
  selected MI `0.0202`, activation MI `0.0028`, identity share `0.360`,
  train TPS `1736.3`
- `tac_semantic_w0p01_p12`: final loss `5.4392`, accuracy `0.1439`,
  selected MI `0.0207`, activation MI `0.0028`, identity share `0.360`,
  train TPS `1619.0`

Rejected:

- `tac_authority_p24`, because identity share `0.507` exceeds the 50% gate.

Identity-share sanity after the fix:

- `tac_base_p8`: `0.304`
- `tac_base_p12`: `0.360`
- `tac_base_p16`: `0.409`
- `tac_base_p24`: `0.486`

Decision:

The right implementation path is p12 `base_semantic`, not the earlier
unsupported p12 `0.05` hand-pick and not authority-gated routing. The best
corrected pathfinder row is `tac_semantic_w0p1_p12`. The top p12 semantic
weights are close, so this is a candidate for a serious Run 5 validation, not a
final scientific promotion.

Implementation update:

- `run5_capability_config` remains p12 `base_semantic`, top-k 2,
  `routing_load_balance_weight=0.05`
- `run5_capability_training_kwargs` now uses `category_route_weight=0.1`,
  `category_route_objective="mi"`, and `warmup_steps=2000`
- `kaggle/train_best_tac_agentic.py --preset run5_capability` now applies
  those training defaults automatically unless the user explicitly overrides
  them

Next proof required:

Run the p12/w0.1 candidate at real scale with periodic specialization,
forced-program matrix, and capability eval. The selected-route MI is non-zero
locally, but activation MI remains small, so functional specialization is still
not proven.

## TAC-092 Run 5 Real-Scale Capability Validation

Date: 2026-06-02

Run 5 has moved from local pathfinder candidate to real Kaggle validation.

Preflight:

- Local trainer smoke used `--preset run5_capability` and confirmed the actual
  run manifest applies p12 `base_semantic`, top-k 2, `category_route_weight=0.1`,
  MI objective, `warmup_steps=2000`, and specialization checkpoint wiring.
- Added `test_agentic_training_bundle_imports_after_extraction` after finding
  the rebuilt code bundle would otherwise fail exactly like the earlier Run 4
  bundle omission. The first red failure was missing `tac_transformer.authority`.
- Fixed the bundle manifest to include `tac_transformer/authority.py`,
  `tac_transformer/capability.py`, capability/pathfinder docs, and the two
  Run 5 research scripts.
- Rebuilt `runs/kaggle_run5_code_2026_06_02/best-tac-agentic-training-bundle.zip`
  and verified the zip contains the required modules.

Kaggle launch:

- Code dataset: `mathewlincoln/tac-run5-capability-code-2026-06-02`
- Data dataset: `mathewlincoln/tac-run4-semantic-mi-data-2026-06-01`
- Kernel: `mathewlincoln/tac-run-5-capability-p12-w0-1-20k`
- URL: <https://www.kaggle.com/code/mathewlincoln/tac-run-5-capability-p12-w0-1-20k>
- Initial status: `RUNNING`
- Heartbeat automation: `monitor-tac-run-5`, every 30 minutes

Run 5 is not a promotion yet. The completion gate is capability recovery first:
language-model eval loss and accuracy must materially improve over Run 4's
near-random result. Functional specialization is a second gate: periodic and
final specialization artifacts must show non-trivial category-program structure
with useful knockout deltas, not only MI from the auxiliary routing objective.

## Research Directions & Experimental Priorities After Run 5

Date: 2026-06-02

This section is a roadmap, not an immediate architecture decision. I have not
yet done a dedicated external literature pass on these tracks. They are
internal research priorities inferred from Run 4/Run 5 evidence, prior TAC
benchmarks, and the unresolved gap between routing specialization and language
model capability.

The useful reframing is:

> TAC is attempting to learn computational organization, not merely statistical
> prediction.

That distinction matters if routing specialization continues to emerge while
capability does not.

### 12. Is Next-Token Prediction Sufficient for TAC?

Current position:

Run 4 is not sufficient evidence that next-token prediction is fundamentally
incompatible with TAC. The test was not clean:

- the language-model backbone was capacity-constrained at roughly 10M
  parameters;
- the identity subsystem consumed roughly 66% of total parameters;
- `category_route_weight=0.5` made routing pressure too strong;
- optimization pressure favored routing specialization over language modeling.

Under those conditions, failure cannot be uniquely attributed to NTP.

Working hypothesis:

| Cause | Estimated Probability |
| --- | ---: |
| Identity field consumed too much capacity | 45% |
| Routing objective too strong | 30% |
| NTP fundamentally mismatched to TAC | 15% |
| Other causes | 10% |

Therefore, TAC has not yet received a fair NTP test. Run 5 should finish before
major objective redesign is considered.

Escalation criterion:

Alternative training objectives become a primary research track only if all of
the following occur:

1. Vanilla baselines learn successfully.
2. Rebalanced TAC learns routing specialization.
3. Routing weight is reduced to regularization levels.
4. Useful program differentiation still fails to emerge.
5. Capability remains inferior to comparable baselines.

If those conditions hold, the question shifts from "Is TAC implemented
correctly?" to "Does NTP provide sufficient pressure for computational
specialization at all?"

### 13. Alternative Objective Research Track (Deferred)

These are exploratory and should not replace Run 5.

Latent-state prediction:

- Train current state to predict a future representation rather than only the
  next token.
- Potential benefit: rewards state transitions and may align better with
  program execution.
- Research question: do TAC programs represent state transitions more naturally
  than token distributions?

Predictive coding objectives:

- Train state -> prediction -> error -> update loops.
- Programs become responsible for modeling deviations from expectation.
- Potential benefit: closer to persistent-state architectures and
  coherence-based processing.

Program contrastive objectives:

- Run 4 showed high program-memory cosine, so direct differentiation pressure is
  plausible.
- The risk is arbitrary difference. The objective must reward useful
  difference, not merely decorrelated programs.

Route-and-reconstruct:

- Require routing decisions to affect future reconstruction quality.
- Correct routes should improve reconstruction; incorrect routes should worsen
  it.
- This pressures routes to correspond to meaningful computation rather than
  category labels.

Computation prediction:

- Predict next thought state, plan state, or semantic state rather than next
  token alone.
- Speculative, but aligned with TAC's long-term goal of learning latent
  computational transitions.

### 14. Neuroscience-Inspired Efficiency Roadmap

Run 4 also exposed an efficiency problem independent of capability:

> TAC pays the cost of maintaining persistent state too frequently.

Current decode inefficiency may be less about persistent memory itself and more
about update frequency. The efficiency roadmap should investigate sparse,
event-driven, hierarchical updates.

Incremental state evolution:

- Current TAC often recomputes routing, coherence, and memory update surfaces at
  every token.
- Research direction: update only state variables that change significantly.
- Targets: coherence matrices, routing distributions, program summaries.

Sparse program activation:

- Activate few programs and leave most dormant.
- Goal: increase effective capacity without proportional compute growth.
- This connects to MoE, cortical sparsity, and conditional computation.

Hierarchical update schedules:

| State | Candidate Update Frequency |
| --- | --- |
| Attention | every token |
| Routing | every few tokens |
| Working memory | event-driven |
| Identity field | sentence-level |
| Long-term memory | rare |

Research question: how much routing quality is lost if routing is updated every
N tokens instead of every token?

Event-driven retrieval:

- Retrieve when a cue appears rather than at every token.
- Potential benefits: lower decode cost, reduced memory noise, and clearer
  retrieval interpretation.

Prediction-error updates:

- Track delta coherence rather than recomputing full coherence when values are
  stable.
- Potential benefit: meaningful decode savings if coherence changes slowly.

Macro-program formation:

- Frequently co-occurring program sequences could be compiled into macro
  programs.
- Potential benefits: reduced routing overhead, faster inference, and emergent
  procedural abstraction.
- This resembles skill compilation in cognitive science and expert human
  performance.

Updated long-term research thesis:

- Capability question: can meaningful computational specialization emerge under
  NTP? Run 5 owns this.
- Efficiency question: can persistent-state architectures avoid
  transformer-style recomputation through sparse, event-driven, hierarchical
  updates? This becomes the central TAC efficiency research program after
  capability is restored.

These tracks are now partially decoupled. Run 5 should focus on capability. The
neuroscience-inspired roadmap should be investigated after the capability path
has clean evidence.

## TAC-093 TAC-Aware Optimizer

Date: 2026-06-02

The architecture previously used normal AdamW directly in each training loop.
TAC now has a shared optimizer factory that keeps AdamW's checkpoint and AMP/DDP
compatibility while making the optimization surface TAC-specific.

Implemented:

- `tac_transformer.optimization.TACOptimizerConfig`
- `tac_transformer.optimization.tac_optimizer_param_groups`
- `tac_transformer.optimization.build_tac_optimizer`
- package-root exports from `tac_transformer`
- documentation in `docs/tac_optimizer.md`

Grouping contract:

- `core`
- `identity`
- `router`
- `memory`
- `head`

Each category is split into decay and no-decay subgroups. The group metadata
records `tac_group`, `tac_category`, `tac_lr_mult`, `base_lr`, and
`tac_param_names`, so schedules and diagnostics can reason about the optimizer.

Integration coverage:

- synthetic training
- language-model training
- chunked-memory training
- capability sanity/pathfinder research
- agentic controller training
- recurrent agentic baseline training
- Kaggle Run trainer
- program-specialization objective benchmark
- Kaggle memory profiler
- long-context efficiency timing experiment

Decision:

Keep default LR multipliers at `1.0` for all groups so historical comparisons
remain fair. Use explicit per-category multipliers in future optimizer sweeps
instead of silently changing the active architecture evidence trail.

## TAC-095 Local Objective And Efficiency Matrix

Date: 2026-06-02

Purpose:

Run a local, bounded research matrix while Kaggle Run 5 continues. This is not a
replacement for the real-scale Run 5 capability validation. It is a fast probe
to test whether the deferred objective ideas show enough signal to become the
next architecture direction, and whether the neuroscience-inspired efficiency
roadmap has concrete local targets.

Artifacts:

- JSON: `runs/benchmarks/tac_research_directions_local_2026_06_02/tac_research_directions_matrix.json`
- Markdown: `runs/benchmarks/tac_research_directions_local_2026_06_02/RESULTS.md`
- Smoke: `runs/benchmarks/tac_research_directions_smoke_2026_06_02`

Local matrix settings:

- corpus: hard agentic prepared corpus
- model: `d_model=48`, `n_layers=2`, `n_heads=4`, `n_programs=12`
- training: 40 steps, `seq_len=64`, batch size 4
- seeds: 11, 23, 37
- evaluation: 3 eval batches per run, CPU, 4 torch threads
- objective variants: NTP reference, Run 5 regularized MI, latent-state
  prediction, predictive coding, program contrastive, route-and-reconstruct,
  computation prediction, and combined light objective
- efficiency modes: full update, serving no-aux, no content updates,
  content-every-4, content-every-8, and event-error update

Objective results:

| Variant | Loss Improvement | Final Loss | Accuracy | Selected MI | Activation MI | Program Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| program_contrastive | 0.9404 | 5.4708 | 0.1042 | 0.0245 | 0.0022 | 0.9167 |
| combined_light | 0.9401 | 5.4711 | 0.1037 | 0.0246 | 0.0021 | 0.9167 |
| run5_regularized_mi | 0.9398 | 5.4714 | 0.1037 | 0.0237 | 0.0020 | 0.9167 |
| route_reconstruct | 0.9388 | 5.4724 | 0.1033 | 0.0252 | 0.0019 | 0.9167 |
| latent_state | 0.9385 | 5.4727 | 0.1033 | 0.0235 | 0.0019 | 0.9167 |
| ntp_reference | 0.9384 | 5.4728 | 0.1033 | 0.0234 | 0.0019 | 0.9167 |
| predictive_coding | 0.9351 | 5.4760 | 0.1033 | 0.0237 | 0.0019 | 0.9167 |
| computation_prediction | 0.7294 | 5.6818 | 0.0738 | 0.0208 | 0.0019 | 0.9167 |

Objective verdict:

The local winner is `program_contrastive`, but the margin over
`combined_light`, `run5_regularized_mi`, and plain NTP is too small to justify
an immediate objective redesign. This result does not show that NTP is
sufficient, but it also does not show that NTP is the blocker. The Run 5 path
remains the right active test.

The most important negative signal is that program-memory cosine remains high at
roughly `0.9167` across all variants, and activation MI remains tiny at roughly
`0.002` bits. The local objective probes do not yet solve useful program
differentiation. `route_reconstruct` slightly improves selected-route MI, but
not capability. `computation_prediction` is the only clear local regression and
should not be promoted without redesign.

Efficiency results:

| Mode | Loss | Accuracy | Tokens/s | Speedup | Update Fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| event_error_update | 3.4183 | 0.1069 | 3998.7 | 1.233x | 0.999 |
| no_content_updates | 3.4183 | 0.1069 | 3983.4 | 1.228x | 0.000 |
| content_every_8 | 3.4183 | 0.1069 | 3657.7 | 1.127x | 0.562 |
| content_every_4 | 3.4183 | 0.1069 | 3527.8 | 1.087x | 0.625 |
| serving_no_aux | 3.4183 | 0.1069 | 3450.8 | 1.064x | 1.000 |
| full_update | 3.4183 | 0.1069 | 3244.3 | 1.000x | 1.000 |

Efficiency verdict:

The efficiency probes are more actionable than the objective probes. Disabling
content-memory updates or updating them periodically produced the same local
loss and accuracy in this benchmark while improving the sequence/decode proxy
throughput. This makes update frequency a high-priority efficiency experiment.

The event-error update mode was not sparse at the tested threshold because it
updated on roughly `99.9%` of positions. That does not invalidate the idea; it
means the threshold and gating signal need a dedicated sweep.

Macro-program probe:

The repeated route-sequence probe found frequent two-program repetitions with
top-sequence fractions around `0.15` to `0.21`, implying a theoretical local
compression upper bound of roughly `0.15` to `0.21` in the sampled sequences.
The current top sequences are mostly repeated single program IDs, so this is a
measurement hook rather than evidence of mature procedural abstraction.

Decision:

Keep Run 5 as the capability gate. Do not integrate alternative objectives as
the main architecture yet. If Run 5 fails after the documented escalation
criteria are satisfied, promote a longer objective matrix focused on
`program_contrastive`, `combined_light`, and a redesigned `route_reconstruct`
objective. Independently, start an efficiency track around true content-update
scheduling, event-error threshold sweeps, and routing/update skip mechanics once
the Run 5 capability result is known.

## TAC-096 Content-Memory Update-Frequency Isolation

Date: 2026-06-02

Purpose:

Isolate whether TAC can reduce content-memory update cost without losing the
memory behavior that made content-addressed reads useful. This study separates
two different questions that TAC-095 bundled together:

- query/decode-phase upkeep: should TAC reconsolidate or write content memory
  while already using a carried context state?
- context-phase writes: how sparse can content writes be while still preserving
  recall?

Artifacts:

- Corrected matrix JSON: `runs/benchmarks/content_update_frequency_local_adapter_2026_06_02/content_update_frequency_matrix.json`
- Corrected matrix Markdown: `runs/benchmarks/content_update_frequency_local_adapter_2026_06_02/RESULTS.md`
- Smoke: `runs/benchmarks/content_update_frequency_smoke_2026_06_02`

Invalid artifact warning:

The earlier broad artifact
`runs/benchmarks/content_update_frequency_local_2026_06_02` should not be used
for capability conclusions. It evaluated raw query logits after training the
promoted memory adapter. The corrected artifact above evaluates through the same
`memory_adapted_logits` path used by `train_chunked_memory`.

Local matrix settings:

- tasks: single-key, multi-key, delayed-query, noisy-key, and multi-hop chunked
  recall
- seeds: 11, 23, 37
- model: `d_model=48`, `n_layers=2`, `n_heads=4`, `n_programs=12`
- training: 120 steps per task/seed, batch size 8
- evaluation: 6 batches, eval batch size 8, `seq_len=32`, CPU, 4 torch threads
- schedules: full update, query skip, no content updates, segmented context
  every 1/2/4/8 segments, and segmented context never

Schedule results:

| Schedule | Phase | Carry | Delta vs Full | Query TPS Ratio | Context Update | Query Update | Content Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| query_skip | full_window | 0.1778 | 0.0000 | 1.083 | 1.000 | 0.000 | 0.1523 |
| segment_every_8 | segmented_context | 0.2014 | 0.0236 | 1.192 | 0.125 | 0.000 | 0.0869 |
| full_update | full_window | 0.1778 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.1523 |
| segment_every_4 | segmented_context | 0.1347 | -0.0431 | 1.187 | 0.250 | 0.000 | 0.1324 |
| segment_every_2 | segmented_context | 0.0125 | -0.1653 | 1.150 | 0.500 | 0.000 | 0.1526 |
| segment_every_1 | segmented_context | 0.0236 | -0.1542 | 1.122 | 1.000 | 0.000 | 0.1538 |
| segment_never | segmented_context | 0.0153 | -0.1625 | 1.398 | 0.000 | 0.000 | 0.0000 |
| no_content_updates | full_window | 0.0153 | -0.1625 | 1.346 | 0.000 | 0.000 | 0.0000 |

Task-level split:

| Task | Full Carry | Query-Skip Carry | Segment-8 Carry | No-Update Carry |
| --- | ---: | ---: | ---: | ---: |
| single_key | 0.2431 | 0.2431 | 0.4028 | 0.0069 |
| multi_key | 0.3194 | 0.3194 | 0.0764 | 0.0278 |
| delayed_query | 0.2639 | 0.2639 | 0.4236 | 0.0139 |
| noisy_key | 0.0347 | 0.0347 | 0.0903 | 0.0069 |
| multi_hop | 0.0278 | 0.0278 | 0.0139 | 0.0208 |

Verdict:

Query/decode-phase content updates are the safe immediate target. `query_skip`
matches `full_update` exactly on mean carry accuracy (`0.1778`), carry-reset
delta (`0.1667`), carry-shuffled delta (`0.1694`), and content hit (`0.1523`),
while improving query throughput by roughly `8.3%`. This supports making
serving/decode paths skip content reconsolidation and content writes once a
context state already exists.

Context-phase sparsity is promising but not safe as a fixed global interval.
`segment_every_8` has the best aggregate carry (`0.2014`) and a `19.2%` query
throughput ratio improvement, but it wins by helping single-key, delayed-query,
and noisy-key while badly hurting multi-key. That means the next architecture
experiment should not be "update every N tokens" as a global rule. It should be
event-driven or route-aware:

- write when a cue/value boundary or retrieval-relevant event is detected;
- skip repeated low-information updates after the content store is populated;
- avoid overwriting multi-key contexts before all relevant pairs are captured;
- expose a content-write gate metric separate from query-phase reconsolidation.

Decision:

Promote `query_skip` as the immediate serving/decode policy and keep full
context writes as the training/default capability path. Open the next efficiency
research gate around event-driven context writes, with `segment_every_8` treated
as a useful positive control rather than a default architecture.

## TAC-097 Prediction-Error Content-Write Threshold Sweep

Date: 2026-06-02

Purpose:

Test the pasted A1 research-plan question directly: can prediction-error
content writes reduce context writes by at least 50% while preserving memory
behavior and context loss? This extends TAC-096 by sweeping several
cross-entropy thresholds instead of evaluating one event-error setting.

Artifacts:

- Corrected threshold matrix JSON: `runs/benchmarks/content_event_error_threshold_local_seg2_2026_06_02/content_update_frequency_matrix.json`
- Corrected threshold matrix Markdown: `runs/benchmarks/content_event_error_threshold_local_seg2_2026_06_02/RESULTS.md`
- Smoke: `runs/benchmarks/content_event_error_smoke_2026_06_02`

Invalid artifact warning:

The folder `runs/benchmarks/content_event_error_threshold_local_2026_06_02`
was stopped and should not be used for conclusions. It used
`context_segment_len=1`, but the current content store writes cue/value pairs
and requires at least two tokens in a context chunk. That run therefore tested
an invalid interface rather than valid prediction-error writes.

Local matrix settings:

- tasks: single-key, multi-key, delayed-query, noisy-key, and multi-hop chunked
  recall
- seeds: 11, 23, 37
- model: `d_model=48`, `n_layers=2`, `n_heads=4`, `n_programs=12`
- training: 120 steps per task/seed, batch size 8
- evaluation: 6 batches, eval batch size 8, `seq_len=32`, CPU, 4 torch threads
- schedules: full update, query skip, segment-every-8 positive control, and
  event-error thresholds `1.5`, `2.0`, `2.5`, `3.0`, `3.5`, `4.0`, `4.5`,
  `5.0`, and `6.0`
- corrected event proxy: two-token segmented context, because one-token chunks
  cannot write cue/value content pairs in the current model

Aggregate results:

| Schedule | Phase | Carry | Delta vs Full | Query TPS Ratio | Context Update | Query Update | Content Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| query_skip | full_window | 0.1778 | 0.0000 | 1.067 | 1.000 | 0.000 | 0.1523 |
| full_update | full_window | 0.1778 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.1523 |
| event_error_ge_1p5 | event_error_context | 0.0208 | -0.1569 | 1.068 | 1.000 | 0.000 | 0.1538 |
| event_error_ge_4p0 | event_error_context | 0.0222 | -0.1556 | 1.062 | 0.985 | 0.000 | 0.1539 |
| segment_every_8 | segmented_context | 0.0111 | -0.1667 | 1.050 | 0.125 | 0.000 | 0.0599 |
| event_error_ge_4p5 | event_error_context | 0.0069 | -0.1708 | 1.039 | 0.116 | 0.000 | 0.0498 |
| event_error_ge_5p0 | event_error_context | 0.0083 | -0.1694 | 1.009 | 0.062 | 0.000 | 0.0126 |
| event_error_ge_6p0 | event_error_context | 0.0083 | -0.1694 | 1.012 | 0.062 | 0.000 | 0.0126 |

Prediction-error gate:

Blocked for the current segmented-context prototype. No threshold achieved the
target of at least 50% write reduction with at most 1% context-loss degradation
and at most 1 percentage point carry degradation. The best sparse threshold was
`event_error_ge_4p5`, which reduced context updates to `0.116` and preserved
context-loss ratio at `1.0011`, but carry accuracy collapsed by `0.1708`
absolute versus full update. Dense low thresholds preserved content-hit but
wrote nearly all segments and still lost carry accuracy because the segmented
prefill interface itself damages the carried-memory behavior.

Decision:

Promote `query_skip` as the only immediate A2 serving/decode policy. Do not
promote segmented prediction-error context writes. The next implementation-ready
research step is to add an in-forward `content_write_mask` / `write_policy`
path that keeps the context in one prefill forward pass while masking
content-store writes per cue/value pair. Only after that exists should A1 be
rerun against the original success criterion.

## TAC-098 In-Forward Prediction-Error Write Mask

Date: 2026-06-02

Purpose:

Close the main limitation from TAC-097 by testing prediction-error writes inside
a full prefill forward instead of splitting context into small segments. This
keeps attention, identity routing, and hidden-state computation on the normal
full-window path while masking only which cue/value pairs are written into the
content store.

Implemented:

- `TACTransformerLM.forward(..., content_write_mask=...)`
- `TACTransformerBlock.forward(..., content_write_mask=...)`
- `IdentityField.forward(..., content_write_mask=...)`
- masked `_update_content_store` writes that preserve per-batch selected
  cue/value pairs without rolling the store for skipped pairs
- benchmark schedules `event_mask_ge_1p5` through `event_mask_ge_6p0`
- `prediction_error_content_write_mask`, which builds a full-context
  cross-entropy write mask and always seeds the first cue/value pair

Artifacts:

- Full masked matrix JSON: `runs/benchmarks/content_event_mask_threshold_local_2026_06_02/content_update_frequency_matrix.json`
- Full masked matrix Markdown: `runs/benchmarks/content_event_mask_threshold_local_2026_06_02/RESULTS.md`
- Smoke: `runs/benchmarks/content_event_mask_smoke_2026_06_02`

Local matrix settings:

- tasks: single-key, multi-key, delayed-query, noisy-key, and multi-hop chunked
  recall
- seeds: 11, 23, 37
- model: `d_model=48`, `n_layers=2`, `n_heads=4`, `n_programs=12`
- training: 120 steps per task/seed, batch size 8
- evaluation: 6 batches, eval batch size 8, `seq_len=32`, CPU, 4 torch threads
- schedules: full update, query skip, and full-context prediction-error masks
  at thresholds `1.5`, `2.0`, `2.5`, `3.0`, `3.5`, `4.0`, `4.5`, `5.0`,
  and `6.0`

Aggregate results:

| Schedule | Phase | Carry | Delta vs Full | Query TPS Ratio | Context Update | Query Update | Content Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| query_skip | full_window | 0.1778 | 0.0000 | 1.012 | 1.000 | 0.000 | 0.1523 |
| event_mask_ge_2p5 | masked_full_context | 0.1778 | 0.0000 | 1.195 | 1.000 | 0.000 | 0.1523 |
| event_mask_ge_3p0 | masked_full_context | 0.1736 | -0.0042 | 1.102 | 0.996 | 0.000 | 0.1521 |
| full_update | full_window | 0.1778 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.1523 |
| event_mask_ge_3p5 | masked_full_context | 0.1528 | -0.0250 | 1.083 | 0.936 | 0.000 | 0.1524 |
| event_mask_ge_4p0 | masked_full_context | 0.1069 | -0.0708 | 1.126 | 0.714 | 0.000 | 0.1502 |
| event_mask_ge_4p5 | masked_full_context | 0.0528 | -0.1250 | 1.143 | 0.353 | 0.000 | 0.1469 |
| event_mask_ge_5p0 | masked_full_context | 0.0208 | -0.1569 | 1.201 | 0.114 | 0.000 | 0.0864 |
| event_mask_ge_6p0 | masked_full_context | 0.0069 | -0.1708 | 1.135 | 0.033 | 0.000 | 0.0135 |

Verdict:

The in-forward mask fixes the invalid segmented-context interface, but raw
prediction-error thresholding still does not satisfy A1. The best
quality-preserving threshold is `event_mask_ge_3p0`, with carry delta
`-0.0042`, context-loss ratio `1.0000`, and content hit `0.1521`, but it writes
`0.996` of cue/value pairs and therefore fails the required 50% write
reduction. Thresholds that do reduce writes by at least 50% lose too much carry:
`event_mask_ge_4p5` writes `0.353` of pairs but drops carry by `0.1250`, while
`event_mask_ge_5p0` writes `0.114` of pairs but drops carry by `0.1569`.

Decision:

Keep the `content_write_mask` API because it is the correct implementation
surface for future sparse-write research. Do not promote raw CE thresholding as
the balanced solution. The next A1 candidate should combine prediction error
with retrieval structure, for example a learned or heuristic gate over
prediction error, content-hit novelty, cue/value boundary confidence, and
multi-key overwrite risk. `query_skip` remains the only immediate serving/decode
promotion.

## TAC-099 Retrieval-Aware Content-Write Mask

Date: 2026-06-03

Purpose:

Continue the A1 research plan after TAC-098 by testing whether a richer
heuristic gate can find a better balance than raw prediction-error thresholding.
The candidate keeps the corrected full-prefill `content_write_mask` surface and
scores each cue/value pair with prediction loss, value loss, and hidden-state
novelty before writing the top fraction of pairs.

Implemented:

- retrieval-aware full-context write-mask schedules: `retrieval_mask_top_25`,
  `retrieval_mask_top_50`, and `retrieval_mask_top_75`
- `retrieval_aware_content_write_mask`, combining row-normalized cue loss,
  value loss, and prior-hidden novelty
- focused tests for top-fraction selection and invalid fraction rejection
- a stricter aggregate A1 decision gate that also requires preserving carried
  state advantage, preventing degenerate low-carry runs from passing

Artifacts:

- Full retrieval-aware matrix JSON: `runs/benchmarks/content_retrieval_mask_local_2026_06_03/content_update_frequency_matrix.json`
- Full retrieval-aware matrix Markdown: `runs/benchmarks/content_retrieval_mask_local_2026_06_03/RESULTS.md`
- Smoke: `runs/benchmarks/content_retrieval_mask_smoke_2026_06_03`

Local matrix settings:

- tasks: single-key, multi-key, delayed-query, noisy-key, and multi-hop chunked
  recall
- seeds: 11, 23, 37
- model: `d_model=48`, `n_layers=2`, `n_heads=4`, `n_programs=12`
- training: 120 steps per task/seed, batch size 8
- evaluation: 6 batches, eval batch size 8, `seq_len=32`, CPU, 4 torch threads
- schedules: full update, query skip, `event_mask_ge_3p0`, and retrieval-aware
  top-k masks at 25%, 50%, and 75%

Aggregate results:

| Schedule | Phase | Carry | Delta vs Full | Query TPS Ratio | Context Update | Query Update | Content Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| query_skip | full_window | 0.1778 | 0.0000 | 1.082 | 1.000 | 0.000 | 0.1523 |
| event_mask_ge_3p0 | masked_full_context | 0.1736 | -0.0042 | 1.074 | 0.996 | 0.000 | 0.1521 |
| full_update | full_window | 0.1778 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.1523 |
| retrieval_mask_top_75 | retrieval_masked_full_context | 0.1556 | -0.0222 | 1.157 | 0.743 | 0.000 | 0.1544 |
| retrieval_mask_top_50 | retrieval_masked_full_context | 0.1417 | -0.0361 | 1.144 | 0.523 | 0.000 | 0.1539 |
| retrieval_mask_top_25 | retrieval_masked_full_context | 0.0778 | -0.1000 | 1.107 | 0.275 | 0.000 | 0.1519 |

Verdict:

Retrieval-aware top-k scoring is a better sparse-write direction than raw CE
thresholding, but it still does not satisfy A1. `retrieval_mask_top_50`
preserves content hit and is much stronger than similarly sparse raw CE masks,
but it writes `0.523` of pairs and drops carry by `0.0361`. `retrieval_mask_top_75`
keeps more carry but writes `0.743` of pairs, while `retrieval_mask_top_25`
achieves strong sparsity but loses `0.1000` carry accuracy. The best
quality-preserving row remains `event_mask_ge_3p0`, which writes almost every
pair and therefore fails the 50% write-reduction requirement.

Decision:

Keep `content_write_mask` and the retrieval-aware heuristic as useful evidence,
not as the final balanced solution. Fixed top-k retrieval scoring improves the
tradeoff but is still too blunt for multi-key overwrite risk. The next A1
candidate should be a learned or calibrated write gate trained against
carried-state preservation, cue/value boundary usefulness, and overwrite risk,
rather than a global threshold or fixed write fraction. `query_skip` remains
the only immediate A2 serving/decode promotion.

## TAC-100 Structural Boundary Sparse-Write Oracle

Date: 2026-06-03

Purpose:

Test whether A1 is blocked because sparse writes are inherently harmful, or
because the previous gates chose the wrong writes. This run adds a deliberately
task-structural oracle mask for the local chunked-recall family: write the early
odd-position cue/value boundary pairs and skip filler-context overwrites.

Implemented:

- `structural_pair_top_4` benchmark schedule on the full-prefill
  `content_write_mask` path
- `structural_pair_content_write_mask`, which writes up to four early
  odd-position cue/value pairs
- focused tests for the mask contract and sparse full-context scheduling
- sparse context-write gate reporting that covers prediction-error,
  retrieval-aware, and structural masks under the same A1 criteria

Artifacts:

- Structural matrix JSON: `runs/benchmarks/content_structural_pair_local_2026_06_03/content_update_frequency_matrix.json`
- Structural matrix Markdown: `runs/benchmarks/content_structural_pair_local_2026_06_03/RESULTS.md`

Aggregate results:

| Schedule | Phase | Carry | Delta vs Full | Query TPS Ratio | Context Update | Query Update | Content Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| query_skip | full_window | 0.1778 | 0.0000 | 1.091 | 1.000 | 0.000 | 0.1523 |
| event_mask_ge_3p0 | masked_full_context | 0.1736 | -0.0042 | 1.145 | 0.996 | 0.000 | 0.1521 |
| structural_pair_top_4 | structural_pair_full_context | 0.2500 | 0.0722 | 1.082 | 0.129 | 0.000 | 0.1172 |
| full_update | full_window | 0.1778 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.1523 |
| retrieval_mask_top_50 | retrieval_masked_full_context | 0.1417 | -0.0361 | 1.101 | 0.523 | 0.000 | 0.1539 |

Task-level structural result:

| Task | Full Carry | Structural Carry | Structural Delta | Structural Update |
| --- | ---: | ---: | ---: | ---: |
| single_key | 0.2431 | 0.3472 | 0.1042 | 0.129 |
| multi_key | 0.3194 | 0.4236 | 0.1042 | 0.129 |
| delayed_query | 0.2639 | 0.3611 | 0.0972 | 0.129 |
| noisy_key | 0.0347 | 0.0833 | 0.0486 | 0.129 |
| multi_hop | 0.0278 | 0.0347 | 0.0069 | 0.129 |

Verdict:

This is the first local A1 pass. `structural_pair_top_4` reduces context writes
by about `87.1%`, keeps context loss unchanged, improves carry by `0.0722`
absolute over full update, and improves carried-state advantage. The lower
aggregate content-hit value (`0.1172` vs `0.1523`) did not prevent better carry,
which suggests the previous content-hit metric is too broad: writing fewer but
more semantically relevant boundary pairs is better than writing every filler
pair.

Decision:

Do not promote this exact structural mask as a general production policy; it
uses the local benchmark's task anatomy. Do promote the architectural direction:
A1 should become a learned cue/value boundary and overwrite-risk gate on top of
`content_write_mask`. The gate should be trained or calibrated to write
retrieval-relevant boundary pairs and skip filler overwrites, with
`structural_pair_top_4` as the positive oracle target and `query_skip` as the
separate A2 decode policy.

## TAC-101 Calibrated Boundary Gate Probe

Date: 2026-06-03

Purpose:

Test whether TAC-100's structural sparse-write oracle can be approximated by a
small learned gate without modifying the core TAC model. This is the first
implementation step toward a production A1 write policy: train a cheap boundary
scorer against structural cue/value targets, then use its top-k predictions
through the existing `content_write_mask` path.

Implemented:

- `calibrated_boundary_top_4` benchmark schedule
- `BoundaryWriteGate`, a tiny linear scorer over pair position, parity, and
  cue/value token features
- `train_boundary_write_gate`, trained against structural sparse-write targets
  from local chunked-recall contexts
- `calibrated_boundary_content_write_mask` and `boundary_gate_features`
- focused tests for feature shape, gate calibration, and schedule evaluation

Artifacts:

- Calibrated boundary matrix JSON: `runs/benchmarks/content_calibrated_boundary_local_2026_06_03/content_update_frequency_matrix.json`
- Calibrated boundary matrix Markdown: `runs/benchmarks/content_calibrated_boundary_local_2026_06_03/RESULTS.md`

Aggregate results:

| Schedule | Phase | Carry | Delta vs Full | Query TPS Ratio | Context Update | Query Update | Content Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| query_skip | full_window | 0.1778 | 0.0000 | 1.103 | 1.000 | 0.000 | 0.1523 |
| structural_pair_top_4 | structural_pair_full_context | 0.2500 | 0.0722 | 1.131 | 0.129 | 0.000 | 0.1172 |
| calibrated_boundary_top_4 | calibrated_boundary_full_context | 0.1917 | 0.0139 | 1.101 | 0.129 | 0.000 | 0.1141 |
| full_update | full_window | 0.1778 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.1523 |

Task-level calibrated result:

| Task | Full Carry | Structural Carry | Calibrated Carry | Calibrated Delta |
| --- | ---: | ---: | ---: | ---: |
| single_key | 0.2431 | 0.3472 | 0.3264 | 0.0833 |
| multi_key | 0.3194 | 0.4236 | 0.2083 | -0.1111 |
| delayed_query | 0.2639 | 0.3611 | 0.3264 | 0.0625 |
| noisy_key | 0.0347 | 0.0833 | 0.0694 | 0.0347 |
| multi_hop | 0.0278 | 0.0347 | 0.0278 | 0.0000 |

Verdict:

The calibrated gate is a partial success, not the final balanced solution. It
passes the aggregate sparse-write gate with the same `0.129` context-update
fraction as the oracle, context-loss ratio `0.9986`, and carry delta `+0.0139`.
However, it fails the most important stress case: multi-key carry falls from
`0.3194` full-update and `0.4236` structural-oracle carry to `0.2083`. That
means a position/parity/token linear probe can learn sparse boundary cadence,
but it cannot yet learn overwrite risk or ensure all relevant multi-key pairs
are preserved.

Decision:

Keep the calibrated boundary probe as implementation evidence that sparse
write-gating is trainable through `content_write_mask`. Do not promote this
linear probe as the production A1 gate. The next implementation-ready gate must
be task/hidden-state aware: it should include retrieval query compatibility,
content-store occupancy/overwrite cost, pair novelty, cue/value boundary
confidence, and a per-task or per-context no-regression check. The positive
control remains `structural_pair_top_4`; the learned gate should be judged
against both aggregate A1 criteria and multi-key no-regression.

## TAC-102 Ranking-Trained Boundary Gate

Date: 2026-06-03

Purpose:

Address TAC-101's main failure mode: the BCE-calibrated boundary probe passed
the aggregate sparse-write gate but regressed multi-key carry. This run changes
the learned gate objective from independent pair classification to top-k
ranking, so positive boundary pairs are explicitly trained to score above
filler pairs.

Implemented:

- `ranked_boundary_top_4` benchmark schedule
- `train_ranked_boundary_write_gate`, using margin ranking plus a small BCE
  stabilizer against structural sparse-write targets
- schedule support for `ranked_boundary_full_context` through the existing
  `content_write_mask` path
- focused tests for ranked gate recovery and schedule evaluation

Artifacts:

- Ranked boundary matrix JSON: `runs/benchmarks/content_ranked_boundary_local_2026_06_03/content_update_frequency_matrix.json`
- Ranked boundary matrix Markdown: `runs/benchmarks/content_ranked_boundary_local_2026_06_03/RESULTS.md`

Aggregate results:

| Schedule | Phase | Carry | Delta vs Full | Query TPS Ratio | Context Update | Query Update | Content Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| query_skip | full_window | 0.1778 | 0.0000 | 1.082 | 1.000 | 0.000 | 0.1523 |
| ranked_boundary_top_4 | ranked_boundary_full_context | 0.2542 | 0.0764 | 1.183 | 0.129 | 0.000 | 0.1157 |
| calibrated_boundary_top_4 | calibrated_boundary_full_context | 0.1917 | 0.0139 | 1.151 | 0.129 | 0.000 | 0.1141 |
| structural_pair_top_4 | structural_pair_full_context | 0.2500 | 0.0722 | 1.134 | 0.129 | 0.000 | 0.1172 |
| full_update | full_window | 0.1778 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.1523 |

Task-level ranked result:

| Task | Full Carry | Structural Carry | Ranked Carry | Ranked Delta |
| --- | ---: | ---: | ---: | ---: |
| single_key | 0.2431 | 0.3472 | 0.3750 | 0.1319 |
| multi_key | 0.3194 | 0.4236 | 0.3611 | 0.0417 |
| delayed_query | 0.2639 | 0.3611 | 0.4444 | 0.1806 |
| noisy_key | 0.0347 | 0.0833 | 0.0694 | 0.0347 |
| multi_hop | 0.0278 | 0.0347 | 0.0208 | -0.0069 |

Verdict:

`ranked_boundary_top_4` is the strongest learned local A1 candidate so far. It
writes only `0.129` of context pairs, keeps query writes disabled, preserves
context loss within the gate (`1.0021` ratio), and improves aggregate carry by
`0.0764` absolute versus full update. It also fixes TAC-101's multi-key
regression: multi-key rises to `0.3611`, above full update's `0.3194`, though
still below the structural oracle's `0.4236`. Multi-hop remains weak and
slightly below full update, but full-update multi-hop is also near the floor.

Decision:

Promote ranking-trained boundary gating as the next implementation-ready A1
candidate for local development. Do not yet promote it as final production
policy. The next architecture step should move this from a benchmark-side probe
into an opt-in model gate with richer hidden-state, retrieval-compatibility,
content-store occupancy, and overwrite-cost features. Promotion criteria should
include the aggregate sparse-write gate, multi-key no-regression versus full
update, and a separate multi-hop capability gate so the sparse-write decision
does not hide base-task weakness.

## TAC-103 Hybrid Ranked Boundary Gate

Date: 2026-06-03

Purpose:

Test the TAC-102 follow-up directly: keep the sparse top-k ranking objective,
but expose richer probe features that can represent retrieval usefulness,
hidden-state novelty, and overwrite pressure.

Implemented:

- `hybrid_ranked_boundary_top_4` benchmark schedule
- `hybrid_boundary_gate_features`, extending boundary features with token loss,
  next-token loss, hidden novelty, cue recurrence, and value recurrence
- `train_hybrid_ranked_boundary_write_gate`, using the same top-k margin
  ranking objective as TAC-102 on the richer features
- `hybrid_ranked_boundary_content_write_mask` through the existing
  `content_write_mask` path
- focused tests for hybrid features, trained recovery, and schedule evaluation

Artifacts:

- Hybrid ranked matrix JSON: `runs/benchmarks/content_hybrid_ranked_boundary_local_2026_06_03/content_update_frequency_matrix.json`
- Hybrid ranked matrix Markdown: `runs/benchmarks/content_hybrid_ranked_boundary_local_2026_06_03/RESULTS.md`

Aggregate results:

| Schedule | Phase | Carry | Delta vs Full | Query TPS Ratio | Context Update | Query Update | Content Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| query_skip | full_window | 0.1778 | 0.0000 | 1.085 | 1.000 | 0.000 | 0.1523 |
| ranked_boundary_top_4 | ranked_boundary_full_context | 0.2542 | 0.0764 | 1.198 | 0.129 | 0.000 | 0.1157 |
| hybrid_ranked_boundary_top_4 | hybrid_ranked_boundary_full_context | 0.2625 | 0.0847 | 1.183 | 0.129 | 0.000 | 0.1163 |
| structural_pair_top_4 | structural_pair_full_context | 0.2500 | 0.0722 | 1.122 | 0.129 | 0.000 | 0.1172 |
| full_update | full_window | 0.1778 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.1523 |

Task-level hybrid result:

| Task | Full Carry | Structural Carry | Ranked Carry | Hybrid Carry | Hybrid Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| single_key | 0.2431 | 0.3472 | 0.3750 | 0.4028 | 0.1597 |
| multi_key | 0.3194 | 0.4236 | 0.3611 | 0.3611 | 0.0417 |
| delayed_query | 0.2639 | 0.3611 | 0.4444 | 0.4653 | 0.2014 |
| noisy_key | 0.0347 | 0.0833 | 0.0694 | 0.0625 | 0.0278 |
| multi_hop | 0.0278 | 0.0347 | 0.0208 | 0.0208 | -0.0069 |

Verdict:

`hybrid_ranked_boundary_top_4` is now the strongest aggregate learned local A1
candidate. It keeps the TAC-102 sparse rate (`0.129` context update fraction),
keeps query writes disabled, passes the local context-loss gate (`1.0021`
ratio), and improves aggregate carry to `0.2625`, or `+0.0847` absolute versus
full update. It improves the strongest slices (`single_key` and
`delayed_query`) and preserves TAC-102's multi-key mean.

It is still not the final balanced production solution. The richer features do
not fix the multi-hop weakness, where hybrid remains at `0.0208` versus full
update `0.0278`, and noisy-key drops slightly versus TAC-102 ranked boundary.
Seed-level multi-key behavior is also mixed even though the task mean remains
above full update.

Decision:

Promote `hybrid_ranked_boundary_top_4` as the best current benchmark-side A1
candidate and the next opt-in model-gate prototype. Do not declare production
completion. The next balanced solution needs a two-part design: hybrid ranked
boundary writes for sparse content memory plus a separate multi-hop capability
gate or retrieval-chain mechanism. Promotion criteria should include aggregate
sparse-write pass, multi-key no-regression, noisy-key no-regression versus
ranked boundary, and multi-hop no-regression versus full update.

## TAC-104 Sparse Write-Budget Sweep

Date: 2026-06-03

Purpose:

Test whether TAC-103's remaining noisy-key and multi-hop weaknesses come from
an overly small write budget. The sweep compares top-4, top-6, and top-8 sparse
context writes for structural, ranked-boundary, and hybrid-ranked gates. During
the first attempt, adding schedules changed learned top-4 results because gate
training consumed the global RNG. That partial run was treated as invalid.
The benchmark now trains learned gates inside a deterministic schedule-specific
RNG fork, and focused tests cover schedule-order independence.

Implemented:

- `structural_pair_top_6` and `structural_pair_top_8`
- `ranked_boundary_top_6` and `ranked_boundary_top_8`
- `hybrid_ranked_boundary_top_6` and `hybrid_ranked_boundary_top_8`
- deterministic learned-gate seeding via `deterministic_gate_seed` and
  `fork_rng_devices`
- focused regression test for learned schedule independence from prior gate
  training

Artifacts:

- Budget sweep JSON: `runs/benchmarks/content_sparse_budget_sweep_local_2026_06_03/content_update_frequency_matrix.json`
- Budget sweep Markdown: `runs/benchmarks/content_sparse_budget_sweep_local_2026_06_03/RESULTS.md`

Aggregate results:

| Schedule | Carry | Delta vs Full | Context Update | Content Hit | Context-Loss Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_update | 0.1778 | 0.0000 | 1.000 | 0.1523 | 1.0000 |
| query_skip | 0.1778 | 0.0000 | 1.000 | 0.1523 | 1.0000 |
| structural_pair_top_4 | 0.2500 | 0.0722 | 0.129 | 0.1172 | 1.0000 |
| structural_pair_top_6 | 0.2000 | 0.0222 | 0.194 | 0.1428 | 1.0000 |
| structural_pair_top_8 | 0.1722 | -0.0056 | 0.258 | 0.1622 | 1.0000 |
| ranked_boundary_top_4 | 0.2542 | 0.0764 | 0.129 | 0.1157 | 1.0021 |
| ranked_boundary_top_6 | 0.2083 | 0.0306 | 0.194 | 0.1418 | 1.0021 |
| ranked_boundary_top_8 | 0.1958 | 0.0181 | 0.258 | 0.1619 | 1.0021 |
| hybrid_ranked_boundary_top_4 | 0.2542 | 0.0764 | 0.129 | 0.1158 | 1.0021 |
| hybrid_ranked_boundary_top_6 | 0.2083 | 0.0306 | 0.194 | 0.1423 | 1.0021 |
| hybrid_ranked_boundary_top_8 | 0.1819 | 0.0042 | 0.258 | 0.1627 | 1.0021 |

Task-level hybrid budget result:

| Task | Full Carry | Hybrid Top-4 | Hybrid Top-6 | Hybrid Top-8 |
| --- | ---: | ---: | ---: | ---: |
| single_key | 0.2431 | 0.3681 | 0.3264 | 0.2917 |
| multi_key | 0.3194 | 0.3403 | 0.2778 | 0.2153 |
| delayed_query | 0.2639 | 0.4583 | 0.3611 | 0.3194 |
| noisy_key | 0.0347 | 0.0833 | 0.0556 | 0.0556 |
| multi_hop | 0.0278 | 0.0208 | 0.0208 | 0.0278 |

Verdict:

Write-budget expansion is not the balanced solution. Larger budgets increase
content-hit, but they dilute the carried-recall benefit and introduce
multi-key interference. `hybrid_ranked_boundary_top_8` reaches multi-hop
no-regression versus full update, but only by giving back almost all aggregate
carry advantage and sharply regressing multi-key. `hybrid_ranked_boundary_top_4`
remains the best learned sparse-write tradeoff under deterministic evaluation.

Decision:

Keep top-4 hybrid/ranked boundary gating as the A1 sparse-write prototype.
Reject top-6/top-8 as production candidates for now. The multi-hop blocker is
not solved by writing more pairs; it needs a separate retrieval-chain or
multi-hop capability mechanism that composes sparse content memories without
expanding writes globally.

## TAC-105 Two-Step Retrieval-Chain Probe

Date: 2026-06-03

Purpose:

Test the TAC-104 follow-up directly: can multi-hop improve by composing two
content-memory reads at query time while keeping top-4 sparse context writes?
The local multi-hop generator stores `first_key -> bridge_key` and
`bridge_key -> value`, so a two-step read is the minimal retrieval-chain
positive control.

Implemented:

- `query_skip_chain_k2`, `ranked_boundary_top_4_chain_k2`, and
  `hybrid_ranked_boundary_top_4_chain_k2` schedules
- `memory_chain_steps` propagation through schedule evaluation
- `chained_memory_readout`, which feeds the first memory prediction back as the
  key for a second memory read
- focused tests for chained readout on a deterministic probe model and for the
  sparse hybrid chain schedule

Artifacts:

- Chain matrix JSON: `runs/benchmarks/content_retrieval_chain_local_2026_06_03/content_update_frequency_matrix.json`
- Chain matrix Markdown: `runs/benchmarks/content_retrieval_chain_local_2026_06_03/RESULTS.md`

Aggregate results:

| Schedule | Carry | Delta vs Full | Context Update | Query TPS Ratio | Content Hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_update | 0.1778 | 0.0000 | 1.000 | 1.000 | 0.1523 |
| query_skip | 0.1778 | 0.0000 | 1.000 | 1.037 | 0.1523 |
| query_skip_chain_k2 | 0.0611 | -0.1167 | 1.000 | 1.054 | 0.1523 |
| ranked_boundary_top_4 | 0.2542 | 0.0764 | 0.129 | 1.083 | 0.1157 |
| ranked_boundary_top_4_chain_k2 | 0.1139 | -0.0639 | 0.129 | 1.012 | 0.1157 |
| hybrid_ranked_boundary_top_4 | 0.2542 | 0.0764 | 0.129 | 1.048 | 0.1158 |
| hybrid_ranked_boundary_top_4_chain_k2 | 0.1083 | -0.0694 | 0.129 | 0.993 | 0.1160 |

Task-level chain result:

| Task | Full Carry | Hybrid Top-4 | Hybrid Chain-k2 | Ranked Top-4 | Ranked Chain-k2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| single_key | 0.2431 | 0.3681 | 0.1319 | 0.3750 | 0.1458 |
| multi_key | 0.3194 | 0.3403 | 0.1250 | 0.3611 | 0.1181 |
| delayed_query | 0.2639 | 0.4583 | 0.1806 | 0.4444 | 0.1736 |
| noisy_key | 0.0347 | 0.0833 | 0.0625 | 0.0694 | 0.0833 |
| multi_hop | 0.0278 | 0.0208 | 0.0417 | 0.0208 | 0.0486 |

Verdict:

Always-on two-step retrieval chaining is not the balanced solution. It proves
there is a usable bridge-read signal for multi-hop: sparse ranked chain-k2
improves multi-hop from `0.0208` to `0.0486`, and sparse hybrid chain-k2
improves multi-hop from `0.0208` to `0.0417`. But it destroys direct recall:
single-key, multi-key, and delayed-query all fall sharply, and aggregate carry
drops below full update.

Decision:

Do not promote chain-k2 as a default query policy. Keep `ranked_boundary_top_4`
and `hybrid_ranked_boundary_top_4` as the sparse-write prototypes. Promote
retrieval chaining only as the next conditional mechanism: a learned
halt/continue or verifier gate must decide when the first memory prediction is
a bridge key that warrants a second hop. TAC-106 should train or probe this
conditional chain gate and require no-regression on single-key, multi-key, and
delayed-query while improving multi-hop over full update.

## TAC-106 Cue-Presence Conditional Chain Gate

Date: 2026-06-03

Purpose:

Test the TAC-105 follow-up: keep the two-step retrieval-chain machinery, but
halt unless the first memory prediction is also a written cue in the context.
This is a lightweight verifier proxy for "the first prediction is a bridge key,
not the final answer."

Implemented:

- `hybrid_ranked_boundary_top_4_cond_chain_k2`
- `memory_chain_policy="predicted_written_cue"`
- propagation of the exact context `content_write_mask` into query evaluation
- `predicted_token_is_written_cue`, selecting a second hop only when the first
  predicted token appears at a written cue position
- `memory_chain_fraction` query metric for chained-read diagnostics
- focused tests for halting on direct values, continuing on bridge keys, and
  sparse conditional-chain schedule evaluation

Artifacts:

- Conditional chain JSON: `runs/benchmarks/content_conditional_chain_local_2026_06_03/content_update_frequency_matrix.json`
- Conditional chain Markdown: `runs/benchmarks/content_conditional_chain_local_2026_06_03/RESULTS.md`

Aggregate results:

| Schedule | Carry | Delta vs Full | Context Update | Query TPS Ratio | Content Hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_update | 0.1778 | 0.0000 | 1.000 | 1.000 | 0.1523 |
| query_skip | 0.1778 | 0.0000 | 1.000 | 1.070 | 0.1523 |
| ranked_boundary_top_4 | 0.2542 | 0.0764 | 0.129 | 1.057 | 0.1157 |
| ranked_boundary_top_4_chain_k2 | 0.1139 | -0.0639 | 0.129 | 0.993 | 0.1157 |
| hybrid_ranked_boundary_top_4 | 0.2542 | 0.0764 | 0.129 | 1.121 | 0.1158 |
| hybrid_ranked_boundary_top_4_chain_k2 | 0.1083 | -0.0694 | 0.129 | 1.056 | 0.1160 |
| hybrid_ranked_boundary_top_4_cond_chain_k2 | 0.2361 | 0.0583 | 0.129 | 1.091 | 0.1160 |

Task-level conditional result:

| Task | Full Carry | Hybrid Top-4 | Always Chain-k2 | Conditional Chain-k2 |
| --- | ---: | ---: | ---: | ---: |
| single_key | 0.2431 | 0.3681 | 0.1319 | 0.3472 |
| multi_key | 0.3194 | 0.3403 | 0.1250 | 0.2986 |
| delayed_query | 0.2639 | 0.4583 | 0.1806 | 0.4444 |
| noisy_key | 0.0347 | 0.0833 | 0.0625 | 0.0694 |
| multi_hop | 0.0278 | 0.0208 | 0.0417 | 0.0208 |

Conditional chain fraction by task/seed:

| Task | Seed 11 | Seed 23 | Seed 37 |
| --- | ---: | ---: | ---: |
| single_key | 0.1875 | 0.0833 | 0.0208 |
| multi_key | 0.0833 | 0.1042 | 0.0208 |
| delayed_query | 0.0833 | 0.0417 | 0.0625 |
| noisy_key | 0.0417 | 0.0625 | 0.0625 |
| multi_hop | 0.0417 | 0.0417 | 0.1250 |

Verdict:

Cue-presence conditional chaining is a useful partial repair, but it is not the
balanced final mechanism. It largely fixes TAC-105's direct-recall collapse:
aggregate carry recovers from always-on hybrid chain `0.1083` to `0.2361`.
However, it still underperforms plain hybrid top-4 (`0.2542`), regresses
multi-key below full update (`0.2986` versus `0.3194`), and fails to improve
multi-hop (`0.0208`, still below full update `0.0278`). The low chain fractions
on multi-hop show the cue-presence proxy is too conservative or often follows
the wrong cue signal.

Decision:

Do not promote cue-presence conditional chaining. Keep top-4 hybrid/ranked
boundary gating as the A1 sparse-write prototype. The next multi-hop mechanism
must be a learned verifier/halt head trained against bridge-needed traces or a
task/query-conditioned chain policy that can continue on true bridge reads
without firing on direct recall. TAC-107 should add supervised chain-target
labels to the benchmark harness and test a learned chain gate against strict
requirements: aggregate sparse-write pass, multi-key no-regression, and
multi-hop improvement over full update.

## TAC-107 Oracle Target-Path Chain Gate

Date: 2026-06-03

Purpose:

Before training a verifier, establish the positive-control upper bound for
halt/continue decisions. Continue to the second memory hop only when the first
memory prediction is a written cue whose written value is the query target.
This deliberately uses `value_targets`, so it is an oracle diagnostic, not a
production policy.

Implemented:

- `hybrid_ranked_boundary_top_4_oracle_chain_k2`
- `memory_chain_policy="oracle_written_target"`
- `predicted_token_reaches_written_target`, which checks written cue/value
  pairs against the query target
- focused tests for oracle target-path detection, direct-value halting,
  bridge-path continuation, and sparse schedule evaluation

Artifacts:

- Oracle chain JSON: `runs/benchmarks/content_oracle_chain_local_2026_06_03/content_update_frequency_matrix.json`
- Oracle chain Markdown: `runs/benchmarks/content_oracle_chain_local_2026_06_03/RESULTS.md`

Aggregate results:

| Schedule | Carry | Delta vs Full | Context Update | Query TPS Ratio | Content Hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_update | 0.2986 | 0.0000 | 1.000 | 1.000 | 0.1548 |
| query_skip | 0.2986 | 0.0000 | 1.000 | 1.078 | 0.1548 |
| hybrid_ranked_boundary_top_4 | 0.3861 | 0.0875 | 0.129 | 1.087 | 0.1187 |
| hybrid_ranked_boundary_top_4_chain_k2 | 0.1208 | -0.1778 | 0.129 | 1.021 | 0.1186 |
| hybrid_ranked_boundary_top_4_cond_chain_k2 | 0.3694 | 0.0708 | 0.129 | 0.983 | 0.1186 |
| hybrid_ranked_boundary_top_4_oracle_chain_k2 | 0.3958 | 0.0972 | 0.129 | 1.017 | 0.1187 |

Task-level oracle result:

| Task | Full Carry | Hybrid Top-4 | Always Chain-k2 | Cue Conditional | Oracle Target-Path |
| --- | ---: | ---: | ---: | ---: | ---: |
| single_key | 0.4792 | 0.6528 | 0.1458 | 0.5903 | 0.6389 |
| multi_key | 0.4722 | 0.5694 | 0.1736 | 0.5625 | 0.5833 |
| delayed_query | 0.4444 | 0.5625 | 0.1250 | 0.5139 | 0.5694 |
| noisy_key | 0.0833 | 0.0694 | 0.0694 | 0.0833 | 0.0903 |
| multi_hop | 0.0139 | 0.0764 | 0.0903 | 0.0972 | 0.0972 |

Oracle chain fraction by task/seed:

| Task | Seed 11 | Seed 23 | Seed 37 | Mean |
| --- | ---: | ---: | ---: | ---: |
| single_key | 0.0000 | 0.0208 | 0.0000 | 0.0069 |
| multi_key | 0.0208 | 0.0000 | 0.0000 | 0.0069 |
| delayed_query | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| noisy_key | 0.0000 | 0.0417 | 0.0000 | 0.0139 |
| multi_hop | 0.0208 | 0.0417 | 0.0208 | 0.0278 |

Verdict:

The target-path oracle is the best local sparse chain positive control so far.
It keeps the TAC-104/TAC-106 write budget (`0.129` context update fraction and
no query writes), improves aggregate carry over full update by `+0.0972`, and
does not reproduce the always-on direct-recall collapse. It also repairs the
TAC-106 multi-key blocker in this local run (`0.5833` versus full `0.4722`)
and improves multi-hop over full update (`0.0972` versus `0.0139`).

Decision:

Do not promote `oracle_written_target` as a final architecture, because it uses
the target label at query evaluation time. Promote the mechanism it validates:
a learned verifier/halt head should approximate this target-path decision from
available query, first-read, confidence, written-cue, recurrence, and content
state features. TAC-108 should train that verifier against oracle
target-path labels and require the same gates without label access: sparse
write pass, aggregate carry at or above plain hybrid top-4, multi-key
no-regression, and multi-hop above full update and plain top-4.

## TAC-108 Learned Verifier Chain Gate

Date: 2026-06-03

Purpose:

Approximate TAC-107's oracle target-path halt/continue decision without using
query labels at inference. The verifier is trained locally against oracle
labels, then evaluated from label-free features: first-read confidence, logit
margin, entropy, written-cue presence, cue recurrence, query cue recurrence,
query/prediction equality, and normalized query/predicted token ids.

Implemented:

- `hybrid_ranked_boundary_top_4_learned_chain_k2`
- `ChainVerifierGate`
- `train_chain_verifier_gate`
- `chain_verifier_features`
- `apply_chain_verifier_gate`
- `memory_chain_policy="learned_verifier"`
- focused tests for verifier features, gate application, learned query
  chaining, and sparse learned-chain schedule evaluation

Artifacts:

- Learned chain JSON: `runs/benchmarks/content_learned_chain_local_2026_06_03/content_update_frequency_matrix.json`
- Learned chain Markdown: `runs/benchmarks/content_learned_chain_local_2026_06_03/RESULTS.md`

Aggregate results:

| Schedule | Carry | Delta vs Full | Context Update | Query TPS Ratio | Content Hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_update | 0.2986 | 0.0000 | 1.000 | 1.000 | 0.1548 |
| query_skip | 0.2986 | 0.0000 | 1.000 | 1.121 | 0.1548 |
| hybrid_ranked_boundary_top_4 | 0.3861 | 0.0875 | 0.129 | 1.273 | 0.1187 |
| hybrid_ranked_boundary_top_4_cond_chain_k2 | 0.3694 | 0.0708 | 0.129 | 1.319 | 0.1186 |
| hybrid_ranked_boundary_top_4_oracle_chain_k2 | 0.3958 | 0.0972 | 0.129 | 1.242 | 0.1187 |
| hybrid_ranked_boundary_top_4_learned_chain_k2 | 0.3931 | 0.0944 | 0.129 | 1.211 | 0.1201 |

Task-level learned result:

| Task | Full Carry | Hybrid Top-4 | Oracle Chain | Learned Chain |
| --- | ---: | ---: | ---: | ---: |
| single_key | 0.4792 | 0.6528 | 0.6389 | 0.6181 |
| multi_key | 0.4722 | 0.5694 | 0.5833 | 0.5764 |
| delayed_query | 0.4444 | 0.5625 | 0.5694 | 0.6389 |
| noisy_key | 0.0833 | 0.0694 | 0.0903 | 0.0833 |
| multi_hop | 0.0139 | 0.0764 | 0.0972 | 0.0486 |

Chain fraction comparison:

| Task | Cue Conditional | Oracle Target-Path | Learned Verifier |
| --- | ---: | ---: | ---: |
| single_key | 0.1111 | 0.0069 | 0.0278 |
| multi_key | 0.0486 | 0.0069 | 0.0069 |
| delayed_query | 0.0694 | 0.0000 | 0.0139 |
| noisy_key | 0.0764 | 0.0139 | 0.0694 |
| multi_hop | 0.0556 | 0.0278 | 0.0972 |

Verdict:

The learned verifier is a strong aggregate approximation but not the balanced
final mechanism. It keeps sparse writes (`0.129`), disables query writes,
nearly matches the oracle aggregate carry (`0.3931` versus `0.3958`), and beats
plain hybrid top-4 on aggregate (`0.3861`). It also satisfies multi-key
no-regression versus full update and plain top-4. The blocker is multi-hop:
learned carry is only `0.0486`, below plain hybrid top-4 `0.0764` and oracle
`0.0972`. Its multi-hop chain fraction is too high (`0.0972` versus oracle
`0.0278`), showing over-continuation and poor bridge precision.

Decision:

Do not promote the learned verifier at threshold `0.0` as final. Keep the
verifier surface and supervised oracle-label training as useful machinery. The
next research step should calibrate verifier thresholds or train with a
precision-weighted objective that reduces false continuation, with explicit
multi-hop precision and noisy-key stability gates.

## TAC-109 Learned Verifier Threshold Sweep

Date: 2026-06-03

Purpose:

Test whether TAC-108's multi-hop blocker is only a calibration problem. The
learned verifier over-continued on multi-hop at threshold `0.0`, so this sweep
kept the same supervised verifier machinery and raised the inference threshold
to `1.0`, `2.0`, and `3.0`.

Implemented:

- `hybrid_ranked_boundary_top_4_learned_chain_t1_k2`
- `hybrid_ranked_boundary_top_4_learned_chain_t2_k2`
- `hybrid_ranked_boundary_top_4_learned_chain_t3_k2`
- `chain_gate_threshold` schedule field propagated into `ChainVerifierGate`
- focused sparse-schedule test for a thresholded learned verifier

Artifacts:

- Threshold sweep JSON: `runs/benchmarks/content_learned_chain_threshold_local_2026_06_03/content_update_frequency_matrix.json`
- Threshold sweep Markdown: `runs/benchmarks/content_learned_chain_threshold_local_2026_06_03/RESULTS.md`

Aggregate results:

| Schedule | Carry | Delta vs Full | Context Update | Query TPS Ratio | Content Hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_update | 0.2986 | 0.0000 | 1.000 | 1.000 | 0.1548 |
| hybrid_ranked_boundary_top_4 | 0.3861 | 0.0875 | 0.129 | 1.058 | 0.1187 |
| hybrid_ranked_boundary_top_4_oracle_chain_k2 | 0.3958 | 0.0972 | 0.129 | 0.957 | 0.1187 |
| hybrid_ranked_boundary_top_4_learned_chain_k2 | 0.3931 | 0.0944 | 0.129 | 0.965 | 0.1201 |
| hybrid_ranked_boundary_top_4_learned_chain_t1_k2 | 0.3833 | 0.0847 | 0.129 | 1.033 | 0.1199 |
| hybrid_ranked_boundary_top_4_learned_chain_t2_k2 | 0.3819 | 0.0833 | 0.129 | 0.968 | 0.1202 |
| hybrid_ranked_boundary_top_4_learned_chain_t3_k2 | 0.3861 | 0.0875 | 0.129 | 0.990 | 0.1202 |

Task-level threshold result:

| Task | Hybrid Top-4 | Oracle | Learned t0 | Learned t1 | Learned t2 | Learned t3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single_key | 0.6528 | 0.6389 | 0.6181 | 0.6042 | 0.6111 | 0.6181 |
| multi_key | 0.5694 | 0.5833 | 0.5764 | 0.5556 | 0.5625 | 0.5625 |
| delayed_query | 0.5625 | 0.5694 | 0.6389 | 0.6250 | 0.6250 | 0.6250 |
| noisy_key | 0.0694 | 0.0903 | 0.0833 | 0.0833 | 0.0764 | 0.0903 |
| multi_hop | 0.0764 | 0.0972 | 0.0486 | 0.0486 | 0.0347 | 0.0347 |

Multi-hop chain fraction:

| Schedule | Chain Fraction | Carry | Shuffled |
| --- | ---: | ---: | ---: |
| oracle_chain_k2 | 0.0278 | 0.0972 | 0.0069 |
| learned_chain_k2 | 0.0972 | 0.0486 | 0.0208 |
| learned_chain_t1_k2 | 0.0556 | 0.0486 | 0.0278 |
| learned_chain_t2_k2 | 0.0000 | 0.0347 | 0.0347 |
| learned_chain_t3_k2 | 0.0000 | 0.0347 | 0.0278 |

Verdict:

Static threshold calibration is not the balanced solution. Threshold `0.0`
remains the best learned aggregate (`0.3931`), while thresholds `1.0` through
`3.0` reduce aggregate carry and still fail the multi-hop gate. Higher
thresholds reduce chain fraction, but they filter out useful bridge cases as
well as false continuations. At thresholds `2.0` and `3.0`, multi-hop chaining
falls to zero and carry drops below plain hybrid top-4.

Decision:

Reject static threshold calibration as the final TAC chain verifier. The
evidence points to a richer query-time validation mechanism, not just a scalar
threshold: a counterfactual second-read validator should compare first-read and
second-read confidence/agreement, written-cue evidence, and carried-state
dependence before selecting the second hop.

## TAC-110 Counterfactual Second-Read Validator

Date: 2026-06-03

Purpose:

Test whether a label-free query-time validator can recover the useful oracle
second hops without over-continuing. Instead of accepting every bridge-like
prediction or tuning one verifier threshold, the counterfactual policy computes
the second read, compares first-read and second-read confidence/margin, and
uses the second read only when the candidate answer changes with enough
confidence gain.

Implemented:

- `hybrid_ranked_boundary_top_4_counterfactual_chain_k2`
- `memory_chain_policy="counterfactual_confidence"`
- `counterfactual_second_read_is_better`
- `counterfactual_min_confidence_gain`
- `counterfactual_min_margin_gain`
- shared `write_gate_seed_group` for hybrid top-4 chain-policy variants
- focused tests for confidence-gain validation, query evaluation, sparse
  schedule evaluation, and shared write-gate seed groups

Artifacts:

- Initial counterfactual JSON, superseded/provisional:
  `runs/benchmarks/content_counterfactual_chain_local_2026_06_03/content_update_frequency_matrix.json`
- Corrected shared-gate JSON:
  `runs/benchmarks/content_counterfactual_chain_shared_gate_local_2026_06_03/content_update_frequency_matrix.json`
- Corrected shared-gate Markdown:
  `runs/benchmarks/content_counterfactual_chain_shared_gate_local_2026_06_03/RESULTS.md`

Methodology correction:

The first TAC-110 matrix exposed a comparison flaw: chain-policy variants used
different deterministic gate seeds from the plain hybrid top-4 schedule, so a
policy that selected zero second reads could still report different carry. The
fix is `write_gate_seed_group`, which makes hybrid top-4, oracle chain, learned
chain, thresholded learned chain, and counterfactual chain share the same
trained sparse write gate. The corrected shared-gate artifact supersedes the
earlier TAC-110 run for policy comparison and should be preferred when
interpreting TAC-107 through TAC-109 chain deltas.

Aggregate corrected results:

| Schedule | Carry | Delta vs Full | Context Update | Query TPS Ratio | Chain Fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_update | 0.2986 | 0.0000 | 1.000 | 1.000 | 0.0000 |
| query_skip | 0.2986 | 0.0000 | 1.000 | 1.033 | 0.0000 |
| hybrid_ranked_boundary_top_4 | 0.3861 | 0.0875 | 0.129 | 1.162 | 0.0000 |
| hybrid_ranked_boundary_top_4_oracle_chain_k2 | 0.3944 | 0.0958 | 0.129 | 1.117 | 0.0167 |
| hybrid_ranked_boundary_top_4_learned_chain_k2 | 0.3833 | 0.0847 | 0.129 | 1.135 | 0.0472 |
| hybrid_ranked_boundary_top_4_learned_chain_t1_k2 | 0.3833 | 0.0847 | 0.129 | 1.130 | 0.0153 |
| hybrid_ranked_boundary_top_4_counterfactual_chain_k2 | 0.3861 | 0.0875 | 0.129 | 1.156 | 0.0000 |

Task-level corrected result:

| Task | Full | Hybrid Top-4 | Oracle | Learned t0 | Learned t1 | Counterfactual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single_key | 0.4792 | 0.6528 | 0.6528 | 0.6250 | 0.6250 | 0.6528 |
| multi_key | 0.4722 | 0.5694 | 0.5694 | 0.5556 | 0.5556 | 0.5694 |
| delayed_query | 0.4444 | 0.5625 | 0.5694 | 0.6181 | 0.6250 | 0.5625 |
| noisy_key | 0.0833 | 0.0694 | 0.0903 | 0.0764 | 0.0694 | 0.0694 |
| multi_hop | 0.0139 | 0.0764 | 0.0903 | 0.0417 | 0.0417 | 0.0764 |

Task-level chain fractions:

| Task | Oracle | Learned t0 | Learned t1 | Counterfactual |
| --- | ---: | ---: | ---: | ---: |
| single_key | 0.0069 | 0.0278 | 0.0139 | 0.0000 |
| multi_key | 0.0069 | 0.0139 | 0.0000 | 0.0000 |
| delayed_query | 0.0139 | 0.0208 | 0.0000 | 0.0000 |
| noisy_key | 0.0278 | 0.0625 | 0.0069 | 0.0000 |
| multi_hop | 0.0278 | 0.1111 | 0.0556 | 0.0000 |

Verdict:

Counterfactual confidence validation is safe but too conservative. It exactly
matches plain hybrid top-4 aggregate carry (`0.3861`) and all task means, with
zero measured chain selections. That makes it a valid guard against harmful
second reads, but not a solution for the multi-hop retrieval gap. The corrected
oracle still shows a small sparse-chain ceiling (`0.3944` aggregate and
`0.0903` multi-hop), but it remains non-deployable because it uses target
labels. The learned verifier variants are also rejected under the corrected
comparison: they chain more often than the oracle and underperform plain
hybrid top-4 on aggregate and multi-hop.

Decision:

Do not promote counterfactual confidence validation, learned verifier threshold
`0.0`, or learned verifier threshold `1.0` as the balanced final chain policy.
Keep `hybrid_ranked_boundary_top_4` as the best current deployable sparse-write
baseline: it writes `0.129` of context pairs, disables query writes, and
improves aggregate carry by `+0.0875` versus full update in the corrected
shared-gate comparison. The next research target should be a deployable
bridge-target verifier trained for oracle-like sparse continuation, using
direct supervision from target-path labels during training but adding
inference-time features that counterfactual confidence lacks: first-read
written-cue identity, query-to-prediction recurrence, second-read agreement
with written values, and carried-state dependence/ablation. Promotion gates
remain: aggregate at or above hybrid top-4, no direct-recall regression,
multi-hop above hybrid top-4, no query writes, and context update fraction at
or near `0.129`.

## TAC-111 Bridge-Target Verifier

Date: 2026-06-03

Purpose:

Test TAC-110's proposed richer deployable verifier: keep oracle target-path
labels for training, but expose candidate second-read evidence at inference.
The bridge verifier adds label-free second-read features to the TAC-108
verifier surface so the gate can learn whether the predicted bridge produces a
plausible written value before selecting the second hop.

Implemented:

- `hybrid_ranked_boundary_top_4_bridge_chain_k2`
- `memory_chain_policy="bridge_verifier"`
- `bridge_verifier_features`
- bridge-verifier training path in `train_chain_verifier_gate`
- focused tests for bridge feature exposure, query-time bridge selection, and
  sparse schedule evaluation

Artifacts:

- Bridge verifier JSON:
  `runs/benchmarks/content_bridge_verifier_chain_local_2026_06_03/content_update_frequency_matrix.json`
- Bridge verifier Markdown:
  `runs/benchmarks/content_bridge_verifier_chain_local_2026_06_03/RESULTS.md`

Aggregate results:

| Schedule | Carry | Delta vs Full | Context Update | Query TPS Ratio | Chain Fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_update | 0.2986 | 0.0000 | 1.000 | 1.000 | 0.0000 |
| query_skip | 0.2986 | 0.0000 | 1.000 | 1.041 | 0.0000 |
| hybrid_ranked_boundary_top_4 | 0.3861 | 0.0875 | 0.129 | 1.116 | 0.0000 |
| hybrid_ranked_boundary_top_4_oracle_chain_k2 | 0.3944 | 0.0958 | 0.129 | 1.029 | 0.0167 |
| hybrid_ranked_boundary_top_4_learned_chain_k2 | 0.3833 | 0.0847 | 0.129 | 1.030 | 0.0472 |
| hybrid_ranked_boundary_top_4_counterfactual_chain_k2 | 0.3861 | 0.0875 | 0.129 | 1.061 | 0.0000 |
| hybrid_ranked_boundary_top_4_bridge_chain_k2 | 0.3847 | 0.0861 | 0.129 | 1.029 | 0.0431 |

Task-level bridge result:

| Task | Hybrid Top-4 | Oracle | Learned | Counterfactual | Bridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| single_key | 0.6528 | 0.6528 | 0.6250 | 0.6528 | 0.6250 |
| multi_key | 0.5694 | 0.5694 | 0.5556 | 0.5694 | 0.5556 |
| delayed_query | 0.5625 | 0.5694 | 0.6181 | 0.5625 | 0.6250 |
| noisy_key | 0.0694 | 0.0903 | 0.0764 | 0.0694 | 0.0764 |
| multi_hop | 0.0764 | 0.0903 | 0.0417 | 0.0764 | 0.0417 |

Task-level chain fractions:

| Task | Oracle | Learned | Bridge |
| --- | ---: | ---: | ---: |
| single_key | 0.0069 | 0.0278 | 0.0278 |
| multi_key | 0.0069 | 0.0139 | 0.0069 |
| delayed_query | 0.0139 | 0.0208 | 0.0208 |
| noisy_key | 0.0278 | 0.0625 | 0.0486 |
| multi_hop | 0.0278 | 0.1111 | 0.1111 |

Verdict:

The bridge-target verifier is a useful diagnostic but not the balanced final
policy. It improves delayed-query over every compared schedule (`0.6250`
versus hybrid top-4 `0.5625` and oracle `0.5694`) and recovers noisy-key
relative to counterfactual/top-4, showing that second-read evidence contains
real signal. But it still underperforms plain hybrid top-4 on aggregate
(`0.3847` versus `0.3861`), regresses direct recall on `single_key` and
`multi_key`, and fails the key multi-hop gate (`0.0417` versus top-4 `0.0764`
and oracle `0.0903`). The multi-hop chain fraction is `0.1111`, four times the
oracle's `0.0278`, so the failure is still over-continuation.

Decision:

Reject the TAC-111 bridge verifier as final. Keep its feature surface as a
useful signal source, but require TAC-112 to add precision pressure rather
than more raw features. The next candidate should train the bridge verifier
with an explicit false-positive cost and a direct-recall guard, then evaluate
promotion only if it beats hybrid top-4 on aggregate, preserves `single_key`
and `multi_key`, and lifts multi-hop above `0.0764` without chain fraction
approaching the learned verifier's `0.1111`.

## TAC-112 Precision-Biased Bridge Gate

Date: 2026-06-03

Purpose:

Test whether TAC-111's over-continuation can be fixed with explicit
false-positive pressure rather than more features. This keeps the
bridge-verifier feature surface but adds schedule-level verifier loss weights
and a higher inference threshold for a precision-biased bridge schedule.

Implemented:

- `hybrid_ranked_boundary_top_4_bridge_precision_chain_k2`
- `chain_gate_positive_weight`
- `chain_gate_negative_weight`
- weighted verifier BCE in `train_chain_verifier_gate`
- focused sparse-schedule test confirming the precision schedule uses a
  negative verifier weight above `1.0`

Artifacts:

- Precision bridge JSON:
  `runs/benchmarks/content_bridge_precision_chain_local_2026_06_03/content_update_frequency_matrix.json`
- Precision bridge Markdown:
  `runs/benchmarks/content_bridge_precision_chain_local_2026_06_03/RESULTS.md`

Aggregate results:

| Schedule | Carry | Delta vs Full | Context Update | Chain Fraction |
| --- | ---: | ---: | ---: | ---: |
| full_update | 0.2986 | 0.0000 | 1.000 | 0.0000 |
| hybrid_ranked_boundary_top_4 | 0.3861 | 0.0875 | 0.129 | 0.0000 |
| hybrid_ranked_boundary_top_4_oracle_chain_k2 | 0.3944 | 0.0958 | 0.129 | 0.0167 |
| hybrid_ranked_boundary_top_4_counterfactual_chain_k2 | 0.3861 | 0.0875 | 0.129 | 0.0000 |
| hybrid_ranked_boundary_top_4_bridge_chain_k2 | 0.3847 | 0.0861 | 0.129 | 0.0431 |
| hybrid_ranked_boundary_top_4_bridge_precision_chain_k2 | 0.3819 | 0.0833 | 0.129 | 0.0042 |

Task-level precision result:

| Task | Hybrid Top-4 | Oracle | Bridge | Precision Bridge |
| --- | ---: | ---: | ---: | ---: |
| single_key | 0.6528 | 0.6528 | 0.6250 | 0.6250 |
| multi_key | 0.5694 | 0.5694 | 0.5556 | 0.5556 |
| delayed_query | 0.5625 | 0.5694 | 0.6250 | 0.6250 |
| noisy_key | 0.0694 | 0.0903 | 0.0764 | 0.0694 |
| multi_hop | 0.0764 | 0.0903 | 0.0417 | 0.0347 |

Task-level chain fractions:

| Task | Oracle | Bridge | Precision Bridge |
| --- | ---: | ---: | ---: |
| single_key | 0.0069 | 0.0278 | 0.0069 |
| multi_key | 0.0069 | 0.0069 | 0.0000 |
| delayed_query | 0.0139 | 0.0208 | 0.0000 |
| noisy_key | 0.0278 | 0.0486 | 0.0000 |
| multi_hop | 0.0278 | 0.1111 | 0.0139 |

Verdict:

Precision weighting is not the balanced solution. It successfully reduces
aggregate chain fraction from `0.0431` to `0.0042`, but the reduction mostly
removes useful continuations while leaving direct-recall regressions in place.
Aggregate carry drops from unweighted bridge `0.3847` to `0.3819`, below plain
hybrid top-4 and counterfactual `0.3861`. Multi-hop also worsens from bridge
`0.0417` to `0.0347`, far below hybrid top-4 `0.0764` and oracle `0.0903`.

Decision:

Reject simple weighted BCE plus thresholding for bridge verification. The
evidence now separates two needs: the sparse write gate is good enough for the
current deployable baseline, but the query-time chain policy needs a stronger
structural guard than a linear verifier. Next research should either add an
explicit direct-answer veto that preserves first-read answers when they are
already written values, or move the chain decision into a constrained
two-stage policy: first detect whether the query itself is a direct written cue
and only then allow bridge continuation on examples with no direct written
value path.

## TAC-092 Run 5 Capability Validation Result

Date: 2026-06-03

Artifacts:

- Pulled Kaggle outputs:
  `runs/kaggle_results/tac_run5_capability_20k_latest`
- Run directory:
  `runs/kaggle_results/tac_run5_capability_20k_latest/best_tac_agentic_run5_capability`
- Inspected files:
  `final_summary.json`, `metrics.jsonl`, `run_manifest.json`,
  `specialization_checkpoints/summaries.jsonl`,
  `specialization_checkpoints/step_002000/program_specialization.json`,
  `specialization_checkpoints/step_002000/program_attribution.csv`,
  `specialization_checkpoints/step_005000/program_specialization.json`,
  and
  `specialization_checkpoints/step_005000/program_attribution.csv`

Run status:

Kaggle reports `KernelWorkerStatus.COMPLETE`. The run stopped for time at
`9586` optimizer steps out of the planned `20000`, after seeing about
`175,998,960` tokens. The manifest confirms the intended Run 5 shape:
`p12`, `base_semantic`, `category_route_weight=0.1`, MI route objective,
`fp16`, dual-worker distributed execution, and `19,943,472` total parameters
with `9,554,224` identity-field parameters (`47.91%`).

Capability result:

Run 5 failed the capability restoration gate.

- Best eval loss: `6.328312933444977`
- Random uniform CE for vocab size 512: `ln(512)=6.238324625039508`
- Latest train loss: `6.4451003074646`
- Latest next-token loss: `6.36531416575114`
- Latest program memory cosine: `0.9118390281995138`
- Latest gradient telemetry: `gradient_norm=0.0`,
  `grad_scaler_scale=0.0`

The best eval loss is worse than the uniform-token CE reference, and the
reported gradient/scaler values are not healthy enough to treat this as a
clean architectural result. The loss trajectory did not show meaningful
capability recovery before the time stop.

Specialization result:

Run 5 recovered partial routing structure, but not functional specialization.

Both the `2k` and `5k` specialization checkpoints report:

- MI: `0.41154661727406927` bits
- Normalized MI: `0.23005711655058203`
- Program entropy: `1.788888878747568` bits
- Attribution rows: `96`
- Sampled categories: `argument_schema`, `memory_counterfactual`,
  `repair_after_failure`, `stale_memory_rejection`, `tool_choice`,
  `verification_planning`
- Dominant top-program count: program `3` on `57/96` records

Final specialization analysis did not run because
`specialization_analysis.enabled=false` with skipped reason `stopped_for_time`.
No 10k/20k checkpoint or final knockout evidence exists for this run.

Verdict:

Run 5 is complete as an execution artifact, but blocked as an architecture
validation. It shows weak routing specialization without useful language-model
capability. This does not prove that next-token prediction is incompatible
with TAC, because the run also shows a concrete optimization/precision warning:
zero reported gradient norm and zero reported GradScaler scale under `fp16`.

Follow-up gate:

Do not promote alternative objectives or large TAC architecture conclusions
from this result. The next capability experiment should be a Run 5B
optimizer/precision sanity repair:

- validate nonzero gradient telemetry before long training;
- use `bf16` or `fp32`, or disable AMP if T4 `fp16` scaler collapse repeats;
- run a vanilla/baseline trainer sanity check under the same Kaggle path;
- keep the Run 5 TAC shape comparable unless the repair requires a bounded
  training-stability change;
- require final capability improvement before interpreting specialization;
- require final specialization and knockout artifacts after capability clears.

## TAC-113 Run 5B Optimizer Precision Sanity Repair

Date: 2026-06-03

Purpose:

Turn the Run 5 failure mode into an explicit trainer gate before spending
another long Kaggle session. The repair keeps the Run 5 architecture comparable
(`p12`, `base_semantic`, `routing_top_k=2`, `category_route_weight=0.1`, MI
route objective), but removes the observed `fp16` GradScaler-collapse ambiguity
by defaulting Run 5B to `fp32` and requiring nonzero optimizer-health telemetry.

Implemented:

- Added `run5b_capability_config` and `run5b_capability_training_kwargs`.
- Added `--preset run5b_capability` to `kaggle/train_best_tac_agentic.py`.
- Run 5B defaults `precision` from `auto` to `fp32` while preserving explicit
  precision overrides.
- Added `optimization_health_status` with gradient-norm, precision, and scaler
  checks.
- Added a Run 5B fail-fast optimizer gate using
  `min_healthy_gradient_norm=1e-12`.
- Persisted optimizer-health gate settings in `run_manifest.json`.
- Added focused tests for preset defaults, trainer selection, and a two-step
  local CPU smoke run.

Local evidence:

The focused Run 5B smoke test completed two CPU optimizer steps with:

- `precision=fp32`
- `grad_scaler_scale=1.0`
- `gradient_norm=1.6381253004074097`
- `optimization_health.status=passed`
- `n_programs=12`
- identity-field share `20294 / 82470 = 24.61%` in the tiny smoke shape

Decision:

Accept TAC-113 as a trainer/experiment-safety repair. This is not architecture
promotion evidence and does not overturn the Run 5 capability failure. It
does, however, close the immediate ambiguity from Run 5's `gradient_norm=0.0`
and `grad_scaler_scale=0.0` by making the next comparable run fail early if
optimization collapses.

Next external gate:

Launch Run 5B only with the repaired preset and require all of the following
before interpreting specialization:

- optimizer health remains passed with nonzero gradient telemetry;
- capability beats the uniform CE reference `ln(512)=6.238324625039508`;
- the run reaches the planned step/time budget or provides a clean resume path;
- final specialization and knockout artifacts are produced after capability
  clears.

## TAC-114 Run 5B Kaggle Launch Attempt

Date: 2026-06-03

Prepared artifacts:

- Code dataset directory:
  `runs/kaggle_run5b_code_2026_06_03`
- Code dataset id:
  `mathewlincoln/tac-run5b-capability-code-2026-06-03`
- Kernel directory:
  `runs/kaggle_run5b_capability_2026_06_03`
- Kernel id:
  `mathewlincoln/tac-run-5b-capability-p12-fp32-20k`

Validation:

- `dataset-metadata.json` parses with `python -m json.tool`.
- `kernel-metadata.json` parses with `python -m json.tool`.
- `run_run5b_capability.py` compiles with `python -m py_compile`.
- The staged code dataset contains `run5b_capability` and
  `optimization_health` in the copied trainer/preset files.

Kaggle action:

The code dataset push succeeded. `kaggle datasets files
mathewlincoln/tac-run5b-capability-code-2026-06-03` lists the uploaded bundle
contents, including `best-tac-agentic-training-bundle/...`.

The kernel push was attempted with:

`kaggle kernels push -p runs/kaggle_run5b_capability_2026_06_03`

Result:

Blocked by Kaggle quota:

`Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.`

Decision:

TAC-114 is blocked on external GPU quota, not on local code readiness. The
ready-to-run Run 5B artifacts should be pushed again after quota resets. Until
that external run completes, no TAC architecture promotion, NTP abandonment, or
specialization interpretation should be made from Run 5B.

## TAC-115 Bridge-Veto Chain Policy And Verifier-Stream Audit

Date: 2026-06-03

Purpose:

Test the TAC-112 follow-up hypothesis: an explicit direct-answer veto might
stop bridge-verifier over-continuation while preserving useful bridge
continuations. Also audit the verifier-chain comparison method after a small
matrix showed chain effects that were inconsistent with reported chain
fractions.

Implemented:

- Added `hybrid_ranked_boundary_top_4_bridge_veto_chain_k2`.
- Added `memory_chain_policy=bridge_verifier_direct_veto`.
- Added `is_direct_written_answer`, which vetoes continuation only when the
  first read is a direct written answer for the query and is not itself a
  written bridge cue. This preserves valid `query -> bridge -> value` paths.
- Added aggregate reporting for `mean_memory_chain_fraction`.
- Fixed a benchmark confound: sparse-gate and chain-verifier training used the
  same `ChunkedRecallBatcher` as evaluation, so verifier schedules consumed
  training batches and then evaluated on a later stream than non-verifier
  schedules. `evaluate_content_update_schedule` now preserves and restores the
  batcher RNG state around gate training.
- Added a regression test proving gate training does not advance the evaluation
  batcher stream.

Artifacts:

- Superseded first matrix:
  `runs/benchmarks/content_bridge_veto_chain_local_2026_06_03`
- Corrected state-preserved matrix:
  `runs/benchmarks/content_bridge_veto_chain_state_preserved_local_2026_06_03`

Corrected matrix:

The corrected matrix covers 5 chunked-recall tasks x 3 seeds x 8 schedules at
120 training steps. It compares `full_update`, `query_skip`, plain
`hybrid_ranked_boundary_top_4`, oracle/counterfactual chain controls, unweighted
bridge, precision bridge, and bridge-veto.

Aggregate corrected results:

- `hybrid_ranked_boundary_top_4`: carry `0.1861`, delta vs full `+0.0542`,
  chain fraction `0.0000`, context update `0.2667`, context-loss ratio `1.0000`
- `hybrid_ranked_boundary_top_4_bridge_chain_k2`: carry `0.1861`, delta
  `+0.0542`, chain fraction `0.0264`, context update `0.2667`, context-loss
  ratio `1.0000`
- `hybrid_ranked_boundary_top_4_bridge_precision_chain_k2`: carry `0.1861`,
  delta `+0.0542`, chain fraction `0.0000`, context update `0.2667`,
  context-loss ratio `1.0000`
- `hybrid_ranked_boundary_top_4_bridge_veto_chain_k2`: carry `0.1861`, delta
  `+0.0542`, chain fraction `0.0264`, context update `0.2667`,
  context-loss ratio `1.0000`

Per-task corrected carry for plain hybrid top-4 versus bridge-veto:

- `single_key`: `0.2708` vs `0.2708`
- `multi_key`: `0.2778` vs `0.2778`
- `delayed_query`: `0.2708` vs `0.2708`
- `noisy_key`: `0.0625` vs `0.0625`
- `multi_hop`: `0.0486` vs `0.0486`

Decision:

Reject the direct-answer veto as a balanced chain solution. It is logically
safe in focused tests, but in the corrected matrix it adds nonzero continuation
overhead without any carry lift over plain hybrid top-4. Precision bridge
selects no continuations and also ties plain hybrid top-4, so it remains a
suppression mechanism rather than a useful chain policy.

Methodology decision:

Treat earlier learned/verifier chain comparisons as superseded unless rerun
with batcher-state preservation. The corrected result changes the local story:
the strongest currently supported sparse-write candidate is plain
`hybrid_ranked_boundary_top_4`; chain policies have not yet shown a deployable
benefit under a same-stream evaluation.

Next local chain direction:

Do not keep adding scalar thresholds or direct-answer vetoes to the current
post-hoc verifier gate. The next credible chain experiment should either:

- train/evaluate a chain head inside the model path so the query distribution
  and continuation decision are learned jointly; or
- build a deterministic graph/path read over the sparse written cue/value
  table, then compare it as a separate retrieval algorithm against the learned
  post-hoc verifier.

## TAC-116 Sparse Graph-Path Read Diagnostic

Date: 2026-06-03

Purpose:

Test the TAC-115 follow-up directly: determine whether the top-4 sparse writer
already stores enough cue/value edges for multi-hop recall, with the remaining
failure caused by learned chain/verifier selection, or whether the sparse write
gate itself is not storing target-aligned paths.

Implemented:

- Added `hybrid_ranked_boundary_top_4_graph_path_k2`.
- Added `memory_chain_policy="sparse_graph_path"`.
- Added `sparse_graph_path_tokens`, which follows written cue/value edges for
  up to `k` hops using the latest matching written edge per cue.
- Added focused tests for direct lookup, two-hop lookup, latest-edge overwrite
  behavior, query-time graph-path readout, and sparse schedule evaluation.

Artifact:

`runs/benchmarks/content_sparse_graph_path_local_2026_06_03`

Matrix:

The matrix covers 5 chunked-recall tasks x 3 seeds x 7 schedules at 120
training steps. It compares `full_update`, `query_skip`, plain
`hybrid_ranked_boundary_top_4`, bridge/precision/veto chain controls, and the
new graph-path diagnostic under the state-preserved benchmark harness.

Aggregate results:

- `full_update`: carry `0.1844`, context update `1.0000`, query update
  `1.0000`
- `query_skip`: carry `0.1844`, context update `1.0000`, query update `0.0000`,
  query TPS ratio `1.0342`
- `hybrid_ranked_boundary_top_4`: carry `0.2448`, delta vs full `+0.0604`,
  chain fraction `0.0000`, context update `0.1290`, context-loss ratio
  `1.0000`
- `hybrid_ranked_boundary_top_4_bridge_chain_k2`: carry `0.2448`, delta
  `+0.0604`, chain fraction `0.0292`, context update `0.1290`
- `hybrid_ranked_boundary_top_4_bridge_precision_chain_k2`: carry `0.2448`,
  delta `+0.0604`, chain fraction `0.0010`, context update `0.1290`
- `hybrid_ranked_boundary_top_4_bridge_veto_chain_k2`: carry `0.2448`, delta
  `+0.0604`, chain fraction `0.0292`, context update `0.1290`
- `hybrid_ranked_boundary_top_4_graph_path_k2`: carry `0.2448`, delta
  `+0.0604`, chain fraction `0.1969`, context update `0.1290`

Task-level carry for full update, plain hybrid top-4, and graph-path:

- `single_key`: `0.2344`, `0.3438`, `0.3438`; graph chain fraction `0.0417`
- `multi_key`: `0.3177`, `0.4167`, `0.4167`; graph chain fraction `0.0000`
- `delayed_query`: `0.2917`, `0.3542`, `0.3542`; graph chain fraction
  `0.0469`
- `noisy_key`: `0.0521`, `0.0885`, `0.0885`; graph chain fraction `0.0000`
- `multi_hop`: `0.0260`, `0.0208`, `0.0208`; graph chain fraction `0.8958`

Decision:

Reject deterministic sparse graph-path readout as the balanced solution. It is
logically correct in focused tests and it finds many traversable paths on
multi-hop, but those paths do not improve carry accuracy. The decisive negative
case is `multi_hop`: graph-path chains on `0.8958` of examples yet remains at
`0.0208` carry, below full update's `0.0260`.

Interpretation:

The current top-4 sparse writer stores traversable edges, but those edges are
not sufficiently target-aligned for answer recovery. This shifts the next local
research target away from more post-hoc chain policies and toward path-aligned
write selection: the writer must learn which cue/value edges preserve answer
paths under noisy and multi-hop conditions, not merely which edges look
structurally writable.

Next local direction:

Keep `hybrid_ranked_boundary_top_4` as the supported sparse-write prototype.
The next credible A1 candidate should add path-supervised or target-alignment
features/losses to the write gate, then rerun the same state-preserved matrix
with explicit no-regression gates for noisy-key and multi-hop.

## TAC-117 Path-Aligned Sparse Write Gate

Date: 2026-06-03

Purpose:

Test whether the TAC-116 failure is caused by the sparse writer optimizing the
wrong structural target. Instead of training only against odd-position
cue/value boundaries, train a sibling hybrid sparse-write gate against explicit
query-to-answer path edges.

Implemented:

- Added `path_aligned_content_write_mask`.
- Added `train_path_aligned_hybrid_write_gate`.
- Added `path_aligned_hybrid_top_4`.
- Added `path_aligned_hybrid_top_4_graph_path_k2`.
- Added focused tests for direct answer edges, multi-hop path edges,
  noisy-key value-ending edges, learned path-edge recovery, and sparse schedule
  evaluation.

Artifact:

`runs/benchmarks/content_path_aligned_hybrid_local_2026_06_03`

Matrix:

The matrix covers 5 chunked-recall tasks x 3 seeds x 6 schedules at 120
training steps. It compares `full_update`, `query_skip`, the current
`hybrid_ranked_boundary_top_4`, the TAC-116 graph-path diagnostic, and the two
path-aligned variants.

Aggregate results:

- `full_update`: carry `0.1844`, context update `1.0000`
- `query_skip`: carry `0.1844`, context update `1.0000`, query update `0.0000`
- `hybrid_ranked_boundary_top_4`: carry `0.2448`, delta vs full `+0.0604`,
  context update `0.1290`, content hit `0.1166`
- `hybrid_ranked_boundary_top_4_graph_path_k2`: carry `0.2448`, delta
  `+0.0604`, chain fraction `0.1969`, context update `0.1290`
- `path_aligned_hybrid_top_4`: carry `0.1792`, delta vs full `-0.0052`,
  context update `0.1290`, content hit `0.1117`
- `path_aligned_hybrid_top_4_graph_path_k2`: carry `0.1792`, delta vs full
  `-0.0052`, chain fraction `0.1177`, context update `0.1290`

Task-level carry for full update, hybrid top-4, and path-aligned top-4:

- `single_key`: `0.2344`, `0.3438`, `0.3385`
- `multi_key`: `0.3177`, `0.4167`, `0.2812`
- `delayed_query`: `0.2917`, `0.3542`, `0.1823`
- `noisy_key`: `0.0521`, `0.0885`, `0.0573`
- `multi_hop`: `0.0260`, `0.0208`, `0.0365`

Decision:

Reject pure path-aligned sparse-write training as the balanced A1 solution. It
does repair the specific multi-hop slice (`0.0365` vs hybrid top-4 `0.0208` and
full update `0.0260`), but the aggregate collapses from hybrid top-4 `0.2448`
to `0.1792`, and it regresses multi-key, delayed-query, and noisy-key.

Interpretation:

The path target contains useful signal, but replacing structural sparse-write
supervision with path-only supervision loses broadly useful recall writes. The
next write-gate candidate should mix structural and path-aligned objectives
rather than choosing one. The likely gate should preserve the current
hybrid-ranked top-4 behavior as a backbone, then add a small path-alignment
bonus or protected path-edge quota only where it does not reduce direct recall
coverage.

Next local direction:

Build a mixed structural-plus-path sparse write gate with explicit no-regression
acceptance criteria:

- aggregate carry at least `hybrid_ranked_boundary_top_4`
- multi-key and delayed-query no worse than hybrid top-4
- noisy-key no worse than hybrid top-4
- multi-hop at least full update and preferably above hybrid top-4

## TAC-118 Mixed Structural-Plus-Path Sparse Write Gate

Date: 2026-06-03

Purpose:

Test the TAC-117 follow-up: combine the useful multi-hop path signal with the
existing structural sparse-write backbone by training one hybrid gate against
the union of structural boundary targets and path-aligned answer-edge targets.

Implemented:

- Added `mixed_path_structural_content_write_mask`.
- Added `train_mixed_path_structural_hybrid_write_gate`.
- Added `mixed_path_structural_hybrid_top_4`.
- Added `mixed_path_structural_hybrid_top_4_graph_path_k2`.
- Added focused tests for mixed-target construction, learned path-edge
  preservation, and sparse schedule evaluation.

Artifact:

`runs/benchmarks/content_mixed_path_structural_local_2026_06_03`

Matrix:

The matrix covers 5 chunked-recall tasks x 3 seeds x 6 schedules at 120
training steps. It compares full update, query skip, hybrid top-4, pure
path-aligned top-4, and the mixed structural/path variants.

Aggregate results:

- `full_update`: carry `0.1844`, context update `1.0000`
- `query_skip`: carry `0.1844`, context update `1.0000`, query update `0.0000`
- `hybrid_ranked_boundary_top_4`: carry `0.2448`, delta vs full `+0.0604`,
  context update `0.1290`, content hit `0.1166`
- `path_aligned_hybrid_top_4`: carry `0.1792`, delta vs full `-0.0052`,
  context update `0.1290`, content hit `0.1117`
- `mixed_path_structural_hybrid_top_4`: carry `0.1771`, delta vs full
  `-0.0073`, context update `0.1290`, content hit `0.1132`
- `mixed_path_structural_hybrid_top_4_graph_path_k2`: carry `0.1771`, delta
  vs full `-0.0073`, chain fraction `0.1156`, context update `0.1290`

Task-level carry for full update, hybrid top-4, pure path, and mixed target:

- `single_key`: `0.2344`, `0.3438`, `0.3385`, `0.2656`
- `multi_key`: `0.3177`, `0.4167`, `0.2812`, `0.3073`
- `delayed_query`: `0.2917`, `0.3542`, `0.1823`, `0.2604`
- `noisy_key`: `0.0521`, `0.0885`, `0.0573`, `0.0417`
- `multi_hop`: `0.0260`, `0.0208`, `0.0365`, `0.0104`

Decision:

Reject the simple mixed structural-plus-path target. It does not preserve the
TAC-117 multi-hop gain and still fails the aggregate/no-regression gates. The
mixed target is worse than pure path on aggregate (`0.1771` vs `0.1792`) and
far below hybrid top-4 (`0.2448`).

Interpretation:

Unioning structural and path positives into one ranking target dilutes both
signals. The structural gate's useful behavior is not recovered by simply
adding more positive labels, and the path signal that helped multi-hop in
TAC-117 is lost. The next credible design should not retrain one scorer on a
broader OR target. It should preserve the hybrid-ranked top-4 scores and add a
small protected path-edge rerank/quota, or train a two-head scorer with separate
structural and path logits whose combination is constrained by no-regression
gates.

Next local direction:

Prototype a protected path-edge quota over the existing hybrid top-4 gate:
start from hybrid scores, reserve at most one slot for a path-scored edge only
when confidence is high, and otherwise keep the original hybrid ranking. This
tests whether the TAC-117 multi-hop signal can be added without erasing the
structural backbone.

## TAC-119 Protected Path-Quota Hybrid Sparse Write Gate

Date: 2026-06-03

Purpose:

Test the TAC-118 follow-up while preserving the current hybrid sparse-write
backbone. Instead of retraining one scorer on structural/path union labels,
train separate hybrid and path-aligned gates, then reserve one of the top-4
write slots for a path-scored edge while leaving the other slots controlled by
the structural hybrid scorer.

Implemented:

- Added `ProtectedPathQuotaGate`.
- Added `protected_path_quota_content_write_mask`.
- Added `protected_path_quota_hybrid_top_4_q1`.
- Added `protected_path_quota_hybrid_top_4_q1_graph_path_k2`.
- Added focused tests for path-slot reservation and protected-quota sparse
  schedule evaluation.

Artifact:

`runs/benchmarks/content_protected_path_quota_local_2026_06_03`

Matrix:

The matrix covers 5 chunked-recall tasks x 3 seeds x 6 schedules at 120
training steps. It compares full update, query skip, hybrid top-4, pure
path-aligned top-4, protected path-quota top-4, and protected path-quota with
graph-path readout.

Aggregate results:

- `full_update`: carry `0.1844`, context update `1.0000`
- `query_skip`: carry `0.1844`, context update `1.0000`, query update `0.0000`
- `hybrid_ranked_boundary_top_4`: carry `0.2448`, delta vs full `+0.0604`,
  context update `0.1290`, content hit `0.1166`
- `path_aligned_hybrid_top_4`: carry `0.1792`, delta vs full `-0.0052`,
  context update `0.1290`, content hit `0.1117`
- `protected_path_quota_hybrid_top_4_q1`: carry `0.2323`, delta vs full
  `+0.0479`, context update `0.1290`, content hit `0.1143`
- `protected_path_quota_hybrid_top_4_q1_graph_path_k2`: carry `0.2323`,
  delta vs full `+0.0479`, chain fraction `0.2313`, context update `0.1290`

Task-level carry for full update, hybrid top-4, pure path, and protected
path-quota:

- `single_key`: `0.2344`, `0.3438`, `0.3385`, `0.3854`
- `multi_key`: `0.3177`, `0.4167`, `0.2812`, `0.3177`
- `delayed_query`: `0.2917`, `0.3542`, `0.1823`, `0.3750`
- `noisy_key`: `0.0521`, `0.0885`, `0.0573`, `0.0677`
- `multi_hop`: `0.0260`, `0.0208`, `0.0365`, `0.0156`

Decision:

Reject protected path-quota top-4 as the balanced A1 solution. It is the best
path-integration variant so far on aggregate (`0.2323` vs pure path `0.1792`
and mixed OR `0.1771`), and it improves `single_key` and `delayed_query`
above plain hybrid top-4. But it still fails the no-regression gates: aggregate
carry remains below hybrid top-4 (`0.2323` vs `0.2448`), `multi_key` falls back
to full-update level, `noisy_key` remains below hybrid, and `multi_hop` drops
below full update, hybrid, and pure path. Graph-path readout adds chain activity
(`0.2313` aggregate and `0.9740` on multi-hop) without improving carry.

Interpretation:

The path scorer is useful, but a fixed protected slot is too blunt. It improves
some direct recall settings by changing which edges survive, yet it steals
coverage from multi-key/noisy/multi-hop cases where the structural hybrid gate
was already making the least bad sparse-write tradeoff. The negative result is
now strong enough to stop spending local cycles on post-hoc path slots, union
targets, or graph traversal as standalone repairs.

Current local A1 position:

Keep `hybrid_ranked_boundary_top_4` as the best supported sparse-write
prototype. The balanced local solution has not yet been found. The next
credible direction should be model-integrated rather than another fixed
reranker: either a jointly learned write objective with task/context-aware
confidence, or a capability run that proves the TAC architecture can learn the
needed routing end-to-end. External Run 5B remains the required promotion gate,
but Kaggle execution is blocked by the weekly GPU quota.

## TAC-120 A1 Selector Upper-Bound Audit

Date: 2026-06-03

Purpose:

Decide whether to keep searching within the current A1 family of fixed sparse
write rules, path quotas, and graph/path readouts, or to close the local A1
variant loop and move to the next research track.

Method:

Use the TAC-119 matrix and compute an oracle task selector over the schedules
already tested in that corrected state-preserved run. This selector is not
deployable because it uses task identity after the fact. It is an upper-bound
diagnostic: if even this selector has small lift, then more fixed variants are
unlikely to produce the balanced solution.

Artifact:

`runs/benchmarks/content_protected_path_quota_local_2026_06_03/content_update_frequency_matrix.json`

Oracle selection over current A1 variants:

| Task | Best schedule | Best carry | Hybrid top-4 carry | Lift |
| --- | --- | ---: | ---: | ---: |
| `single_key` | `protected_path_quota_hybrid_top_4_q1` | `0.3854` | `0.3438` | `+0.0417` |
| `multi_key` | `hybrid_ranked_boundary_top_4` | `0.4167` | `0.4167` | `+0.0000` |
| `delayed_query` | `protected_path_quota_hybrid_top_4_q1` | `0.3750` | `0.3542` | `+0.0208` |
| `noisy_key` | `hybrid_ranked_boundary_top_4` | `0.0885` | `0.0885` | `+0.0000` |
| `multi_hop` | `path_aligned_hybrid_top_4` | `0.0365` | `0.0208` | `+0.0156` |

Aggregate upper bound:

- Oracle task selector over current variants: `0.2604`
- Plain `hybrid_ranked_boundary_top_4`: `0.2448`
- Upper-bound lift: `+0.0156`

Decision:

Close the current local A1 variant loop. A non-deployable oracle selector over
the tested variants only adds `+0.0156` aggregate carry, and each gain comes
from a different specialized policy. That is too small and too fragmented to
justify more fixed-rule variants as the route to a perfect balanced solution.

Track-level conclusion:

For local inference-speed research, `hybrid_ranked_boundary_top_4` remains the
best evidence-backed sparse-write prototype. It achieves the intended write
reduction (`0.1290` context update fraction and no query writes) while retaining
the best balanced aggregate in the corrected A1 family. The remaining gap is no
longer a local fixed-gate tuning problem; it should be addressed through the
broader plan's next tracks:

- A2: integrate `query_skip` and sparse prefill writes behind a single
  `write_policy` surface.
- A4: formalize speed measurement so prefill/decode gains are reported against
  a stable baseline.
- C: retry external Run 5B when Kaggle GPU quota resets, because architecture
  promotion still depends on an end-to-end capability-plus-specialization gate.

## TAC-121 A2 Write-Policy Integration Prototype

Date: 2026-06-03

Purpose:

Implement the Track A2 integration prototype from the pasted plan: make
`query_skip` and sparse/event-style prefill writes compose behind one
write-policy surface instead of requiring callers to coordinate scattered
`update_content_memory` flags and masks manually.

Implemented:

- Added `ContentWritePolicy`.
- Added public TAC forward support for `write_policy`.
- Added `ContentWritePolicy.MASKED_PREFILL_QUERY_SKIP`, which writes during
  multi-token prefill using the supplied `content_write_mask` and skips
  content-memory writes during single-token decode/query calls.
- Exported `ContentWritePolicy` from `tac_transformer`.
- Added `write_policy` metadata to the efficiency research variants.
- Added `masked_prefill_query_skip` to `EFFICIENCY_RESEARCH_VARIANTS`.
- Added `write_policy_from_settings` to the research benchmark harness so
  sequence and decode efficiency evaluations pass a single policy into the
  model call.

Focused tests:

- `test_masked_prefill_query_skip_write_policy_composes_sparse_prefill_and_decode_skip`
- `test_invalid_content_write_policy_is_rejected`
- `test_efficiency_variants_expose_write_policy_surface`

Plan-scale synthetic validation:

Ran a 512-token prefill plus 64-token single-token decode probe on a tiny TAC
model with a deterministic sparse prefill mask standing in for event-driven
prediction-error writes.

Result:

- Prefill under `write_policy=masked_prefill_query_skip` exactly matched the
  explicit `content_write_mask` path for content cues, values, and mask.
- Decode ran for 64 single-token steps with no content cue/value/mask mutation.
- Model parameters were unchanged during decode.
- Identity routing activations remained present during decode.
- Sparse prefill write fraction was `0.1761`; all 32 content slots were filled.

Decision:

Accept A2 as a local integration prototype. This does not claim a final serving
API or a KV-cache speedup, but it proves the two A2 speed mechanisms compose at
the public model boundary and in the research benchmark harness. The next local
speed step should be A4: establish a stable benchmark harness/report for
prefill, carried-query, and decode throughput against TAC dense/update-disabled
and vanilla baselines.

## TAC-122 A4 Local Inference Benchmark Harness

Date: 2026-06-03

Purpose:

Complete the Track A4 local benchmark requirement from the pasted plan: measure
time-to-first-token/prefill throughput, sustained single-token decode
throughput, peak memory, and ratios against a vanilla transformer of comparable
backbone size.

Implemented:

- Extended `kaggle/benchmark_inference.py` with peak-memory fields:
  `prefill_peak_memory_bytes`, `carried_query_peak_memory_bytes`, and
  `decode_peak_memory_bytes`.
- CUDA runs use `torch.cuda.max_memory_allocated`.
- CPU runs use `tracemalloc` as a dependency-free local allocation proxy. This
  is not a full native tensor RSS measurement, but it gives local runs a stable
  peak field and keeps CUDA memory exact when available.
- Switched benchmark decode calls to `write_policy=ContentWritePolicy.QUERY_SKIP`,
  aligning the A4 decode path with the A2 policy surface.
- Updated Markdown output with mean prefill/decode peak memory and per-row
  prefill/decode peak memory.
- Added focused test coverage for CPU peak-memory reporting.

Artifact:

`runs/benchmarks/inference_profile_a4_local_2026_06_03`

Command:

```powershell
python kaggle/benchmark_inference.py --output-dir runs/benchmarks/inference_profile_a4_local_2026_06_03 --seq-lens 16 64 --variants vanilla_matched base_program_memory content_addressed_k1 current_best --content-store-sizes 8 --batch-size 2 --decode-steps 8 --warmup 1 --iters 2 --vocab-size 64 --d-model 32 --n-layers 1 --n-programs 8 --device cpu
```

Summary results:

| Variant | Mean prefill vs vanilla | Mean carried-query vs vanilla | Mean decode vs vanilla | Mean prefill peak | Mean decode peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vanilla_matched` | `1.0000` | `1.0000` | `1.0000` | `3.5 KiB` | `6.0 KiB` |
| `base_program_memory` | `1.8174` | `0.8150` | `0.4683` | `18.1 KiB` | `21.5 KiB` |
| `content_addressed_k1` | `1.6548` | `0.5143` | `0.3445` | `17.9 KiB` | `20.5 KiB` |
| `current_best` | `2.7550` | `1.7583` | `0.7968` | `17.9 KiB` | `20.5 KiB` |

Decision:

Accept A4 as locally satisfied: the repository now has a benchmark harness that
reports prefill, carried-query, decode, peak-memory fields, and vanilla ratios,
and the harness has been run locally to establish a compact baseline.

Interpretation:

The CPU smoke baseline is directionally useful, not a final performance claim.
`current_best` has the best aggregate local throughput profile among tested TAC
variants, with strong prefill and carried-query ratios versus vanilla. Decode
still trails vanilla on average (`0.7968x`), and TAC peak-memory proxy values
are higher than vanilla. This supports the pasted plan's ordering: A3
identity-state/KV-cache work is the next speed bottleneck to attack, because
A2 has removed content writes from decode but the identity/program field is
still recomputed on every generated token.

Next local direction:

Proceed to A3 with an identity-state cache prototype. Acceptance should require
identical logits/states under no program switch, explicit invalidation when a
switch is detected, and a local decode-speed comparison against the A4 harness.

## TAC-123 A3 Identity Decode Cache Feasibility Diagnostic

Date: 2026-06-03

Purpose:

Start Track A3 by measuring whether identity-field caching is worth pursuing
and whether a simple "reuse routed identity output until program switch" policy
is safe enough to prototype.

Implemented:

- Added `experiments/benchmark_identity_decode_cache.py`.
- The script times decode with `write_policy=ContentWritePolicy.QUERY_SKIP`.
- It instruments `IdentityFieldLayer.forward` calls during decode.
- It reports identity-field decode time share, theoretical speedup ceiling if
  identity-field work were removed, and program assignment switch fraction.
- Added focused tests for speedup-ceiling and switch-rate summarization.

Artifact:

`runs/benchmarks/identity_decode_cache_local_2026_06_03`

Command:

```powershell
python experiments/benchmark_identity_decode_cache.py --output-dir runs/benchmarks/identity_decode_cache_local_2026_06_03 --seq-len 64 --decode-steps 64 --batch-size 2 --iters 3 --warmup 1 --vocab-size 64 --d-model 32 --n-layers 1 --n-programs 8 --device cpu
```

Results:

- Decode throughput: `335.67` tok/s
- Identity-field time fraction: `0.5802`
- Identity-cache speedup ceiling: `2.3820x`
- Identity-field calls per iteration: `64.0`
- Program switch fraction: `0.8665`
- Program stable fraction: `0.1335`
- Diagnostic decision: `diagnostic_only`

Decision:

Do not implement a naive whole-identity-output cache. The speed opportunity is
large: identity-field work accounts for about `58%` of local decode time, so
identity caching could theoretically clear the A3 `>=20%` speedup target. But
program assignments switch on `86.65%` of adjacent decode comparisons in this
profile, so a "reuse until route switch" cache would invalidate almost every
step and would not reliably preserve behavior.

Implementation-ready next step:

Build a lower-level exact decode cache instead of a frozen identity-output
cache:

- cache per-layer identity state tensors from prefill/decode (`stability`,
  program memory tiers, ages, content stores)
- reuse content-memory state under A2 `QUERY_SKIP`
- avoid recomputing store maintenance when `update_content_memory=False`
- keep token-dependent routing logits/program weights fresh for correctness
- profile whether bypassing store/reconsolidation/allocation work, while still
  computing routing for the current token, delivers meaningful decode speedup
- only consider whole-output reuse later if a learned stability guard shows low
  switch/drift on real decode data

This converts A3 from "cache routed program embedding wholesale" into a safer
two-stage plan: first exact state-maintenance bypass, then optional approximate
whole-output reuse with explicit drift guards.

## TAC-124 A3 Exact Decode State-Maintenance Bypass

Date: 2026-06-03

Purpose:

Implement the safer A3 path from TAC-123: keep current-token routing fresh for
correctness, but skip recurrent identity-state maintenance during single-token
decode when serving uses an explicit decode-skip write policy.

Implemented:

- Added `ContentWritePolicy.DECODE_STATE_SKIP`.
- Multi-token prefill under this policy keeps normal dense state updates.
- Single-token decode under this policy skips content-store writes,
  program-memory writes, memory allocation, reconsolidation, memory-tier
  updates, program age/frequency updates, and engram-store updates.
- The bypass preserves incoming recurrent state exactly, including optional
  `None` state fields, while still using freshly computed token routing and
  program context for logits.
- Extended `experiments/benchmark_identity_decode_cache.py` to compare
  `query_skip` and `decode_state_skip` from the same warmed model/context.
- Added focused tests proving first-token decode logits match `query_skip`,
  returned recurrent state is unchanged, and decode-update helper policy logic
  treats `decode_state_skip` as a no-update decode policy.

Artifact:

`runs/benchmarks/identity_decode_state_skip_local_2026_06_03`

Command:

```powershell
python experiments/benchmark_identity_decode_cache.py --output-dir runs/benchmarks/identity_decode_state_skip_local_2026_06_03 --seq-len 64 --decode-steps 64 --batch-size 2 --iters 3 --warmup 1 --device cpu
```

Results:

| Policy | Decode tok/s | Identity-field fraction | Cache speedup ceiling | Program switch fraction |
| --- | ---: | ---: | ---: | ---: |
| `query_skip` | `317.89` | `0.5819` | `2.3916x` | `0.8665` |
| `decode_state_skip` | `417.77` | `0.5331` | `2.1419x` | `0.8665` |

Decision:

Accept `decode_state_skip` as the current exact A3 serving policy prototype.
It improves local CPU decode throughput by about `31%` over `query_skip` in
the compact profile while preserving first-token logits and returned recurrent
state. It does not solve whole-output identity caching: the program switch
fraction remains `0.8665`, so approximate reuse of routed identity outputs is
still rejected without a stronger learned drift guard.

Next local direction:

Wire `decode_state_skip` into the A4 inference harness as an opt-in decode
variant, then compare against vanilla and `query_skip` across longer contexts
and CUDA when quota/resources are available. Keep A3 exact-state bypass as a
serving optimization candidate; leave approximate identity-output caching on
hold until routing/drift diagnostics become stable enough to guard it.

## TAC-125 A4 Decode-Policy Harness Integration

Date: 2026-06-03

Purpose:

Move `decode_state_skip` out of the standalone A3 diagnostic and into the main
inference benchmark so decode policy choices can be compared against vanilla,
BASE, and current-best TAC variants in one artifact.

Implemented:

- Added `--decode-policies` to `kaggle/benchmark_inference.py`.
- Added a `decode_policy` dimension to row output and Markdown tables.
- Summary keys now use `variant:policy` for TAC rows, while vanilla remains
  `vanilla_matched`.
- `decode_vs_base` now compares TAC rows against the BASE row with the same
  decode policy.
- Added focused test coverage for decode-policy parsing and profile labeling.

Artifact:

`runs/benchmarks/inference_profile_decode_state_skip_local_2026_06_03`

Command:

```powershell
python kaggle/benchmark_inference.py --output-dir runs/benchmarks/inference_profile_decode_state_skip_local_2026_06_03 --seq-lens 16 64 --variants vanilla_matched base_program_memory current_best --content-store-sizes 8 --batch-size 2 --decode-steps 8 --warmup 1 --iters 2 --vocab-size 64 --d-model 32 --n-layers 1 --n-programs 8 --device cpu --decode-policies query_skip,decode_state_skip
```

Summary results:

| Profile | Mean decode vs vanilla | Mean decode vs BASE | Mean decode peak |
| --- | ---: | ---: | ---: |
| `vanilla_matched` | `1.0000` | `n/a` | `6.0 KiB` |
| `base_program_memory:query_skip` | `0.2967` | `1.0000` | `20.8 KiB` |
| `base_program_memory:decode_state_skip` | `0.2281` | `1.0000` | `19.9 KiB` |
| `current_best:query_skip` | `0.1606` | `0.7522` | `20.7 KiB` |
| `current_best:decode_state_skip` | `0.1980` | `1.0815` | `19.5 KiB` |

Decision:

Keep `decode_state_skip` as an opt-in decode policy for the current-best TAC
serving path, not a blanket default for every TAC variant. In this compact CPU
A4 run, `current_best:decode_state_skip` improves mean decode-vs-vanilla from
`0.1606` to `0.1980` and mean decode-vs-BASE from `0.7522` to `1.0815`, while
also lowering mean decode peak memory. The simpler `base_program_memory`
variant regresses under the same policy, so promotion should be variant-gated.

Next local direction:

Run a longer A4 policy matrix when practical, ideally on CUDA after the Kaggle
quota reset or on another GPU. The local CPU evidence is enough to keep
`decode_state_skip` in the serving candidate set, but not enough to claim final
throughput superiority over vanilla.

## TAC-126 B2 Route-and-Reconstruct Diagnostic

Date: 2026-06-03

Purpose:

Start Track B by answering the pasted plan's B2 question: did route-and-
reconstruct fail because routed programs are functionally interchangeable, or
because gradients do not reach the program/routing machinery?

Implemented:

- Added `experiments/benchmark_route_reconstruct_diagnostic.py`.
- The script trains a compact TAC plus route decoder with a light
  route-reconstruction objective.
- It evaluates counterfactual reconstruction per token by comparing the hidden
  reconstruction loss for the routed program against every other program.
- It reports routed-is-best fraction, routed-minus-best loss gap,
  other-minus-routed gap, best-program margin, and route/program gradient norms.
- Added focused tests for the counterfactual reconstruction statistics and
  diagnostic verdict classification.

Artifact:

`runs/benchmarks/route_reconstruct_diagnostic_local_2026_06_03`

Command:

```powershell
python experiments/benchmark_route_reconstruct_diagnostic.py --output-dir runs/benchmarks/route_reconstruct_diagnostic_local_2026_06_03 --steps 20 --seq-len 64 --batch-size 4 --eval-batches 4 --eval-batch-size 4 --vocab-size 512 --d-model 48 --n-layers 2 --n-programs 12 --device cpu
```

Results:

- Routed-is-best fraction: `0.0967`
- Mean routed reconstruction loss: `1.0554`
- Mean best-program reconstruction loss: `0.9634`
- Mean routed-minus-best loss gap: `0.0920`
- Mean other-minus-routed loss gap: `-0.0152`
- Max route/program gradient norm from route-reconstruction loss: `0.006445`
- Verdict: `routing_not_functionally_aligned`

Decision:

Keep iterating on Track B, but do not treat this as a pure gradient-flow bug.
The diagnostic shows nonzero route/program gradients, so the route-reconstruct
loss can reach program parameters. The stronger failure is functional
misalignment: the currently routed program is best for only about `9.7%` of
sampled token decisions, and the average best counterfactual program
reconstructs hidden state materially better than the routed program.

Implementation-ready next step:

Run B1 as a utility-aligned contrastive/routing refinement rather than a
generic program-separation loss:

- derive counterfactual best-program labels from the B2 diagnostic
- train a small auxiliary objective that pulls routed activations toward
  low-reconstruction-loss programs and away from hard-negative programs
- include hard negative mining and temperature annealing as requested in B1
- gate success on both mean off-diagonal program cosine `<0.85` and improved
  routed-is-best fraction versus the TAC-126 diagnostic baseline

## TAC-127 B1 Program Contrastive Refinement

Date: 2026-06-03

Purpose:

Test the pasted plan's B1 refinements after TAC-126 showed functional route
misalignment with live gradients. The goal is to see whether hard negatives,
task-conditioned utility labels, or temperature annealing can push program
memory cosine below `0.85` while improving routed program usefulness.

Implemented:

- Added `experiments/benchmark_program_contrastive_refinement.py`.
- Added four compact local variants:
  - `route_reconstruct_reference`
  - `hard_negative`
  - `task_conditioned`
  - `annealed_utility`
- The benchmark derives best-program labels from counterfactual reconstruction
  losses, then trains route probabilities toward those utility labels.
- Reports program-memory cosine, raw program-embedding cosine, routed-is-best
  fraction, and routed-minus-best reconstruction gap.
- Added focused tests for hard-negative loss behavior, annealed temperature,
  and B1 row summarization.

Artifact:

`runs/benchmarks/program_contrastive_refinement_local_2026_06_03`

Command:

```powershell
python experiments/benchmark_program_contrastive_refinement.py --output-dir runs/benchmarks/program_contrastive_refinement_local_2026_06_03 --steps 20 --seq-len 64 --batch-size 4 --eval-batches 4 --eval-batch-size 4 --vocab-size 512 --d-model 48 --n-layers 2 --n-programs 12 --device cpu
```

Results:

| Variant | Program-memory cosine | Routed is best | Routed-best gap | Pass |
| --- | ---: | ---: | ---: | --- |
| `route_reconstruct_reference` | `0.9167` | `0.0850` | `0.089851` | `False` |
| `hard_negative` | `0.9167` | `0.1680` | `0.084857` | `False` |
| `task_conditioned` | `0.9167` | `0.2754` | `0.052210` | `False` |
| `annealed_utility` | `0.9167` | `0.1133` | `0.063345` | `False` |

Decision:

Reject loss-only B1 promotion as the balanced solution. `task_conditioned` is
the best local refinement because it improves routed-is-best from the B2
baseline near `0.0967` to `0.2754` and cuts routed-minus-best reconstruction
gap from about `0.0920` to `0.0522`. But every variant leaves program-memory
cosine stuck at `0.9167`, missing the pasted B1 success criterion of `<0.85`.

Implementation-ready next step:

Treat useful route alignment and memory differentiation as separate levers:

- keep the task-conditioned utility objective as the best routing-alignment
  auxiliary candidate
- add a direct program-memory diversification mechanism, not just an embedding
  or route-probability loss
- candidate mechanisms: stronger `memory_separation_weight`, program-specific
  write-slot allocation, or a decorrelation penalty on updated `program_memory`
  tensors after writes
- gate the next B1/B3 attempt on both `program_memory_cosine < 0.85` and
  routed-is-best improvement over TAC-126

## TAC-128 Program-Memory Separation Probe

Date: 2026-06-03

Purpose:

Follow the TAC-127 finding that task-conditioned utility alignment improves
routed usefulness but does not fix the `0.9167` program-memory cosine plateau.
This probe asks whether simply adding direct program-memory separation loss to
the best task-conditioned variant is enough.

Implemented:

- Extended `experiments/benchmark_program_contrastive_refinement.py` with:
  - `task_conditioned_memsep_0p1`
  - `task_conditioned_memsep_1p0`
- Added the model's own `output.aux.losses["separation"]` to the training
  objective for those rows.
- Kept the corrected B1 gate on program-memory cosine, not raw embedding
  cosine.
- Added focused test coverage that the direct-memory-separation variants are
  present in the B1 variant surface.

Artifact:

`runs/benchmarks/program_contrastive_memsep_local_2026_06_03`

Command:

```powershell
python experiments/benchmark_program_contrastive_refinement.py --output-dir runs/benchmarks/program_contrastive_memsep_local_2026_06_03 --steps 20 --seq-len 64 --batch-size 4 --eval-batches 4 --eval-batch-size 4 --vocab-size 512 --d-model 48 --n-layers 2 --n-programs 12 --device cpu
```

Results:

| Variant | Program-memory cosine | Routed is best | Routed-best gap | Pass |
| --- | ---: | ---: | ---: | --- |
| `route_reconstruct_reference` | `0.9167` | `0.0850` | `0.089851` | `False` |
| `hard_negative` | `0.9167` | `0.1680` | `0.084857` | `False` |
| `task_conditioned` | `0.9167` | `0.2754` | `0.052210` | `False` |
| `annealed_utility` | `0.9167` | `0.1133` | `0.063345` | `False` |
| `task_conditioned_memsep_0p1` | `0.9167` | `0.4502` | `0.039961` | `False` |
| `task_conditioned_memsep_1p0` | `0.9167` | `0.3535` | `0.030467` | `False` |

Decision:

Reject direct loss-weight tuning as the full B1/B3 solution. The
`task_conditioned_memsep_0p1` row is the best route-utility candidate so far,
raising routed-is-best to `0.4502` and reducing the routed-best gap to
`0.0400`, but program-memory cosine remains exactly `0.9167` across all rows.

Implementation-ready next step:

Inspect and alter the memory-write/update path rather than adding more loss
weight. The next Track B ticket should instrument per-program write gates,
allocation masks, program age/frequency, and memory deltas during training.
If only a few programs receive meaningful writes, add program-specific
allocation pressure or anti-collapse write balancing before testing another
objective-only variant.

## TAC-129 Program-Memory Write/Allocation Diagnostic

Date: 2026-06-03

Purpose:

Instrument the Track B memory-collapse failure below the objective layer. TAC-128
showed that task-conditioned utility plus memory-separation loss improves
routed usefulness but leaves program-memory cosine pinned at `0.9167`. This
diagnostic asks whether the problem is concentrated/insufficient writes or a
collapsed update formulation.

Implemented:

- Added `experiments/benchmark_program_memory_write_diagnostic.py`.
- Compared three allocation/write variants under the best current
  task-conditioned memory-separation auxiliary:
  - `stability_task_memsep`
  - `creb_k1_task_memsep`
  - `creb_k2_task_memsep`
- The diagnostic reports program-memory cosine, dead-program fraction, selected
  load entropy/gini, write-frequency entropy/gini, memory norm distribution,
  age distribution, and per-program selected/write/norm vectors.
- Added tests for entropy/gini summarization, dead-program classification, and
  refusing to recommend low-cosine rows that are only low because programs are
  dead.

Artifact:

`runs/benchmarks/program_memory_write_diagnostic_local_2026_06_03`

Command:

```powershell
python experiments/benchmark_program_memory_write_diagnostic.py --output-dir runs/benchmarks/program_memory_write_diagnostic_local_2026_06_03 --steps 20 --seq-len 64 --batch-size 4 --eval-batches 4 --eval-batch-size 4 --vocab-size 512 --d-model 48 --n-layers 2 --n-programs 12 --device cpu
```

Results:

| Variant | Program-memory cosine | Dead frac | Write entropy | Selected entropy | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `stability_task_memsep` | `0.9167` | `0.0000` | `1.0000` | `0.9295` | `memory_update_collapsed_despite_broad_writes` |
| `creb_k1_task_memsep` | `0.0000` | `0.6667` | `0.5579` | `0.9671` | `memory_dead_or_underwritten` |
| `creb_k2_task_memsep` | `0.0139` | `0.5417` | `0.6451` | `0.9538` | `memory_dead_or_underwritten` |

Decision:

Do not promote the CREB allocation variants despite their low cosine. Their
low cosine is an artifact of dead or underwritten programs: dead fraction is
`0.6667` for `creb_k1` and `0.5417` for `creb_k2`. The default stability
allocation writes broadly (`write_frequency_entropy=1.0000`, `dead=0.0`) but
still produces collapsed program-memory vectors (`cosine=0.9167`).

The best current diagnosis is therefore:

- routing utility can be improved by task-conditioned objectives
- write coverage is broad under the default allocation
- memory vector updates are not program-specific enough, so broad writes still
  converge to highly similar program memories
- sparse CREB-style allocation reduces cosine by starving programs, which is
  not a valid solution

Implementation-ready next step:

Modify the program-memory update candidate itself. The next Track B prototype
should make `candidate_memory` program-specific, for example by combining the
pooled hidden update with each program embedding or routed program identity
before blending. Acceptance should require:

- no dead/underwritten programs (`dead_program_fraction < 0.2`)
- `program_memory_cosine < 0.85`
- routed-is-best fraction remains above the TAC-126 baseline and preferably
  near the TAC-128 best (`>=0.45`)
- full regression gates pass

## TAC-130 Program-Conditioned Memory Update Prototype

Date: 2026-06-03

Purpose:

Resolve the Track B tension exposed by TAC-126 through TAC-129: route-utility
objectives improved routed usefulness but left program memory collapsed, while
sparse CREB allocation lowered cosine by starving programs. TAC-130 tests
whether the update candidate itself should be program-specific.

Implemented:

- Added `TACConfig.program_memory_update_type` with opt-in values:
  - `shared`, preserving the existing broadcast pooled-hidden update
  - `program_conditioned`, which combines pooled hidden state with each program
    embedding through a trainable update projection
- Preserved default behavior and parameter counts for existing configs.
- Updated the parameter estimator and added focused model tests for invalid
  config rejection, program-specific candidate shapes, gradient flow, and
  parameter accounting.
- Extended the write/allocation diagnostic with program-conditioned stability
  and CREB top-k variants.
- Tightened the write diagnostic's viability gate so low-cosine rows are only
  recommendable when `dead_program_fraction < 0.2`.
- Extended the B1 refinement harness with
  `program_conditioned_creb_k6_task_memsep` and ran it against the previous
  best route-utility row.

Artifacts:

`runs/benchmarks/program_conditioned_memory_budget_sweep_local_2026_06_03`

`runs/benchmarks/program_conditioned_b1_balance_local_2026_06_03`

Commands:

```powershell
python experiments/benchmark_program_memory_write_diagnostic.py --output-dir runs/benchmarks/program_conditioned_memory_budget_sweep_local_2026_06_03 --variants program_conditioned_stability_task_memsep program_conditioned_creb_k2_task_memsep program_conditioned_creb_k4_task_memsep program_conditioned_creb_k6_task_memsep --steps 20 --seq-len 64 --batch-size 4 --eval-batches 4 --eval-batch-size 4 --d-model 48 --n-layers 2 --n-programs 12
python experiments/benchmark_program_contrastive_refinement.py --output-dir runs/benchmarks/program_conditioned_b1_balance_local_2026_06_03 --variants task_conditioned_memsep_0p1 program_conditioned_creb_k6_task_memsep --steps 20 --seq-len 64 --batch-size 4 --eval-batches 4 --eval-batch-size 4 --d-model 48 --n-layers 2 --n-programs 12
```

Write/allocation results:

| Variant | Program-memory cosine | Dead frac | Write entropy | Verdict |
| --- | ---: | ---: | ---: | --- |
| `program_conditioned_stability_task_memsep` | `0.9072` | `0.0000` | `1.0000` | `memory_update_collapsed_despite_broad_writes` |
| `program_conditioned_creb_k2_task_memsep` | `0.0137` | `0.5208` | `0.6733` | `memory_dead_or_underwritten` |
| `program_conditioned_creb_k4_task_memsep` | `0.0824` | `0.1875` | `0.8781` | `memory_diversification_viable` |
| `program_conditioned_creb_k6_task_memsep` | `0.2064` | `0.0833` | `0.9294` | `memory_diversification_viable` |

B1 balance results:

| Variant | Program-memory cosine | Routed is best | Routed-best gap | Pass |
| --- | ---: | ---: | ---: | --- |
| `task_conditioned_memsep_0p1` | `0.9167` | `0.3535` | `0.052607` | `False` |
| `program_conditioned_creb_k6_task_memsep` | `0.2057` | `0.3359` | `0.049223` | `True` |

Decision:

Promote `program_conditioned_creb_k6_task_memsep` as the current balanced local
Track B prototype. It is the first tested row to pass the corrected local B1
gate: program-memory cosine is well below `0.85`, dead-program fraction is below
`0.2`, write-frequency entropy remains high, and routed-is-best remains above
the TAC-126 baseline and the B1 local threshold.

Tradeoff:

This is not the absolute best route-alignment row. TAC-128's
`task_conditioned_memsep_0p1` reached routed-is-best `0.4502` in the broader B1
matrix, while the balanced top-6 prototype reaches `0.3359` in the head-to-head
run. The accepted balance is justified because the previous route-best row left
memory cosine pinned at `0.9167`, while the new row materially fixes memory
collapse without underwriting most programs.

Implementation-ready next step:

Keep `program_memory_update_type="program_conditioned"` opt-in and promote
CREB top-6 allocation as the next Track B candidate for larger local and GPU
validation. Before making it a default, run the full Run 5B external matrix when
GPU quota is available and add a longer-seed local stability sweep to confirm
that dead-program fraction stays below `0.2` while routed-is-best does not
regress toward the TAC-126 baseline.

## TAC-131 Track C Vanilla Baseline Pipeline and Failure Protocol

Date: 2026-06-03

Purpose:

Complete the contingency path from the pasted plan. If Run 5/5B does not recover
capability, the project now has a Kaggle-ready vanilla transformer baseline that
uses the same prepared JSONL corpus path, tokenizer/vocab, sequence length,
training token budget, eval cadence, and artifact layout before TAC is blamed.

Implemented:

- Added `kaggle/train_vanilla_baseline.py`.
- Supports `same_backbone` and `parameter_matched` comparison modes:
  - `same_backbone` keeps TAC `d_model`, `n_layers`, `n_heads`, and seq length.
  - `parameter_matched` uses `parameter_matched_baseline_config` to move vanilla
    closer to the TAC parameter budget.
- Uses the same local/Kaggle prepared JSONL discovery convention as the TAC
  training scripts.
- Writes `run_manifest.json`, `metrics.jsonl`, `last.pt`, `best.pt`, and
  `final_summary.json`.
- Added tests that lock the fairness semantics for same-backbone and
  parameter-matched modes.
- Added `docs/run5_failure_protocol.md` with Gate 1 criteria, required vanilla
  comparisons, failure labels, and minimum artifacts to archive.
- Added the vanilla trainer and failure protocol to the Kaggle bundle builder.

Artifact:

`runs/benchmarks/vanilla_baseline_100step_smoke_local_2026_06_03`

Command:

```powershell
python kaggle/train_vanilla_baseline.py --output-dir runs/benchmarks/vanilla_baseline_100step_smoke_local_2026_06_03 --scale smoke --baseline-mode same_backbone --steps 100 --batch-size 2 --grad-accum-steps 1 --eval-every 50 --eval-batches 1 --eval-batch-size 2 --checkpoint-every 50 --log-every 50 --device cpu --precision fp32 --max-seconds 900 --stop-buffer-seconds 0
```

Smoke result:

- Completed `100/100` steps.
- Wrote valid `run_manifest.json` and `final_summary.json`.
- Wrote `last.pt` and `best.pt`.
- Best eval loss: `5.3748`.
- Latest eval loss: `5.5753`.
- Latest eval accuracy: `0.0703`.
- Training throughput: `3528.70` tokens/sec on local CPU.

Decision:

Track C is locally satisfied. A vanilla baseline can now be launched on Kaggle
with a single script/flag path before declaring TAC at fault. The failure
protocol requires both `same_backbone` and `parameter_matched` vanilla runs for
any serious Run 5/5B failure decision.

Implementation-ready next step:

When Kaggle GPU quota is available, launch Run 5B first. If it fails Gate 1 or
is inconclusive, run:

```bash
python kaggle/train_vanilla_baseline.py --preset run5b_capability --scale base --baseline-mode same_backbone --steps 20000 --warmup-steps 500 --batch-size 12 --grad-accum-steps 3 --eval-every 500 --eval-batches 8 --checkpoint-every 250 --output-dir /kaggle/working/vanilla_run5b_same_backbone --device auto --precision fp16 --max-seconds 30600 --stop-buffer-seconds 1200
python kaggle/train_vanilla_baseline.py --preset run5b_capability --scale base --baseline-mode parameter_matched --steps 20000 --warmup-steps 500 --batch-size 12 --grad-accum-steps 3 --eval-every 500 --eval-batches 8 --checkpoint-every 250 --output-dir /kaggle/working/vanilla_run5b_parameter_matched --device auto --precision fp16 --max-seconds 30600 --stop-buffer-seconds 1200
```

## TAC-142 External Run 5B Program-Conditioned CREB Validation

Date: 2026-06-04

Purpose:

Validate the local program-conditioned CREB top-6 TAC candidate against completed
external same-backbone and parameter-matched vanilla baselines, using capability,
memory-health, optimizer-health, selected-route MI, and category-conditioned
knockout gates.

External baselines:

- Same-backbone vanilla 20k: best eval loss `0.1092`, latest eval accuracy
  `0.9530`, train tokens `183,600,000`.
- Parameter-matched vanilla 20k: best eval loss `0.1048`, latest eval accuracy
  `0.9549`, train tokens `183,600,000`.

TAC run:

- Kernel: `jeffkolo/tac-run5b-program-conditioned-creb-k6-20k`.
- Final run reached `11021/20000` steps and stopped for time, with latest eval
  loss `0.1525`, latest eval accuracy `0.9423`, and eval program-memory cosine
  `0.0025`.
- The fair-token comparison point is step `10000`, because it saw exactly
  `183,600,000` training tokens, matching both vanilla baselines.

Fair-token step-10000 result:

| Metric | Value |
| --- | ---: |
| Eval loss | `0.1672` |
| Eval accuracy | `0.9414` |
| Same-backbone loss gap | `0.0580` |
| Parameter-matched loss gap | `0.0624` |
| Program-memory cosine | `0.0031` |
| Selected-route MI | `0.2821` |
| Max knockout loss delta | `0.3401` |
| Max knockout selectivity span | `0.1868` |

Decision:

Promote the following as the current capability-preserving specialization
solution candidate:

```text
program_memory_update_type=program_conditioned
memory_allocation_type=creb
memory_allocation_k=6
memory_separation_weight=0.1
routing_type=base_semantic
routing_top_k=2
category_route_objective=mi
category_route_weight=0.1
```

Important caveat:

Do not use the step-6000 `best.pt` checkpoint as the specialization checkpoint.
It has better eval loss (`0.1451`) but sequence-level top-program MI is `0.0`.
Use the step-10000 specialization checkpoint for the promoted solution.

Artifacts:

- `runs/benchmarks/external_run5b_program_conditioned_creb_k6_step10000_fair_token_validation_2026_06_04/RESULTS.md`
- `runs/analysis/tac_run5b_program_conditioned_creb_k6_step10000_specialization_2026_06_04/program_specialization.json`
- `docs/program_specialization_solution_research.md`

## TAC-143 TAC-Control-v1 Next-Stage Research Automation

Date: 2026-06-04

Purpose:

Convert the Run 5B promote decision into a repeatable research contract so the
next phase is driven by artifact checks rather than prose memory.

Implemented:

- Added `tac_transformer.research_plan`.
- Added `experiments/advance_tac_research_plan.py`.
- Exported the new public research-plan API from `tac_transformer`.
- Added `train_best_tac_agentic` CLI controls for `memory_read_type`,
  `content_read_steps`, and `content_read_gate_type` so Phase B seed commands
  can reproduce the frozen content-read path.
- Added the new research-plan module, CLI, and contract docs to the Kaggle
  agentic training bundle so package imports still work after extraction.
- Added focused unit coverage for the Phase A freeze audit, Phase B seed plan,
  Phase D benchmark protocol, and generated Markdown contract.

Generated artifact:

`runs/benchmarks/tac_control_v1_next_stage_2026_06_04`

The CLI wrote:

- `phase_a_freeze.json`
- `phase_b_replication_plan.json`
- `phase_d_benchmark_protocol.json`
- `tac_next_stage_research_contract.json`
- `RESULTS.md`

It also wrote:

- `docs/tac_control_v1_research_contract.md`

Phase A freeze decision:

`freeze_ready`

Resolved gaps:

- `content_addressed_store`: resolved by manifest config plus nonzero
  `content_addressed_hit=0.17204046994447708` and
  `content_synthesis_gate=0.5098107680678368`.
- `identity_first_path`: resolved by manifest config
  `identity_attention_type=identity_first`.
- `fair_token_checkpoint`: resolved at step `10000` with `183,600,000`
  tokens seen.
- `specialization_signal`: resolved with selected-route MI
  `0.28210237165562335`, program-memory cosine `0.00248082383768633`,
  and max knockout loss delta `0.34013200163220364`.

Open gaps carried forward:

- `multi_hop`: check whether BASE routing still dominates multi-hop behavior.
- `long_context`: check whether specialization transfers beyond short-context
  training.
- `seed_stability`: run the Phase B seed replicas before claiming stable
  program identities.
- `decode_economics`: pair capability deltas with wall-clock, tokens/s, and
  cost-normalized scores because TAC still carries the decode penalty.

Phase B contract:

Run seeds `11`, `23`, and `37` with the frozen TAC-Control-v1 config:

```text
identity_attention_type=identity_first
memory_read_type=content_addressed
content_read_steps=2
content_read_gate_type=synthesis
program_memory_update_type=program_conditioned
memory_allocation_type=creb
memory_allocation_k=6
memory_separation_weight=0.1
routing_type=base_semantic
routing_top_k=2
category_route_objective=mi
category_route_weight=0.1
```

Seed success gates:

- `program_memory_cosine <= 0.25`
- `selected_route_mi >= 0.15`
- `max_knockout_loss_delta >= 0.05`
- `eval_accuracy >= 0.93`

Phase D contract:

Capability claims require TAC to beat the parameter-matched vanilla control on
memory-intensive and agentic benchmarks, with `tac_shuffled_state` and
`tac_base_routing_ablation` controls included. The first task IDs are:

- `multi_hop_chain_retrieval`
- `long_context_retrieval_4096`
- `episodic_fact_update`
- `tool_selection`
- `delayed_goal_binding`

Verification:

- `python -m unittest tests_py.test_tac_research_plan -v`
- `python -m unittest tests_py.test_tac_transformer.TACTransformerArchitectureTest.test_train_best_tac_agentic_accepts_content_read_flags -v`
- `python -m unittest tests_py.test_tac_transformer.TACTransformerArchitectureTest.test_agentic_training_bundle_imports_after_extraction -v`
- `python experiments/advance_tac_research_plan.py`
- `python -m py_compile experiments/advance_tac_research_plan.py tac_transformer/research_plan.py`

## TAC-144 TAC-Control-v1 Phase B External Seed Launch

Date: 2026-06-04

Purpose:

Move from the Phase A freeze contract into actual Phase B replication by
launching independent TAC-Control-v1 seeds on Kaggle.

Refreshed code dataset:

- `jeffkolo/tac-run5b-capability-code-2026-06-04`
- Built from `runs/kaggle_run5b_code_jeffkolo_2026_06_04`.
- Includes the trainer content-read flags, `tac_transformer.research_plan`,
  `experiments/advance_tac_research_plan.py`, and
  `docs/tac_control_v1_research_contract.md`.

Staged kernels:

`runs/kaggle_tac_control_v1_phase_b_2026_06_04`

| Seed | Kernel | Status |
| ---: | --- | --- |
| 11 | `jeffkolo/tac-control-v1-phase-b-seed-11-20k` | `KernelWorkerStatus.RUNNING` |
| 23 | `jeffkolo/tac-control-v1-phase-b-seed-23-20k` | `KernelWorkerStatus.RUNNING` |
| 37 | `jeffkolo/tac-control-v1-phase-b-seed-37-20k` | staged, not pushed |

Seed 37 blocker:

Kaggle rejected the seed-37 push with:

```text
Maximum batch GPU session count of 2 reached.
```

Continuation command:

```powershell
python experiments\monitor_phase_b_kaggle_seeds.py --push-missing-when-slot --pull-complete
```

The monitor writes:

`runs/kaggle_results/tac_control_v1_phase_b_2026_06_04/phase_b_kaggle_status.json`

It checks all staged seed statuses, pushes missing kernels only when a GPU slot
is available, and pulls outputs for completed kernels.

Phase B aggregation:

- Added `summarize_phase_b_seed_result`.
- Added `aggregate_phase_b_seed_results`.
- Added `format_phase_b_seed_results_markdown`.
- Added `experiments/aggregate_phase_b_seed_results.py`.

The aggregate CLI writes:

- `runs/benchmarks/tac_control_v1_phase_b_2026_06_04/phase_b_seed_results.json`
- `runs/benchmarks/tac_control_v1_phase_b_2026_06_04/RESULTS.md`

Current aggregate decision:

`pending`

Reason:

No completed Phase B seed `final_summary.json` artifacts have been pulled yet.
The aggregator is deliberately strict about incomplete evidence: if a seed has
step-10000 accuracy, memory cosine, and selected-route MI but no category
knockout evidence, its status becomes `pending_knockout` rather than `pass`.

Codex automation:

- `tac-phase-b-monitor`
- Heartbeat attached to this thread.
- Re-runs the monitor and aggregate CLI every 30 minutes so seed 37 can be
  pushed when one of the two running Kaggle GPU sessions frees, completed seed
  outputs can be pulled, and Phase B evidence can be updated without relying on
  manual memory.

2026-06-04 heartbeat update:

- Seeds 11 and 23 completed on `jeffkolo` and were pulled into
  `runs/kaggle_results/tac_control_v1_phase_b_2026_06_04`.
- The Phase B aggregate currently reports `fail`: seed 11 failed the
  selected-route-MI hard gate at the fair step-10000 checkpoint, while seed 23
  passes accuracy, program-memory cosine, and route-MI but remains
  `pending_knockout`.
- The final seed was rerouted to the user-supplied Kaggle credential at
  `C:\Users\warit\Downloads\kaggle.json`. The local monitor now supports a
  row-specific `kaggle_config_dir` plus isolated `USERPROFILE` so seed 37 can
  use the `eweewee2` account without disturbing completed `jeffkolo` seed
  artifacts.
- Uploaded refreshed code dataset
  `eweewee2/tac-run5b-capability-code-2026-06-04` from
  `runs/kaggle_run5b_code_eweewee2_2026_06_04`; the data dataset
  `eweewee2/tac-run5b-capability-data-2026-06-03` already existed.
- Pushed `eweewee2/tac-control-v1-phase-b-seed-37-20k`. Version 1 launched
  before the private dataset reroute was fixed; version 2 is the usable run and
  attaches both `eweewee2` datasets. Latest monitor status:
  `KernelWorkerStatus.RUNNING`.
- Tightened `experiments/run_phase_d_benchmark_matrix.py` so the matrix emits
  `blocked_by_phase_b` without checkpoint inference whenever the Phase B
  aggregate says `ready_for_phase_d=false`.

Verification:

- `python -m unittest tests_py.test_tac_research_plan.TACResearchPlanTests.test_phase_b_kaggle_kernel_uses_frozen_seed_command -v`
- `python experiments/stage_phase_b_kaggle_seeds.py`
- `python -m py_compile runs/kaggle_tac_control_v1_phase_b_2026_06_04/seed_11/run_tac_control_v1_phase_b.py runs/kaggle_tac_control_v1_phase_b_2026_06_04/seed_23/run_tac_control_v1_phase_b.py runs/kaggle_tac_control_v1_phase_b_2026_06_04/seed_37/run_tac_control_v1_phase_b.py`
- `python experiments/monitor_phase_b_kaggle_seeds.py --push-missing-when-slot --pull-complete`
- `python experiments/aggregate_phase_b_seed_results.py`
- `npm test` passed with JS tests plus 262 Python discovery tests after the Phase B aggregation additions.

## TAC-145 TAC-Control-v1 Phase D Benchmark Gate Automation

Date: 2026-06-04

Purpose:

Prepare the Phase D consequence gate while Phase B external seed runs are still
running, so benchmark rows can be judged automatically once Phase B clears.

Implemented:

- `aggregate_phase_d_benchmark_results`
- `format_phase_d_benchmark_results_markdown`
- `experiments/aggregate_phase_d_benchmarks.py`

Phase D aggregate artifacts:

- `runs/benchmarks/tac_control_v1_phase_d_2026_06_04/phase_d_benchmark_results.json`
- `runs/benchmarks/tac_control_v1_phase_d_2026_06_04/RESULTS.md`

Current decision:

`blocked_by_phase_b`

Reason:

Phase B has not yet produced enough complete passing seed evidence. The Phase D
gate also has no benchmark rows yet, so both required families are missing:

- `memory_intensive`
- `agentic`

Decision semantics:

- Phase D is `blocked_by_phase_b` until the Phase B aggregate reports
  `ready_for_phase_d=true`.
- Once unblocked, Phase D requires TAC-Control-v1 to beat
  `parameter_matched_vanilla` on the required memory-intensive and agentic task
  families.
- Missing TAC or parameter-matched vanilla evidence produces `pending`, not
  `pass`.

Heartbeat update:

`tac-phase-b-monitor` now runs:

```powershell
python experiments\monitor_phase_b_kaggle_seeds.py --push-missing-when-slot --pull-complete
python experiments\aggregate_phase_b_seed_results.py
python experiments\aggregate_phase_d_benchmarks.py
```

Verification:

- `python -m unittest tests_py.test_tac_research_plan -v`
- `python -m py_compile experiments/aggregate_phase_d_benchmarks.py tac_transformer/research_plan.py tests_py/test_tac_research_plan.py`
- `python experiments/aggregate_phase_d_benchmarks.py`
- `npm test` passed with JS tests plus 264 Python discovery tests after the Phase D gate additions.

## TAC-146 TAC-Control-v1 Phase C Identity Stability Automation

Date: 2026-06-04

Purpose:

Turn the seed-stability question into an automated gate instead of leaving it
as an informal Phase C note.

Implemented:

- `summarize_phase_c_identity_seed`
- `aggregate_phase_c_identity_stability_results`
- `format_phase_c_identity_stability_markdown`
- `experiments/aggregate_phase_c_identity_stability.py`
- `program_memory_summary` in `kaggle/analyze_program_specialization.py`

Phase C aggregate artifacts:

- `runs/benchmarks/tac_control_v1_phase_c_2026_06_04/phase_c_identity_stability.json`
- `runs/benchmarks/tac_control_v1_phase_c_2026_06_04/RESULTS.md`

Current decision:

`blocked_by_phase_b`

Reason:

Phase C requires completed Phase B seed evidence before identity stability can
be assessed. No completed Phase B seed `program_specialization.json` artifacts
have been pulled yet.

Decision semantics:

- Phase C is `blocked_by_phase_b` until Phase B reports
  `ready_for_phase_d=true`.
- Once unblocked, Phase C requires at least two complete seed profiles.
- Identity stability only passes when cross-seed program-role alignment meets
  the minimum similarity threshold and route, memory, and knockout components
  are all present.

Kaggle dataset refresh:

- Rebuilt `runs/kaggle_agentic_training_bundle/best-tac-agentic-training-bundle.zip`.
- Refreshed `jeffkolo/tac-run5b-capability-code-2026-06-04`.
- Kaggle file listing confirms
  `best-tac-agentic-training-bundle/experiments/aggregate_phase_c_identity_stability.py`
  is visible in the refreshed dataset.

Heartbeat update:

`tac-phase-b-monitor` now runs:

```powershell
python experiments\monitor_phase_b_kaggle_seeds.py --push-missing-when-slot --pull-complete
python experiments\aggregate_phase_b_seed_results.py
python experiments\aggregate_phase_c_identity_stability.py
python experiments\aggregate_phase_d_benchmarks.py
```

Verification:

- `python -m py_compile tac_transformer\research_plan.py tac_transformer\__init__.py kaggle\analyze_program_specialization.py kaggle\make_agentic_training_bundle.py experiments\aggregate_phase_c_identity_stability.py`
- `python -m unittest tests_py.test_tac_research_plan -v`
- `python -m unittest tests_py.test_tac_transformer.TACTransformerArchitectureTest.test_program_specialization_analysis_reports_attribution_mi_and_knockouts -v`
- `python experiments\aggregate_phase_c_identity_stability.py`
- Phase B/C/D heartbeat command stack passed.
- `npm test` passed with JS tests plus 267 Python discovery tests after the Phase C contract additions.

## TAC-147 TAC-Control-v1 Phase D Benchmark Suite Harness

Date: 2026-06-04

Purpose:

Move Phase D from an aggregate-only gate to a concrete shared benchmark suite
and scorer, so TAC and control predictions can be compared immediately after
Phase B clears.

Implemented:

- `tac_transformer.phase_d_benchmarks`
- `experiments/stage_phase_d_benchmark_suite.py`
- `experiments/score_phase_d_predictions.py`

Phase D task suite:

- `multi_hop_chain_retrieval`
- `long_context_retrieval_4096`
- `episodic_fact_update`
- `tool_selection`
- `delayed_goal_binding`

Artifacts:

- `runs/benchmarks/tac_control_v1_phase_d_suite_2026_06_04/phase_d_benchmark_manifest.json`
- `runs/benchmarks/tac_control_v1_phase_d_suite_2026_06_04/RESULTS.md`
- `runs/benchmarks/tac_control_v1_phase_d_suite_2026_06_04/seed_11/tasks.jsonl`
- `runs/benchmarks/tac_control_v1_phase_d_suite_2026_06_04/seed_23/tasks.jsonl`
- `runs/benchmarks/tac_control_v1_phase_d_suite_2026_06_04/seed_37/tasks.jsonl`

Current staged suite:

- Seeds: `11`, `23`, `37`
- Examples per task: `8`
- Total examples: `120`
- Context length target for long-context task: `4096`

Scoring semantics:

- Prediction rows require `example_id`, `control_id`, and `prediction`.
- Exact-match task accuracy becomes `primary_score`.
- Missing predictions count as incorrect and are reported.
- `tokens_per_second` and `wall_clock_seconds` are carried into the row so
  decode economics can be reported beside capability.

Kaggle dataset refresh:

- Refreshed `jeffkolo/tac-run5b-capability-code-2026-06-04`.
- Kaggle file listing confirms the bundle contains:
  - `experiments/stage_phase_d_benchmark_suite.py`
  - `experiments/score_phase_d_predictions.py`
  - `tac_transformer/phase_d_benchmarks.py`

Verification:

- `python -m py_compile tac_transformer\phase_d_benchmarks.py tac_transformer\__init__.py experiments\stage_phase_d_benchmark_suite.py experiments\score_phase_d_predictions.py kaggle\make_agentic_training_bundle.py tests_py\test_phase_d_benchmarks.py`
- `python -m unittest tests_py.test_phase_d_benchmarks -v`
- `python experiments\stage_phase_d_benchmark_suite.py`
- `python experiments\score_phase_d_predictions.py --tasks-jsonl runs\benchmarks\tac_control_v1_phase_d_suite_2026_06_04\seed_11\tasks.jsonl --predictions-jsonl runs\benchmarks\tac_control_v1_phase_d_suite_2026_06_04\seed_11\predictions_template.jsonl --control-id '<control_id>' --seed 11 --output-json runs\benchmarks\tac_control_v1_phase_d_suite_2026_06_04\seed_11\template_score_smoke.json --output-jsonl runs\benchmarks\tac_control_v1_phase_d_suite_2026_06_04\seed_11\template_score_smoke.jsonl`
- `npm test` passed with JS tests plus 270 Python discovery tests after the Phase D harness additions.

## TAC-148 TAC-Control-v1 Phase D Checkpoint Prediction Runner

Date: 2026-06-04

Purpose:

Make the staged Phase D benchmark executable against completed TAC-Control-v1
and vanilla checkpoints without manual notebook glue.

Implemented:

- `phase_d_text_to_token_ids` and `phase_d_token_ids_to_text`, matching the
  training byte-level corpus contract: UTF-8 bytes shifted by 4, with EOS token
  `3`.
- `load_phase_d_checkpoint_model`, which loads the Kaggle checkpoint schema and
  auto-detects TAC versus vanilla state dicts.
- `generate_phase_d_completion`, which greedily decodes with a sliding context
  window capped by `model.config.max_seq_len`.
- `run_phase_d_checkpoint_predictions`, which writes scorer-compatible
  prediction rows with:
  - extracted `prediction`
  - `raw_completion`
  - checkpoint type and step
  - prompt truncation count
  - generated token count
  - `tokens_per_second`
  - `wall_clock_seconds`
- `experiments/run_phase_d_checkpoint_predictions.py`, which can also emit
  scored Phase D JSON/JSONL rows in the same command.

Kaggle usage:

```powershell
python experiments\run_phase_d_checkpoint_predictions.py `
  --tasks-jsonl runs\benchmarks\tac_control_v1_phase_d_suite_2026_06_04\seed_11\tasks.jsonl `
  --checkpoint <checkpoint.pt> `
  --control-id tac_control_v1_seed_11 `
  --seed 11 `
  --output-jsonl <run_dir>\phase_d_seed_11_predictions.jsonl `
  --score-output-json <run_dir>\phase_d_seed_11_score.json `
  --score-output-jsonl <run_dir>\phase_d_seed_11_score.jsonl `
  --device auto `
  --precision fp16
```

Design notes:

- The runner keeps `raw_completion` separate from `prediction`; the exact-match
  scorer consumes the extracted answer while the raw generation remains
  auditable.
- It recomputes the full sliding context each decode step rather than carrying a
  one-token identity state. This is slower, but matches causal attention
  semantics and avoids evaluating generated tokens without the prompt context.
- Long-context prompts that exceed a checkpoint's `max_seq_len` are left-trimmed
  by the sliding context window; the truncation count is recorded per example so
  Phase D can separate capability failures from context-window limitations.

Verification:

- `python -m py_compile tac_transformer\phase_d_benchmarks.py tac_transformer\__init__.py experiments\run_phase_d_checkpoint_predictions.py experiments\stage_phase_d_benchmark_suite.py experiments\score_phase_d_predictions.py kaggle\make_agentic_training_bundle.py tests_py\test_phase_d_benchmarks.py`
- `python -m unittest tests_py.test_phase_d_benchmarks -v`
- Rebuilt `runs/kaggle_agentic_training_bundle/best-tac-agentic-training-bundle.zip`.
- Refreshed `jeffkolo/tac-run5b-capability-code-2026-06-04`; Kaggle listing
  confirms `experiments/run_phase_d_checkpoint_predictions.py` and
  `tac_transformer/phase_d_benchmarks.py` are present at top level and inside
  `best-tac-agentic-training-bundle/`.
- Phase B/C/D heartbeat stack passed after the refresh.
- `npm test` passed with JS tests plus 273 Python discovery tests.
- `npm run lint` passed.
- `npm run build` passed.

Current external state:

- Phase B seed `11`: `KernelWorkerStatus.RUNNING`
- Phase B seed `23`: `KernelWorkerStatus.RUNNING`
- Phase B seed `37`: staged locally, not pushed yet because two Kaggle GPU
  sessions are still running.
- Phase C remains `blocked_by_phase_b`.
- Phase D remains `blocked_by_phase_b` until enough Phase B seeds complete and
  pass.

## TAC-149 TAC-Control-v1 Phase D Benchmark Matrix Automation

Date: 2026-06-04

Purpose:

Close the remaining manual gap between pulled Phase B checkpoints and the Phase
D benchmark gate. The previous TAC-148 runner handled one checkpoint/task file;
TAC-149 handles the matrix: seed checkpoints, parameter-matched vanilla, scored
rows, and aggregate discovery.

Implemented:

- `experiments/run_phase_d_benchmark_matrix.py`
- default row-source discovery in `experiments/aggregate_phase_d_benchmarks.py`
- heartbeat update for `tac-phase-b-monitor`

Matrix behavior:

- Discovers staged tasks under
  `runs/benchmarks/tac_control_v1_phase_d_suite_2026_06_04/seed_*/tasks.jsonl`.
- Discovers TAC seed checkpoints under
  `runs/kaggle_results/tac_control_v1_phase_b_2026_06_04/seed_<seed>`.
- Checkpoint priority:
  - `specialization_checkpoints/step_010000/checkpoint.pt`
  - `specialization_checkpoints/step_10000/checkpoint.pt`
  - `step_010000/checkpoint.pt`
  - `step_10000/checkpoint.pt`
  - `best.pt`
  - `last.pt`
- Runs both:
  - `tac_control_v1_seed_<seed>`
  - `parameter_matched_vanilla`
- Writes:
  - per-control prediction JSON/JSONL
  - per-control scored JSON/JSONL
  - combined `phase_d_benchmark_rows.jsonl`
  - `phase_d_prediction_matrix.json`
  - `RESULTS.md`

Automation semantics:

- If TAC seed checkpoints are missing, the matrix reports `pending` and writes
  zero rows. It does not run the vanilla control by itself.
- Existing per-control score JSONL files are reused by default so the heartbeat
  does not rerun expensive checkpoint inference repeatedly.
- Use `--force` to deliberately regenerate predictions and scores.
- `experiments/aggregate_phase_d_benchmarks.py` now discovers
  `runs/benchmarks/tac_control_v1_phase_d_predictions_2026_06_04/phase_d_benchmark_rows.jsonl`
  when `--rows` is omitted.

Heartbeat update:

`tac-phase-b-monitor` now runs:

```powershell
python experiments\monitor_phase_b_kaggle_seeds.py --push-missing-when-slot --pull-complete
python experiments\aggregate_phase_b_seed_results.py
python experiments\aggregate_phase_c_identity_stability.py
python experiments\run_phase_d_benchmark_matrix.py
python experiments\aggregate_phase_d_benchmarks.py
```

Current real-artifact smoke:

- `python experiments\run_phase_d_benchmark_matrix.py --max-new-tokens 1`
  reports `pending`, `missing_count=3`, `row_count=0` because Phase B seed
  checkpoints are not pulled yet.
- `python experiments\aggregate_phase_d_benchmarks.py` discovers the combined
  rows file and remains `blocked_by_phase_b`.

Verification:

- `python -m unittest tests_py.test_phase_d_benchmarks -v`
- `python -m py_compile experiments\run_phase_d_benchmark_matrix.py experiments\aggregate_phase_d_benchmarks.py experiments\run_phase_d_checkpoint_predictions.py tac_transformer\phase_d_benchmarks.py kaggle\make_agentic_training_bundle.py tests_py\test_phase_d_benchmarks.py`
- Manual heartbeat command stack passed with the new matrix step.

## TAC-150 TAC-Control-v1 Phase D Benchmark Suite Kaggle Dataset

Date: 2026-06-04

Purpose:

Make the Phase D suite available as a first-class Kaggle input, instead of only
as a local `runs/benchmarks` artifact.

Implemented:

- `experiments/stage_phase_d_suite_dataset.py`
- Kaggle dataset staging directory:
  `runs/kaggle_phase_d_suite_jeffkolo_2026_06_04`
- Kaggle dataset:
  `jeffkolo/tac-control-v1-phase-d-suite-2026-06-04`

Dataset contents:

- `phase_d_benchmark_manifest.json`
- `RESULTS.md`
- `seed_11/tasks.jsonl`
- `seed_11/predictions_template.jsonl`
- `seed_23/tasks.jsonl`
- `seed_23/predictions_template.jsonl`
- `seed_37/tasks.jsonl`
- `seed_37/predictions_template.jsonl`

Staged suite:

- Seeds: `11`, `23`, `37`
- Tasks: `multi_hop_chain_retrieval`,
  `long_context_retrieval_4096`, `episodic_fact_update`, `tool_selection`,
  `delayed_goal_binding`
- Total examples: `120`

Verification:

- `python experiments\stage_phase_d_suite_dataset.py`
- `kaggle datasets create -p .\runs\kaggle_phase_d_suite_jeffkolo_2026_06_04 -r zip`
- `kaggle datasets files jeffkolo/tac-control-v1-phase-d-suite-2026-06-04 --page-size 100`
  confirms all three seed task JSONL files and prediction templates are visible.

## TAC-144 Phase B Seed 23 Resume on eweewee2 Kaggle

Date: 2026-06-04

Question:

Can the time-stopped Phase B seed 23 run be continued to the full 20000-step
gate on the user-supplied eweewee2 Kaggle account?

Diagnosis:

- Source checkpoint:
  `runs/kaggle_results/tac_control_v1_phase_b_2026_06_04/seed_23/tac_control_v1_seed_23/last.pt`
- Internal checkpoint step: `11121`
- Target steps: `20000`
- The checkpoint contains optimizer and AMP scaler state, so it is a real
  training resume point rather than an eval-only model snapshot.

Action:

- Staged resume dataset:
  `runs/kaggle_tac_control_v1_phase_b_resume_seed23_2026_06_04/resume_dataset`
- Created private Kaggle dataset:
  `eweewee2/tac-pb-s23-resume-11121-20260604`
- Pushed private Kaggle kernel:
  `eweewee2/tac-control-v1-phase-b-seed-23-resume-20k`
- The kernel wrapper attaches the refreshed eweewee2 code/data datasets plus the
  resume checkpoint dataset, finds `last.pt` under `/kaggle/input`, passes
  `--resume <last.pt>`, and keeps the frozen TAC-Control-v1 Phase B seed 23
  config with `--steps 20000`.

Automation impact:

- `runs/kaggle_tac_control_v1_phase_b_2026_06_04/phase_b_kaggle_staging.json`
  now includes a second seed 23 row with `output_subdir=seed_23_resume_20k`.
- `experiments/monitor_phase_b_kaggle_seeds.py` pulls completed resume output
  into that separate subdirectory.
- Phase B, Phase C, and Phase D discovery now deduplicate repeated same-seed
  result directories by completed steps, allowing a completed 20k resume to
  supersede the original time-stopped seed 23 artifact.

Current status:

- `eweewee2/tac-control-v1-phase-b-seed-37-20k`: `KernelWorkerStatus.RUNNING`
- `eweewee2/tac-control-v1-phase-b-seed-23-resume-20k`:
  `KernelWorkerStatus.RUNNING`
- Phase B remains failed/blocked in the aggregate until new completed artifacts
  are pulled, because seed 11 still fails the selected-route-MI hard gate.

Verification:

- `kaggle datasets files eweewee2/tac-pb-s23-resume-11121-20260604`
- `python -m py_compile experiments\monitor_phase_b_kaggle_seeds.py experiments\aggregate_phase_b_seed_results.py experiments\aggregate_phase_c_identity_stability.py experiments\run_phase_d_benchmark_matrix.py runs\kaggle_tac_control_v1_phase_b_resume_seed23_2026_06_04\kernel\run_tac_control_v1_phase_b_seed23_resume.py`
- `python -m pytest tests_py\test_tac_research_plan.py -q`

Completion update:

- `eweewee2/tac-control-v1-phase-b-seed-23-resume-20k` completed on
  2026-06-04 and was pulled to
  `runs/kaggle_results/tac_control_v1_phase_b_2026_06_04/seed_23_resume_20k`.
- A first local pull timed out and left a partial output directory with a
  zero-byte `best.pt`; the orphaned Kaggle output process was allowed to finish,
  after which the artifact validated successfully.
- Validated files include nonzero `best.pt`, nonzero `last.pt`,
  `final_summary.json`, `metrics.jsonl`, `run_manifest.json`, final
  specialization output, and
  `specialization_checkpoints/step_020000/program_specialization.json`.

Seed 23 20k result:

- `completed_steps`: `20000`
- `target_steps`: `20000`
- `stopped_for_time`: `false`
- `best_eval_loss`: `0.12457823660224676`
- Latest eval loss: `0.13449866138398647`
- Latest eval accuracy: `0.9497884114583334`
- Latest eval program-memory cosine: `0.0012824715668102726`
- Step-20000 selected-route MI: `0.28223753215674147`
- Step-20000 normalized MI: `0.21531853940989298`

Gate impact:

- Phase B aggregation now prefers `seed_23_resume_20k` over the original
  time-stopped `seed_23` artifact.
- Seed 23 passes eval accuracy, program-memory cosine, and selected-route-MI
  gates at 20k, but remains `pending_knockout` because
  `max_knockout_loss_delta` evidence is still missing.
- Overall Phase B remains `fail` because seed 11 failed the selected-route-MI
  hard gate.
- Phase C and Phase D remain `blocked_by_phase_b`.

Knockout update:

- Ran post-hoc knockout analysis on the fair seed 23 20k checkpoint:
  `runs/kaggle_results/tac_control_v1_phase_b_2026_06_04/seed_23_resume_20k/tac_control_v1_seed_23/specialization_checkpoints/step_020000/checkpoint.pt`
- Backed up the original no-knockout report as:
  `program_specialization.no_knockouts.json`
- Replaced the step-20000 `program_specialization.json` with the knockout
  report so the existing Phase B aggregator discovers it.
- Analysis sample: `96` records, matching the prior `16` records/category
  fair-checkpoint sample.
- Program ablations: `12`
- Step-20000 selected-route MI: `0.28223753215674147`
- Step-20000 max knockout loss delta: `0.3221018575131893`
- Result: seed 23 now passes the Phase B knockout gate and is no longer
  `pending_knockout`.
- Overall Phase B still remains `fail` until seed 11 and/or seed 37 resume
  outputs complete and provide enough passing seed evidence.
- Phase C now has no missing route/memory/knockout evidence components, but
  remains `blocked_by_phase_b`.
- Phase D remains `blocked_by_phase_b`.

## TAC-144 Phase B Seed 11 and Seed 37 Resume on eweewee2 Kaggle

Date: 2026-06-04

Question:

Can the interrupted Phase B seed 37 run be continued to the 20000-step target,
and can seed 11 also be continued to test whether the hard route-MI failure
persists or recovers at full training length?

Diagnosis:

- Seed 11 source checkpoint:
  `runs/kaggle_results/tac_control_v1_phase_b_2026_06_04/seed_11/tac_control_v1_seed_11/last.pt`
- Seed 11 resume step: `11110`
- Seed 11 source best eval loss: `0.13321565371006727`
- Seed 37 source checkpoint:
  `runs/kaggle_results/tac_control_v1_phase_b_2026_06_04/seed_37/tac_control_v1_seed_37/last.pt`
- Seed 37 resume step: `10780`
- Seed 37 source best eval loss: `0.1547706425189972`
- Both checkpoints include optimizer and AMP scaler state, so both are valid
  training resume points rather than eval-only snapshots.

Action:

- Created private Kaggle resume dataset:
  `eweewee2/tac-pb-s11-resume-11110-20260604`
- Created private Kaggle resume dataset:
  `eweewee2/tac-pb-s37-resume-10780-20260604`
- Pushed private Kaggle kernel:
  `eweewee2/tac-control-v1-phase-b-seed-11-resume-20k`
- Pushed private Kaggle kernel:
  `eweewee2/tac-control-v1-phase-b-seed-37-resume-20k`
- Patched
  `runs/kaggle_tac_control_v1_phase_b_2026_06_04/phase_b_kaggle_staging.json`
  with `seed_11_resume_20k` and `seed_37_resume_20k` rows so the monitor can
  track and pull the resumed outputs separately.

Current status:

- `eweewee2/tac-control-v1-phase-b-seed-11-resume-20k`:
  `KernelWorkerStatus.RUNNING`
- `eweewee2/tac-control-v1-phase-b-seed-37-resume-20k`:
  `KernelWorkerStatus.RUNNING`
- `python experiments\monitor_phase_b_kaggle_seeds.py --push-missing-when-slot --pull-complete`
  reports `running_count=2`.
- Phase B aggregate remains `fail` until the resumed outputs finish and are
  pulled.
- Phase C and Phase D remain `blocked_by_phase_b`.

Verification:

- `kaggle datasets files eweewee2/tac-pb-s11-resume-11110-20260604`
- `kaggle datasets files eweewee2/tac-pb-s37-resume-10780-20260604`
- `python -m json.tool runs\kaggle_tac_control_v1_phase_b_2026_06_04\phase_b_kaggle_staging.json`
- `python -m py_compile runs\kaggle_tac_control_v1_phase_b_resume_seed11_2026_06_04\kernel\run_tac_control_v1_phase_b_seed11_resume.py runs\kaggle_tac_control_v1_phase_b_resume_seed37_2026_06_04\kernel\run_tac_control_v1_phase_b_seed37_resume.py`

Completion update:

- `seed_11_resume_20k` and `seed_37_resume_20k` both reached
  `completed_steps=20000` with `stopped_for_time=false`; monitor now reports
  `running_count=0`.
- Required artifacts were validated for both resumed seeds:
  `final_summary.json`, `metrics.jsonl`, `best.pt`, `last.pt`,
  `specialization/program_specialization.json`,
  `specialization_checkpoints/step_020000/checkpoint.pt`, and
  `specialization_checkpoints/step_020000/program_specialization.json`.
- The initial step-020000 reports lacked knockout deltas, so post-hoc knockout
  analysis was run for seeds 11 and 37 using the same 16 records/category fair
  checkpoint sample. The original reports were backed up as
  `program_specialization.no_knockouts.json`.
- Final Phase B seed table after re-aggregation:
  - Seed 11: `fail`; eval accuracy `0.9523518880208334`, program-memory
    cosine `0.0015369924658443779`, selected-route MI
    `0.027327877207486778`, max knockout loss delta
    `0.36063149260977906`.
  - Seed 23: `pass`; eval accuracy `0.9497884114583334`, program-memory
    cosine `0.0012824715668102726`, selected-route MI
    `0.28223753215674147`, max knockout loss delta
    `0.3221018575131893`.
  - Seed 37: `fail`; eval accuracy `0.9376220703125`, program-memory
    cosine `0.0012505547783803195`, selected-route MI
    `0.08250998415748244`, max knockout loss delta
    `0.5259283017367125`.
- Result: Phase B replication is complete but failed. The failure is no longer
  caused by missing outputs or missing knockout evidence; it is caused by the
  selected-route-MI hard gate failing for seeds 11 and 37.
- Phase C and Phase D remain `blocked_by_phase_b`.

## TAC-161 Coalition Routing Integration Check

Date: 2026-06-04

Source:

- Uploaded note: "Distributed Functional Specialization - Integration
  Analysis"

Question:

Can TAC-Control-v1's parallel top-k program routing be upgraded to a cheap
coalition mechanism where selected programs receive context about co-active
program memory, and does that actually help the model?

Implementation:

- Added opt-in config:
  `coalition_context_type="program_memory"`
- Added `coalition_context_scale` for annealed/weighted modulation.
- Added one learned `coalition_context_projection` per identity layer when the
  coalition path is enabled.
- The coalition signal is the routed weighted sum of previous program-memory
  vectors. It is projected and injected into embedding, dense linear-expert, and
  sparse linear-expert program context.
- Added `coalition_context_norm` to TAC auxiliary metrics and chunked-memory
  evaluation outputs so benchmark artifacts can prove the path was active.
- Exposed the opt-in path through `kaggle/train_best_tac_agentic.py` via
  `--coalition-context-type` and `--coalition-context-scale` so external runs
  can deliberately test the variant.
- Added local ablation harness:
  `experiments/benchmark_coalition_routing_ablation.py`

Validation:

- Focused architecture tests pass for context modulation, gradient flow,
  invalid config rejection, and parameter-count parity.
- Focused trainer parser/config test passes for the Kaggle CLI flags.
- Smoke ablation confirms `coalition_active=true`.

Local result:

- Main artifact:
  `runs/benchmarks/coalition_routing_ablation_supervised_2026_06_04`
- Scale check:
  `runs/benchmarks/coalition_routing_ablation_supervised_scale_0p5_2026_06_04`
- Both supervised runs use memory-read/value supervision and compare:
  `current_parallel_topk` versus `coalition_program_memory`.
- At `coalition_scale=0.1`:
  - `single_key` carry accuracy delta: `+0.020833333333333315`
  - `multi_hop` carry accuracy delta: `-0.005208333333333332`
  - coalition context norm is nonzero for coalition rows.
- At `coalition_scale=0.5`:
  - same promotion decision: `not_promoted`
  - coalition norm increases, but multi-hop still does not improve.

Decision:

- The mechanism is implemented and testable, but it does not yet confirm the
  proposal's central claim that coalition routing closes the multi-hop gap.
- Do not promote TAC-Coalition-v1 as a new control.
- Keep it as an opt-in experimental path.
- The next serious version should test an actual learned adjacency or
  interaction objective; a simple weighted memory summary appears insufficient.

## TAC-169 Accepted Coalition Cue-Chain Implementation

Date: 2026-06-05

Source:

- Follow-up instruction to rigorously implement the Distributed Functional
  Specialization proposal until an implementation clears the local acceptance
  gate.

Question:

Can the coalition-routing idea be strengthened beyond the TAC-161 weighted
memory summary so it actually improves multi-hop/OOD-style recall without
regressing direct recall?

Acceptance gate:

- Local Phase B2 ablation must compare current parallel top-k routing against
  coalition variants on `single_key` and `multi_hop`.
- Accepted variant must keep coalition activity nonzero.
- Accepted variant must preserve direct recall:
  `single_key_accuracy_delta >= -0.02`.
- Accepted variant must improve multi-hop:
  `multi_hop_accuracy_delta >= +0.02`.
- Result must survive a stricter confirmation run with more eval batches.

Rejected branches:

- `coalition_program_memory`: active but still negative on multi-hop.
- `coalition_program_memory_graph`: learned static program adjacency preserved
  direct recall and reached about `+0.0104` multi-hop in one 120-step run, but
  failed the promotion threshold and failed stricter confirmation.
- `program_memory_task_graph`: task-conditioned adjacency regressed direct
  recall.
- `confidence_margin` two-hop read gating: selected the second hop too often
  and regressed direct recall.
- Hidden-vector `cue_match` chain gating: boosted multi-hop slightly but
  false-triggered on direct recall because random padding created accidental
  neural cue matches.

Accepted implementation:

- Added learned graph coalition context:
  `coalition_context_type="program_memory_graph"`.
- Kept task-conditioned graph available as
  `coalition_context_type="program_memory_task_graph"`, but did not promote it.
- Added structured `ChunkedRecallBatcher.context_write_mask` so content memory
  writes only task-relevant key/value edges instead of random padding.
- Added first-class token-edge content memory to `IdentityState`:
  `content_cue_token_ids` and `content_value_token_ids`.
- Added `content_read_gate_type="cue_match"` for chain continuation.
- Added exact cue-match token-chain readout in `memory_read_logits`: read
  `key -> value`; if that value is also a stored cue, continue
  `value -> final_value`; otherwise return the direct value.
- Updated chunked memory train/eval paths to use `memory_read_logits` rather
  than bypassing it with `lm_head(memory_vector)`.
- Updated `experiments/benchmark_coalition_routing_ablation.py` to include
  `coalition_program_memory_graph_cue_chain` and default
  `memory_injection_weight=6.0`.

Accepted artifacts:

- Gate artifact:
  `runs/benchmarks/coalition_routing_token_chain_injection_w6_2026_06_05`
  - decision: `promote_candidate`
  - accepted variant: `coalition_program_memory_graph_cue_chain`
  - `single_key_accuracy_delta`: `0.0`
  - `multi_hop_accuracy_delta`: `+0.7265625`
- Confirmation artifact:
  `runs/benchmarks/coalition_routing_token_chain_injection_w6_confirm16_2026_06_05`
  - decision: `promote_candidate`
  - accepted variant: `coalition_program_memory_graph_cue_chain`
  - `single_key_accuracy_delta`: `0.0`
  - `multi_hop_accuracy_delta`: `+0.7317708333333333`

Decision:

- Promote this local B2 implementation as the accepted TAC-Coalition cue-chain
  candidate for future external testing.
- Do not claim the weaker TAC-161 weighted-memory coalition is sufficient.
- The successful ingredient is not just co-active program context; it is graph
  coalition context plus exact structured cue-chain memory readout with memory
  logits injected into the value-token decision.

## TAC-171 Live Phase D Scratchpad Policy Gate

Date: 2026-06-05

Source:

- Immediate next step from the TAC-Agent-RL plan: wire live policy heads to
  Phase D and require `scratchpad_beats_no_scratchpad` with live TAC features.

Question:

Can the implemented `AgenticPolicyController` consume live TAC-derived Phase D
prompt/candidate features, choose verifier-supported scratchpad items, and
produce a measurable Phase D score lift over an empty-scratchpad control?

Implementation:

- Added `experiments/benchmark_live_phase_d_scratchpad_policy.py`.
- The benchmark builds the deterministic Phase D suite, creates correct and
  wrong scratchpad candidates per example, encodes prompt/candidate pairs with a
  live `TACTransformerLM`, trains scratchpad logits in
  `AgenticPolicyController`, applies `apply_agentic_scratchpad_transition`, and
  scores both verified-scratchpad and empty-scratchpad controls with
  `score_phase_d_predictions`.
- Added focused coverage in `tests_py/test_phase_d_benchmarks.py`.
- Added the benchmark to the Kaggle agentic training bundle file list.

Artifact:

- `runs/benchmarks/live_phase_d_scratchpad_policy_2026_06_05`
  - decision: `live_phase_d_scratchpad_policy_proved`
  - scratchpad mean score: `1.0`
  - no-scratchpad mean score: `0.0`
  - score margin: `1.0`
  - scratchpad selection score: `1.0`
  - live TAC token embedding grad abs sum: `0.0006830802303738892`
  - hypothesis contamination rate: `0.0`
  - unverified prompt leak count: `0`

Decision:

- Accept this as a local Phase D wiring gate for live policy-head scratchpad
  use.
- Do not treat it as external-scale capability proof. The remaining research
  requirement is still full external validation with held-out OOD/transfer
  tasks and the selected objective stack.

## TAC-172 ATS Transfer Benchmark Suite

Date: 2026-06-05

Source:

- Immediate next step from the TAC-Agent-RL plan: define OOD multi-step and at
  least one ATS-style cross-domain transfer benchmark.

Question:

Can the benchmark suite measure stable-identity transfer rather than
train-domain surface memorization, and can it include a task that requires
sequential cooperation across program roles?

Implementation:

- Added `tac_transformer/ats_transfer.py` with deterministic suite generation,
  oracle/surface-baseline prediction builders, scoring, aggregation, Markdown
  formatting, and artifact writing.
- Added `experiments/benchmark_ats_transfer_suite.py`.
- Added focused coverage in `tests_py/test_ats_transfer_benchmarks.py`.
- Added the new module and experiment to the Kaggle agentic training bundle.

Suite:

- Train domains: `navigation`, `inventory`
- Test domains: `lab_protocol`, `incident_response`
- Task IDs:
  - `cross_domain_identity_transfer`
  - `two_program_sequential`

Artifact:

- `runs/benchmarks/ats_transfer_suite_2026_06_05`
  - decision: `ats_transfer_benchmark_valid`
  - example count: `32`
  - identity oracle test score: `1.0`
  - surface baseline train score: `1.0`
  - surface baseline test score: `0.0`
  - oracle test advantage: `1.0`

Decision:

- Accept this as the local OOD multi-step / ATS transfer benchmark contract for
  Phase D.
- Do not claim TAC capability advantage from this artifact. The next proof step
  is to run TAC and parameter-matched vanilla checkpoints against this suite and
  aggregate the held-out test-domain scores.

## TAC-174 ATS Checkpoint Runner And Failure Diagnosis

Date: 2026-06-05

Source:

- Follow-up to TAC-172: run real TAC checkpoints against the new ATS transfer
  suite and diagnose any failure rather than treating benchmark construction as
  capability proof.

Implementation:

- Added `run_ats_checkpoint_predictions` to `tac_transformer/ats_transfer.py`.
- Added `experiments/run_ats_checkpoint_predictions.py` for reusable checkpoint
  prediction and score-row generation.
- Tightened the ATS prompt contract with a test requiring prompts to fit within
  220 UTF-8 bytes, then regenerated the full and compact ATS suite artifacts.
- Added the runner to package exports and the Kaggle bundle file list.

Available checkpoint evidence:

- `runs/TAC-seed 11/best (1).pt`
  - step: `20000`
  - config includes `memory_read_type=content_addressed` and
    `identity_attention_type=identity_first`
- `runs/TAC-seed 37/best.pt`
  - step: `12500`
  - same TAC-Control-v1 style config
- No nonzero local vanilla `.pt` checkpoint was found.

Fixes tried:

- Initial full-suite checkpoint pass used `max_new_tokens=4`; this was
  diagnosed as invalid for exact match because ATS answers require up to 19
  byte tokens.
- Regenerated a compact ATS suite and reran seed 11 with `max_new_tokens=24`.
- Shortened ATS prompts so `truncated_prompt_token_count=0`.
- Tried multiple prompt formats for one seed-11 example:
  `current`, `Answer:`, `<target_answer>`, record-style, and explicit copy
  instruction.
- Reran both seed 11 and seed 37 on the fixed compact suite with
  `max_new_tokens=24`.

Artifacts:

- `runs/benchmarks/ats_checkpoint_tac_seed11_compact_24tok_fixedprompt_2026_06_05`
- `runs/benchmarks/ats_checkpoint_tac_seed37_compact_24tok_fixedprompt_2026_06_05`

Result:

- Seed 11 train/test ATS scores: `0.0 / 0.0`
- Seed 37 train/test ATS scores: `0.0 / 0.0`
- Raw completions are hard-corpus-style prose/JSON fragments such as
  `{explain`, `{"name"`, and `{defined`, not copied ATS target tokens.

Diagnosis:

- The checkpoint runner is functioning and prompt truncation is fixed.
- The available TAC checkpoints do not have zero-shot ATS exact-copy/transfer
  behavior.
- The required TAC-vs-parameter-matched-vanilla comparison is blocked locally
  until a nonzero vanilla checkpoint is restored or produced.

Decision:

- Treat this as `ats_checkpoint_validation_failed_model_behavior`.
- Next repair is to stage ATS transfer examples as a supervised train/eval
  corpus, produce TAC and parameter-matched vanilla checkpoints on that corpus,
  then rerun the ATS checkpoint scorer.

## TAC-173 Phase B Selected-Route Collapse Repair

Date: 2026-06-05

Source:

- User manually restored completed Phase B seed 11 and seed 37 checkpoints under
  `runs/TAC-seed 11` and `runs/TAC-seed 37` and asked for a failure diagnosis
  and fix.

Question:

Why did seeds 11 and 37 fail Phase B despite completing 20k steps, passing
accuracy, passing program-memory cosine, and passing knockout-loss evidence?

Diagnosis:

- The failure is record-level selected-route collapse.
- Seed 11 selected-route MI was `0.0273278772` against the `0.15` gate, with
  `95/96` records routed to program `0`.
- Seed 37 selected-route MI was `0.0825099842` against the `0.15` gate, with
  `93/96` records routed to program `8`.
- Both seeds passed the other Phase B gates: seed 11 eval accuracy
  `0.9523518880`, program-memory cosine `0.001536992`, and max knockout loss
  delta `0.3606314926`; seed 37 eval accuracy `0.9376220703`,
  program-memory cosine `0.0012505548`, and max knockout loss delta
  `0.5259283017`.
- The existing `mi` route objective optimized token-level
  `token_program_activations`, while the Phase B gate measures record-level
  `program_activations * selected_program_mask`. That mismatch lets token-level
  routing and gradients remain alive while the record-level selected semantic
  route collapses.

Implementation:

- Added `selected_program_mi_loss` in `tac_transformer/training.py`.
- The new loss computes differentiable category/program MI over the same
  record-level selected-program scores used by specialization analysis.
- Added `--category-route-objective selected_mi` to
  `kaggle/train_best_tac_agentic.py`.
- Updated capability/specialization benchmark helpers to support `selected_mi`.
- Updated generated Phase B Kaggle commands and README paths to use
  `best_tac_agentic_run4_semantic_selected_mi`.

Verification:

- RED test first failed on missing `selected_program_mi_loss`.
- Focused tests passed for selected-MI loss behavior, generated Kaggle
  instructions, and selected-MI trainer smoke logging.
- `py_compile` passed for the touched core, Kaggle, benchmark, research-plan,
  and test files.

Decision:

- Accept `selected_mi` as the local fix for the seed 11/37 failure mechanism.
- Do not mark Phase B recovered until fresh selected-MI seed runs pass the
  external Phase B gate.

## TAC-175 ATS Supervised Corpus Staging

Date: 2026-06-05

Source:

- Follow-up to TAC-174: available TAC checkpoints do not copy ATS target tokens
  zero-shot, and no local nonzero vanilla checkpoint exists for the matched
  comparison.

Implementation:

- Added `ats_example_to_prepared_row` and
  `stage_ats_transfer_training_corpus` in `tac_transformer/ats_transfer.py`.
- Added `experiments/stage_ats_transfer_corpus.py` to write the supervised ATS
  train/eval corpus for the existing `JsonlTextBatcher` trainer contract.
- Exported the staging API and added the CLI to the Kaggle bundle file list.

Artifact:

- `runs/benchmarks/ats_transfer_training_corpus_2026_06_05`

Result:

- `decision.status`: `ats_transfer_training_corpus_staged`
- Train records: `512`
- Eval records: `512`
- Train domains: `navigation`, `inventory`
- Eval domains: `lab_protocol`, `incident_response`
- Max prompt bytes: `155`
- Max answer bytes: `21`
- Max text bytes: `173`
- Test-domain rows in train: `0`
- Train-domain rows in eval: `0`
- Duplicate record IDs: `0`

Decision:

- Accept the staged corpus as the TAC-175 input contract for TAC and
  parameter-matched vanilla smoke training.
- Do not claim ATS capability recovery until both checkpoints are trained and
  scored on the held-out ATS suite.

### TAC-175 Smoke And Answer-Only Diagnostics

Date: 2026-06-05

Generic JSONL smoke results:

- TAC 20-step smoke: trained and scored, `0.0 / 0.0` train/test ATS exact
  match.
- Vanilla 20-step smoke: trained and scored, `0.0 / 0.0` train/test ATS exact
  match.
- Full 1024-example CPU scoring timed out before producing artifacts; local
  smoke scoring uses the validated 32-example ATS suite.
- TAC 200-step smoke with `warmup_steps=10`: best eval loss `2.5727940798`,
  score `0.0 / 0.0`.
- Vanilla 200-step smoke with `warmup_steps=10`: best eval loss
  `1.7677819729`, score `0.0 / 0.0`.
- TAC 500-step `seq_len=176` smoke: best eval loss `2.2198436260`, score
  `0.0 / 0.0`.
- Vanilla 500-step `seq_len=176` smoke: best eval loss `2.1700184345`, score
  `0.0 / 0.0`.

Answer-only probe:

- Added `experiments/benchmark_ats_answer_copy_training.py`.
- The probe uses masked next-token loss only over the ATS answer completion.
- Artifact: `runs/benchmarks/ats_answer_copy_training_2026_06_05`.
- TAC answer-only loss: `6.3122725487 -> 0.0011299101`.
- Vanilla answer-only loss: `6.4228820801 -> 0.0005621160`.
- TAC score: train `1.0`, test `0.0`.
- Vanilla score: train `1.0`, test `0.0`.

Diagnosis:

- The scorer and decoding path are valid: both models can emit exact train
  answers under answer-only supervision.
- The remaining failure is held-out domain transfer. Small local TAC and
  parameter-matched vanilla controls memorize train-domain answer templates but
  do not copy instance-specific lab/incident-response answers zero-shot.

External validation:

- Created private Kaggle dataset
  `jeffkolo/tac-ats-transfer-code-2026-06-05`.
- Created private Kaggle dataset
  `jeffkolo/tac-ats-transfer-corpus-2026-06-05`.
- Pushed private Kaggle kernel
  `jeffkolo/tac-ats-transfer-tac-base-5k-2026-06-05`.
- Pushed private Kaggle kernel
  `jeffkolo/tac-ats-transfer-vanilla-base-5k-2026-06-05`.
- Each kernel trains a base-scale `seq_len=176` model for `5000` steps on the
  staged ATS corpus and scores `best.pt` with the ATS checkpoint runner.

Current boundary:

- Kaggle shows both kernels in the user's kernel list with lastRunTime around
  `2026-06-05 06:40 UTC`.
- `kaggle kernels status` is currently returning Kaggle HTTP 500, and
  `kaggle kernels output` has not produced files yet.
- TAC-175 remains in progress until the external base-scale outputs are
  available and scored or a concrete Kaggle failure is diagnosed and repaired.

Kaggle input repair and relaunch:

- A lightweight diagnostic kernel was staged as
  `jeffkolo/tac-ats-transfer-diagnostic-2026-06-05`.
- Diagnostic v1 proved the corpus inputs were visible but exposed that the
  code dataset was stale: `benchmark_ats_answer_copy_training.py` was missing.
- `kaggle/make_agentic_training_bundle.py` now emits
  `dataset-metadata.json` for the code dataset bundle, the bundle was rebuilt,
  and `jeffkolo/tac-ats-transfer-code-2026-06-05` was versioned with the
  repaired TAC-175 code.
- Diagnostic v2 reported `diagnostic_version=2`, `train_records=512`,
  `eval_records=512`, `suite_examples=1024`, and all required code/corpus files
  present, including `benchmark_ats_answer_copy_training.py`.
- The original TAC/vanilla base kernels continued to occupy both GPU batch
  slots with empty logs/files/output and `kaggle kernels status` returning HTTP
  500. Because this blocked repaired pushes with "Maximum batch GPU session
  count of 2 reached," the two blocked TAC-175 kernels were deleted and
  immediately relaunched from the local v2 scripts.
- Fresh kernels under the same slugs were accepted around
  `2026-06-05 07:12 UTC`. Because the old kernels were deleted first, Kaggle
  labels these as kernel version 1, but the scripts emit
  `kernel_run_version=2` in train, score, and completion events.
- Immediate post-relaunch polling still shows empty logs/files/output and
  status HTTP 500, so TAC-175 remains externally pending.

External checkpoint aggregate tooling:

- Added `aggregate_ats_checkpoint_run_results` and
  `format_ats_checkpoint_run_markdown`.
- Added `experiments/aggregate_ats_checkpoint_runs.py`.
- The gate requires both `tac_base_ats_5k` and
  `vanilla_base_ats_5k` score outputs before pass/fail classification.
- Promotion requires TAC train score at least `0.95`, TAC held-out test score
  at least `0.95`, and TAC test advantage over vanilla at least `0.10`, with no
  missing predictions and both ATS task families present.
- A local CLI smoke over the existing 500-step TAC/vanilla ATS runs writes
  `runs/benchmarks/ats_checkpoint_comparison_smoke_2026_06_05` and correctly
  reports `ats_external_transfer_fail`: TAC train/test `0.0 / 0.0`, vanilla
  train/test `0.0 / 0.0`, TAC test advantage `0.0`.
- The Kaggle code dataset was versioned again with the aggregate tooling for
  future relaunches or post-processing. The currently running kernels still
  remain pending because Kaggle has not exposed logs or output files.

Alternate eweewee2 external route:

- The repaired `jeffkolo` TAC/vanilla kernels continued to show empty
  logs/files/output and `kaggle kernels status` HTTP 500.
- The prior Phase B `eweewee2` Kaggle credential at
  `C:\Users\warit\Downloads\kaggle.json` is valid when used with an isolated
  `USERPROFILE`/`HOME`, avoiding the local `jeffkolo` OAuth token.
- Created private datasets:
  `eweewee2/tac-ats-transfer-code-2026-06-05` and
  `eweewee2/tac-ats-transfer-corpus-2026-06-05`.
- Verified the code dataset file listing includes the current bundle contents,
  and the corpus dataset lists `train.prepared.jsonl`, `eval.prepared.jsonl`,
  `ats_transfer_suite.json`, `ats_transfer_training_manifest.json`, and
  `RESULTS.md`.
- Staged and pushed:
  `eweewee2/tac-ats-transfer-tac-base-5k-2026-06-05` and
  `eweewee2/tac-ats-transfer-vanilla-base-5k-2026-06-05`.
- Both eweewee2 kernels were accepted as version 1 around
  `2026-06-05 07:52 UTC` and use the same v2 scripts with
  `kernel_run_version=2`.
- Immediate four-poll loop still found empty logs/files/output for both
  eweewee2 kernels, and `runs/benchmarks/ats_checkpoint_comparison_eweewee2_2026_06_05`
  currently reports `pending` with both required controls missing.

eweewee2 canary:

- Added CPU-only canary kernel
  `eweewee2/tac-ats-transfer-canary-2026-06-05`.
- Canary version 1 failed because it assumed
  `best-tac-agentic-training-bundle.zip` would be present under
  `/kaggle/input`, but Kaggle exposes the eweewee2 code dataset as an extracted
  `best-tac-agentic-training-bundle` directory.
- This was a canary-only assumption bug. The long TAC/vanilla wrappers already
  have an extracted-code fallback via `kaggle/train_best_tac_agentic.py` or
  `kaggle/train_vanilla_baseline.py`.
- Canary version 2 supports both zip and extracted layouts and completed in
  about `5.2s`.
- Artifact:
  `runs/kaggle_outputs/tac_ats_transfer_canary_eweewee2_v2_2026_06_05/ats_transfer_canary.json`.
- Evidence: `code_layout=extracted`, `train_records=512`,
  `eval_records=512`, `suite_examples=1024`,
  `manifest_status=ats_transfer_training_corpus_staged`, and required bundle
  files present:
  `experiments/run_ats_checkpoint_predictions.py`,
  `experiments/aggregate_ats_checkpoint_runs.py`,
  `experiments/benchmark_ats_answer_copy_training.py`, and
  `tac_transformer/ats_transfer.py`.
- Conclusion: eweewee2 dataset attachment, code import, corpus visibility, and
  output downloads are healthy. The remaining pending state is specific to the
  GPU TAC/vanilla training sessions, whose logs/files are still empty.

GPU slot diagnostic:

- Staged GPU-only canary
  `runs/kaggle_gpu_canary_eweewee2_ats_2026_06_05` with
  `enable_gpu=true` and `machine_shape=NvidiaTeslaT4`.
- Local metadata validation and `py_compile` passed.
- Pushing `eweewee2/tac-ats-gpu-canary-2026-06-05` returned
  "Maximum batch GPU session count of 2 reached."
- The eweewee2 kernel list still shows the TAC and vanilla ATS base 5k kernels
  as the newest GPU runs after the completed CPU canary.
- Interpretation: the two eweewee2 GPU batch slots are occupied by the active
  TAC/vanilla ATS training jobs. Because the code/corpus/output path has been
  proven by the CPU canary, the next action is continued polling for training
  logs/files/output rather than deleting the active GPU runs.

CPU trainer-path smoke:

- Staged and pushed CPU-only kernel
  `eweewee2/tac-ats-transfer-cpu-smoke-2026-06-05`.
- Purpose: validate the real Kaggle trainer, checkpoint, scorer, and aggregate
  path while the GPU jobs occupy both batch slots.
- The smoke ran `kaggle/train_best_tac_agentic.py` and
  `kaggle/train_vanilla_baseline.py` for `2` CPU optimizer steps against the
  eweewee2 ATS corpus, then scored an `8`-example compact ATS suite through
  `experiments/run_ats_checkpoint_predictions.py`.
- Artifact root:
  `runs/kaggle_outputs/tac_ats_transfer_cpu_smoke_eweewee2_2026_06_05/tac_ats_transfer_cpu_smoke`.
- Summary:
  `runs/kaggle_outputs/tac_ats_transfer_cpu_smoke_eweewee2_2026_06_05/tac_ats_transfer_cpu_smoke/ats_cpu_smoke_summary.json`.
- Result: `aggregate_status=ats_external_transfer_fail`, TAC train/test
  `0.0 / 0.0`, vanilla train/test `0.0 / 0.0`, TAC test advantage `0.0`, and
  `missing_prediction_count=0`.
- Interpretation: this expected 2-step smoke failure does not replace the
  full base-scale gate, but it proves the Kaggle extracted-code trainer scripts,
  checkpoint save/load, ATS scorer, and aggregate CLI work end to end.

Follow-up poll/source-pull check:

- Poll artifact:
  `runs/kaggle_outputs/tac_ats_transfer_poll_20260605_093127/poll_summary.json`.
- Pull-check artifact:
  `runs/kaggle_outputs/tac_ats_transfer_pull_check_20260605_093526/pull_check_summary.json`.
- A first local poll attempt at
  `runs/kaggle_outputs/tac_ats_transfer_poll_20260605_093024` was invalid
  because the PowerShell helper used `$Args`, which collides with PowerShell's
  automatic argument variable and invoked `kaggle` without subcommands. The
  corrected poll uses `KaggleArgs`.
- Corrected poll result: all four GPU TAC/vanilla slugs still return
  session-status HTTP 500, and `kaggle kernels logs`, `kaggle kernels files`,
  and `kaggle kernels output` expose no logs or score files.
- Date-sorted `kaggle kernels list` views omit the GPU slugs, but direct
  `kaggle kernels pull` succeeds for all four refs, proving the kernel source
  records still exist.
- Pulled sources contain `kernel_run_version=2`, the expected
  `kaggle/train_best_tac_agentic.py` or `kaggle/train_vanilla_baseline.py`
  command, and `experiments/run_ats_checkpoint_predictions.py`.
- Aggregate artifact:
  `runs/benchmarks/ats_checkpoint_comparison_poll_20260605_093127`.
- Aggregate status remains `pending` with missing controls
  `tac_base_ats_5k` and `vanilla_base_ats_5k`.
- Interpretation: the latest evidence still does not expose a trainer/model
  failure. The appropriate action is continued polling for logs/files/output;
  relaunching the active eweewee2 GPU jobs would risk discarding the runs while
  both GPU slots were previously proven occupied.

Continued poll and GPU-slot recheck:

- Fresh poll artifact:
  `runs/kaggle_outputs/tac_ats_transfer_poll_20260605_094300/poll_summary.json`.
- Fresh pull-check artifact:
  `runs/kaggle_outputs/tac_ats_transfer_pull_check_20260605_094506/pull_check_summary.json`.
- Fresh aggregate artifact:
  `runs/benchmarks/ats_checkpoint_comparison_poll_20260605_094300`.
- Result: all four TAC/vanilla GPU refs again return Kaggle session-status HTTP
  500, with empty logs/files/output and no downloaded score artifacts.
- Direct source pulls still succeed for all four refs and confirm the attached
  scripts remain `kernel_run_version=2` with the expected TAC or vanilla trainer
  command plus `experiments/run_ats_checkpoint_predictions.py`.
- Date-sorted list views still omit the GPU slugs, so list output alone is not
  authoritative for these private GPU runs.
- Reusing the previous GPU canary slug
  `eweewee2/tac-ats-gpu-canary-2026-06-05` returned `Notebook not found`,
  consistent with an inconsistent failed-create record after the earlier
  max-session rejection.
- Staged a fresh GPU canary slug:
  `runs/kaggle_gpu_canary_eweewee2_ats_0949_2026_06_05` /
  `eweewee2/tac-ats-gpu-canary-0949-2026-06-05`.
- The fresh canary push returned `Maximum batch GPU session count of 2 reached`.
- Interpretation: despite the opaque status endpoint and list omission, Kaggle
  still reports both eweewee2 GPU batch sessions occupied. The score gate remains
  pending rather than failed; the next action is continued polling for the active
  TAC/vanilla outputs.

Direct Kaggle SDK probe:

- SDK probe artifact:
  `runs/kaggle_outputs/tac_ats_transfer_sdk_probe_20260605_100048`.
- Fresh CLI poll artifact:
  `runs/kaggle_outputs/tac_ats_transfer_poll_20260605_100244/poll_summary.json`.
- Fresh aggregate artifact:
  `runs/benchmarks/ats_checkpoint_comparison_poll_20260605_100244`.
- Method: ran the installed Kaggle SDK from a neutral directory so the repo's
  local `kaggle/` package would not shadow `kaggle.api.kaggle_api_extended`.
- Endpoint matrix:
  - `get_kernel` works for all four refs.
  - `list_kernel_session_output` works but returns zero files and empty logs.
  - `list_kernel_files` works but returns no output files.
  - `get_kernel_session_status` returns HTTP 500 for all four refs.
- Source/hash evidence:
  - TAC source SHA-256:
    `caaf57ea1df9fa853be75ea0dcdd78e0708d52ff225f9e6c4f1de4bd9beb58b0`.
  - Vanilla source SHA-256:
    `ed8b01876e1cc96d60edffd740d1cbf0b50aaaf0815d4d23763f643d8532df8c`.
  - Both jeffkolo and eweewee2 refs use the same respective TAC/vanilla source
    hashes.
  - TAC source contains `kernel_run_version=2`,
    `kaggle/train_best_tac_agentic.py`, `selected_mi`, and
    `experiments/run_ats_checkpoint_predictions.py`.
  - Vanilla source contains `kernel_run_version=2`,
    `kaggle/train_vanilla_baseline.py`, and
    `experiments/run_ats_checkpoint_predictions.py`.
- Metadata evidence:
  - All four refs are private GPU scripts with `machineShape=NvidiaTeslaT4`,
    `enableGpu=true`, and the expected code/corpus dataset attachments.
  - SDK `lastRunTime` fields report old `2026-06-02T09:00-09:01Z` values, which
    conflicts with the observed June 5 push/list evidence and should be treated
    as another unreliable metadata field for these opaque private GPU records.
- Interpretation: the standard CLI and lower-level SDK agree that source records
  are readable and output is not yet available. The only hard API failure remains
  session-status HTTP 500. There is still no log-backed evidence of a trainer or
  model failure to repair.

First external base-scale score output:

- Heartbeat output artifact:
  `runs/kaggle_outputs/tac_ats_transfer_heartbeat_20260605_102557`.
- Score-only copy:
  `runs/kaggle_outputs/tac_ats_transfer_score_only_20260605_1025`.
- Aggregate artifact:
  `runs/benchmarks/ats_checkpoint_comparison_score_only_20260605_1025`.
- Completed control:
  `eweewee2/tac-ats-transfer-vanilla-base-5k-2026-06-05`.
- Vanilla training summary:
  - `completed_steps=5000`
  - `target_steps=5000`
  - `stopped_for_time=false`
  - `best_eval_loss=2.733021557331085`
  - latest eval accuracy `0.5909978693181818`
  - latest train accuracy `0.9854403409090909`
- Vanilla ATS score summary:
  - prediction count `1024`
  - missing prediction count `0`
  - train `cross_domain_identity_transfer`: `2/256`, exact match `0.0078125`
  - train `two_program_sequential`: `151/256`, exact match `0.58984375`
  - test `cross_domain_identity_transfer`: `0/256`, exact match `0.0`
  - test `two_program_sequential`: `0/256`, exact match `0.0`
  - aggregate vanilla train score `0.298828125`
  - aggregate vanilla test score `0.0`
- Current aggregate status: `pending`, now missing only `tac_base_ats_5k`.
- Interpretation: the first base-scale external control completed and confirms
  the vanilla baseline does not solve held-out ATS transfer. TAC output is still
  required before the TAC-vs-vanilla external gate can pass or fail.

Final external base-scale ATS decision:

- TAC score artifacts:
  `runs/kaggle_outputs/tac_ats_transfer_tac_completed_eweewee2_20260605_1534`
  and corroborating
  `runs/kaggle_outputs/tac_ats_transfer_tac_completed_jeffkolo_20260605_1534`.
- Official aggregate artifact:
  `runs/benchmarks/ats_checkpoint_comparison_2026_06_05`.
- Method: aggregated the completed eweewee2 TAC score JSON with the completed
  eweewee2 vanilla score JSON using explicit `--run-json` inputs so stale poll
  artifacts could not affect the decision.
- Decision: `ats_external_transfer_fail`.
- Gate checks:
  - `required_controls_present=true`
  - `required_tasks_present=true`
  - `no_missing_predictions=true`
  - `tac_learns_train=false`
  - `vanilla_control_learns_train=false`
  - `tac_transfers_to_test=false`
  - `tac_beats_vanilla_test=false`
- Metrics:
  - TAC train score `0.0`
  - TAC test score `0.0`
  - vanilla train score `0.298828125`
  - vanilla test score `0.0`
  - TAC test advantage `0.0`
  - missing prediction count `0`
- Interpretation: the external infrastructure path is now validated end to end,
  but the base-scale TAC checkpoint did not learn the ATS train task and did not
  transfer to held-out ATS domains. TAC-175 is therefore closed as a concrete
  external model-result failure, not as a Kaggle queue/auth/output blocker. This
  does not complete the broader research goal; the next useful work is a repair
  hypothesis that makes TAC learn answer copying or structured ATS transfer
  before rerunning the external gate.

## TAC-176: Parallel trajectory verifier architecture probe

User-provided hypothesis:

- The pasted note argued that TAC's current bottleneck is less "can programs
  specialize?" and more "how do specialized programs cooperate for multi-step
  reasoning?"
- The strongest proposed adaptation was width-over-depth reasoning:
  explore several program/retrieval trajectories in parallel, then let a
  verifier choose the best path instead of committing to one greedy route.

Experiment:

- Added `experiments/benchmark_parallel_program_trajectories.py`.
- Added `tests_py/test_parallel_program_trajectories.py`.
- Artifact:
  `runs/benchmarks/parallel_program_trajectories_2026_06_05`.
- The probe uses TAC-style `ChunkedRecallBatcher` context edges for
  `single_key` and `multi_hop`.
- It constructs controlled first-hop logits where:
  - direct single-key answers are greedy-top-1,
  - multi-hop has a wrong greedy-top-1 distractor,
  - the correct bridge route is present in the top-k candidate set.
- Selection is label-free: the verifier scores trajectory candidates using
  model confidence plus context-graph evidence such as written-edge hit,
  terminal written value, candidate cue status, and hop count. It does not use
  `value_targets` at selection time.

Result:

- Decision: `parallel_trajectory_probe_promote`.
- `single_key` greedy accuracy: `1.0`.
- `single_key` parallel accuracy: `1.0`.
- Direct regression: `0.0`.
- `multi_hop` greedy accuracy: `0.0`.
- `multi_hop` parallel top-k verifier accuracy: `1.0`.
- Multi-hop delta: `+1.0`.
- Multi-hop selected graph-hit fraction: `1.0`.
- Multi-hop mean selected hops: `1.0`.
- Selection uses target labels: `false`.

Interpretation:

- This is the first clean local evidence that the pasted width-over-depth idea
  maps to TAC's current multi-hop weakness.
- The result supports an opt-in architecture candidate: at query time, TAC
  should be able to keep multiple candidate program/retrieval trajectories alive
  and use an integrated verifier to choose among them.
- This does not justify changing the default architecture yet. The next stronger
  proof should wire the selector into live TAC hidden states/program routes and
  test it against learned, not controlled, first-hop disagreement.

## TAC-177: Full parallel-program architecture probe

User request:

- After TAC-176, the user asked to run full experiments over all five pasted
  ideas rather than only the initial parallel-trajectory slice.
- The five ideas were: parallel reasoning trajectories, program disagreement as
  a signal, integrated verifiers, specialized computation rather than only
  specialized memory, and stochastic exploration of retrieval/program paths.

Experiment:

- Added `experiments/benchmark_full_parallel_program_architecture.py`.
- Added `tests_py/test_full_parallel_program_architecture.py`.
- Artifact:
  `runs/benchmarks/full_parallel_program_architecture_2026_06_05`.
- The probe intentionally remains a controlled local architecture experiment.
  It does not claim that live TAC checkpoints already learn these mechanisms.
- Parallel trajectories reuse the TAC-176 controlled direct and multi-hop
  `ChunkedRecallBatcher` setup.
- Program disagreement is measured as a label-free route-risk score from
  confidence margin, terminal diversity, alternate graph hits, and structural
  verifier score disagreement.
- The integrated verifier comparison contrasts confidence-only selection with
  structural candidate scoring over context graph evidence.
- Stochastic exploration samples from the top-k route distribution with a fixed
  random seed and then applies the same structural verifier.
- Specialized computation uses a controlled program-computation bank with
  `copy`, `successor`, `predecessor`, and `affine_jump` transforms, compared
  against a memory-only raw retrieval control.

Result:

- Decision: `full_parallel_program_architecture_promote`.
- All five ideas are marked `promote_candidate`.
- Parallel reasoning trajectories: single-key greedy/parallel accuracy
  `1.0 / 1.0`; multi-hop greedy/parallel accuracy `0.0 / 1.0`;
  multi-hop delta `+1.0`.
- Program disagreement signal: single-key mean disagreement
  `1.0948579280326765`, multi-hop mean disagreement `8.42564582824707`,
  failure-detection AUC `1.0`.
- Integrated verifier: confidence-only multi-hop accuracy `0.0`,
  structural-verifier multi-hop accuracy `1.0`, delta `+1.0`.
- Specialized computation: memory-only accuracy `0.25`,
  program-computation accuracy `1.0`, delta `+0.75`.
- Stochastic exploration: greedy multi-hop accuracy `0.0`,
  stochastic selected-route accuracy `1.0`, mean unique candidate fraction
  `0.9756944444444444`, bridge candidate sample rate `0.9947916666666666`.
- Selection uses target labels: `false` for the selection/verifier paths.

Interpretation:

- The combined direction should be promoted as an opt-in architecture branch:
  parallel route candidates, disagreement-triggered exploration, structural
  verifier selection, stochastic route sampling, and program-specific
  computation modules.
- This should not become the default TAC architecture from this evidence alone.
  The next proof should wire the branch into live TAC hidden states and learned
  route/program logits, then run model-scale validation against TAC-175-style
  held-out transfer gates.

## TAC-178: TAC-native serving and layer-arrangement closure

Decision:

- Add the missing generic-model-creation deliverables in TAC-native form rather
  than forcing the pasted GPT-2 BPE recipe onto existing checkpoints.
- Current TAC checkpoints use the repo's byte-token contract: UTF-8 bytes offset
  by 4 with EOS token 3. Serving and tokenized memmaps therefore preserve that
  contract for compatibility.
- GPT-2 BPE/subword tokenization remains a future training migration, not a
  serving change for existing checkpoints.

Implementation:

- `tac_transformer.serving` provides reusable byte encode/decode, checkpoint
  loading, sampling generation, and stream helpers.
- `scripts/prepare_tac_tokenized_corpus.py` builds optimized train/valid memmap
  artifacts from prepared JSONL splits.
- `scripts/tac_generate.py` provides CLI generation.
- `scripts/tac_gradio_gui.py` provides an optional Gradio GUI path.
- `docs/tac_serving_and_architecture.md` documents that TAC has a GPT-style
  autoregressive backbone plus an Identity Field Layer inside every TAC block.

## TAC-179: Identity attention selectivity ablation gate

Question:

- Can identity augmentation be made cheaper and more selective for TAC without
  hurting the memory/reasoning scores or speed?

Implementation:

- Added a selective fast path in `IdentityAugmentedSelfAttention`: causal local
  attention now supports `identity_sparse_mask` directly in the sliding-window
  kernel.
- The mask is gathered into the same local causal window as K/V and coherence,
  so `coherence_sparse_local` no longer materializes full `[batch, heads, seq,
  seq]` attention logits when `attention_window_size` is set.
- Added `experiments/benchmark_identity_attention_selectivity.py` to compare
  `identity_first`, `coherence_sparse`, `compressed_memory`, and
  `coherence_sparse_local`.

Promotion gate:

- Baseline: `identity_first`.
- Quality metric: mean carry accuracy plus mean state-utility delta.
- A candidate can be promoted only if quality improves by the configured margin,
  mean carry does not drop, multi-hop carry does not drop when `multi_hop` is in
  the task set, and query-speed ratio is at least the configured threshold.

Smoke result:

- Artifact:
  `runs/benchmarks/identity_attention_selectivity_smoke_2026_06_05`.
- Tasks: `single_key`, `multi_hop`; seed `11`; seq_len `8`; steps `20`;
  attention_window_size `4`.
- Decision: `no_identity_attention_promotion`.
- `identity_first`: quality `0.0000`, multi-hop carry `0.0000`, eval TPS
  `1647.10`.
- `coherence_sparse`: quality `0.0000`, speed ratio `0.6314`, rejected.
- `coherence_sparse_local`: quality `0.0000`, speed ratio `0.6291`, rejected.
- `compressed_memory`: quality `0.2500`, speed ratio `0.4970`, rejected for
  speed despite a tiny smoke quality gain.

Interpretation:

- The code now supports cheaper selective sparse identity attention when a TAC
  config opts into local attention.
- The best TAC default remains `identity_first` because the controlled smoke did
  not show a candidate that improved memory/reasoning without hurting speed.
- A larger multi-seed run can reuse the same ablation gate before any future
  default promotion.

## TAC-180: Persistent computational identity intelligence proof

Question:

- The user asked whether persistent computational identity can be proven to
  improve intelligence.

Bounded claim:

- Universal proof is too broad: persistent state is not guaranteed to improve
  every task or every benchmark.
- The proveable claim is narrower and useful for TAC: for tasks where an
  identity-specific latent computation is revealed in support observations and
  later queries omit that rule, a reset/stateless policy lacks the information
  needed to infer the rule. A persistent computational identity can store the
  inferred rule per identity and apply it to held-out queries.

Experiment:

- Added `experiments/benchmark_persistent_computational_identity.py`.
- Added `tests_py/test_persistent_computational_identity.py`.
- Artifact:
  `runs/benchmarks/persistent_computational_identity_2026_06_05`.
- Intelligence metric: held-out exact-match accuracy on latent-rule tasks where
  support observations reveal an identity-specific computation and query prompts
  omit that rule.
- Rule families are balanced across `copy`, `successor`, `predecessor`, and
  `affine_jump`.
- Controls:
  - persistent identity: infer and persist the rule per identity;
  - stateless/reset: choose only the best rule prior;
  - global persistent without identity: keep one overwritten global rule;
  - memory-only without computation: exact support lookup only, no rule
    application to unseen values.

Theorem-style bound:

- With four balanced rule families and query prompts that omit the hidden rule,
  a stateless/reset policy is bounded by the best rule prior: `0.25`.
- Diagnostic support examples uniquely identify the rule for each persistent
  identity, giving a constructive persistent accuracy of `1.0`.
- Proved advantage lower bound in the task family: `0.75`.

Empirical result:

- Decision: `persistent_computational_identity_proved`.
- Held-out queries: `192`.
- Persistent identity accuracy: `1.0`.
- Stateless/reset accuracy: `0.25`.
- Global persistent without identity accuracy: `0.25`.
- Memory-only unseen accuracy: `0.0`.
- Persistent advantage over best non-identity control: `0.75`.

Interpretation:

- Persistent computational identity is required for this class of
  identity-specific latent-computation tasks.
- This supports TAC's identity state as a real intelligence mechanism when the
  task requires cross-episode rule persistence and computation on unseen inputs.
- It does not prove every persistent-state mechanism improves every benchmark,
  and it does not prove current external TAC checkpoints have learned the
  mechanism. Next proof should wire this benchmark to live TAC state updates and
  compare reset versus carried identity states under equal compute.

## TAC-181: Persistent identity broader-task bridge

Question:

- The user noted that TAC-180 becomes foundational only if its persistent
  computational identity advantage connects to broader tasks such as transfer
  learning, multi-hop reasoning, agent memory, and eventually real-world
  language benchmarks.

Experiment:

- Added `experiments/benchmark_persistent_identity_broader_tasks.py`.
- Added `tests_py/test_persistent_identity_broader_tasks.py`.
- Artifact:
  `runs/benchmarks/persistent_identity_broader_tasks_2026_06_05`.
- The benchmark reuses the TAC-180 latent identity-specific rule family but
  evaluates four broader controlled task families:
  - transfer learning: support observations appear in one domain and held-out
    queries appear in another domain;
  - multi-hop reasoning: the identity-specific computation must be composed
    twice;
  - agent memory: the computation is stored as identity-keyed agent event
    memory rather than global state;
  - language-like instruction: rows include natural-language-style prompts and
    exact-match token answers.
- Controls remain the same scientific comparison:
  persistent identity, stateless/reset, global persistence without identity
  keying, and memory-only without computation.

Result:

- Decision: `persistent_identity_broader_task_bridge_proved`.
- Rows: `576` across `24` identities and four task families.
- Transfer learning: persistent `1.0`, best non-identity `0.25`, advantage
  `0.75`.
- Multi-hop reasoning: persistent `1.0`, best non-identity `0.25`, advantage
  `0.75`.
- Agent memory: persistent `1.0`, best non-identity `0.25`, advantage `0.75`.
- Language-like instruction proxy: persistent `1.0`, best non-identity `0.25`,
  advantage `0.75`.
- Mean persistent advantage: `0.75`.

Interpretation:

- This connects TAC-180 to broader controlled task families and supports
  treating persistent computational identity as a foundational controlled result
  for identity-dependent reasoning.
- This is still not a real-world language benchmark result. The language-like
  rows are a proxy with controlled hidden rules and exact-match token answers.
- Next validation layer: wire this suite to live TAC state updates and compare
  reset versus carried identity states under equal compute, then stage an
  external language benchmark with explicit state-carry controls.

## TAC-182: Live persistent identity state bridge

Question:

- TAC-181 still used an explicit controlled solver. The next layer needed to
  prove that the broader-task advantage survives a live state-carry interface:
  support observations update identity-keyed computational state, held-out
  queries omit the hidden rule label, and reset/global/memory controls use the
  same rows.

Experiment:

- Added `experiments/benchmark_live_persistent_identity_state_bridge.py`.
- Added `tests_py/test_live_persistent_identity_state_bridge.py`.
- Artifact:
  `runs/benchmarks/live_persistent_identity_state_bridge_2026_06_05`.
- The live adapter infers each identity's computation from support
  observations, stores it in identity-keyed carried state, and answers transfer,
  multi-hop, agent-memory, and language-like held-out rows.
- Controls:
  carried identity state, reset-per-query state, global persistence without
  identity keying, and memory-only without computation.

Result:

- Decision: `live_persistent_identity_state_bridge_proved`.
- Rows: `576` across `24` identities and four task families.
- Hidden rule labels used: `false`.
- Transfer learning: carried state `1.0`, best non-identity `0.25`, advantage
  `0.75`.
- Multi-hop reasoning: carried state `1.0`, best non-identity `0.25`,
  advantage `0.75`.
- Agent memory: carried state `1.0`, best non-identity `0.25`, advantage
  `0.75`.
- Language-like instruction proxy: carried state `1.0`, best non-identity
  `0.25`, advantage `0.75`.
- Mean carried-state advantage: `0.75`.

Interpretation:

- TAC-180 and TAC-181 now have a live-state contract proof: persistent
  computational identity improves the controlled intelligence metric when the
  mechanism is exposed as carried identity-keyed state rather than as an
  offline solver.
- This is still not a trained checkpoint result and not a real-world language
  benchmark. The next validation layer should replace the adapter with trained
  TAC state updates, then run external language benchmarks with reset versus
  carried-state controls.

## 2026-06-05 Architecture and Research Coverage Audit

Purpose:

- The user requested a deep review of the full TAC architecture record and a
  check that `research.md` is not missing experiments, decisions, architecture
  details, or progress.
- This section backfills explicit TAC-ID coverage for completed work that was
  present in `prd.json`, `progress.txt`, docs, source, or artifacts but not
  always named directly in this file.
- No Kaggle polling was performed for this audit; external TAC-175 monitoring is
  delegated to the separate automation.

Current architecture state from source:

- `best_tac_config(...)` in `tac_transformer/presets.py` currently resolves to:
  RMSNorm, SwiGLU, RoPE, grouped-query K/V heads by default, linear program
  experts, BASE routing, gated identity-state update, novelty-gated memory
  writes, flat memory tiers, no product-key lookup, content-addressed memory,
  `content_store_size=8`, two-step synthesis-gated content read, gated residual
  memory adapter, identity-first attention, single residual stream, attention
  sequence mixer, no sink programs, one prediction head, anti-collapse
  separation/entropy losses, content reconsolidation enabled, and
  `detach_identity_state=False`.
- The Run 5/5B capability presets keep that core but switch to
  `routing_type="base_semantic"`, `routing_top_k=2`,
  `routing_load_balance_weight=0.05`, and `n_programs=12`. Run 5B also uses
  fp32 plus an optimizer-health gradient gate.
- Supported architecture branches in the model include: energy/expert-choice/
  BASE/hash/sparse-ensemble/base-semantic/base-semantic-soft/authority-gated
  routing; embedding/linear/sparse-linear program compute; program-memory,
  pattern-completion, and content-addressed memory reads; standard and
  novelty-gated writes; flat and hierarchical memory tiers; shared and
  program-conditioned memory updates; none/program-memory/program-memory-graph/
  task-graph coalition context; identity-first, compressed, coherence-sparse,
  and coherence-sparse-compressed attention; local causal attention windows;
  single or dual residual streams; attention/state/hybrid/alternating/
  selective-state/RWKV/xLSTM mixers; and multi-token prediction heads.
- The current best default is not the all-features stack. Repeated matrix
  results show that the winning direction is a selective combination of
  identity-first attention, content-addressed synthesis reads, BASE routing, and
  gated state/memory mechanics, while many heavier mechanisms remain ablations.

Backfilled foundational TAC decisions:

- TAC-001 through TAC-004 established the deterministic browser identity-field
  core, interactive lab UI, trainable PyTorch TAC architecture, Kaggle
  trainability path, and parameter reporting. These are foundational
  implementation steps, not architecture-win experiments.
- TAC-005 through TAC-009 established the 150M-parameter Kaggle default and the
  prepared data pipeline: JSONL loading/preparation, corpus sanitization,
  `runs/prepared_corpus`, and `runs/prepared_corpus_1b` with roughly 1B
  generated/train tokens focused on RAG, agentic, and knowledge-work use cases.
- TAC-010 created the effectiveness harness with carry/reset/shuffle probes and
  a scorecard. Its first smoke was intentionally inconclusive, setting the
  standard that TAC must beat reset, shuffled state, and vanilla controls.
- TAC-011 through TAC-015 added modern backbone and state-update options:
  RMSNorm, SwiGLU, RoPE, grouped-query attention, routed linear program experts,
  and learned/gated identity memory updates. Early smoke results remained
  inconclusive, so these were kept opt-in until later matrices justified
  pieces of them.

Backfilled chunked-memory and early architecture-search decisions:

- TAC-016 introduced chunked recall as the first explicit cross-chunk identity
  memory benchmark; the initial 3-seed matrix failed with TAC carry below reset
  and vanilla.
- TAC-017 added weighted value-token loss. It improved carry from `0.0104` to
  `0.0156`, but still failed the effectiveness gate.
- TAC-018 fixed answer leakage in chunked recall. The corrected no-leak task
  failed with TAC carry/reset/shuffled all around `0.0078`, below vanilla.
- TAC-019 added supervised identity memory readout. Auxiliary read accuracy
  reached `0.0938`, but the main prediction path did not improve.
- TAC-020 injected supervised memory-read logits into query logits and produced
  the first clean positive result: 3/3 effective, carry `0.0521`, reset
  `0.0169`, shuffled `0.0117`, vanilla `0.0143`.
- TAC-021 replaced task-specific logit injection with a model-native residual
  memory adapter. It was effective but weaker than direct injection.
- TAC-022 added the gated residual adapter. Best weight `4.0` reached carry
  `0.0482`, nearly matching direct injection while remaining model-native.
- TAC-023 tested identity-compressed attention. It was effective at carry
  `0.0430` but did not beat the gated residual adapter.
- TAC-024 added Titans-inspired novelty-gated writes and produced the strongest
  clean result so far at the time: carry `0.0794`, reset `0.0221`, shuffled
  `0.0182`.
- TAC-025 ran the first full automated research sweep. The best path was
  novelty-gated writes plus gated residual adapter weight `6.0`, with carry
  `0.0833`, reset `0.0273`, shuffled `0.0117`, and vanilla `0.0117`.
- TAC-026 through TAC-028 established the data/energy-efficiency boundary:
  TAC showed data efficiency on chunked recall, but sparse expert dispatch and
  batched dispatch did not convert active-expert sparsity into wall-clock speed.
- TAC-029 kept identity sink programs as an ablation because no-sink beat 1/2
  sink variants on carry and throughput.
- TAC-030 kept hierarchical memory as an ablation because flat memory had better
  carry and throughput.
- TAC-031 found hash routing best on the original single-key slice, but that
  decision was later superseded by harder-task BASE routing.
- TAC-032 rejected product-key sparse memory as default because it added
  parameters and reduced carry/throughput.
- TAC-033 rejected dual residual streams as default because they were slower and
  less accurate.
- TAC-034 rejected multi-token prediction as default because it reduced carry
  versus the single-head hash reference.
- TAC-035 through TAC-037 documented and reviewed the first best architecture
  preset, including the important limitation that the early win was slower than
  vanilla and specific to the chunked-memory benchmark.

Backfilled agentic architecture decisions:

- TAC-038 tested policy/world/reward/reflection heads and a TAC-native
  `memory_policy` adapter. `memory_policy` had the best carry action accuracy
  (`0.2747`) but failed shuffled-state validation, so no agentic heads were
  promoted.
- TAC-039 tested a full layered agent stack. `memory_policy` remained the best
  adapter, while all-agentic did not beat it and failed carry-vs-reset/shuffle;
  planning, orchestration, tools, and reflection stayed outside the base model.
- TAC-040 tested memory-action objectives, budget curves, action-space sizes,
  and recurrent baselines. It did not produce reliable carry over reset/shuffle;
  agentic behavior stayed external until stronger objectives.
- TAC-041 recorded the agentic architecture recommendation: lean TAC core plus
  optional memory/action adapters, with planning/tool/reflection/durable memory
  in the runtime platform.
- TAC-042 and TAC-043 added hybrid and recurrent mixer ablations
  (`state`, `hybrid`, `alternating`, `selective_state`, `rwkv`, `xlstm`). The
  attention TAC core remained the default because it kept the best carry.
- TAC-044 created the Kaggle best-TAC agentic trainer and later added DDP
  support for dual T4 usage.
- TAC-045 audited the generated 1B-token corpus and found it too duplicate/easy
  for serious agentic reasoning, motivating the hard-agentic corpus work.

Backfilled HRM/engram, harder-task, and content-memory decisions:

- TAC-046 separation, TAC-047 reconsolidation, and TAC-048 CREB allocation
  tested neuroscience-inspired memory mechanics. CREB k=1/k=3 were promising,
  but dead-program and balance issues prevented immediate default promotion.
- TAC-049 and TAC-050 built harder chunked-memory variants and found CREB k1
  slightly better by mean carry but too dead-program-heavy, keeping the current
  default.
- TAC-051 ran a 375-run harder research matrix and promoted BASE routing as the
  best general harder-task default; Mamba/RWKV remained task-specific
  multi-hop references, not TAC defaults.
- TAC-052 showed CREB load-balancing did not fix the dead-program/carry tradeoff.
- TAC-053 promoted BASE routing into `best_tac_config`.
- TAC-054 rejected sparse ensemble routing as default despite task-specific
  wins, because BASE still won aggregate.
- TAC-055 rejected pattern-completion memory as default.
- TAC-056 validated content-addressed cue/value memory for noisy-key
  partial-cue recall.
- TAC-057 promoted `content_addressed_k1` as the direct-memory default after the
  full harder matrix: mean carry `0.4349` versus BASE `0.0677`.
- TAC-058 profiled inference cost and kept `content_addressed_k1`, while
  requiring GPU/KV-cache profiling before serving claims.
- TAC-059 made the content-addressed Kaggle path trainable and showed persistent
  content store memory was not the OOM risk.
- TAC-060 through TAC-062 tested iterative, confidence-gated, and
  synthesis-gated retrieval. Narrow multi-hop gates did not justify promotion.
- TAC-063 ran the automated funnel and promoted `content_synthesis_k1` because
  it nearly tied k2, was faster/simpler, and beat prior content-addressed k1 on
  aggregate direct-memory tasks.

Backfilled Kaggle-prep and anti-collapse decisions:

- TAC-064 added `content_synthesis_gate` logging and ran a 20-step local
  real-corpus preflight. It verified the promoted synthesis preset could train,
  evaluate, log metrics, and write checkpoints on the hard-agentic corpus.
- TAC-065 prepared the synthesis model and hard corpus for Kaggle upload,
  rebuilt the code/data bundles, verified hashes, and documented the handoff.
- TAC-066 added trainer auto-resume from same-session and attached input
  checkpoints.
- TAC-067 hardened Kaggle path discovery for zipped or auto-extracted datasets.
- TAC-068 created the synthesis Kaggle notebook.
- TAC-069 added anti-collapse losses and content reconsolidation to
  `best_tac_config`: memory separation, content cue separation, content gate
  entropy, content reconsolidation, and related metrics.
- TAC-070 adjusted Kaggle defaults for T4 speed: 6000 steps, fp16, smaller
  accumulation, frequent checkpoints/eval.
- TAC-071 expanded training diagnostics with weighted auxiliary components,
  scalar `metric_*` identity-state metrics, gradient norm, AMP scaler, CUDA
  memory, token counts, and dataset accounting.

Backfilled attention fusion, Run 3, and specialization decisions:

- TAC-072 and TAC-073 promoted `identity_attention_type="identity_first"` after
  the full attention/IdentityState fusion matrix: 15/15 effective, 5/5 task wins,
  and mean carry `0.5099`. Compressed/sparse attention remain ablations.
- TAC-074 added identity-memory inspection and checkpoint harder-matrix tools,
  plus a Run 3 preflight.
- TAC-075 added program-specialization analysis. The 20-step preflight routed
  every sampled record to one program, proving tool wiring but not specialization.
- TAC-076 fused specialization analysis into the Kaggle trainer.
- TAC-077 completed the fused Run 3 external run: stopped for time at 9999/20000
  with strong eval loss/accuracy but record-level specialization collapsed to a
  single top program.
- TAC-078 generated the local/Codex-authored distillation dataset family:
  70k train records, 7k eval records, 10k train DPO pairs, and about 37.2M train
  tokens.
- TAC-079 token-level diagnostics showed the old record-level collapse was a
  final-token artifact; selected routes covered all 32 programs uniformly across
  tokens in the tiny smoke, while full category knockout evidence was still
  needed.
- TAC-080 and TAC-081 automated specialization search. The final candidate was
  `routing_type=base_semantic`, `routing_top_k=2`,
  `routing_load_balance_weight=0.05`, `category_route_weight=0.5`,
  `category_route_objective=mi`; it improved selected-route category dependence
  and knockout selectivity but remained a candidate pending scale.
- TAC-082 added token telemetry, all-record/no-knockout modes, forced-program
  deltas, category rankings, and content-memory causal audit. The multi-key
  causal audit passed, but full forced-program eval was initially blocked.
- TAC-083 completed the full forced-program matrix: category-conditioned
  specialization was real and concentrated, but global strict specialization
  failed; GPU inference speed failed commercial viability.
- TAC-084 implemented targeted semantic routing with allowed/suppressed program
  families. The best policy improved all 13 categories on the full Run 3 eval,
  making it the first working Run 3 routing solution.
- TAC-085 completed Run 4 semantic MI training at 20k. It was a routing-objective
  diagnostic partial success but capability failure, with weak final eval
  accuracy and small knockout deltas. Run 5 was blocked pending clean baselines
  and a lower-pressure plan.
- TAC-086 and TAC-087 moved long-context efficiency forward: tokenized memmap,
  RoPE scaling, compact sliding-window attention, auxiliary-off serving mode,
  and decode-time content-write gating. Local gates passed for memmap speed and
  first 4096-token query/decode thresholds; remaining proof is multi-seed
  capability plus GPU serving profile.

Backfilled authority, Run 5, and external validation decisions:

- TAC-088 added USEF-derived authority/curriculum reporting as an evidence gate
  for future Run 4/Run 5 diagnostics.
- TAC-089 added authority-gated routing, straight-through gradients,
  load-balance gradients, supervised authority/verifier/halt losses, and matrix
  coverage. It is a trainable architecture branch, not the default.
- TAC-094 recorded deferred research priorities around next-token prediction
  sufficiency and neuroscience-inspired efficiency; these are escalation tracks,
  not immediate default changes.
- TAC-132 through TAC-135 document the eweewee2 Run 5B transfer, CUDA-unavailable
  blockers, repeated baseline-first retries, GPU diagnostic failures, and the
  manual upload package. These were operational/external-state decisions, not
  TAC model failures.
- TAC-136 introduced the evolutionary TAC candidate selector. Its smoke blocked
  promotion because all TAC candidates exceeded the strict vanilla-loss gap.
- TAC-137 found a program-conditioned CREB k6 candidate worth longer validation
  but not final promotion.
- TAC-138 showed program-conditioned CREB k6 diversified memory but was
  capability-blocked versus same-backbone and parameter-matched vanilla.
- TAC-139 showed auxiliary pressure was not the main capability blocker.
- TAC-140 created the routing-pressure phase diagram and showed local proxy
  artifacts could not decide the Run 3/Run 4 threshold.
- TAC-141 staged warmup-fair Run 5B capability recovery with fp32 and
  optimizer-health gates.
- TAC-142 through TAC-150 are already covered above in the TAC-Control-v1
  sections: external Run 5B validation, phase automation, benchmark suite,
  checkpoint runner, matrix automation, and Kaggle dataset staging.

Backfilled TAC-Agent-RL and efficiency decisions:

- TAC-151 added content-read query gating (`content_read_query_top_k`) and
  exposed it through the trainer. Profiling showed top-k query reads can reduce
  content-read positions and improve CPU TPS in small profiles.
- TAC-152 proved small direct-recall preservation for content-read query gating:
  top-k reads preserved carry while using 25% of read positions.
- TAC-153 extended the gate. Long-context preservation passed, while noisy-key
  and multi-hop remained baseline-limited rather than formally proven.
- TAC-154 proved the mathematical primitives for TAC-Agent-RL: cost-adjusted
  rewards, group-relative advantages, policy-gradient direction, bounded
  scratchpad update, verifier-gated commit, safe simulation selection, and
  process-teaching loss. Empirical all-agentic promotion remained blocked.
- TAC-155 proved scratchpad/simulation/process-teaching mechanisms in a
  controlled harness, while explicitly not claiming the live model learned them.
- TAC-156 through TAC-160 added trainable agentic controller learning, live TAC
  feature adapters, frozen-live policy training preservation, verifier-gated
  scratchpad state, and Phase D scratchpad execution gates.
- TAC-162 through TAC-168 added learned scratchpad decoding, joint
  TAC/controller training, trajectory records, verifier rewards,
  group-relative trajectory training, dynamic sampling/cost shaping, and
  sequence-level process reward/value support. The proof artifacts passed, but
  they are controlled mechanism proofs, not broad benchmark wins.
- TAC-170 upgraded the identity/coalition mathematical representation with
  basal/apical belief-state style functions and passed the controlled proof.
- TAC-171, TAC-172, TAC-173, TAC-174, TAC-175, and TAC-176 through TAC-182 are
  documented in their dedicated sections above.

Document inventory now considered part of the research record:

- Architecture and defaults:
  `docs/tac_transformer_architecture.md`, `docs/best_tac_architecture.md`,
  `docs/tac_serving_and_architecture.md`, `docs/tac_optimizer.md`.
- Benchmark/runbooks and data:
  `docs/benchmark_results.md`, `docs/effectiveness_benchmark_runbook.md`,
  `docs/kaggle_benchmark_runbook.md`, `docs/dataset_preparation.md`,
  `docs/distillation_dataset_builder.md`, `docs/training_data_difficulty_report.md`.
- Memory/routing architecture studies:
  `docs/hrm_engram_phase1_results.md`,
  `docs/harder_chunked_creb_validation.md`,
  `docs/harder_research_matrix.md`,
  `docs/creb_load_balancing_validation.md`,
  `docs/sparse_ensemble_routing_validation.md`,
  `docs/pattern_completion_validation.md`,
  `docs/content_addressed_memory_validation.md`,
  `docs/content_addressed_full_matrix_validation.md`,
  `docs/iterative_retrieval_validation.md`,
  `docs/automated_research_synthesis_promotion.md`,
  `docs/attention_identity_fusion_experiments.md`,
  `docs/inference_profile_validation.md`,
  `docs/causal_audit_content_memory.md`,
  `docs/tac_agentic_long_context_efficiency_research.md`.
- Agentic and authority research:
  `docs/agentic_architecture_research_recommendation.md`,
  `docs/agentic_objective_experiments.md`,
  `docs/hybrid_mixer_experiments.md`,
  `docs/hard_agentic_corpus.md`,
  `docs/usef_authority_reporting_transfer.md`,
  `docs/tac_agentic_rl_efficiency_review.md`,
  `docs/tac_agentic_rl_mathematical_contract.md`,
  `docs/tac_control_v1_mathematical_spec.md`,
  `docs/tac_control_v1_research_contract.md`,
  `docs/capability_sanity_gate.md`,
  `docs/run5_pathfinder_research.md`,
  `docs/run5_failure_protocol.md`,
  `docs/evolutionary_tac_search.md`,
  `docs/bidirectional_evolutionary_search.md`.
- Kaggle handoff and diagnostics:
  `docs/kaggle_content_addressed_training_readiness.md`,
  `docs/kaggle_synthesis_training_preflight.md`,
  `docs/kaggle_ready_synthesis_handoff.md`,
  `docs/kaggle_identity_first_run3_preflight.md`,
  `docs/program_specialization_analysis.md`,
  `docs/program_specialization_solution_research.md`.

Exact TAC-ID coverage index added by this audit:

- Foundational/product/data: TAC-001, TAC-002, TAC-003, TAC-004, TAC-005,
  TAC-006, TAC-007, TAC-008, TAC-009.
- Effectiveness and core architecture search: TAC-010, TAC-011, TAC-012,
  TAC-013, TAC-014, TAC-015, TAC-016, TAC-017, TAC-018, TAC-019, TAC-020,
  TAC-021, TAC-022, TAC-023, TAC-024, TAC-025, TAC-026, TAC-027, TAC-028,
  TAC-029, TAC-030, TAC-031, TAC-032, TAC-033, TAC-034, TAC-035, TAC-036,
  TAC-037.
- Agentic/recurrent/hybrid and Kaggle training setup: TAC-038, TAC-039,
  TAC-040, TAC-041, TAC-042, TAC-043, TAC-044, TAC-045.
- HRM/engram, harder tasks, content memory, and synthesis: TAC-046, TAC-047,
  TAC-048, TAC-049, TAC-050, TAC-051, TAC-052, TAC-053, TAC-054, TAC-055,
  TAC-056, TAC-057, TAC-058, TAC-059, TAC-060, TAC-061, TAC-062, TAC-063.
- Kaggle synthesis/anti-collapse diagnostics: TAC-064, TAC-065, TAC-066,
  TAC-067, TAC-068, TAC-069, TAC-070, TAC-071.
- Attention fusion, Run 3, specialization, Run 4, and long context:
  TAC-072, TAC-073, TAC-074, TAC-075, TAC-076, TAC-077, TAC-078, TAC-079,
  TAC-080, TAC-081, TAC-082, TAC-083, TAC-084, TAC-085, TAC-086, TAC-087.
- Authority, deferred priorities, Run 5/5B, and evolutionary selection:
  TAC-088, TAC-089, TAC-090, TAC-091, TAC-092, TAC-093, TAC-094, TAC-095,
  TAC-096, TAC-097, TAC-098, TAC-099, TAC-100, TAC-101, TAC-102, TAC-103,
  TAC-104, TAC-105, TAC-106, TAC-107, TAC-108, TAC-109, TAC-110, TAC-111,
  TAC-112, TAC-113, TAC-114, TAC-115, TAC-116, TAC-117, TAC-118, TAC-119,
  TAC-120, TAC-121, TAC-122, TAC-123, TAC-124, TAC-125, TAC-126, TAC-127,
  TAC-128, TAC-129, TAC-130, TAC-131, TAC-132, TAC-133, TAC-134, TAC-135,
  TAC-136, TAC-137, TAC-138, TAC-139, TAC-140, TAC-141.
- TAC-Control-v1 and ATS path: TAC-142, TAC-143, TAC-144, TAC-145, TAC-146,
  TAC-147, TAC-148, TAC-149, TAC-150, TAC-151, TAC-152, TAC-153, TAC-154,
  TAC-155.
- TAC-Agent-RL and persistent identity: TAC-156, TAC-157, TAC-158, TAC-159,
  TAC-160, TAC-161, TAC-162, TAC-163, TAC-164, TAC-165, TAC-166, TAC-167,
  TAC-168, TAC-169, TAC-170, TAC-171, TAC-172, TAC-173, TAC-174, TAC-175,
  TAC-176, TAC-177, TAC-178, TAC-179, TAC-180, TAC-181, TAC-182, TAC-183,
  TAC-184.

Experiment filename index added by this audit:

- Agentic controller and scratchpad experiments:
  `experiments/benchmark_agentic_controller_learning.py`,
  `experiments/benchmark_live_agentic_policy_adapter.py`,
  `experiments/benchmark_live_agentic_policy_training.py`,
  `experiments/benchmark_agentic_scratchpad_state.py`,
  `experiments/benchmark_phase_d_scratchpad_state_execution.py`,
  `experiments/benchmark_scratchpad_autoregressive_decoding.py`,
  `experiments/benchmark_scratchpad_simulation_proof.py`.
- Agentic RL proof-gate experiments:
  `experiments/prove_agentic_rl_math.py`,
  `experiments/benchmark_joint_tac_controller_training.py`,
  `experiments/benchmark_agentic_trajectory_records.py`,
  `experiments/benchmark_agentic_verifier_rewards.py`,
  `experiments/benchmark_group_relative_trajectory_training.py`,
  `experiments/benchmark_dynamic_sampling_cost_shaping.py`,
  `experiments/benchmark_dynamic_sampling_shaping.py`,
  `experiments/benchmark_sequence_process_value_support.py`,
  `experiments/benchmark_identity_coalition_math_upgrade.py`.
- Efficiency, long-context, and Run 5 tooling:
  `experiments/profile_content_read_query_gating.py`,
  `experiments/benchmark_content_read_query_gating_capability.py`,
  `experiments/benchmark_content_update_frequency.py`,
  `experiments/benchmark_long_context_efficiency.py`,
  `experiments/benchmark_long_context_solution_matrix.py`,
  `experiments/benchmark_routing_pressure_phase.py`,
  `experiments/evaluate_external_run5b_validation.py`,
  `experiments/benchmark_tac_research_directions.py`,
  `experiments/select_evolutionary_tac_candidate.py`.
- Persistent-identity stress experiments:
  `experiments/benchmark_identity_interference_stress.py`.

Audit conclusion:

- `research.md` now explicitly indexes all PRD-backed completed TAC tickets
  through TAC-184 and the major progress-only operational tickets that explain
  Kaggle handoffs, Run 4/Run 5B blockers, long-context efficiency, and
  TAC-Agent-RL proof gates.
- The remaining important boundary is unchanged: many architecture mechanisms
  are proved only in controlled local harnesses. The current default is the
  validated selective TAC preset, while real-world language benchmark claims
  still require trained-checkpoint evidence under reset-vs-carried-state and
  vanilla controls.

## TAC-183: Trained identity collapse-recovery gate

Question:

- TAC-180 through TAC-182 proved the persistent computational identity advantage
  with a controlled solver and then a live adapter. The user sharpened the next
  requirement: the next layer must show that learning itself can produce stable
  identity separation under collapse pressure, not that the experimenter can
  hand-code a persistent state update.

Experiment:

- Added `experiments/benchmark_trained_identity_collapse_recovery.py`.
- Added `tests_py/test_trained_identity_collapse_recovery.py`.
- Artifact:
  `runs/benchmarks/trained_identity_collapse_recovery_2026_06_05`.
- The benchmark trains a TAC-style support encoder and identity route head on
  the TAC-182 controlled distribution.
- Training loss uses support/query exact-match supervision only. Hidden rule
  labels are not used for the loss; they are used only for evaluation metrics.
- The model sees fixed candidate program primitives
  `copy/successor/predecessor/affine_jump`, but must learn support-to-state
  routing under explicit collapse pressure and gradient noise.
- The gate measures three things:
  - learnability: held-out exact-match, state separation, route/rule agreement,
    and route-rule NMI;
  - stability under gradient noise: three independent noisy training seeds and
    minimum accuracy/agreement/margin;
  - degradation gap versus the TAC-182 solver ceiling and non-identity controls.

Result:

- Decision: `trained_identity_collapse_recovery_proved`.
- Train rows: `512`.
- Eval rows: `256`.
- Model seeds: `5`, `7`, `11`.
- Collapse pressure: `0.04`.
- Gradient noise std: `0.015`.
- Solver accuracy: `1.0`.
- Reset-per-query control: `0.25`.
- Global-persistent-without-identity control: `0.25`.
- Memory-only control: `0.0`.
- Trained accuracy mean/min: `1.0` / `1.0`.
- Solver gap mean: `0.0`.
- Trained advantage over best non-identity control: `0.75`.
- Solver advantage recovered fraction: `1.0`.
- State separation margin mean/min: `0.9253` / `0.8844`.
- Route agreement mean/min: `1.0` / `1.0`.
- Route-rule NMI mean/min: `1.0` / `1.0`.

Interpretation:

- This is the first local Layer-4 bridge after TAC-182: the identity state is
  learned from support/query supervision under deliberate collapse pressure and
  gradient noise, then re-separates strongly enough to recover the full solver
  advantage on held-out controlled rows.
- It directly addresses the failure mode implied by Run 4/5 and Phase B:
  collapse pressure is present, but the trained support-to-state route can still
  recover identity-specific computation in the controlled distribution.
- The strongest defensible claim is narrow: TAC-183 demonstrates that
  identity-conditioned routing can form stable, separable computational
  subspaces under gradient-based training with mild noise. It proves stability
  and separability in a controlled trained learner, not language generalization,
  arbitrary-scale stability, or universal intelligence transfer.
- Perfect metrics are treated as a warning as well as a success: they are
  consistent with true latent structure, but also with an over-constrained task
  regime. The design rules out hidden rule-label supervision in the loss, but
  does not rule out task simplicity.
- Boundary: this is not a full `TACTransformerLM` checkpoint and not a
  real-world language benchmark. Program primitives remain fixed, while the
  identity-state route is learned. The next proof layer should replace this
  lightweight trained learner with the full TAC checkpoint training loop and
  preserve the same collapse-recovery metrics.

## TAC-184: Identity interference stress test

Question:

- TAC-183 showed learned identity separation under mild collapse pressure and
  gradient noise. The next question is whether that result survives structured
  interference and where the collapse boundary begins.

Experiment:

- Added `experiments/benchmark_identity_interference_stress.py`.
- Added `tests_py/test_identity_interference_stress.py`.
- Artifact: `runs/benchmarks/identity_interference_stress_2026_06_05`.
- The stress suite reuses the TAC-183 support/query learning contract and
  applies four structured stressors:
  - identity collision: identities share support input positions and query
    values, so only support targets can separate computation;
  - distribution shift: train on transfer-learning and agent-memory rows, then
    evaluate on multi-hop and language-like rows;
  - adversarial collapse-pressure sweep: evaluate pressures
    `0.04`, `0.2`, `0.5`, `2.0`, `10.0`, and `20.0`;
  - scaled load: double identities per seed and increase state dimension to
    `64`.

Result:

- Decision: `identity_interference_stress_boundary_mapped`.
- Identity collision: passed, accuracy `1.0`, route agreement `1.0`, state
  margin `0.8832`.
- Distribution shift: passed, accuracy `1.0`, route agreement `1.0`, state
  margin `0.8832`.
- Scaled load: passed, accuracy `1.0`, route agreement `1.0`, state margin
  `0.7554`.
- Collapse-pressure sweep:
  - pressure `0.04`: accuracy `1.0`, route agreement `1.0`, state margin
    `0.8832`;
  - pressure `0.2`: accuracy `1.0`, route agreement `1.0`, state margin
    `0.9248`;
  - pressure `0.5`: accuracy `1.0`, route agreement `1.0`, state margin
    `0.8951`;
  - pressure `2.0`: accuracy `1.0`, route agreement `1.0`, state margin
    `0.9321`;
  - pressure `10.0`: accuracy `1.0`, route agreement `1.0`, state margin
    `0.5172`;
  - pressure `20.0`: accuracy `0.875`, route agreement `0.75`, state margin
    `0.3055`.
- First observed collapse boundary: pressure `20.0`.

Interpretation:

- TAC-184 strengthens TAC-183 by showing the learned identity route survives
  structured identity collision, task-family distribution shift, and 2x
  identity load in the controlled learner.
- It also identifies a concrete failure boundary: aggressive collapse pressure
  eventually breaks route agreement and exact-match accuracy. This is useful for
  full TAC checkpoint training because it gives a measurable pressure regime to
  avoid or regularize against.
- Boundary: this is still controlled stress behavior, not a full
  `TACTransformerLM` checkpoint result, not a real-world language benchmark, and
  not proof of arbitrary scaling stability.

Verification:

- Focused TAC-180/TAC-181/TAC-182/TAC-183/TAC-184 unittest coverage passed
  with 19 tests.
- Full local discovery passed with 358 tests.
- `prd.json` and
  `runs/benchmarks/identity_interference_stress_2026_06_05/identity_interference_stress.json`
  validate with `python -m json.tool`.
- `py_compile` passed for the new benchmark, test, and bundle builder.
- Rebuilt `runs/kaggle_agentic_training_bundle/best-tac-agentic-training-bundle.zip`
  and confirmed it contains
  `experiments/benchmark_identity_interference_stress.py`.
- The mechanical research coverage audit remains clean: no missing PRD TAC IDs,
  progress TAC IDs, experiment filenames, or docs filenames.
- No Kaggle polling or Kaggle status commands were run during this local
  verification pass.

## 2026-06-05 TAC-185 Relaxed Identity Routing Memory

Question:

- Can the current identity architecture move away from explicit identity rules
  and hand-defined route selection while preserving the core TAC idea:
  persistent identity state, learnable routing, and long-horizon consistency?

Implemented:

- Added `experiments/benchmark_relaxed_identity_routing_memory.py`.
- Added `tests_py/test_relaxed_identity_routing_memory.py`.
- Added `experiments/benchmark_relaxed_identity_routing_memory.py` to the
  Kaggle code bundle file list.

Experiment contract:

- Support/query supervision only.
- No explicit route labels or hidden rule labels are used in the training loss.
- Support observations are encoded into a trainable recurrent memory state.
- A learned carry update advances memory across query windows.
- A soft router reads memory and selects latent candidate program outputs.
- Hidden rule labels are used only after training to measure whether the learned
  routing structure approximates the data-generating identity programs.

Default artifact:

- `runs/benchmarks/relaxed_identity_routing_memory_2026_06_05/relaxed_identity_routing_memory.json`
- `runs/benchmarks/relaxed_identity_routing_memory_2026_06_05/RESULTS.md`

Result:

- `decision.status=relaxed_identity_routing_memory_promote_candidate`
- carried accuracy mean: `0.9167`
- horizon-tail accuracy mean: `0.9167`
- reset accuracy mean: `0.25`
- shuffled-memory accuracy mean: `0.0833`
- carried advantage over best control: `0.6667`
- route-rule NMI min: `0.75`
- route consistency min: `1.0`
- model seed count: `3`

Interpretation:

- This is a useful relaxation of TAC-183/TAC-184. The route is no longer
  supervised by labels or selected by hand-defined routing logic; it emerges
  from support/query loss and can be audited posthoc against the hidden rule
  structure.
- Persistent memory is now represented by a trainable update/carry subsystem,
  and the long-horizon gate requires carried memory to beat reset and shuffled
  controls.
- The next architectural step should replace the fixed candidate program bank
  with learned experts inside `TACTransformerLM` while preserving this same
  reset/shuffle/horizon evaluation loop.

Boundary:

- This remains a controlled local probe. It is not a full
  `TACTransformerLM` checkpoint result, not evidence on real-world language
  benchmarks, and not a claim that fixed candidate programs are sufficient for
  open-ended routing.

## 2026-06-05 TAC-186 Phase Boundary Quantification

Question:

- Where does TAC-185's identity-conditioned routing stop being useful and
  become reset-equivalent or harmful under memory/routing/task/horizon stress?

Implemented:

- Added `experiments/benchmark_phase_boundary_quantification.py`.
- Added `tests_py/test_phase_boundary_quantification.py`.
- Added `experiments/benchmark_phase_boundary_quantification.py` to the Kaggle
  code bundle file list.

Measurement contract:

- TAC-186 is not an accuracy optimization run.
- It trains the relaxed TAC-185 local model once, then perturbs evaluation
  memory coherence, routing entropy, task entropy, and horizon depth.
- Primary order parameter:
  `performance_gap = carried_accuracy - reset_accuracy`.
- Collapse index:
  `1 - shuffled_memory_accuracy / carried_accuracy`.
- Routing stability:
  NMI between clean identity routes and perturbed-memory routes.
- Phase sharpness:
  finite-difference absolute performance-gap slope along each axis.

Default artifact:

- `runs/benchmarks/phase_boundary_quantification_2026_06_05/phase_boundary_quantification.json`
- `runs/benchmarks/phase_boundary_quantification_2026_06_05/phase_heatmaps.json`
- `runs/benchmarks/phase_boundary_quantification_2026_06_05/RESULTS.md`

Default grid:

- Memory levels: `0, 1, 2, 3`
- Routing levels: `0, 2, 4`
- Task levels: `0, 2, 4`
- Horizon levels: `0, 2, 4`
- Total cells: `108`

Result:

- `decision.status=phase_boundary_mapped`
- mean performance gap: `0.3001`
- min performance gap: `-0.25`
- max performance gap: `0.75`
- harmful-memory cells: `17`
- mapped boundary slices: `9`
- max phase sharpness: `1.0`
- memory mean absolute slope: `0.1764`
- routing mean absolute slope: `0.15625`
- task mean absolute slope: `0.00152`
- horizon mean absolute slope: `0.00217`

Interpretation:

- TAC-186 turns TAC-185 from a pass/fail architecture probe into a measurement
  instrument. The strongest phase cliff in this default run is memory
  coherence, followed by routing entropy. Task and horizon axes are weaker in
  this controlled grid, which means the next useful expansion should increase
  task/horizon difficulty rather than further optimizing the model.
- The presence of negative performance-gap cells confirms the critical risk in
  the attached notes: identity memory can become actively harmful when memory
  coherence and routing are misaligned.

Boundary:

- This is a controlled local phase diagram over the TAC-185 probe, not a full
  `TACTransformerLM` checkpoint result, not a real-world language benchmark,
  and not evidence that the current default architecture should change.

## 2026-06-05 TAC-187 ATS TAC FP32 Optimizer-Health Repair

Failure diagnosis:

- TAC-175 external validation reached a real pass/fail decision, but the TAC
  control failed because the TAC training run did not optimize.
- Completed TAC manifests report `precision=fp16`,
  `fail_on_unhealthy_optimization=false`, and `min_gradient_norm=0.0`.
- Metrics show the fp16 grad scaler decayed to `0.0` from about step `200`
  onward, with `gradient_norm=0.0` and
  `optimization_health.status=failed` / `grad_scaler_scale_collapsed`.
- TAC eval accuracy stayed around `0.001-0.003`, best eval loss was
  `6.359759747982025`, and ATS raw predictions were degenerate byte fragments
  such as `ؘ��`.
- The vanilla checkpoint used the same ATS scorer and emitted coherent answer
  strings, so this is not an answer-extraction or Kaggle output-download bug.

Implemented repair:

- Updated ATS recommended TAC command generation in
  `tac_transformer/ats_transfer.py` to use base 5k, `seq_len=176`,
  `--precision fp32`, `--min-healthy-gradient-norm 1e-12`, and
  `--fail-on-unhealthy-optimization`.
- Updated selected-MI Kaggle instructions in
  `kaggle/make_agentic_training_bundle.py` and `kaggle/README.md` to use the
  same fp32/fail-fast health contract.
- Regenerated
  `runs/benchmarks/ats_transfer_training_corpus_2026_06_05` so
  `ats_transfer_training_manifest.json` and `RESULTS.md` carry the repaired
  command.
- Rebuilt
  `runs/kaggle_agentic_training_bundle/best-tac-agentic-training-bundle.zip`
  and verified the zip contains the repaired strings.
- Staged repaired Kaggle TAC kernels:
  - `runs/kaggle_ats_transfer_tac_base_fp32_eweewee2_2026_06_05`
  - `runs/kaggle_ats_transfer_tac_base_fp32_jeffkolo_2026_06_05`

Verification:

- Focused ATS transfer tests and the selected-MI bundle instruction test passed.
- `py_compile` passed for the touched Python files and staged wrappers.
- `json.tool` passed for staged kernel metadata and aggregate artifacts.
- A two-step local CPU smoke at
  `runs/benchmarks/tac_ats_fp32_health_smoke_2026_06_05` passed the optimizer
  health gate with `precision=fp32`, `grad_scaler_scale=1.0`, and gradient norm
  about `1.7`.

External blocker:

- Pushing `eweewee2/tac-ats-transfer-tac-base-fp32-5k-2026-06-05` returned
  `Maximum weekly GPU quota of 30.00 hours reached`.
- Pushing `jeffkolo/tac-ats-transfer-tac-base-fp32-5k-2026-06-05` returned the
  same quota error.
- On the 2026-06-06 heartbeat, retrying the original repaired slugs returned
  `Notebook not found` for both eweewee2 and jeffkolo, consistent with stale
  failed-create records.
- Fresh unique repair stages were created:
  - `runs/kaggle_ats_transfer_tac_base_fp32_eweewee2_0033_2026_06_06`
  - `runs/kaggle_ats_transfer_tac_base_fp32_jeffkolo_0033_2026_06_06`
- `eweewee2/tac-ats-transfer-tac-base-fp32-0033-2026-06-06` pushed
  successfully as version 1.
- Source pull artifact:
  `runs/kaggle_outputs/tac187_fp32_source_pull_eweewee2_0033_20260606`.
- Source evidence confirms `kernel_run_version=3`, fp32 training/scoring,
  `--min-healthy-gradient-norm 1e-12`, `--fail-on-unhealthy-optimization`, and
  output directory `tac_ats_transfer_tac_base_fp32_5k`.
- Initial files/logs/output were empty at launch and `kaggle kernels status`
  still returned HTTP 500, so the repaired external TAC run was initially
  monitored through direct output pulls.

Completed fp32 external result:

- Completed output was downloaded from
  `eweewee2/tac-ats-transfer-tac-base-fp32-0033-2026-06-06` to
  `runs/kaggle_outputs/tac187_fp32_completed_eweewee2_0033_20260606`.
- The run completed `5000/5000` steps with `stopped_for_time=false`.
- The fp32 health repair worked: final `optimization_health.status=passed`,
  `precision=fp32`, `grad_scaler_scale=1.0`, and
  `gradient_norm=0.038461752235889435`.
- The model optimized normally enough to reach latest eval accuracy
  `0.6187855113636364` with best eval loss `2.428375542163849`.
- Official aggregate artifact:
  `runs/benchmarks/ats_checkpoint_comparison_tac187_fp32_2026_06_06`.
- Decision: `ats_external_transfer_fail`.
- Scores: TAC train/test `0.0/0.0`; vanilla train/test
  `0.298828125/0.0`; TAC test advantage `0.0`; missing predictions `0`.
- Best checkpoint was step `250` and scored `0/256` exact matches on every
  ATS split/task. Raw generations were answer-shaped but wrong/malformed
  strings rather than the previous fp16 byte-fragment corruption.
- Downloaded `last.pt` and ran a compact 8-example CPU diagnostic at
  `runs/benchmarks/tac187_fp32_last_compact_score_2026_06_06`; final step
  `5000` also scored `0/8` exact matches, ruling out a best-checkpoint-only
  selection failure.

Conclusion:

- TAC-187 fixed the original failure mechanism, fp16 optimizer collapse.
- The remaining ATS failure is a capability/objective/output-alignment issue:
  a healthy TAC checkpoint learns token statistics but does not exact-copy the
  ATS answer strings under the current training and generation contract.

## 2026-06-05 TAC-188 Research-Integrated Memory-Advantage Model Version

Attachment question:

- The highest-leverage experiment is whether persistent computational identity
  creates a measurable long-horizon memory advantage over transformer plus
  retrieval baselines under equal resource constraints.
- The concrete success story should be a simple curve such as
  `Context Tokens Required vs Task Success` or
  `Days Since Instruction vs Accuracy`.

Implemented model version:

- Added the opt-in `memory_advantage_config(...)` preset in
  `tac_transformer/presets.py` and exported it through `tac_transformer`.
- The preset composes the best/promoted research mechanisms into one candidate:
  RMSNorm, SwiGLU, RoPE, linear experts, `base_semantic` top-2 routing,
  route load balancing, program-conditioned persistent memory writes, CREB
  memory allocation, content-addressed two-step synthesis reads, content
  reconsolidation, `identity_first` attention, a gated residual memory adapter,
  and `program_memory_graph` coalition context.
- Added `memory_advantage_training_kwargs(...)` with selected-MI routing
  supervision, chunked-memory support weights, `precision=fp32`,
  `min_healthy_gradient_norm=1e-12`, and fail-fast optimization health.
- Added `--preset memory_advantage` to `kaggle/train_best_tac_agentic.py` so
  the candidate can be launched by name instead of reconstructing a long flag
  list.

Benchmark contract:

- Added `experiments/benchmark_memory_advantage_model_version.py`.
- Artifact:
  `runs/benchmarks/memory_advantage_model_version_2026_06_05/memory_advantage_model_version.json`.
- Decision: `memory_advantage_model_version_ready`.
- Default parameter counts:
  - `memory_advantage_tac`: `2097684`
  - `parameter_matched_vanilla`: `2106264`
  - `current_best_tac`: `1899796`
- Equal-resource controls recorded:
  - parameter-matched vanilla transformer with fixed context window
  - parameter-matched vanilla transformer plus retrieval
  - parameter-matched vanilla transformer plus memory database
  - current `best_tac_config` as an internal TAC ablation

Boundary:

- This is a model-version and benchmark-contract artifact, not a trained
  checkpoint result and not a real-world memory advantage claim.
- The next evidence step is to train this preset and the equal-resource
  controls on long-horizon sparse-reminder, identity-continuity,
  multi-session reasoning, context-efficiency, and adversarial stress suites.

## 2026-06-05 TAC-189 Controlled Long-Horizon Memory Advantage Benchmark

Attachment question answered in controlled proxy:

- Question: Does persistent computational identity create a measurable
  long-horizon memory advantage over transformer plus retrieval baselines under
  equal resource constraints?
- Controlled answer: yes on aggregate mean in the local proxy benchmark.
- Decision: `controlled_long_horizon_memory_advantage_observed`.

Benchmark:

- Added `experiments/benchmark_long_horizon_memory_advantage.py`.
- Artifact:
  `runs/benchmarks/long_horizon_memory_advantage_2026_06_05/long_horizon_memory_advantage.json`.
- Graph CSVs:
  - `context_tokens_required_vs_task_success.csv`
  - `days_since_instruction_vs_accuracy.csv`
- The benchmark trains the TAC-185-style soft-router/recurrent-memory learner
  with support/query supervision only, no explicit route-label loss, and then
  evaluates carried memory against reset and shuffled-state controls.
- Transformer controls are evaluated on the same task rows with explicit
  context-token accounting:
  - transformer current-window context
  - transformer plus retrieval, with retrieved support charged as context
  - transformer plus identity-keyed memory database, with returned observations
    charged as context

Main graph result:

| Control | Context tokens required for 90% success |
| --- | ---: |
| `tac_carried_identity_state` | `6` |
| `transformer_memory_db` | `14` |
| `transformer_retrieval` | `22` |
| `transformer_window` | `62` |

Aggregate metrics:

- TAC carried accuracy mean: `0.9166666666666666`
- TAC carried accuracy min across default model seeds: `0.75`
- TAC reset-state accuracy mean: `0.25`
- TAC shuffled-memory accuracy mean: `0.08333333333333333`
- Best transformer control at TAC's 6-token context budget: `0.25`
- TAC advantage at the 6-token context budget: `0.6666666666666666`
- Nearest control token savings at 90% success: `8`

Interpretation:

- The benchmark produces the requested investor-facing curve:
  `Context Tokens Required vs Task Success`.
- In this controlled proxy, TAC's carried identity memory reaches the 90%
  success threshold with only query-token context, while retrieval and memory-db
  controls also succeed but must spend extra context tokens to re-present
  support observations.
- The result is evidence for a context-efficiency memory advantage, not a claim
  that current external TAC checkpoints have learned the behavior.

Seed robustness caveat:

- Default seeds are `5`, `7`, and `11`.
- Seeds `5` and `11` reach `1.0` carried accuracy; seed `7` reaches `0.75`.
- The aggregate mean clears the benchmark gate, but not every default seed
  clears the 90% success threshold. The next run should improve seed robustness
  before using this as a strong fundraising claim.

Boundary:

- This is a controlled local benchmark over synthetic long-horizon identity
  tasks.
- It is not a trained external `TACTransformerLM` checkpoint result, not a
  product benchmark, and not evidence that TAC beats production retrieval
  systems in the wild.

## 2026-06-05 TAC-190 Kaggle TAC Training Speed Profile

Problem:

- User observed that TAC training on Kaggle takes much longer than the vanilla
  LLM control.
- Prior local research already showed the risk: dense TAC identity fields and
  content-addressed reads add real per-step work; sparse routing alone did not
  produce speedups without fused sparse kernels.

Implemented speed profile:

- Added the opt-in `kaggle_fast_tac_config(...)` preset.
- The preset keeps the Run5B semantic TAC path:
  `base_semantic` top-2 routing, load balancing, content-addressed two-step
  synthesis reads, identity-first attention, and gated residual memory adapter.
- It reduces avoidable work with:
  - `content_read_query_top_k=8`
  - `attention_window_size=128`
- Added `kaggle_fast_tac_training_kwargs(...)` with selected-MI routing
  pressure, `precision=fp32`, nonzero-gradient health gate, and fail-fast
  optimization defaults.
- Exposed the profile through `kaggle/train_best_tac_agentic.py` as
  `--preset kaggle_fast_tac`, plus `--attention-window-size` for explicit
  override.

Benchmark:

- Added `experiments/benchmark_kaggle_tac_training_speed_profile.py`.
- Artifact:
  `runs/benchmarks/kaggle_tac_training_speed_profile_2026_06_05/kaggle_tac_training_speed_profile.json`.
- Local CPU benchmark shape:
  `vocab_size=512`, `d_model=64`, `n_heads=4`, `n_layers=1`,
  `n_programs=12`, `seq_len=64`, `batch_size=2`, `iters=3`,
  `torch_threads=1`.

Local result:

| Variant | Tokens/s | Read-query fraction |
| --- | ---: | ---: |
| `base_tac_run5b` | `1707.3959` | `1.0` |
| `kaggle_fast_tac` | `1860.3669` | `0.25` |
| `parameter_matched_vanilla` | `3445.0678` | `0.0` |

Interpretation:

- Decision: `kaggle_fast_tac_profile_ready_for_external_validation`.
- Structural read-work reduction vs full content reads: `0.75`.
- Fast TAC local TPS ratio vs base TAC: `1.0896`.
- Fast TAC local TPS ratio vs parameter-matched vanilla: `0.5400`.
- Vanilla remains `1.8518x` faster on the local CPU microbenchmark, so this
  narrows the gap but does not eliminate it.

Boundary:

- This proves the opt-in speed profile and local training-step telemetry.
- It does not prove Kaggle T4 wall-clock speedup or capability preservation.
- Next evidence step is to run Kaggle with `--preset kaggle_fast_tac`, compare
  `metrics.jsonl` `tokens_per_second` against the current TAC and vanilla jobs,
  and then decide whether deeper kernel work is needed.

## 2026-06-05 TAC-191 Lossless Local TAC Chunked-State Speedup

Problem:

- User reported TAC is still slow on local runs and asked for a speedup without
  losing capability.
- Local environment for this pass: PyTorch `2.12.0+cpu`, CUDA unavailable.
- Baseline artifact:
  `runs/benchmarks/local_speed_baseline_2026_06_05/kaggle_tac_training_speed_profile.json`.
- Baseline local shape:
  `vocab_size=512`, `d_model=64`, `n_heads=4`, `n_layers=1`,
  `n_programs=12`, `seq_len=64`, `batch_size=2`, `iters=3`,
  `torch_threads=1`.

Profiling finding:

- TAC local training was dominated by many small auxiliary/state operations
  (`mul`, `sum`, `_to_copy`, vector norms, metric reductions), not a single
  matmul or attention kernel.
- In chunked-state training, the context half is used only for context
  next-token loss and identity-state handoff into the query half.
- The context half's auxiliary losses and metrics were being computed even
  though they are not returned, not included in the optimizer auxiliary loss,
  not used by selected-MI route loss, and not logged.

Implemented lossless cleanup:

- `forward_language_model_window(...)` now calls the chunked context half with
  `collect_auxiliary=False` and `collect_metrics=False`.
- The query half still collects auxiliary losses by default, preserving the
  training objective and route-loss inputs.
- `TACTransformerLM`, `TACTransformerBlock`, and `IdentityFieldLayer` now
  accept `collect_metrics`; when false, auxiliary losses are still computed but
  logging-only metrics use the minimal metric path.
- `kaggle/train_best_tac_agentic.py` now collects train metrics only on
  observable steps: log, eval, checkpoint, specialization checkpoint, and final
  step.

Capability preservation checks:

- Added `tests_py/test_lossless_local_tac_speedup.py`.
- Tests verify the context half skips auxiliary collection while preserving
  state handoff, returned logits, and weighted next-token loss.
- Real TAC test verifies context `collect_auxiliary=True` vs `False` produces
  identical carried identity state, query logits, and query loss.
- Real TAC metric-deferral test verifies `collect_metrics=False` preserves
  logits, total loss, and auxiliary losses.

Measured result:

| Artifact | Fast TAC tokens/s | Fast/vanilla | Notes |
| --- | ---: | ---: | --- |
| `local_speed_baseline_2026_06_05` | `1149.0545` | `0.4318` | before TAC-191 |
| `local_lossless_context_aux_speedup_2026_06_05` | `1643.4754` | `0.4545` | same 3-iter shape after context aux skip |

Interpretation:

- Fast TAC absolute local throughput improved by about `1.43x` against the
  saved baseline artifact on the same shape.
- The vanilla gap remains; this is a real local speedup, not a complete kernel
  parity fix.
- Short metric-deferral microbenchmarks were noisy and did not produce a
  separate reliable fast-preset gain, so metric deferral is retained as
  trainer overhead control rather than claimed as the headline win.

Boundary:

- This is lossless with respect to capability-critical outputs tested here:
  logits, next-token loss, auxiliary losses, route-loss inputs, and carried
  identity state.
- It does not remove TAC mechanisms, reduce routing/memory/attention capacity,
  or change the optimizer objective.

## 2026-06-05 TAC-192 Local TAC Efficiency Experiment Matrix

Problem:

- User supplied a concrete optimization list after TAC-191: fuse many small
  TAC ops, amortize auxiliary losses, sparsify/cache routing and memory,
  reduce two-pass overhead, reallocate parameters toward dense compute, and
  apply CPU dispatch/thread fixes.
- Current local environment: PyTorch `2.12.0+cpu`, CUDA unavailable,
  `torch.compile` available, `torch-threads=1`, `interop-threads=1` in the
  final matrix run.
- Fresh current-code baseline before edits:
  `runs/benchmarks/local_efficiency_matrix_baseline_2026_06_05`.
  With deferred train metrics, fast TAC reported `1777.1245` tokens/s and
  parameter-matched vanilla reported `3347.2132` tokens/s.

Implemented matrix:

- Added `experiments/benchmark_local_tac_efficiency_matrix.py`.
- Artifact:
  `runs/benchmarks/local_tac_efficiency_matrix_2026_06_05/local_tac_efficiency_matrix.json`.
- The matrix compares:
  - `eager_full_aux`
  - `eager_metrics_deferred`
  - `eager_aux_every_2`
  - `eager_aux_every_4`
  - `torch_compile_reduce_overhead`
  - `vanilla_reference`
- The artifact also records explicit non-promotions:
  - `triton_identity_kernel`: not applicable on the local CPU-only run
  - `foreach_identity_ops`: deferred because current identity tensors are
    already stacked rather than many independent tensor-list ops
  - `routing_cache_or_hard_routing`: deferred because route dynamics change
    and need long-horizon capability validation
  - `two_pass_amortized_state`: deferred because single-pass cached-state
    training changes the batch objective
  - `parameter_reallocation`: deferred because it changes capacity allocation
    and requires equal-parameter retraining

Longer local CPU matrix result:

| Variant | Tokens/s | Speed vs full aux | Held-out loss delta | Status |
| --- | ---: | ---: | ---: | --- |
| `eager_full_aux` | `1982.4279` | `1.0000` | `0.0000` | baseline |
| `eager_metrics_deferred` | `1813.9325` | `0.9150` | `0.0000` | not promoted in this longer run |
| `eager_aux_every_2` | `2099.8426` | `1.0592` | `-0.00004` | opt-in candidate |
| `eager_aux_every_4` | `2276.8946` | `1.1485` | `-0.0006` | opt-in candidate |
| `vanilla_reference` | `4307.4206` | `2.1728` | not comparable | reference |

Compile finding:

- `torch.compile(mode="reduce-overhead", backend="inductor")` was attempted.
- It did not run locally because CPU Inductor could not find the MSVC compiler
  executable `cl.exe`.
- Dynamo also reported a graph break at
  `tac_transformer/model.py::_ensure_at_least_one_adaptive_route` due
  `if not bool(missing.any())`, so even with a compiler installed the routing
  path needs scalar-control-flow cleanup before fullgraph-style fusion is likely
  to work well.

Trainer integration:

- Added `forward_language_model_window(..., collect_auxiliary=True)` so the
  query half can skip auxiliary-loss construction on non-cadence steps while
  preserving the TAC-191 context-half skip.
- Added `--aux-loss-cadence N` to `kaggle/train_best_tac_agentic.py`.
- Default is `1`, preserving the existing every-step objective.
- A local two-step trainer smoke under
  `runs/benchmarks/local_aux_cadence_trainer_smoke_2026_06_05` completed with
  `--aux-loss-cadence 4`, `auxiliary_loss_collected=false`, nonzero gradient
  norm, fp32 GradScaler scale `1.0`, and optimizer health passed.

Interpretation:

- The best measured local candidate is auxiliary loss cadence every 4 steps:
  about `1.15x` over eager full auxiliary collection on the longer CPU
  microbenchmark.
- This is not enough to match vanilla: vanilla is still about `1.89x` faster
  than aux-every-4 TAC on the same matrix shape.
- The highest-impact next fix remains true fusion/compiler work, but local
  Windows CPU needs MSVC `cl.exe` for Inductor and the TAC routing code has
  scalar graph breaks that should be cleaned before expecting compile wins.

Boundary:

- Aux cadence changes the regularization cadence, so it is an opt-in speed
  candidate, not a lossless default.
- The held-out loss proxy stayed within tolerance in the local matrix, but this
  does not prove long-horizon memory/routing capability preservation.
- Triton and GPU fusion claims were not tested because this machine is CPU-only.

## 2026-06-05 TAC-193 Opt-In CPU Research TAC Version

User clarification:

- This is research, so a separate TAC version is acceptable as long as it does
  not disturb the main TAC architecture.
- The target is a CPU-testable experimental version that applies CPU-compatible
  forms of the TAC-192 ideas even when `torch.compile`, Triton, routing cache,
  hard routing, foreach refactors, and parameter reallocation were previously
  deferred.

Implemented version:

- Added `cpu_research_tac_config(...)` and
  `cpu_research_tac_training_kwargs(...)`.
- Exposed the preset through `tac_transformer` and
  `kaggle/train_best_tac_agentic.py` as `--preset cpu_research_tac`.
- The preset uses the same `TACTransformerLM` class and remains opt-in. Main
  presets are unchanged.

CPU-compatible tactics applied:

- Hard/lower-k routing: `routing_type=base_semantic`, `routing_top_k=1`.
- Smaller program bank: `n_programs=8` instead of the fast profile's `12`.
- Sparse content reads: `content_read_query_top_k=4`.
- Single-step content read: `content_read_steps=1`.
- Local attention: `attention_window_size=64`.
- Cheaper memory adapter: `memory_adapter_type=residual` instead of
  `gated_residual`.
- Auxiliary loss amortization: training default `aux_loss_cadence=4`.
- CPU thread pinning: training defaults `torch_threads=1` and
  `torch_interop_threads=1`.
- Category-route supervision is disabled by default with
  `category_route_weight=0.0` to avoid extra labeled-route work in local CPU
  research runs.

Benchmark:

- Added `experiments/benchmark_cpu_research_tac_version.py`.
- Artifact:
  `runs/benchmarks/cpu_research_tac_version_2026_06_05/cpu_research_tac_version.json`.
- Local CPU benchmark shape:
  `vocab_size=512`, `d_model=64`, `n_heads=4`, `n_layers=1`,
  `seq_len=64`, `batch_size=2`, `warmup=2`, `iters=10`,
  `torch_threads=1`, `interop_threads=1`.

Same-run ablation result:

| Variant | Technique stack | Tokens/s | Speed vs fast full-aux TAC | Parameters | Held-out loss delta |
| --- | --- | ---: | ---: | ---: | ---: |
| `kaggle_fast_tac_reference` | Baseline fast TAC, full aux | `1682.9961` | `1.0000` | `259110` | `0.0000` |
| `kaggle_fast_tac_aux_every_4` | Aux every 4 only | `1819.1413` | `1.0809` | `259110` | `-0.0001` |
| `cpu_research_arch_full_aux` | CPU research architecture only | `2031.9437` | `1.2073` | `183641` | `0.0257` |
| `cpu_research_tac` | CPU research architecture + aux every 4 | `2051.5230` | `1.2190` | `183641` | `0.0258` |
| `vanilla_reference` | Vanilla reference | `4478.1487` | `2.6608` | `179120` | `-0.0786` |

Combination analysis:

| Comparison | Speed ratio |
| --- | ---: |
| Aux every 4 on fast TAC | `1.0809` |
| CPU research architecture with full aux | `1.2073` |
| CPU research architecture + aux every 4 | `1.2190` |
| Aux every 4 gain on CPU research architecture | `1.0096` |
| CPU architecture gain after aux every 4 | `1.1277` |
| Vanilla speed vs combined CPU research TAC | `2.1828` |

Prior TAC-192 local efficiency reference added to the artifact:

| Variant | Tokens/s | Speed vs full-aux TAC | Held-out loss delta | Scope |
| --- | ---: | ---: | ---: | --- |
| `tac_aux_every_4` | `2276.89` | `1.15` | `-0.0006` | Prior local efficiency matrix result. The `1.15x` ratio is against the TAC eager full-aux baseline from TAC-192, not against the current CPU research benchmark baseline. |

Parameter effect:

- Fast TAC identity-field parameters: `77414`.
- CPU research TAC identity-field parameters: `39129`.
- Identity-field parameter share dropped by about `0.0857`, moving the local
  version away from fragmented controller/memory overhead and toward cheaper
  dense backbone work.

Trainer smoke:

- Ran a real two-step trainer smoke under
  `runs/benchmarks/cpu_research_tac_trainer_smoke_2026_06_05`.
- Command used `--preset cpu_research_tac` without manually setting thread or
  cadence flags, proving preset defaults apply.
- Manifest recorded `torch_threads=1`, `torch_interop_threads=1`,
  `routing_top_k=1`, `content_read_steps=1`, `content_read_query_top_k=4`,
  `memory_adapter_type=residual`, and `aux_loss_cadence=4`.
- Training completed 2/2 steps with `auxiliary_loss_collected=false` on those
  non-cadence steps, nonzero gradient norms, fp32 GradScaler scale `1.0`, and
  `optimization_health.status=passed`.

Interpretation:

- `cpu_research_tac` is a practical CPU research branch: about `1.22x` faster
  than the current fast full-aux TAC reference on the local benchmark shape.
- In this same-run ablation, the CPU architecture simplification contributes
  most of the speedup (`1.2073x`), while aux every 4 adds only `1.0096x` after
  the CPU architecture is already applied.
- It is still not vanilla-speed: vanilla remains about `2.18x` faster than
  `cpu_research_tac` on the same artifact.
- The TAC-192 `tac_aux_every_4` result remains recorded as a prior reference,
  but the current artifact now also measures aux every 4 inside the same
  benchmark family.
- This version intentionally trades capacity/regularization cadence for local
  experimentation speed. It is not a no-capability-loss claim.

Boundary:

- `cpu_research_tac` does not change the main TAC architecture or defaults.
- It does not prove long-horizon memory, ATS transfer, or external benchmark
  preservation.
- Any individual tactic should only be promoted into main TAC after passing the
  long-horizon memory and ATS capability gates.

## 2026-06-06 TAC-194 Run 5B Best-Capability Fast Launch Preset

User request:

- Run the Run 5B line with the best capability version researched locally and
  include the implemented training speedup.
- Use the lessons from TAC-160 through TAC-189 rather than relaunching the
  stale Run 5B or plain `memory_advantage` preset unchanged.

Applicable research synthesis:

- TAC-160 through TAC-168 agentic scratchpad/controller work remains useful for
  Phase D policy/runtime evaluation, but it is not part of the plain LM trainer
  preset.
- TAC-169 adds the missing capability ingredient for coalition memory:
  `program_memory_graph` context plus `content_read_gate_type="cue_match"` for
  structured cue-chain continuation.
- TAC-173 replaces token-level `mi` with record-level `selected_mi` for the
  same selected-program surface measured by the Phase B specialization gate.
- TAC-188 provides the strongest named model-version stack:
  base-semantic top-2 routing, 24 programs, program-conditioned memory writes,
  CREB k=6 allocation, larger content store, identity-first attention, gated
  residual memory adapter, and graph coalition context.
- TAC-189 gives controlled local evidence for the memory-advantage hypothesis,
  but not external checkpoint validation.
- The implemented speed path used for this launch is the trainer-side speed
  implementation already available after the later speed work: lossless
  chunked context auxiliary skipping/metric deferral plus opt-in auxiliary-loss
  cadence. The capacity-reducing `cpu_research_tac` architecture is not used
  because it is explicitly a speed research branch, not the best capability
  candidate.

Implemented preset:

- Added `run5b_best_capability_fast_config(...)`.
- Added `run5b_best_capability_fast_training_kwargs(...)`.
- Exposed `--preset run5b_best_capability_fast` through
  `kaggle/train_best_tac_agentic.py`.
- The preset resolves to:
  - `routing_type="base_semantic"`
  - `routing_top_k=2`
  - `n_programs=24`
  - `program_memory_update_type="program_conditioned"`
  - `memory_allocation_type="creb"`
  - `memory_allocation_k=6`
  - `memory_read_type="content_addressed"`
  - `content_read_steps=2`
  - `content_read_gate_type="cue_match"`
  - `content_read_query_top_k=8`
  - `coalition_context_type="program_memory_graph"`
  - `memory_adapter_type="gated_residual"`
  - `identity_attention_type="identity_first"`
  - `attention_window_size=128`
- Training defaults resolve to:
  - `category_route_objective="selected_mi"`
  - `category_route_weight=0.1`
  - `precision="fp32"`
  - `min_healthy_gradient_norm=1e-12`
  - `fail_on_unhealthy_optimization=1`
  - `aux_loss_cadence=4`

Launch command:

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

Verification:

- Focused preset and CLI tests pass.
- A real two-step CPU trainer smoke with `--preset run5b_best_capability_fast`
  completed and wrote a manifest with `preset="run5b_best_capability_fast"`,
  `content_read_gate_type="cue_match"`, graph coalition context,
  `category_route_objective="selected_mi"`, `category_route_weight=0.1`,
  `aux_loss_cadence=4`, fp32 precision, fail-fast optimizer health, nonzero
  gradient norm, and `optimization_health.status="passed"`.
- The generated Kaggle bundle was rebuilt at
  `runs/kaggle_agentic_training_bundle/best-tac-agentic-training-bundle.zip`;
  `RUN_ON_KAGGLE.md` contains the new launch command.

Decision:

- Decision: `run5b_best_capability_fast_launch_ready`.
- Boundary: this is a smoke-tested launch preset and command path. It is not an
  external Phase B recovery, ATS transfer pass, long-horizon real checkpoint
  proof, or product benchmark result until the Kaggle run completes and clears
  the same downstream gates.

External launch addendum:

- Created private code dataset
  `jeffkolo/tac-run5b-best-capability-fast-code-2026-06-06` from the rebuilt
  bundle.
- Pushed private GPU kernel
  `jeffkolo/tac-run5b-best-capability-fast-20k-2026-06-06` as version 1.
- Kernel metadata attaches the fresh code dataset and
  `jeffkolo/tac-run5b-capability-data-2026-06-03`.
- Source pull artifact:
  `runs/kaggle_outputs/run5b_best_capability_fast_source_pull_jeffkolo_20260606`.
- Source pull confirms `kernel_run_version=1`,
  `--preset run5b_best_capability_fast`, `--aux-loss-cadence 4`,
  `--precision fp32`, `--min-healthy-gradient-norm 1e-12`, and
  `--fail-on-unhealthy-optimization`.
- Kaggle session status endpoint returned HTTP 500 immediately after push, but
  `kaggle kernels list -m` shows the kernel with fresh `lastRunTime`
  `2026-06-06 06:20:12.140000` UTC.
- Output pull is currently empty, so the run is accepted but not yet locally
  validated from completed outputs.

Failure and retry addendum:

- User confirmed the upload succeeded but the Kaggle run failed.
- Pulled failed output to
  `runs/kaggle_outputs/run5b_best_capability_fast_failed_output_jeffkolo_20260606`.
- The failure log shows two `Tesla T4` devices were available and the script
  started distributed training, so the failure was not a GPU allocation issue.
- Both DDP ranks failed with
  `Expected to have finished reduction in the prior iteration before starting a new one`.
- The unused parameter index was `270`, which maps in the same local preset to
  `blocks.7.identity_field.raw_energy_costs`.
- Cause: `aux_loss_cadence=4` skips auxiliary-loss collection on non-cadence
  steps. During those steps, `raw_energy_costs` can be excluded from the loss
  graph even though DDP expects every trainable parameter to participate before
  the next reduction.
- Fix: `forward_language_model_window` now adds a zero-weight safety term from
  `output.aux.used_energy` in both non-chunked and chunked paths. This preserves
  the objective and auxiliary cadence while keeping the energy-cost parameter in
  the autograd graph for DDP.
- Added regression coverage proving non-auxiliary chunked forward keeps the
  energy-cost gradient path alive.
- Verification after the fix:
  - `python -m unittest tests_py.test_lossless_local_tac_speedup`
  - `python -m py_compile tac_transformer\training.py tests_py\test_lossless_local_tac_speedup.py`
  - `python -m unittest tests_py.test_memory_advantage_model_version tests_py.test_lossless_local_tac_speedup tests_py.test_kaggle_tac_training_speed_profile`
- Rebuilt the Kaggle bundle, published a new private code dataset version for
  `jeffkolo/tac-run5b-best-capability-fast-code-2026-06-06`, and pushed kernel
  `jeffkolo/tac-run5b-best-capability-fast-20k-2026-06-06` as version 2.
- Source pull artifact:
  `runs/kaggle_outputs/run5b_best_capability_fast_v2_source_pull_jeffkolo_20260606`.
- Source pull confirms `kernel_run_version=2`,
  `--preset run5b_best_capability_fast`, and `--aux-loss-cadence 4`.
- `kaggle kernels list --user jeffkolo --sort-by dateRun` shows fresh
  `lastRunTime` `2026-06-06 06:43:44.460000` UTC for the v2 push.
- Kaggle's status endpoint still returns HTTP 500 for this kernel, and output
  pulls are currently empty, so v2 is accepted/pushed but not yet externally
  validated from completed artifacts.

## 2026-06-06 Parallel Reasoning, Search, And Intelligence Research Restart

User request:

- Continue the reasoning, search, and intelligence research in parallel.

Read-only subagent split:

- Reasoning track: TAC-189 is strong evidence for controlled long-horizon memory
  efficiency, but it is still vulnerable to the interpretation that TAC improved
  recall/context accounting rather than compositional reasoning. The next CPU
  gate should hold direct recall constant and test matched 2-hop/3-hop
  composition against retrieval, memory-database, reset, shuffled, and
  recall-oracle controls.
- Search track: TAC-176/TAC-177 prove controlled parallel trajectory mechanics,
  but prior search probes do not yet consume live TAC output/state in a runtime
  loop. The next CPU gate should keep planning external: generate candidates,
  score with a label-free structural verifier, commit only verified scratchpad
  state, and avoid adding planner/value/world heads into `TACTransformerLM`.
- Intelligence/capability track: TAC-194 v2 is the immediate external blocker.
  The latest confirmed state remains source-pull evidence for
  `kernel_run_version=2` and the intended preset/flags; completed output
  artifacts are not yet available. Phase D and ATS checkpoint claims remain
  blocked until external checkpoint evidence exists.

PRD split:

- TAC-195: controlled multi-hop reasoning-vs-recall advantage benchmark.
- TAC-196: live TAC-state runtime search planner.
- TAC-197: Run 5B best-capability external validation gate.

Boundary:

- This restart is a coordination and gating step. It does not claim TAC already
  improves general reasoning, runtime search, or broad intelligence. The three
  streams are designed to turn those claims into falsifiable local/external
  artifacts with no-go criteria.

## TAC-197 Run 5B Best-Capability External Validation Gate

Implemented a monitor/status artifact for the TAC-194 v2 external run.

Added:

- `experiments/monitor_run5b_best_capability_external_validation.py`
- `tests_py/test_external_run5b_best_capability_validation.py`

Live external check:

- `kaggle kernels status jeffkolo/tac-run5b-best-capability-fast-20k-2026-06-06`
  still returns HTTP 500.
- `kaggle kernels output ... -p
  runs/kaggle_outputs/run5b_best_capability_fast_v2_completed_jeffkolo_20260606`
  produced no completed output files.

Artifact:

- `runs/benchmarks/external_run5b_best_capability_fast_v2_validation_2026_06_06/external_run5b_best_capability_status.json`
- `runs/benchmarks/external_run5b_best_capability_fast_v2_validation_2026_06_06/RESULTS.md`

Decision:

- `decision.status = external_pending`
- Source pull passes: `kernel_run_version=2` and
  `preset=run5b_best_capability_fast`.
- Missing required outputs: `final_summary.json`, `metrics.jsonl`, `best.pt`,
  and `last.pt`.
- No capability claim is allowed from this external run yet.

Unblock status:

- Phase B: still blocked by external pending output.
- Phase D: still blocked by external pending output.
- ATS checkpoint scoring: still blocked by external pending output.
- Long-horizon checkpoint validation: still blocked by external pending output.

## TAC-195 Controlled Multi-Hop Reasoning-vs-Recall Advantage Benchmark

Question:

- Does the carried identity-state mechanism support compositional multi-hop
  reasoning beyond direct edge recall?

Implemented:

- `experiments/benchmark_multihop_reasoning_advantage.py`
- `tests_py/test_multihop_reasoning_advantage.py`
- Added the benchmark to `kaggle/make_agentic_training_bundle.py`.

Benchmark contract:

- Same task rows across controls.
- Chain lengths: 1, 2, and 3.
- Direct recall is held constant at chain length 1.
- Recall-only controls can recover one edge but cannot compose chains.
- TAC carried identity state follows the carried identity graph for the
  configured chain length.
- Selection/training flags report no target-label or hidden-route-label use.

Artifact:

- `runs/benchmarks/multihop_reasoning_advantage_2026_06_06/multihop_reasoning_advantage.json`
- `runs/benchmarks/multihop_reasoning_advantage_2026_06_06/RESULTS.md`

Result:

- `decision.status = controlled_multihop_reasoning_advantage_observed`
- Chain length 1: TAC `1.0000`, recall oracle `1.0000`, direct regression
  `0.0000`.
- Multi-hop TAC mean: `1.0000`.
- Best recall-only multi-hop mean: `0.0000`.
- Reasoning lift over best recall-only control: `1.0000`.
- Min-seed TAC multi-hop: `1.0000`.

Boundary:

- This proves a controlled graph-composition proxy: carried identity state can
  represent and follow multi-hop edges when recall-only controls are limited to
  direct edges. It still does not prove a trained external TACTransformerLM
  checkpoint has learned general reasoning.

## TAC-196 Live TAC-State Runtime Search Planner

Question:

- Can search/planning stay outside the base model while still using TAC-state
  surfaces to improve controlled multi-hop behavior?

Implemented:

- `tac_transformer/runtime_search.py`
- `experiments/benchmark_live_tac_runtime_search.py`
- `tests_py/test_live_tac_runtime_search.py`
- Runtime-search exports through `tac_transformer/__init__.py`
- Added the benchmark to `kaggle/make_agentic_training_bundle.py`.

Benchmark contract:

- Candidate generation uses top-k first-hop logits.
- Selection uses a label-free structural verifier over the context graph.
- Verified graph hits are committed as scratchpad items.
- Target labels are used only after selection to score the benchmark.
- `hypothesis_contamination = 0.0`.
- No planner/value/world/reflection heads are added to `TACTransformerLM`.

Artifact:

- `runs/benchmarks/live_tac_runtime_search_2026_06_06/live_tac_runtime_search.json`
- `runs/benchmarks/live_tac_runtime_search_2026_06_06/RESULTS.md`

Result:

- `decision.status = runtime_search_useful`
- `single_key`: greedy `1.0000`, runtime search `1.0000`, direct regression
  `0.0000`.
- `multi_hop`: greedy `0.0000`, runtime search `1.0000`, gain `1.0000`.
- Mean committed scratchpad items on multi-hop: `32.0`.
- Hypothesis contamination: `0.0000`.

Boundary:

- This validates an external runtime-search loop on a controlled TAC-state
  surface. It does not prove external checkpoint search skill and does not
  justify promoting planner heads into the base model.

## TAC-198 ATS Answer-Only Supervision Repair

Trigger:

- TAC-187 proved the fp32 optimizer-health repair worked, but the repaired
  external TAC checkpoint still scored `0.0/0.0` train/test ATS exact match.
- The new failure shape was not optimizer collapse: final health passed,
  gradient norm was nonzero, and generations were answer-shaped but wrong.

Diagnosis:

- ATS training rows were staged as full `prompt + answer + newline` LM text.
- The external gate scores only the generated answer string, so full-LM loss
  mostly rewards prompt reconstruction and token-distribution learning.
- The existing local answer-copy probe already showed masked answer-only
  supervision is the relevant scorer-aligned objective.

Implemented local repair:

- Added `JsonlCompletionBatcher` and `JsonlLabeledCompletionBatcher` in
  `tac_transformer/training.py`.
- The new batchers read separate `prompt` and `answer` fields, feed
  `prompt + answer`, and mask labels with `-100` except answer bytes plus EOS.
- Updated chunked window loss and evaluation accuracy to weight/count only
  non-ignored labels, preventing all-ignored chunk halves from producing NaN or
  diluting answer loss.
- Added `--supervision-mode answer_only`, `--prompt-field`, and
  `--completion-field` to `kaggle/train_best_tac_agentic.py`.
- The default remains `--supervision-mode full_lm`, so normal hard-agentic
  training is unchanged.
- Preserved selected-MI routing supervision by adding the labeled completion
  batcher path.
- Updated `ats_example_to_prepared_row(...)` to persist `prompt`, then
  regenerated `runs/benchmarks/ats_transfer_training_corpus_2026_06_05`.
- Updated the ATS recommended TAC command, `kaggle/README.md`, and generated
  `RUN_ON_KAGGLE.md` instructions with the TAC-198 answer-only command.

Local artifact:

- `runs/benchmarks/tac198_ats_answer_only_trainer_smoke_2026_06_06`
- The real trainer completed `2/2` CPU steps on the staged ATS corpus with
  `supervision_mode=answer_only`, `train_records=512`, `eval_records=512`,
  `category_route_objective=selected_mi`, and optimizer health passed.
- Latest smoke gradient norm: `1.2313588857650757`.

External launch:

- Rebuilt and versioned the eweewee2 code dataset
  `eweewee2/tac-ats-transfer-code-2026-06-05`.
- Versioned the eweewee2 ATS corpus dataset
  `eweewee2/tac-ats-transfer-corpus-2026-06-05` with regenerated rows that
  include `prompt`.
- First long-slug push returned a bare Kaggle HTTP 400; shortening the slug and
  title fixed the metadata rejection.
- Accepted kernel: `eweewee2/tac-ats-ao-1240-20260606`, version 1.
- Source pull artifact:
  `runs/kaggle_outputs/tac198_answer_only_source_pull_eweewee2_1240_20260606`.
- Source confirms `kernel_run_version=4`, output dir
  `tac_ats_transfer_tac_base_answer_only_5k`, and trainer flags
  `--supervision-mode answer_only --prompt-field prompt --completion-field answer`.
- Initial output pull is empty and `kaggle kernels status` returns HTTP 500,
  matching prior Kaggle status behavior while a GPU script is queued/running.

Boundary:

- TAC-198 is locally repaired and externally launched, but it is not an ATS
  success claim yet.
- The active heartbeat `monitor-tac-198-ats-answer-only-kaggle-run` must pull
  completed output, aggregate against the existing vanilla score, and record a
  pass/fail decision before this external repair can be closed.

## 2026-06-06 TAC-194 Run 5B Continuation Addendum

The jeffkolo Run 5B best-capability fast Kaggle run stopped because of wall
time, not because of optimizer failure. The completed v2 output was pulled to
`runs/kaggle_outputs/run5b_best_capability_fast_v2_final_output_retry_jeffkolo_20260606`.

Observed state:

- `completed_steps`: 12031 of 20000.
- `stopped_for_time`: true.
- `best_eval_loss`: 0.15196701139211655.
- Optimizer health: passed.
- Latest gradient norm: 0.2333480566740036.
- Precision: fp32.
- `last.pt`: step 12031.
- `best.pt`: step 11000.

Continuation action:

- Created private resume dataset
  `jeffkolo/tac-run5b-fast-resume-12031-20260606`.
- Dataset contains `last.pt`, `best.pt`, previous `final_summary.json`,
  previous `metrics.jsonl`, previous `run_manifest.json`, and
  `resume_manifest.json`.
- Patched the existing jeffkolo kernel wrapper to copy `last.pt` and `best.pt`
  into `/kaggle/working/run5b_best_capability_fast` before launch.
- Added `--auto-resume` to the trainer command and set
  `kernel_run_version=3`.
- Pushed `jeffkolo/tac-run5b-best-capability-fast-20k-2026-06-06` as Kaggle
  kernel version 3.

Verification:

- `python -m py_compile` passed for the staged Kaggle runner.
- `python -m json.tool` passed for the staged kernel metadata.
- `kaggle datasets files` verified the resume dataset files.
- Source pull to
  `runs/kaggle_outputs/run5b_best_capability_fast_v3_source_pull_jeffkolo_20260606`
  confirms the attached resume dataset, `--auto-resume`, and
  `kernel_run_version=3`.
- `kaggle kernels list` shows fresh `lastRunTime`
  `2026-06-06 15:42:17.527000 UTC`.
- `kaggle kernels status` still returns Kaggle HTTP 500, matching previous
  status-endpoint behavior.

Boundary: this is a successful continuation launch from the step-12031
checkpoint. It is not yet the final Run 5B validation result; the continued run
still needs completed output to be pulled and evaluated.

## 2026-06-07 TAC-197 Run 5B Completed Validation

The resumed jeffkolo Run 5B best-capability fast kernel completed and now clears
the repository's external validation gate.

Pulled output:

- Kernel: `jeffkolo/tac-run5b-best-capability-fast-20k-2026-06-06`.
- Output path:
  `runs/kaggle_outputs/run5b_best_capability_fast_v3_completed_jeffkolo_20260607`.
- Source pull:
  `runs/kaggle_outputs/run5b_best_capability_fast_v3_source_pull_jeffkolo_20260606`.
- Validation artifact:
  `runs/benchmarks/external_run5b_best_capability_fast_v3_validation_2026_06_07`.
- Status artifact:
  `runs/benchmarks/external_run5b_best_capability_fast_v3_status_2026_06_07`.

Run result:

- `completed_steps`: 20000 of 20000.
- `stopped_for_time`: false.
- `start_step`: 12031.
- `auto_resume`: true.
- `best_eval_loss`: 0.14900434762239456.
- Latest eval accuracy: 0.9442471590909091.
- Optimizer health: passed.
- Latest gradient norm: 0.31759101152420044.
- Precision: fp32.
- `best.pt`: step 20000.
- `last.pt`: step 20000.

Fair-baseline gate:

- Same-backbone vanilla best eval loss: 0.1091884383931756.
- Parameter-matched vanilla best eval loss: 0.1047834949567914.
- TAC same-backbone loss gap: 0.03981590922921896.
- TAC parameter-matched loss gap: 0.04422085266560316.
- Both gaps clear the configured thresholds.

Memory and specialization gate:

- Program-memory cosine: 0.0010435144261767466.
- Specialization MI: 0.41774966834003563 bits.
- Max knockout loss delta: 0.5417789249913767.
- Max knockout selectivity span: 0.25858329934999347.
- Specialization source: standalone step-20000 report with 384 records.

Decision:

- `decision.status`: `promote`.
- `capability_claim_allowed`: true.
- Reason: TAC completed, preserved near-same-backbone capability, avoided
  program-memory collapse, and produced specialization evidence.

Downstream meaning:

- Phase B: `candidate_unblocked_for_seed_replication_audit`.
- Phase D: `candidate_unblocked_pending_phase_b_gate`.
- ATS checkpoint scoring: `candidate_unblocked`.
- Long-horizon checkpoint validation: `candidate_unblocked`.

Boundary:

- This is a real external checkpoint validation success for the Run 5B
  best-capability fast TAC variant.
- It is still not a full business/investor proof, not independent reproduction,
  and not evidence that TAC beats vanilla on every downstream product task.
  The next step is seed replication and task-specific checkpoint scoring.

## 2026-06-07 Run 5B ATS, LM, and Intelligence Evaluation

Artifact:

- `runs/benchmarks/run5b_ats_lm_intelligence_eval_2026_06_07`
- Summary: `evaluation_summary.json`
- Human report: `RESULTS.md`

Checkpoint:

- `runs/kaggle_outputs/run5b_best_capability_fast_v3_completed_jeffkolo_20260607/run5b_best_capability_fast/best.pt`
- Step: 20000.
- Model type: TAC.
- Parameters: 26,067,272.

ATS exact-match requirement:

- Requirement used by `experiments/aggregate_ats_checkpoint_runs.py`:
  train score >= 0.95, held-out test score >= 0.95, and TAC test advantage
  >= 0.10.
- A balanced compact ATS suite was built with 32 examples: train/test,
  both ATS tasks, and all four ATS domains.
- Result: 0/32 exact matches.
- Train split: 0/16.
- Held-out test split: 0/16.
- Raw generations are mostly hard-agentic schema/tag fragments such as
  `{"name": ...`, `<edge_cases>`, and malformed tool JSON, not ATS answer IDs.

ATS answer-only likelihood:

- Train answer-only loss: 6.2802.
- Train answer-only perplexity: 533.9.
- Train answer byte accuracy: 0.1330.
- Test answer-only loss: 6.8723.
- Test answer-only perplexity: 965.2.
- Test answer byte accuracy: 0.1130.

This means the ATS failure is not only a decoding-format issue. Under teacher
forcing, the checkpoint still assigns poor probability to the correct ATS answer
bytes.

Held-out hard-agentic language modelling:

- Kaggle final eval loss: 0.1490.
- Kaggle final eval perplexity: 1.1607.
- Kaggle final eval accuracy: 0.9442.
- Fresh chunked-state sampled eval loss: 0.1609.
- Fresh chunked-state sampled eval perplexity: 1.1745.
- Fresh chunked-state sampled eval accuracy: 0.9384.
- Fresh plain no-chunked-state sample loss: 0.4462.
- Fresh plain no-chunked-state sample perplexity: 1.5623.
- Fresh plain no-chunked-state sample accuracy: 0.8801.

The model is therefore a strong in-distribution byte-level language model for
the hard-agentic corpus, especially when evaluated with the same chunked-state
path used during training.

Phase D compact exact-answer probe:

- Five examples, one per Phase D task family:
  multi-hop chain retrieval, long-context retrieval, episodic fact update, tool
  selection, and delayed goal binding.
- Result: 0/5 exact matches.
- Outputs again looked like generic training-format fragments rather than exact
  answers.

Decision:

- ATS status: fail.
- Language-model status: strong in-distribution byte-level LM.
- Intelligence/task-transfer status: weak exact-answer transfer.

Boundary:

- The completed Run 5B checkpoint is useful evidence that TAC can train stably,
  maintain low program-memory cosine, produce specialization evidence, and model
  its hard-agentic corpus well.
- It is not evidence that the current checkpoint can satisfy ATS, act as a
  general assistant, or solve exact-answer reasoning tasks without additional
  supervised answer-only training, decoding repair, and task-specific scoring.

## 2026-06-07 TAC-199 Identity Weight Ratio Validation

User question: find and properly validate the optimal ratio between transformer
weights and IdentityState weights for the current Run5B best-capability-fast
architecture family.

New benchmark:

- `experiments/benchmark_identity_weight_ratio_validation.py`
- Focused tests: `tests_py/test_identity_weight_ratio_validation.py`
- Primary artifact:
  `runs/benchmarks/identity_weight_ratio_validation_2026_06_07`
- Sensitivity artifact:
  `runs/benchmarks/identity_weight_ratio_validation_step80_2026_06_07`

Method:

- Held backbone shape fixed at `d_model=48`, `n_heads=4`, `n_layers=2`,
  `seq_len=64`, `content_store_size=16`.
- Held training/eval data generation, optimizer, selected-MI route weight
  (`0.1`), batch size, and eval settings fixed.
- Swept identity capacity through program count.
- Reported total parameters, `.identity_field` parameters, transformer-side
  parameters, identity share, identity-to-transformer ratio, final eval loss,
  loss improvement, accuracy, selected-route MI, activation MI, program-memory
  cosine, active programs, and training throughput.
- Ranked both raw capability and cost-adjusted capability.

40-step result:

| Variant | Identity share | I:T ratio | Final loss | Accuracy | Selected MI | Memory cosine | TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p8 | 0.3274 | 0.4867 | 5.1622 | 0.2170 | 0.0159 | 0.5129 | 1153.8 |
| p16 | 0.4296 | 0.7533 | 5.1877 | 0.2209 | 0.0290 | 0.1577 | 837.7 |
| p20 | 0.4699 | 0.8866 | 5.2154 | 0.1944 | 0.0425 | 0.0965 | 741.6 |
| p12 | 0.3827 | 0.6200 | 5.2625 | 0.2040 | 0.0264 | 0.2458 | 1038.1 |
| p24 | 0.5049 | 1.0199 | 5.2288 | 0.2088 | 0.0531 | 0.0735 | 896.2 |

80-step sensitivity result:

| Variant | Identity share | I:T ratio | Final loss | Accuracy | Selected MI | Memory cosine | TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p8 | 0.3274 | 0.4867 | 4.1151 | 0.2309 | 0.0066 | 0.4435 | 993.3 |
| p12 | 0.3827 | 0.6200 | 4.2018 | 0.2448 | 0.0261 | 0.2225 | 580.1 |
| p16 | 0.4296 | 0.7533 | 4.1534 | 0.2218 | 0.0218 | 0.1351 | 584.8 |
| p24 | 0.5049 | 1.0199 | 4.1681 | 0.2318 | 0.0512 | 0.0629 | 682.8 |

Decision:

- Under this local controlled validation, p8 wins both raw-capability and
  cost-adjusted scoring at both 40 and 80 steps.
- The validated local budget recommendation is therefore roughly
  `identity:transformer = 0.49:1`, or about `33%` identity-field parameters and
  `67%` transformer/backbone parameters.
- The earlier p24/63% identity split should be described only as the completed
  external Run5B checkpoint allocation, not as a validated optimum.

Boundary:

- The p24 external checkpoint remains the only completed full Run5B external
  checkpoint. It also shows better route specialization and lower program-memory
  cosine in the local sweep.
- Replacing p24 at checkpoint scale requires a full p8-vs-p16-vs-p24 training
  replication with the same external validation gates, not only this CPU local
  proxy.

## 2026-06-07 TAC-200 External 5k Identity Ratio Validation

User correction: the external Kaggle validation should train for 5k steps, not
20k.

Launch:

- Code dataset:
  `jeffkolo/tac-identity-ratio-code-2026-06-07`
- p8 kernel:
  `jeffkolo/tac-identity-ratio-p8-5k-2026-06-07`
- p24 kernel:
  `jeffkolo/tac-identity-ratio-p24-5k-2026-06-07`

Controlled variables:

- Preset: `run5b_best_capability_fast`
- Data: `jeffkolo/tac-run5b-capability-data-2026-06-03`
- Scale: `base`
- `seq_len=176`
- `steps=5000`
- `batch_size=12`
- `grad_accum_steps=3`
- `eval_every=1000`
- `eval_batches=4`
- `checkpoint_every=500`
- `aux_loss_cadence=4`
- fp32, optimizer-health fail-fast, dual-T4 distributed launch
- Specialization checkpoints: 2000 and 5000

Experimental variable:

- p8: `n_programs=8`, identity share `0.3274`,
  identity-to-transformer ratio `0.4867`.
- p24: `n_programs=24`, identity share `0.5049`,
  identity-to-transformer ratio `1.0199`.

Current external state:

- Kernel version 1 for both p8 and p24 was accepted by Kaggle but failed before
  training started.
- The shared failure was `AttributeError: 'DistributedDataParallel' object has
  no attribute 'config'` from `tac_transformer/training.py` while
  `forward_language_model_window` computed masked LM loss after DDP wrapping.
- Added regression coverage:
  `tests_py.test_lossless_local_tac_speedup.LosslessLocalTacSpeedupTests.test_chunked_forward_reads_config_through_ddp_style_module_wrapper`.
- Fixed `forward_language_model_window` to read vocab size through
  `model.module.config` when a DDP-style wrapper is present.
- Focused verification passed for the new regression, py_compile, and
  `tests_py.test_lossless_local_tac_speedup` plus
  `tests_py.test_identity_weight_ratio_validation`.
- Rebuilt the Kaggle bundle, versioned
  `jeffkolo/tac-identity-ratio-code-2026-06-07`, and pushed p8/p24 kernel
  version 2.
- Immediate v2 output pulls returned no files yet.
- `kaggle kernels status` still returns Kaggle HTTP 500.
- Heartbeat `monitor-tac-200-identity-ratio-kaggle` is active to pull and
  validate completed outputs when Kaggle exposes them.
- Source pulls under
  `runs/kaggle_outputs/identity_ratio_p8_5k_v2_source_pull_jeffkolo_20260607`
  and
  `runs/kaggle_outputs/identity_ratio_p24_5k_v2_source_pull_jeffkolo_20260607`
  confirm the v2 wrappers still use 5k steps, the expected p8/p24 program
  counts, the fixed code dataset, the shared capability data dataset, and
  NvidiaTeslaT4.
- A delayed burn-in output pull returned no files or immediate crash logs.
- p8 version 2 completed and validated from
  `runs/kaggle_outputs/identity_ratio_p8_5k_v2_completed_jeffkolo_20260607`.
  It reached `completed_steps=5000/5000`, `stopped_for_time=false`,
  `n_programs=8`, and optimizer health passed. Key p8 metrics:
  `best_eval_loss=0.17896727100014687`, latest step-5000 eval
  `loss=0.18265898525714874`, eval `accuracy=0.9324100378787878`,
  eval `program_memory_cosine=0.008388867601752281`, train
  `tokens_per_second=6008.939585492242`, end specialization
  `mi_bits=0.23269702577624624`, `normalized_mi=0.12403733983642708`, and
  max knockout loss delta `0.145131423487328`.
- p24 version 2 has not exposed completed output artifacts yet, so the p8-vs-p24
  external ratio decision remains pending.
- p24 version 2 later completed and validated from
  `runs/kaggle_outputs/identity_ratio_p24_5k_v2_completed_jeffkolo_20260607`.
- Official comparison artifact:
  `runs/benchmarks/external_identity_ratio_5k_validation_2026_06_07`.
- Final decision:
  `external_ratio_5k_specialization_increases_capability_approximately_flat`.
- Actual full-scale run-manifest ratios are:
  p8 `identity_share=0.4529`, `identity_to_transformer=0.8277`,
  `transformer_to_identity=1.2082`; p24 `identity_share=0.6317`,
  `identity_to_transformer=1.7151`.
- p8 is clearly more parameter-efficient and faster: train throughput
  `6008.94` vs `5412.93` tokens/s, with 32.7% fewer total parameters.
- General LM capability appears approximately unchanged between p8 and p24 at
  5k steps. The best-eval-loss gap is only `0.000331`, and final eval accuracy
  differs by only about `0.25` percentage points, so those gaps should not be
  treated as meaningful without multiple seeds and confidence intervals.
- p24 has the strongest measured effect: specialization MI `0.473754` vs
  `0.232697`, max knockout loss delta `0.353776` vs `0.145131`, and lower
  final eval program-memory cosine `0.002664` vs `0.008389`.
- Identity parameters increased from `7.95M` to `16.47M`, about `2.07x`, while
  specialization MI increased about `2.04x` and loss/accuracy stayed roughly
  flat. This is closer to identity-state redistribution than to a simple
  general-capability scaling story.
- Recommendation: p8 is the parameter-efficient 5k full-scale allocation:
  about 54.7% transformer-side / 45.3% identity-field parameters. Current
  evidence supports the claim that additional identity capacity improves
  specialization, but does not yet demonstrate that the specialization yields
  better downstream capability.

Decision boundary:

- No external ratio decision is claimed yet.
- The next validation step is to pull completed outputs, verify
  `final_summary.json`, `run_manifest.json`, `metrics.jsonl`, checkpoints, and
  specialization artifacts, then compare p8 against p24 on loss, accuracy,
  optimizer health, route/specialization signal, memory cosine, throughput, and
  stopped-for-time status.

## 2026-06-07 TAC EBM Hybrid Architecture Research

Question:

- Validate whether TAC should become a pure energy-based model or remain a
  language model with an added scalar energy critic used for routing, memory,
  verification, and reranking.

Evidence:

- Residual EBMs for text generation explicitly avoid replacing the pretrained
  locally normalized LM. They make EBM training tractable by learning the
  residual of a pretrained language model and using noise-contrastive
  estimation. This directly supports a TAC path of "keep LM, add sequence
  energy" rather than "replace LM with pure EBM." Source:
  https://arxiv.org/abs/2004.11714
- Energy-based reranking for neural machine translation trains an energy model
  so lower energy corresponds to better task-measure samples, then reranks
  candidates from an autoregressive Transformer. Reported gains include +4 BLEU
  on IWSLT'14 German-English, +3.0 BLEU on Sinhala-English, and +1.2 BLEU on
  WMT'16 English-German. This supports using TAC energy first as a reranker or
  verifier signal. Source: https://arxiv.org/abs/2009.13267
- FUDGE and PPLM both validate the broader hybrid-control idea: a pretrained LM
  remains the generator while a smaller predictor/classifier adjusts generation.
  FUDGE only needs LM logits and uses a future discriminator over partial
  sequences; PPLM combines a pretrained LM with small attribute classifiers and
  steers hidden activations at decoding time. Sources:
  https://arxiv.org/abs/2104.05218 and https://arxiv.org/abs/1912.02164
- DExperts validates product-of-experts style decoding-time steering: combine a
  pretrained LM with small expert and anti-expert LMs, so tokens are favored
  when they are likely under good experts and unlikely under anti-experts.
  Source: https://arxiv.org/abs/2105.03023
- NCE is a standard way to train unnormalized models by discriminating observed
  data from artificial noise, which matches the local TAC EBM probe's
  clean-vs-corrupt sequence setup. Source:
  https://proceedings.mlr.press/v9/gutmann10a.html
- Current EBMs still carry real training/sampling risk. Du and Mordatch note
  EBMs are appealing but traditionally difficult to train, and Nijkamp et al.
  show that MCMC-based EBM outcomes depend heavily on sampling behavior: short
  runs can produce realistic samples, but non-convergent chains may not define a
  valid steady-state density. Sources: https://arxiv.org/abs/1903.08689 and
  https://arxiv.org/abs/1903.12370
- JEM shows a classifier can be reinterpreted as an EBM and that hybrid
  discriminative/energy training can improve calibration, robustness, and OOD
  detection with little overhead compared to standard classification training.
  This supports an energy critic as an auxiliary/judgment layer, not necessarily
  a standalone generator. Source: https://arxiv.org/abs/1912.03263
- Adaptive compute literature supports making compute input-dependent. ACT lets
  recurrent networks learn how many computational steps to spend, and Universal
  Transformers add dynamic per-position halting. This supports using TAC energy
  to allocate more routing/memory/verification compute to hard or suspicious
  states. Sources: https://arxiv.org/abs/1603.08983 and
  https://arxiv.org/abs/1807.03819
- MoE routing work supports treating routing as a first-class architecture
  problem rather than a fixed fuel meter. Switch Transformer simplified routing
  and reported up to 7x pretraining speedups with the same compute; BASE layers
  formulate expert assignment as balanced allocation; Expert Choice Routing
  allows variable experts per token while fixing expert bucket size and reports
  more than 2x convergence-speed improvement over Switch/GShard references.
  Sources: https://arxiv.org/abs/2101.03961,
  https://arxiv.org/abs/2103.16716, and https://arxiv.org/abs/2202.09368
- Reward-model work in summarization and instruction following validates a
  separate scoring model trained from comparisons/rankings as a practical way to
  improve model behavior. This is conceptually close to a learned energy critic
  used for reranking or policy improvement. Sources:
  https://arxiv.org/abs/2009.01325 and https://arxiv.org/abs/2203.02155

Decision:

- Do not convert TAC into a pure EBM first.
- Recommended next architecture slice is a hybrid:
  `LM next-token loss + scalar data_energy head + existing compute_energy`.
- Keep current routing energy as `compute_energy`, but add `data_energy` trained
  on clean/corrupt, answer/gold-vs-bad, or candidate ranking pairs.
- Use `data_energy` first for candidate reranking and verifier triggering; then
  test dynamic energy-budget routing where high data energy or uncertainty buys
  more program routes, memory reads, or verifier passes.

Risks:

- Negative sample quality controls what the energy head learns; easy corruptions
  will create a cheap detector, not a useful judge.
- A scalar energy can become miscalibrated even if pairwise ranking improves.
- If energy affects routing too early, it can destabilize the base LM. Gate it
  behind reranking/verifier experiments before changing default training.

## 2026-06-07 TAC Local EBM And Compression Experiment Results

Question:

- Validate the practical TAC implementation path for energy-based training and
  representation compression using local CPU experiments.

Artifacts:

- EBM probe:
  `runs/benchmarks/energy_based_model_probe_2026_06_07`
- Balanced energy matrix:
  `runs/benchmarks/energy_balanced_tac_strong_2026_06_07`
- Energy plus compression matrix:
  `runs/benchmarks/energy_compression_tac_2026_06_07`
- Activation-L1 confirmation:
  `runs/benchmarks/energy_compression_tac_activation_l1_confirm_2026_06_07`

Local EBM probe result:

- Existing TAC routing energy alone is not a useful data energy. Its
  best-direction pair accuracy was 0.5469, near chance.
- Adding a scalar sequence energy head over TAC hidden/identity features worked:
  final learned-energy pair accuracy 0.7344, energy gap 1.5500, and verdict
  `yes_with_scalar_energy_head_not_routing_energy_alone`.
- Decision: keep routing energy as `compute_energy`; add separate
  `data_energy`.

Balanced energy-training matrix:

- Tested `lm_only`, `energy_only`, weak hybrid variants, strong hybrid variants,
  and compute-regularized strong hybrids.
- The 80-step weak-hybrid matrix was inconclusive.
- The two-seed 200-step strong matrix promoted `hybrid_energy_strong`.
- `hybrid_energy_strong` metrics:
  LM accuracy 0.5719, energy pair accuracy 0.8906, rerank accuracy 0.6406,
  energy gap 1.5216, compute energy 2.8083.
- `lm_only` had stronger LM accuracy but no useful energy/reranking signal.
- `energy_only` damaged LM ability and is not viable as the base TAC path.
- Light/heavy compute-energy regularization passed thresholds but reduced
  reranking enough to lose the balanced score.
- Decision: use `LM loss + full-strength data_energy contrastive loss`; track
  compute energy as a metric rather than defaulting to compute-energy pressure.

Compression matrix:

- Tested the three best energy variants from the balanced matrix:
  `hybrid_energy_strong`,
  `hybrid_energy_strong_compute_regularized`, and
  `hybrid_energy_strong_compute_heavy`.
- Crossed each with compression variants:
  `none`, `activation_l1`, and `sparse_balanced`.
- Each full matrix cell ran 500 training steps.
- Full seed-7 3x3 matrix promoted
  `hybrid_energy_strong__activation_l1`.
- Full matrix winner metrics:
  LM accuracy 0.8542, energy pair accuracy 0.9688, rerank accuracy 0.9375,
  activation density 0.3299, active-program fraction 0.4635, compression score
  0.4332, balanced score 0.6622.
- `activation_l1` was the best compression pressure for all three energy
  variants in the full matrix.
- Seed-19 500-step activation-L1 confirmation promoted
  `hybrid_energy_strong_compute_heavy__activation_l1` by a tiny balanced-score
  margin, while `hybrid_energy_strong__activation_l1` remained the
  compression-only winner.
- Two-seed activation-L1 aggregate:
  `hybrid_energy_strong__activation_l1` balanced 0.6608, compression 0.4357,
  active-program fraction 0.4818, LM accuracy 0.8677, energy pair accuracy
  0.9844, rerank accuracy 0.8906.
- Two-seed activation-L1 aggregate for the compute-heavy alternative:
  balanced 0.6607, compression 0.3636, active-program fraction 0.8333, LM
  accuracy 0.8750, energy pair accuracy 0.9844, rerank accuracy 0.9062.

Revised interpretation:

- These results are an encouraging local signal, not a final conclusion.
- The full compression matrix used a tiny model (`d_model=24`, one layer), one
  full-matrix seed, 500 steps, and four eval batches. The seed-19 confirmation
  covered only `activation_l1`, not the full 3x3 matrix.
- The strongest supported claim is narrow: activation-L1 compression pressure
  can reduce activation density while preserving strong LM, energy-pair, and
  reranking behavior in the tested local setting.
- The stronger claim that TAC has learned compact persistent identity
  representations is not yet demonstrated because the benchmark does not test
  identity retention after distractor tasks.

Decision:

- Best current candidate compression-valuing TAC implementation:
  `hybrid_energy_strong + activation_l1`.
- Treat `hybrid_energy_strong + activation_l1` as the next candidate to test,
  not as a promoted default architecture.
- Compute-heavy activation-L1 is reasonable if reranking or compute cost is
  prioritized. In the seed-19 confirmation, compute-heavy activation-L1 won the
  balanced score by a tiny margin and improved rerank accuracy from 0.8438 to
  0.9063, but it kept a much larger active-program fraction.
- The interesting open question is whether high active-program fraction with
  low activation density means TAC is learning distributed selective
  participation rather than few-expert sparsity.

Recommended implementation contract:

- Add `data_energy` as a separate scalar head/objective.
- Train with normal next-token LM loss plus full-strength clean-vs-corrupt
  data-energy contrastive loss.
- Add an activation-L1 compression term on TAC identity/program activations.
- Continue logging compute energy, active-program fraction, assignment entropy,
  activation density, energy pair accuracy, and rerank accuracy.
- Do not make compute-energy regularization default until a larger multi-seed
  run proves the compute savings are worth the compactness/reranking tradeoff.

Required next validation before claiming identity compression:

- Add an identity-retention benchmark that measures identity accuracy after
  distractor task counts `N = 0, 5, 10, 20, 50`.
- Sweep compression strength, not only compression type.
- Report identity retention, routing entropy, activation density, active-program
  fraction, energy pair accuracy, rerank accuracy, and LM accuracy together.
- Identify the phase boundary where compression continues to improve but
  identity retention starts to fall.
- Use that breakpoint as the real "identity compression phase boundary" rather
  than relying on a single balanced-score winner.

## 2026-06-07 TAC-203 Identity Compression Phase Boundary

Question:

- Find the activation-L1 compression strength where TAC identity retention
  begins to fall while compression continues to improve.

Artifacts:

- Initial sweep:
  `runs/benchmarks/identity_compression_phase_boundary_2026_06_07`
- Extended sweep:
  `runs/benchmarks/identity_compression_phase_boundary_extended_2026_06_07`
- Combined phase-boundary artifact:
  `runs/benchmarks/identity_compression_phase_boundary_combined_2026_06_07`

Protocol:

- Fixed energy training to the TAC-201/TAC-202 candidate:
  `hybrid_energy_strong` plus activation-L1 compression.
- Swept activation-L1 strengths:
  `0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.40, 0.80`.
- Used three seeds: `7, 19, 31`.
- Ran 500 training steps per strength/seed cell.
- Measured identity retention after distractor task counts:
  `N = 0, 5, 10, 20, 50`.
- Reported identity retention, LM accuracy, energy pair accuracy, rerank
  accuracy, routing entropy, activation density, active-program fraction,
  compute energy, and compression score.

Result:

- Phase boundary status: `crossed`.
- Boundary strength: activation-L1 `0.20`.
- Baseline strength `0.0`:
  retention 0.4583, compression score 0.3635, activation density 0.4687,
  LM accuracy 0.8542, energy pair accuracy 0.9740, rerank accuracy 0.8542.
- Best retention occurred at strength `0.05`:
  retention 0.5917, compression score 0.4389, activation density 0.3232,
  LM accuracy 0.8472, energy pair accuracy 0.9531, rerank accuracy 0.8854.
- Boundary strength `0.20`:
  retention 0.3667, compression score 0.5002, activation density 0.1671,
  LM accuracy 0.8271, energy pair accuracy 0.9531, rerank accuracy 0.8750.
- The retention drop from the best-retention point to `0.20` is 0.2250 while
  compression continues to improve, so `0.20` is the observed identity
  compression phase boundary in this local setup.
- Stronger compression continues to improve compactness but lowers retention:
  at `0.80`, retention is 0.3333 and compression score is 0.5370.

Interpretation:

- This is stronger than the TAC-202 compression matrix because it isolates
  what fails first under compression.
- The key transition is from activation-L1 `0.05` to `0.20`:
  identity retention falls from 0.5917 to 0.3667, which is about a 38%
  relative drop, while compression improves from 0.4389 to 0.5002, about a
  14% relative gain.
- That tradeoff means the model gives up a large amount of identity capacity
  for a comparatively small amount of extra compression after `0.05`.
- Energy pair accuracy and rerank accuracy remain high at and beyond the
  boundary, so the observed failure mode is identity-retention degradation, not
  energy-learning collapse.
- The result suggests two separable TAC information regimes in this local
  setup: energy/ranking machinery survives aggressive activation sparsification,
  while persistent identity state is much more compression-sensitive.
- At activation-L1 `0.80`, activation density falls to 0.067 and energy/rerank
  still score 0.9531/0.9167, but identity retention is only 0.3333. This is the
  clearest evidence that identity retention is harder than energy evaluation in
  this benchmark.

Limitations:

- This remains a tiny local model (`d_model=24`, one layer).
- Identity retention is a controlled proxy based on carried TAC state and
  candidate energy ranking, not an external long-horizon checkpoint benchmark.
- Absolute retention values are low enough that larger models and a richer
  identity task should be used before turning this into a default architecture
  rule.

Decision:

- Candidate compression strength for future TAC runs: activation-L1 `0.05`.
- Avoid activation-L1 strengths at or above `0.20` unless the explicit goal is
  to study identity degradation under compression.
- Paper-style claim supported by the current evidence:
  increasing activation-L1 pressure induces a measurable identity-compression
  phase transition. Beyond activation-L1 about `0.20`, compression continues
  improving while identity retention degrades disproportionately. Energy
  learning metrics remain largely unchanged, indicating the transition reflects
  loss of persistent identity representations rather than collapse of the
  underlying energy model.

Next validation:

- TAC-204 should narrow the critical threshold with strengths:
  `0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20`.
- Run 5-10 seeds, still at 500 steps per cell unless compute budget changes.
- Fit identity-retention and activation-density curves and estimate the
  critical compression threshold automatically, e.g. `activation_l1 = 0.13 +/-
  0.02` if the data supports that interval.

Validation commands:

- `python -m unittest tests_py.test_identity_compression_phase_boundary tests_py.test_energy_compression_tac tests_py.test_energy_balanced_tac`
- `python -m py_compile experiments\benchmark_identity_compression_phase_boundary.py tests_py\test_identity_compression_phase_boundary.py experiments\benchmark_energy_compression_tac.py experiments\benchmark_energy_balanced_tac.py`
- `python -m json.tool prd.json`

## 2026-06-07 TAC-204 Fine-Grained Identity Compression Critical Threshold

Question:

- Refine the TAC-203 coarse boundary with a denser activation-L1 sweep and
  estimate the critical compression threshold with uncertainty.

Artifact:

- `runs/benchmarks/identity_compression_critical_fit_tac204_2026_06_07`

Protocol:

- Fixed objective: `hybrid_energy_strong` plus activation-L1 compression.
- Swept activation-L1 strengths:
  `0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20`.
- Used five seeds: `7, 19, 31, 43, 59`.
- Ran 500 training steps per strength/seed cell.
- Measured identity retention after distractor counts:
  `N = 0, 5, 10, 20, 50`.
- Added a fitted-threshold estimator:
  piecewise-linear interpolation of the post-peak identity-retention drop, with
  seed-bootstrap uncertainty over 200 bootstrap samples.

Result:

| L1 | Identity | Compression | Density | Energy | Rerank | LM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.050 | 0.630 | 0.444 | 0.322 | 0.966 | 0.906 | 0.845 |
| 0.075 | 0.565 | 0.461 | 0.279 | 0.959 | 0.913 | 0.843 |
| 0.100 | 0.540 | 0.476 | 0.242 | 0.953 | 0.894 | 0.851 |
| 0.125 | 0.400 | 0.486 | 0.216 | 0.947 | 0.888 | 0.845 |
| 0.150 | 0.450 | 0.500 | 0.195 | 0.934 | 0.900 | 0.850 |
| 0.175 | 0.470 | 0.501 | 0.178 | 0.941 | 0.913 | 0.842 |
| 0.200 | 0.445 | 0.510 | 0.165 | 0.959 | 0.906 | 0.834 |

Critical threshold fit:

- Grid crossing strength: activation-L1 `0.125`.
- Estimated critical strength: activation-L1 `0.1018`.
- 95% bootstrap interval: `0.0807` to `0.1225`.
- Bootstrap crossed fraction: `0.985`.
- Identity post-peak slope: `-1.1929`.
- Activation-density slope: `-1.0346`.
- Compression-score slope: `0.4315`.

Interpretation:

- TAC-203's coarse boundary at `0.20` was too high because the grid skipped the
  sharper drop between `0.10` and `0.125`.
- The best operating point remains activation-L1 `0.05`: it has the highest
  identity retention, strong energy/rerank scores, and meaningful compression.
- The critical threshold is better stated as activation-L1 about `0.10`, with
  current local uncertainty roughly `0.08-0.12`.
- The failure pattern remains the same: identity retention drops before
  energy-pair accuracy, reranking, or LM accuracy collapse.
- Compression continues improving after the threshold, but the extra
  compression is not worth the identity loss if persistent identity is the goal.

Decision:

- Recommended compression strength for TAC identity-preserving runs:
  activation-L1 `0.05`.
- Boundary claim for this local setup:
  identity-compression critical threshold is approximately activation-L1
  `0.102`, with a bootstrap interval of `0.081-0.123`.
- Avoid activation-L1 strengths above `0.10` for identity-preserving training
  unless the goal is to deliberately study identity degradation.

Validation commands:

- `python -m unittest tests_py.test_identity_compression_phase_boundary tests_py.test_energy_compression_tac tests_py.test_energy_balanced_tac`
- `python -m py_compile experiments\benchmark_identity_compression_phase_boundary.py tests_py\test_identity_compression_phase_boundary.py experiments\benchmark_energy_compression_tac.py experiments\benchmark_energy_balanced_tac.py`
- `python -m json.tool prd.json`
- `python -m json.tool runs\benchmarks\identity_compression_critical_fit_tac204_2026_06_07\identity_compression_phase_boundary.json`

## 2026-06-07 TAC-205 Larger-Model Identity Compression Validation

Question:

- Test whether the TAC-204 identity-compression threshold near activation-L1
  `0.10` survives a larger local TAC configuration.

Artifact:

- `runs/benchmarks/identity_compression_larger_tac205_2026_06_07`

Protocol:

- Model size: `d_model=48`, `n_layers=2`, `n_programs=8`,
  `energy_budget=4.0`.
- Swept activation-L1 strengths: `0.0, 0.05, 0.10, 0.125`.
- Used three seeds: `7, 19, 31`.
- Ran 500 training steps per strength/seed cell.
- Used 12 identity trials and distractor counts `N = 0, 5, 10, 20, 50`.

Result:

| L1 | Identity | Compression | Density | Energy | Rerank | LM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.528 | 0.381 | 0.461 | 0.943 | 0.927 | 0.924 |
| 0.050 | 0.528 | 0.526 | 0.166 | 0.984 | 0.958 | 0.927 |
| 0.100 | 0.478 | 0.557 | 0.076 | 0.984 | 0.948 | 0.924 |
| 0.125 | 0.350 | 0.569 | 0.059 | 0.984 | 0.927 | 0.925 |

Critical threshold fit:

- Grid crossing strength: activation-L1 `0.125`.
- Estimated critical strength: activation-L1 `0.1098`.
- 95% bootstrap interval: `0.0625` to `0.1188`.
- Bootstrap crossed fraction: `1.0`.
- Identity post-peak slope: `-1.2542`.
- Activation-density slope: `-3.1635`.
- Compression-score slope: `1.4486`.

Interpretation:

- The larger model preserves the same main pattern as TAC-204: identity
  retention fails before energy-pair accuracy, rerank accuracy, or LM accuracy
  collapse.
- Activation-L1 `0.05` is the best practical setting in this run: it preserves
  identity retention relative to baseline while cutting activation density from
  0.461 to 0.166 and improving energy/rerank metrics.
- Activation-L1 `0.10` is near the edge: identity retention falls from 0.528 to
  0.478 while compression continues improving.
- Activation-L1 `0.125` crosses the degradation boundary: identity retention
  falls to 0.350 while energy pair accuracy remains 0.984 and rerank remains
  0.927.
- The fitted larger-model threshold, `0.1098`, agrees with TAC-204's fitted
  threshold, `0.1018`, within the current uncertainty.

Decision:

- Keep activation-L1 `0.05` as the identity-preserving compression candidate.
- Treat activation-L1 around `0.10` as the start of the risk region.
- Do not promote activation-L1 `0.125` or higher for identity-preserving TAC
  training, even though it preserves energy/rerank/LM metrics.
- The current evidence now supports a stronger local claim:
  TAC has an identity-compression phase boundary near activation-L1 `0.10`
  across both tiny and larger local configurations.

Validation commands:

- `python -m unittest tests_py.test_identity_compression_phase_boundary tests_py.test_energy_compression_tac tests_py.test_energy_balanced_tac`
- `python -m py_compile experiments\benchmark_identity_compression_phase_boundary.py tests_py\test_identity_compression_phase_boundary.py experiments\benchmark_energy_compression_tac.py experiments\benchmark_energy_balanced_tac.py`
- `python -m json.tool prd.json`
- `python -m json.tool runs\benchmarks\identity_compression_larger_tac205_2026_06_07\identity_compression_phase_boundary.json`

## 2026-06-07 TAC-206 Routing-Structure Mechanism Test

Question:

- Determine whether the identity-compression boundary is caused by route
  simplification, route concentration, or loss of program-utilization diversity.

Artifact:

- `runs/benchmarks/identity_compression_route_mechanism_tac206_2026_06_07`

Protocol:

- Reused the TAC-205 larger local setup:
  `d_model=48`, `n_layers=2`, `n_programs=8`, `energy_budget=4.0`.
- Swept activation-L1 strengths: `0.0, 0.05, 0.10, 0.125`.
- Used three seeds: `7, 19, 31`.
- Ran 500 training steps per strength/seed cell.
- Added route-structure diagnostics:
  per-program route utilization, activation utilization, effective program
  count, top-program share, route-load standard deviation, underused-program
  fraction, and selected programs per token.

Result:

| L1 | Identity | Density | Energy | Rerank | Route Eff | Route Top1 | Route Std | Route Underused | Act Eff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.528 | 0.461 | 0.943 | 0.927 | 7.267 | 0.201 | 0.186 | 0.062 | 7.914 |
| 0.050 | 0.528 | 0.166 | 0.984 | 0.958 | 7.355 | 0.205 | 0.175 | 0.073 | 7.760 |
| 0.100 | 0.478 | 0.076 | 0.984 | 0.948 | 7.008 | 0.202 | 0.209 | 0.125 | 7.840 |
| 0.125 | 0.350 | 0.059 | 0.984 | 0.927 | 6.757 | 0.208 | 0.228 | 0.167 | 7.751 |

Per-program route utilization:

- `0.000`: `[0.645, 0.594, 0.516, 0.529, 0.426, 0.363, 0.193, 0.279]`
- `0.050`: `[0.691, 0.579, 0.385, 0.421, 0.385, 0.443, 0.342, 0.279]`
- `0.100`: `[0.685, 0.674, 0.499, 0.552, 0.478, 0.260, 0.253, 0.214]`
- `0.125`: `[0.719, 0.650, 0.624, 0.433, 0.472, 0.357, 0.275, 0.105]`

Interpretation:

- Identity collapse does not look like a simple single-program routing collapse.
  Route top-1 share stays nearly flat from 0.201 to 0.208, and selected
  programs per token stays around 3.5-3.6.
- Activation-side program usage also remains broadly distributed:
  activation effective programs stay near 7.75-7.91 out of 8 and activation
  underused-program fraction stays 0.0.
- There is still meaningful route-structure erosion:
  selected-route effective program count falls from 7.355 at `0.05` to 6.757
  at `0.125`, route-load standard deviation rises from 0.175 to 0.228, and
  route underused-program fraction rises from 0.073 to 0.167.
- The strongest mechanism signal is activation magnitude collapse:
  activation density falls from 0.166 at the safe `0.05` operating point to
  0.059 at the identity-collapse point `0.125`, while energy/rerank/LM metrics
  remain stable.
- The likely mechanism is therefore not "TAC routes everything to one expert."
  It is better described as:
  identity representations need enough activation mass plus enough selected-route
  diversity. Energy/ranking can survive with much thinner activations and mild
  route concentration, but persistent identity cannot.

Decision:

- Keep activation-L1 `0.05` as the recommended identity-preserving compression
  setting.
- The next mechanism experiment should directly preserve route diversity or
  identity-specific activation mass while still applying activation-L1, to test
  which component prevents the boundary.
- Do not treat routing entropy alone as sufficient evidence; TAC-206 shows the
  scalar entropy can stay high while selected-route underuse and activation
  density reveal the identity failure.

Validation commands:

- `python -m unittest tests_py.test_identity_compression_phase_boundary tests_py.test_energy_compression_tac tests_py.test_energy_balanced_tac`
- `python -m py_compile experiments\benchmark_identity_compression_phase_boundary.py tests_py\test_identity_compression_phase_boundary.py experiments\benchmark_energy_compression_tac.py experiments\benchmark_energy_balanced_tac.py`
- `python -m json.tool prd.json`
- `python -m json.tool runs\benchmarks\identity_compression_route_mechanism_tac206_2026_06_07\identity_compression_phase_boundary.json`

## 2026-06-07 TAC-207 Representational Thinning Mechanism Test

Question:

- Test whether identity failure is caused by representational thinning:
  many programs remain active, but each carries less activation/state mass.

Artifact:

- `runs/benchmarks/identity_compression_representation_thinning_tac207_2026_06_07`

Protocol:

- Reused the TAC-205/TAC-206 larger local setup:
  `d_model=48`, `n_layers=2`, `n_programs=8`, `energy_budget=4.0`.
- Swept activation-L1 strengths: `0.0, 0.05, 0.10, 0.125`.
- Used three seeds: `7, 19, 31`.
- Ran 500 training steps per strength/seed cell.
- Added representation-thinning diagnostics:
  selected activation mean/L2, selected activation by program, identity-state
  norm, selected identity-state norm, identity-state norm by program,
  energy-feature norm, and marker-identity nearest-centroid probes.
- Marker probes estimate how much rule/identity information remains in carried
  identity memory versus the energy-head feature vector. The MI values are
  empirical normalized probe estimates, not exact information-theoretic bounds.

Result:

| L1 | Identity | Density | Sel Act | Id Norm | Sel Id Norm | Id Probe MI | Energy Norm | Energy Probe MI | Energy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.528 | 0.461 | 0.600 | 0.173 | 0.267 | 0.629 | 3.641 | 0.472 | 0.943 |
| 0.050 | 0.528 | 0.166 | 0.234 | 0.120 | 0.131 | 0.446 | 3.702 | 0.397 | 0.984 |
| 0.100 | 0.478 | 0.076 | 0.102 | 0.095 | 0.062 | 0.326 | 3.691 | 0.509 | 0.984 |
| 0.125 | 0.350 | 0.059 | 0.078 | 0.092 | 0.054 | 0.287 | 3.673 | 0.462 | 0.984 |

Important deltas from baseline:

- At activation-L1 `0.05`, identity retention is unchanged, but activation
  density is down 64%, selected activation mean is down 61%, selected
  identity-state norm is down 51%, and identity-state probe MI is down 29%.
  This means the safe operating point is already thinner internally, even
  though the retention benchmark still passes.
- At activation-L1 `0.10`, identity retention is down 9.5%, selected activation
  mean is down 83%, selected identity-state norm is down 77%, and
  identity-state probe MI is down 48%.
- At activation-L1 `0.125`, identity retention is down 33.7%, selected
  activation mean is down 86.9%, selected identity-state norm is down 79.9%,
  and identity-state probe MI is down 54.4%.
- Energy-feature norm is nearly unchanged across the same range:
  3.641 -> 3.673 at the failure point.
- Energy-feature probe MI is comparatively stable:
  0.472 -> 0.462 at the failure point.

Interpretation:

- TAC-207 directly supports the representational-thinning hypothesis.
- Identity failure tracks selected activation mass, selected identity-state
  norm, and identity-state marker information more strongly than it tracks
  expert takeover or energy-feature degradation.
- The identity state becomes progressively less informative before and during
  the boundary, while the energy feature space remains norm-stable and keeps
  comparable marker information.
- This explains why energy pair accuracy and reranking survive compression:
  the energy head can still use a stable coarse feature representation, while
  persistent identity requires richer distributed state amplitude.
- The safe `0.05` operating point should be described carefully: it is a
  Pareto improvement at the task level, but it already reduces internal
  identity-state information. It is safe only relative to the current retention
  benchmark.

Decision:

- Keep activation-L1 `0.05` as the recommended local compression setting.
- Treat activation-L1 `0.10` as the beginning of identity-state information
  risk, even before the larger retention collapse at `0.125`.
- The mechanism claim is now stronger:
  TAC identity failure under compression is best explained by representational
  thinning of distributed identity state, not by routing monopoly or collapse
  of the learned energy model.
- The next intervention should test whether preserving identity-state mass or
  identity-specific information can push the boundary upward without giving up
  the useful activation-density gains.

Validation commands:

- `python -m unittest tests_py.test_identity_compression_phase_boundary tests_py.test_energy_compression_tac tests_py.test_energy_balanced_tac`
- `python -m py_compile experiments\benchmark_identity_compression_phase_boundary.py tests_py\test_identity_compression_phase_boundary.py experiments\benchmark_energy_compression_tac.py experiments\benchmark_energy_balanced_tac.py`
- `python -m json.tool prd.json`
- `python -m json.tool runs\benchmarks\identity_compression_representation_thinning_tac207_2026_06_07\identity_compression_phase_boundary.json`

## 2026-06-07 TAC-209 Identity Thinning Rescue Intervention

Question:

- Test whether restoring identity-state mass or marker-identity information can
  recover identity retention in the activation-L1 `0.125` failing regime while
  keeping the same sparsity pressure.

Artifact:

- `runs/benchmarks/identity_thinning_rescue_tac209_2026_06_07`

Protocol:

- Used the larger TAC setup from TAC-205 through TAC-207:
  `d_model=48`, `n_layers=2`, `n_programs=8`, `energy_budget=4.0`.
- Fixed activation-L1 compression strength at `0.125`.
- Used three seeds: `7, 19, 31`.
- Ran 500 training steps per variant/seed cell.
- Compared four variants:
  `compressed_control`, `norm_floor_rescue`, `marker_info_rescue`,
  and `combined_rescue`.
- Kept sparsity pressure unchanged across all variants.
- Rescue losses:
  norm-floor rescue penalizes selected identity-state norm below `0.13`;
  marker-info rescue trains an auxiliary marker classifier over carried
  identity memory; combined rescue uses both.
- The benchmark was rerun with train/eval generator offsets aligned to TAC-207
  after an initial non-aligned run failed to reproduce the control failure.

Result:

| Variant | Identity | Compression | Density | Sel Act | Sel Id Norm | Id Probe MI | Energy MI | Energy | Rerank | LM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compressed_control | 0.350 | 0.569 | 0.059 | 0.078 | 0.054 | 0.287 | 0.462 | 0.984 | 0.927 | 0.925 |
| norm_floor_rescue | 0.550 | 0.561 | 0.062 | 0.081 | 0.188 | 0.312 | 0.461 | 0.964 | 0.958 | 0.930 |
| marker_info_rescue | 0.417 | 0.559 | 0.069 | 0.090 | 0.923 | 0.899 | 0.484 | 0.984 | 0.917 | 0.930 |
| combined_rescue | 0.372 | 0.557 | 0.069 | 0.090 | 1.017 | 0.928 | 0.538 | 0.974 | 0.969 | 0.928 |

Decision:

- Status: `identity_rescue_supported`.
- Winner: `norm_floor_rescue`.
- Retention gain over compressed control: `+0.200`.
- Compression delta: `-0.008`.
- Energy-pair delta: `-0.021`.
- Rerank delta: `+0.031`.
- LM accuracy delta: `+0.005`.

Interpretation:

- TAC-209 turns the representational-thinning story into an interventionally
  supported mechanism: restoring selected identity-state magnitude recovers
  identity retention in the `0.125` failure regime while preserving nearly the
  same compression and energy/rerank/LM behavior.
- The norm-floor rescue does not need to restore the full baseline identity
  probe MI. It raises selected identity-state norm from 0.054 to 0.188 and
  retention from 0.350 to 0.550, while identity probe MI only moves from 0.287
  to 0.312. That suggests usable identity amplitude is more important for this
  retention task than the current marker-probe MI estimate.
- Marker-info rescue and combined rescue strongly increase probe MI and
  selected identity norm, but they do not restore functional retention as well.
  This means the probe can create recoverable marker information that is not
  aligned with the energy-based identity-retention behavior.
- The causal claim should therefore be specific:
  preserving identity-state magnitude rescues identity retention; maximizing
  auxiliary probe information alone does not.

Updated theory:

- Compression still drives representational thinning.
- Identity retention fails when selected identity-state mass becomes too low.
- A small selected-identity-norm floor can push the system back into a
  functional identity regime without giving up most compression.
- Energy computation remains comparatively robust and is not the limiting
  failure mode.

Validation commands:

- `python -m unittest tests_py.test_identity_thinning_rescue tests_py.test_identity_compression_phase_boundary tests_py.test_energy_compression_tac tests_py.test_energy_balanced_tac`
- `python -m py_compile experiments\benchmark_identity_thinning_rescue.py tests_py\test_identity_thinning_rescue.py experiments\benchmark_identity_compression_phase_boundary.py tests_py\test_identity_compression_phase_boundary.py experiments\benchmark_energy_compression_tac.py experiments\benchmark_energy_balanced_tac.py`
- `python -m json.tool prd.json`
- `python -m json.tool runs\benchmarks\identity_thinning_rescue_tac209_2026_06_07\identity_thinning_rescue.json`

## 2026-06-07 TAC-211 Functional Identity Transfer Rescue Test

Question:

- Determine whether the TAC-209 norm-floor rescue restores dynamically usable
  identity state, or only restores the original identity-retention behavior.

Artifact:

- `runs/benchmarks/identity_transfer_rescue_tac211_2026_06_07`

Protocol:

- Reused the TAC-209 rescue benchmark and larger local setup:
  `d_model=48`, `n_layers=2`, `n_programs=8`, `energy_budget=4.0`.
- Fixed activation-L1 compression strength at `0.125`.
- Used three seeds: `7, 19, 31`.
- Ran 500 training steps per variant/seed cell.
- Added functional transfer probes:
  after identity context plus distractors, score whether the carried identity
  state can select the correct identity rule for a novel start value and a
  longer offset horizon.
- Compared `compressed_control`, `norm_floor_rescue`,
  `marker_info_rescue`, and `combined_rescue`.

Result:

| Variant | Retention | Transfer | Long Transfer | Compression | Sel Id Norm | Id Probe MI | Energy | Rerank | LM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compressed_control | 0.350 | 0.533 | 0.550 | 0.569 | 0.054 | 0.287 | 0.984 | 0.927 | 0.925 |
| norm_floor_rescue | 0.550 | 0.456 | 0.439 | 0.561 | 0.188 | 0.312 | 0.964 | 0.958 | 0.930 |
| marker_info_rescue | 0.417 | 0.422 | 0.483 | 0.559 | 0.923 | 0.899 | 0.984 | 0.917 | 0.930 |
| combined_rescue | 0.372 | 0.506 | 0.467 | 0.557 | 1.017 | 0.928 | 0.974 | 0.969 | 0.928 |

Interpretation:

- TAC-211 does not support the stronger claim that the TAC-209 norm-floor
  rescue restores generalized functional identity transfer.
- Norm-floor rescue still improves the original retention metric:
  0.350 -> 0.550.
- But norm-floor rescue lowers the new transfer metrics relative to the
  compressed control:
  transfer 0.533 -> 0.456 and long transfer 0.550 -> 0.439.
- Marker-info and combined rescue again show that high probe-decodable marker
  information does not imply functional identity behavior. Their probe MI is
  high, but transfer is not.
- The transfer probes are near chance for all variants, so this should not be
  treated as evidence that the compressed control has strong transfer. The
  correct conclusion is narrower: the current rescue improves same-task
  identity retention but does not yet produce robust novel-start or
  long-horizon identity-rule transfer.

Updated theory:

- Persistent identity retention and identity-rule transfer are separable.
- Selected identity-state magnitude is sufficient to rescue the existing
  retention benchmark under high compression.
- Functional transfer likely requires an additional mechanism: the model must
  use identity state as an active rule/working-state representation, not merely
  preserve enough state mass for the original energy-reranking behavior.
- Probe-decodable marker information remains a poor proxy for functional
  identity transfer.

Decision:

- Keep TAC-209's causal claim, but scope it to identity retention.
- Do not claim that the norm-floor rescue restores generalized identity
  transfer.
- Next intervention should train or evaluate an identity-rule-use objective
  directly, then test whether transfer rises above chance while retaining the
  compression benefits.

Validation commands:

- `python -m unittest tests_py.test_identity_thinning_rescue tests_py.test_identity_compression_phase_boundary tests_py.test_energy_compression_tac tests_py.test_energy_balanced_tac`
- `python -m py_compile experiments\benchmark_identity_thinning_rescue.py tests_py\test_identity_thinning_rescue.py experiments\benchmark_identity_compression_phase_boundary.py tests_py\test_identity_compression_phase_boundary.py experiments\benchmark_energy_compression_tac.py experiments\benchmark_energy_balanced_tac.py`
- `python -m json.tool prd.json`
- `python -m json.tool runs\benchmarks\identity_transfer_rescue_tac211_2026_06_07\identity_thinning_rescue.json`

## 2026-06-07 External Capability Dataset For TAC Seq512

Question:

- If the local clean curriculum is too small, which public datasets can be
  added without exceeding the local machine constraints?

Artifact:

- `runs/capability_balanced_external_seq512_2026_06_07`
- Validation:
  `runs/benchmarks/external_capability_dataset_validation_2026_06_07`

Sources and conversion:

- `HuggingFaceH4/ultrachat_200k`: assistant Q&A from last user/assistant
  turn, MIT.
- `Open-Orca/SlimOrca-Dedup`: assistant/instruction Q&A from human/gpt turns,
  MIT.
- `HuggingFaceFW/fineweb_edu_100BT-shuffled`: bounded English continuation
  rows, ODC-BY.
- `HuggingFaceTB/cosmopedia`: bounded textbook-style continuation rows,
  Apache-2.0.
- `openai/gsm8k`: math problems converted to final-answer-only targets, MIT.
- `glaiveai/glaive-function-calling-v2`: compact tool/function-call
  next-action targets, Apache-2.0.

Result:

- Train rows: `89,645`.
- Eval rows: `6,549`.
- Train streams:
  - assistant Q&A: `30,998`
  - English LM continuation: `23,000`
  - private-reasoning final answer: `19,237`
  - agentic/tool next-action: `11,410`
  - ATS exact answer: `5,000`
- Eval streams:
  - assistant Q&A: `2,543`
  - ATS exact answer: `2,000`
  - private-reasoning final answer: `2,000`
  - local compact agentic next-action: `6`

Validation:

- Decision: `PASS`.
- No strict red-team reject rows after post-build filtering.
- No answer role markers.
- No visible think/thinking/reasoning tags.
- No empty prompt or answer rows.
- No train/eval prompt overlap.
- No train/eval prompt-answer overlap.
- Max prompt+answer bytes: `512`.

Decision:

- This is the recommended local-machine dataset when the goal is to teach the
  Run5B checkpoint broader English fluency, assistant answering, final-answer
  reasoning, tool-use action format, and retain ATS pressure.
- It is still not a replacement for true large-scale pretraining; it is a
  bounded answer-only repair/instruction dataset sized to fit the current
  machine and Kaggle workflow.

## 2026-06-08 20M From-Scratch Pretraining Dataset

Question:

- Build a dataset that can train an approximately 20M-parameter TAC model from
  scratch, rather than fine-tune an existing checkpoint.

Artifact:

- `runs/pretrain_20m_from_scratch_seq512_2026_06_08`
- Validation:
  `runs/benchmarks/pretrain_20m_dataset_validation_2026_06_08`

Sources:

- `HuggingFaceFW/fineweb_edu_100BT-shuffled`
- `HuggingFaceTB/cosmopedia`
- validated local external capability mix as a seed stream

Result:

- Train records: `104,632`.
- Train TAC byte tokens: `190,107,567`.
- Eval records: `4,755`.
- Eval TAC byte tokens: `4,477,028`.
- Train streams:
  - `pretrain_english`: `39,117`
  - `pretrain_textbook`: `16,117`
  - `pretrain_seed`: `49,398`
- Tokenized memmaps were written with `uint16` TAC byte IDs.

Validation:

- Decision: `PASS`.
- No empty text rows.
- No strict red-team reject rows.
- No train/eval text-hash overlap.
- Token file sizes match manifest token counts.
- `TokenizedMemmapBatcher` sampled train/eval batches at `[2, 512]`.

Interpretation:

- This is enough for a practical local/Kaggle 20M-parameter from-scratch run.
- It gives about `9.5` tokens per parameter for a 20M model.
- A compute-optimal 20M run would prefer about `400M` tokens, so this corpus is
  not saturation-scale pretraining. It is a useful first from-scratch corpus
  that fits the current machine.
- The closest checked TAC config is `d_model=224`, `n_layers=8`,
  `n_heads=8`, `n_programs=24`, with `20,013,416` parameters.

## 2026-06-07 TAC-208 Ratio-Controlled p16/p24 Kaggle Launch

Question:

- Does increasing identity program count still increase specialization when
  the p8 full-scale transformer:identity parameter ratio is held approximately
  fixed?

Design:

- Prior TAC-200 p8 full-scale manifest ratio:
  identity_to_transformer=`0.8276879516`,
  transformer_to_identity=`1.2081847972`, identity_share=`0.4528606488`.
- Added `low_rank_linear_expert` as a trainable program-expert implementation
  so program count can increase without forcing identity parameters above the
  p8 ratio.
- p16-ratio-controlled uses `n_programs=16`, `program_expert_rank=63`,
  expected total parameters `17,547,400`, identity parameters `7,946,632`,
  transformer-side parameters `9,600,768`, and identity_to_transformer
  `0.8277079500`.
- p24-ratio-controlled uses `n_programs=24`, `program_expert_rank=41`,
  expected total parameters `17,514,824`, identity parameters `7,914,056`,
  transformer-side parameters `9,600,768`, and identity_to_transformer
  `0.8243148881`. This is the closest integer-rank match to the p8 ratio.

External launch:

- Code dataset:
  `jeffkolo/tac-identity-ratio-rc-code-2026-06-07`.
- Kernels:
  `jeffkolo/tac-identity-ratio-p16-rc-5k-2026-06-07` and
  `jeffkolo/tac-identity-ratio-p24-rc-5k-2026-06-07`.
- Both use 5,000 steps, Run5B-best-capability-fast settings, fp32 fail-fast
  optimizer health, dual T4 torchrun, eval every 1,000 steps, checkpoints every
  500 steps, specialization checkpoints at 2,000 and 5,000, and end
  specialization analysis.

Interpretation rule:

- General loss and accuracy differences are not to be overclaimed without
  multiple seeds and confidence intervals.
- The primary question is whether p16/p24 increase specialization MI,
  knockout dependence, or identity organization at a matched parameter ratio.

## 2026-06-07 TAC-210 Transformer-Expanded Full-Rank Ratio Launch

Correction:

- The intended control was not to reduce identity expert rank. It was to keep
  full-rank identity programs and increase transformer-side parameters until the
  p16/p24 identity:transformer ratio matches the p8 full-scale ratio.
- The TAC-208 low-rank p16/p24 kernels remain useful as an additional control,
  but they are not the corrected interpretation of the requested experiment.

Design:

- Prior TAC-200 p8 full-scale manifest ratio:
  identity_to_transformer=`0.8276879516`,
  transformer_to_identity=`1.2081847972`, identity_share=`0.4528606488`.
- Added `--mlp-ratio` to `kaggle/train_best_tac_agentic.py` so the corrected
  kernels can increase transformer MLP parameters while leaving
  `program_compute_type="linear_expert"` intact.
- p16-transformer-expanded uses `n_programs=16`, `mlp_ratio=7`,
  expected total parameters `26,932,104`, identity parameters `12,206,472`,
  transformer-side parameters `14,725,632`, and identity_to_transformer
  `0.8289268671`.
- p24-transformer-expanded uses `n_programs=24`, `mlp_ratio=10`,
  expected total parameters `36,317,000`, identity parameters `16,466,504`,
  transformer-side parameters `19,850,496`, and identity_to_transformer
  `0.8295260733`.

External launch:

- Alternate Kaggle account/profile verified as `eweewee2`.
- Code dataset:
  `eweewee2/tac-identity-ratio-tx-code-2026-06-07`.
- Kernels:
  `eweewee2/tac-identity-ratio-p16-tx-5k-2026-06-07` and
  `eweewee2/tac-identity-ratio-p24-tx-5k-2026-06-07`.
- Both use 5,000 steps, Run5B-best-capability-fast settings, fp32 fail-fast
  optimizer health, dual T4 torchrun, eval every 1,000 steps, checkpoints every
  500 steps, specialization checkpoints at 2,000 and 5,000, and end
  specialization analysis.

Current status:

- Both transformer-expanded kernels were accepted by Kaggle as version 1.
- `kaggle kernels status` currently returns HTTP 500 for both, but
  `kaggle kernels list --mine` under the eweewee2 profile shows fresh
  lastRunTime entries on 2026-06-07.
- Immediate output pulls returned no artifacts yet, consistent with queued or
  running kernels.
- The active heartbeat `monitor-tac-208-ratio-controlled-kaggle` now monitors
  all four kernels: TAC-208 low-rank p16/p24 under `jeffkolo` and TAC-210
  transformer-expanded full-rank p16/p24 under `eweewee2`.

Interpretation rule:

- Compare all four variants, not just the corrected pair.
- Treat small loss and accuracy differences as approximate without repeated
  seeds or confidence intervals.
- The strongest evidence to look for is whether specialization MI, knockout
  locality, program-memory organization, or throughput changes when the p8
  parameter ratio is matched by reducing identity capacity versus expanding
  transformer capacity.

Validation commands:

- `python -m unittest tests_py.test_transformer_expanded_identity_kaggle_staging tests_py.test_ratio_controlled_identity_kaggle_staging -v`
- `python -m unittest tests_py.test_tac_transformer.TACTransformerArchitectureTest.test_best_tac_agentic_accepts_specialization_training_knobs -v`
- `python -m py_compile kaggle\train_best_tac_agentic.py experiments\stage_transformer_expanded_identity_kaggle.py tests_py\test_transformer_expanded_identity_kaggle_staging.py`
- `python -m json.tool prd.json`

## 2026-06-07 TAC-212 P8/P24 Identity Robustness Causal Benchmark

Question:

- For the first completed external p8/p24 5k set, does p24's higher
  specialization MI translate into causally useful downstream identity behavior?

Artifact:

- `runs/benchmarks/p8_p24_identity_robustness_2026_06_07`

Protocol:

- Loaded the completed TAC-200 `last.pt` checkpoints:
  p8 from `runs/kaggle_outputs/identity_ratio_p8_5k_v2_completed_jeffkolo_20260607`
  and p24 from
  `runs/kaggle_outputs/identity_ratio_p24_5k_v2_completed_jeffkolo_20260607`.
- Generated 48 synthetic byte-pair identity retrieval cases.
- Wrote the target identity fact in an earlier segment, then queried in a later
  active segment where the target fact is absent.
- Scored forced-choice next-byte identity retrieval over one correct value and
  five alternatives.
- Compared carry state against reset/no-carry state.
- Swept interference levels: `0, 4, 8, 16` intervening distractor identity
  pairs.
- Swept active query context budgets: `4, 8, 12, 16, 24, 32, 48` tokens.
- Ran every p8 and p24 program knockout on 12 representative cases, measuring
  retrieval accuracy drop, retrieval margin drop, memory-read margin drop, and
  program-memory norm drop.

Result:

| Metric | p8 | p24 |
| --- | ---: | ---: |
| Carry retrieval accuracy | 0.1667 | 0.1875 |
| Reset retrieval accuracy | 0.1875 | 0.2083 |
| Carry mean margin | -0.9437 | -0.9010 |
| Memory-read accuracy | 0.5000 | 0.5208 |
| Minimum context for 50% accuracy | none | none |
| Minimum context for 75% accuracy | none | none |
| Max knockout retrieval-accuracy drop | 0.0000 | 0.0000 |
| Knockout drop concentration | 0.0000 | 0.0000 |

Interference accuracy:

| Interference pairs | p8 | p24 |
| ---: | ---: | ---: |
| 0 | 0.3333 | 0.1667 |
| 4 | 0.0000 | 0.0833 |
| 8 | 0.2500 | 0.3333 |
| 16 | 0.0833 | 0.1667 |

Interpretation:

- This is a negative causal result for the first trained p8/p24 set.
- p24's specialization MI advantage does not produce reliable downstream
  identity retrieval in this benchmark.
- Neither model beats its own reset baseline:
  p8 carry is `0.1667` vs reset `0.1875`, and p24 carry is `0.1875` vs reset
  `0.2083`.
- Neither model reaches 50% retrieval accuracy under any tested active-context
  budget, so there is no evidence that p24 reduces active-context requirements.
- Program knockouts do not cause positive retrieval-accuracy drops for either
  p8 or p24, so the existing p24 knockout/loss sensitivity does not transfer to
  this identity retrieval task.
- Memory-read forced-choice accuracy is higher than downstream query accuracy,
  especially for p24, but memory-read margins remain negative and do not become
  usable next-token retrieval behavior.

Decision:

- Keep the prior cautious p8/p24 statement:
  additional identity capacity improves specialization metrics, but the first
  trained set does not show that specialization is causally useful for identity
  retention, interference resistance, or active-context compression.
- The next external causal test should train or evaluate on explicit identity
  carry/retrieval tasks if the goal is to prove downstream utility, rather than
  relying on hard-agentic specialization MI alone.

Validation commands:

- `python -m unittest tests_py.test_p8_p24_identity_robustness -v`
- `python -m py_compile experiments\benchmark_p8_p24_identity_robustness.py tests_py\test_p8_p24_identity_robustness.py`
- `python experiments\benchmark_p8_p24_identity_robustness.py --device cpu --case-count 48 --knockout-case-count 12 --torch-threads 4`
- `python -m json.tool runs\benchmarks\p8_p24_identity_robustness_2026_06_07\identity_robustness.json`
- `python -m json.tool prd.json`

## 2026-06-07 TAC-213 Forced Identity-State Objective Causality Test

Question:

- TAC-212 showed that p24 specialization is not causally useful for identity
  retrieval. Is the bottleneck simply the objective, and can a forced
  identity-state objective make identity programs causally necessary?

Artifact:

- `runs/benchmarks/forced_identity_objective_tac213_2026_06_07`

Protocol:

- Trained a small local TAC model on synthetic key/value identity pairs.
- Compared two objectives:
  - `context_visible_lm`: support key/value pairs and query key are in the same
    active sequence, so the transformer can solve directly from context.
  - `forced_state`: support key/value pairs are processed in a prefill segment;
    query contains only `QUERY,key`, so the answer must come from carried state.
- Used three seeds: `7, 19, 31`.
- Ran 240 steps per variant/seed.
- Measured:
  full-context accuracy, carry accuracy, reset accuracy, shuffled-state
  accuracy, direct `memory_read_logits` accuracy, active-context compression,
  all-program knockout retrieval drops, and identity/program gradient share.

Result:

| Metric | context-visible LM | forced-state |
| --- | ---: | ---: |
| Full-context accuracy | 0.3359 | 0.0768 |
| Carry accuracy | 0.0794 | 0.0807 |
| Reset accuracy | 0.0794 | 0.0807 |
| Carry - reset | 0.0000 | 0.0000 |
| Direct memory-read accuracy | 0.3529 | 0.3529 |
| Identity grad share | 0.1087 | 0.0353 |
| Program grad share | 0.1087 | 0.0353 |
| Max knockout accuracy drop | 0.0078 | 0.0104 |

Interpretation:

- TAC-213 is another negative result for causal identity programs under the
  current architecture/objective wiring.
- The context-visible objective learns some direct in-context lookup
  (`0.3359` full-context accuracy), but removing support from active context
  collapses to chance and carry equals reset.
- The forced-state objective does not learn usable carried-state answer
  behavior either: carry and reset are both `0.0807`.
- Direct memory-read accuracy is `0.3529`, higher than the downstream query
  accuracy. This means some support signal exists inside the state/readout
  path, but it is not being converted into answer logits.
- Identity/program gradients are nonzero, but forced-state training actually
  has lower identity-gradient share than the context-visible objective
  (`0.0353` vs `0.1087`), and program knockouts barely affect retrieval.
- The bottleneck is therefore narrower than "add a forced query objective."
  It appears to be the bridge from identity memory/readout into answer logits
  plus localized program causality.

Decision:

- Do not expect more programs, matched ratios, or a plain carried-state LM loss
  to solve the TAC-212 failure mode by themselves.
- The next experiment should add an explicit identity-readout-to-answer bridge
  or supervised program-local memory slot objective, then require:
  carry > reset, direct readout -> logits transfer, and nonzero localized
  knockout drops.

Validation commands:

- `python -m unittest tests_py.test_forced_identity_objective -v`
- `python -m py_compile experiments\benchmark_forced_identity_objective.py tests_py\test_forced_identity_objective.py`
- `python experiments\benchmark_forced_identity_objective.py --steps 240 --batch-size 32 --eval-batches 8 --seeds 7 19 31 --n-pairs 3 --torch-threads 4`
- `python -m json.tool runs\benchmarks\forced_identity_objective_tac213_2026_06_07\forced_identity_objective.json`
- `python -m json.tool prd.json`

## 2026-06-07 TAC-214/TAC-215 Identity Readout Bridge Diagnostic

Question:

- TAC-213 showed direct memory-read accuracy above downstream answer accuracy.
  Does the frozen identity readout already contain the answer, and can a direct
  readout-to-logit bridge make carried state control generation?

Artifact:

- `runs/benchmarks/identity_readout_bridge_tac214_215_2026_06_07`

Protocol:

- Trained the same small forced-state TAC base used in TAC-213 for 240 steps,
  then froze all TAC parameters.
- TAC-214 trained a tiny linear oracle probe from `memory_read_vector` to the
  12-way value-token answer class.
- TAC-215 trained a tiny linear bridge whose output is added to the frozen
  query value-token logits.
- Used three seeds: `7, 19, 31`.
- Scored all bridge/probe metrics as forced-choice accuracy over the 12 value
  tokens.
- Compared base carry, base reset, direct memory read, oracle probe, bridge,
  reset bridge, zero-read bridge, shuffled-read bridge, and every-program bridge
  knockout.

Result:

| Metric | Mean |
| --- | ---: |
| Base carry value accuracy | 0.1029 |
| Base reset value accuracy | 0.1029 |
| Base carry - reset | 0.0000 |
| Direct memory-read accuracy | 0.3789 |
| Oracle readout probe accuracy | 0.3190 |
| Logit bridge accuracy | 0.3073 |
| Reset bridge accuracy | 0.0898 |
| Zero-read bridge accuracy | 0.0898 |
| Shuffled-read bridge accuracy | 0.1068 |
| Bridge - base carry | 0.2044 |
| Bridge - reset bridge | 0.2174 |
| Bridge - shuffled bridge | 0.2005 |
| Max bridge knockout drop | 0.0130 |

Interpretation:

- TAC-214 does not show a high oracle ceiling. The identity readout carries
  above-chance answer information, but probe accuracy around `0.3190` is far
  below the `90%+` outcome that would imply a mostly solved memory
  representation with only a broken router to logits.
- TAC-215 shows that an explicit direct bridge can make some carried readout
  information affect answer logits: accuracy rises from `0.1029` base carry to
  `0.3073`, while reset and zero-read controls stay near chance and shuffled
  readout falls to `0.1068`.
- The bridge result is partial, not a full rescue. The bottleneck is therefore
  mixed: the readout signal is real but weak-to-moderate, and the native LM path
  does not couple it into generation.
- Program knockouts remain non-causal under the bridge: max accuracy drop is
  only `0.0130`, with no harmful program above 5 percentage points. That means
  the bridge uses carried state globally but does not create localized
  program-specific memory structures.

Decision:

- The next architecture experiment should not assume memory is already strong.
  It should jointly improve readout quality and add a native readout-to-logit or
  readout-to-hidden coupling.
- A positive next result should require all three: substantially higher
  readout/probe ceiling, bridge/native carry > reset, and nonzero localized
  knockout drops.

Validation commands:

- `python -m unittest tests_py.test_identity_readout_bridge -v`
- `python -m py_compile experiments\benchmark_identity_readout_bridge.py tests_py\test_identity_readout_bridge.py`
- `python experiments\benchmark_identity_readout_bridge.py --base-steps 240 --probe-steps 200 --bridge-steps 200 --batch-size 32 --eval-batches 8 --seeds 7 19 31 --n-pairs 3 --torch-threads 4`
- `python -m json.tool runs\benchmarks\identity_readout_bridge_tac214_215_2026_06_07\identity_readout_bridge.json`
- `python -m json.tool prd.json`

## 2026-06-07 TAC-216 Program-Specific Supervision

Question:

- Can explicit assignment of identity keys to specific programs make identity
  state modular and causally local, rather than merely distributed and weakly
  bridgeable?

Artifact:

- `runs/benchmarks/program_specific_supervision_tac216_2026_06_07`

Protocol:

- Built a synthetic key/value task where key `KEY_START+p` is assigned to
  program `p`.
- Compared three variants:
  - `forced_state_baseline`: original TAC-213 forced-state setup with base
    routing.
  - `semantic_baseline`: activation-aware `base_semantic` routing without
    program assignment supervision.
  - `program_supervised`: activation-aware `base_semantic` routing plus
    support-token route supervision and target-program-slot value supervision.
- Trained each base for 240 steps across seeds `7, 19, 31`.
- Froze each TAC base, then trained post-hoc heads:
  global readout bridge, target-slot probe, and target-slot bridge.
- Measured target-slot versus wrong-slot bridge accuracy, route target argmax,
  route target selected rate, targeted program knockout drop, nontarget
  knockout drop, and localized drop gap.

Result:

| Metric | forced-state baseline | semantic baseline | program-supervised |
| --- | ---: | ---: | ---: |
| Base carry accuracy | 0.0924 | 0.0872 | 0.0807 |
| Direct memory-read accuracy | 0.4023 | 0.4023 | 0.4023 |
| Global readout bridge accuracy | 0.3099 | 0.3086 | 0.3255 |
| Target-slot probe accuracy | 0.1862 | 0.1654 | 0.2617 |
| Target-slot bridge accuracy | 0.1654 | 0.1549 | 0.2643 |
| Wrong-slot bridge accuracy | 0.1771 | 0.1589 | 0.2617 |
| Target-slot bridge - wrong slot | -0.0117 | -0.0039 | 0.0026 |
| Route target argmax rate | 0.1220 | 0.1385 | 0.9390 |
| Route target selected rate | 0.1220 | 0.2678 | 0.9364 |
| Targeted knockout drop | -0.0013 | -0.0061 | -0.0022 |
| Nontarget knockout drop | 0.0030 | -0.0030 | 0.0056 |
| Localized drop gap | -0.0043 | -0.0030 | -0.0078 |
| Max targeted knockout drop | 0.0104 | 0.0278 | 0.0208 |

Interpretation:

- TAC-216 is a negative result for modular causal identity programs, but it is
  not a negative result for route controllability.
- The corrected run matters because the original base router ignores learned
  activations when selecting programs. Adding the `semantic_baseline` showed
  that activation-aware routing alone only raises target selected rate to
  `0.2678`, while assignment supervision raises it to `0.9364`.
- Therefore the experiment successfully forced program selection: the assigned
  program is usually selected.
- That did not make the selected program causally necessary. Target-slot bridge
  accuracy rises from `0.1549` to `0.2643`, but wrong-slot bridge accuracy rises
  similarly to `0.2617`, so the target slot is not uniquely informative.
- Targeted knockouts still do not hurt: `targeted_knockout_drop=-0.0022`,
  `localized_drop_gap=-0.0078`, and no program exceeds a 5 percentage-point
  targeted drop.

Decision:

- Explicit program assignment can control routing and weakly improve slot
  decodability, but it does not create causal program-local memory under this
  objective.
- The next bottleneck is not simply "choose the right program." It is the
  write/read semantics of the program slot itself: the assigned slot receives
  responsibility but does not become uniquely necessary for the value.
- A stronger next test should force information flow through the selected
  program slot, for example by masking global content memory, training with a
  per-slot contrastive wrong-slot loss, or generating answers only from the
  selected slot during training.

Validation commands:

- `python -m unittest tests_py.test_program_specific_supervision -v`
- `python -m py_compile experiments\benchmark_program_specific_supervision.py tests_py\test_program_specific_supervision.py`
- `python experiments\benchmark_program_specific_supervision.py --base-steps 240 --head-steps 200 --batch-size 32 --eval-batches 8 --knockout-batches 3 --n-pairs 3 --seeds 7 19 31 --torch-threads 4`
- `python -m json.tool runs\benchmarks\program_specific_supervision_tac216_2026_06_07\program_specific_supervision.json`
- `python -m json.tool prd.json`

## 2026-06-07 TAC-217 Representation Binding And Causal Scrubbing

Question:

- Direction A: can identity be made locally persistent under architectural
  representation constraints such as fixed orthogonal program subspaces?
- Direction B: if identity survives program-level interventions, does it die
  under representation-level scrubbing?

Artifact:

- `runs/benchmarks/representation_binding_scrubbing_tac217_2026_06_07`

Protocol:

- Reused the TAC-216 key-to-program task.
- Compared:
  - `semantic_baseline`: activation-aware `base_semantic` routing.
  - `subspace_bound`: same routing plus fixed orthogonal program subspace masks
    over `d_model`, route supervision, target-slot value supervision, outside
    subspace penalty, and wrong-slot contrastive pressure.
- Trained each base for 240 steps across seeds `7, 19, 31`.
- Froze each TAC base, then trained global bridge, raw-slot bridge, bound-slot
  probe, and bound-slot bridge heads for 200 steps.
- Direction A measured route selection, outside-subspace ratio, bound-slot
  accuracy, wrong-slot separation, and targeted knockout sensitivity.
- Direction B measured:
  learned bridge rowspace projection removal, random subspace projection
  removal, top-gradient-dimension scrubbing, random-dimension scrubbing, and
  broad attention-output ablation.

Direction A result:

| Metric | semantic baseline | subspace-bound |
| --- | ---: | ---: |
| Route target selected rate | 0.2717 | 1.0000 |
| Outside subspace ratio | 0.8934 | 0.8619 |
| Global bridge accuracy | 0.3268 | 0.0898 |
| Raw slot bridge accuracy | 0.1367 | 0.1055 |
| Bound slot probe accuracy | 0.0781 | 0.0768 |
| Bound slot bridge accuracy | 0.1016 | 0.0924 |
| Wrong bound slot bridge accuracy | 0.1003 | 0.0872 |
| Bound slot bridge - wrong | 0.0013 | 0.0052 |
| Targeted knockout drop | -0.0039 | -0.0043 |
| Localized knockout gap | -0.0069 | -0.0039 |

Direction B result:

| Metric | semantic baseline | subspace-bound |
| --- | ---: | ---: |
| Baseline bound slot bridge accuracy | 0.0885 | 0.0846 |
| Slot subspace projection drop | -0.0013 | -0.0065 |
| Random subspace projection drop | -0.0065 | 0.0065 |
| Gradient scrub drop | 0.0013 | 0.0000 |
| Random dimension scrub drop | -0.0026 | -0.0052 |
| Attention ablation drop | 0.0052 | 0.0234 |

Interpretation:

- Direction A is not supported. The subspace-bound intervention makes target
  routing perfect (`1.0000` selected rate), but it does not make bound slots
  useful or local. Bound-slot bridge accuracy remains near chance (`0.0924`),
  wrong-slot separation is almost zero (`0.0052`), and targeted knockout drop
  remains effectively zero (`-0.0043`).
- The intervention also damages the useful global bridge path: global bridge
  accuracy falls from `0.3268` in the semantic baseline to `0.0898` in the
  subspace-bound variant. This suggests the first orthogonal subspace binding
  attempt over-constrains or misaligns the useful distributed signal rather
  than turning it into local program memory.
- Direction B is also not supported in this setup. Learned bridge-rowspace
  projection and top-gradient scrubbing do not selectively remove useful signal;
  their drops are `-0.0065` and `0.0000`. Attention-output ablation has only a
  small effect (`0.0234`).
- The negative Direction B result should be read cautiously because the
  subspace-bound bound-slot bridge itself is weak. Scrubbing a near-chance path
  cannot strongly prove where a high-quality representation lives.

Decision:

- TAC-217 does not validate either fork as implemented:
  - A: fixed orthogonal program subspaces do not create local persistent
    identity/action slots.
  - B: scrubbing the learned weak bound-slot representation does not reveal a
    stronger causal representation-level dependency.
- The result does not overturn the broader distributed-representation model; it
  says this particular attempt to make local subspaces causal failed, and this
  particular scrub target was too weak to carry decisive evidence.

Validation commands:

- `python -m unittest tests_py.test_representation_binding_scrubbing -v`
- `python -m py_compile experiments\benchmark_representation_binding_scrubbing.py tests_py\test_representation_binding_scrubbing.py`
- `python experiments\benchmark_representation_binding_scrubbing.py --base-steps 240 --head-steps 200 --batch-size 32 --eval-batches 8 --knockout-batches 3 --n-pairs 3 --seeds 7 19 31 --torch-threads 4`
- `python -m json.tool runs\benchmarks\representation_binding_scrubbing_tac217_2026_06_07\representation_binding_scrubbing.json`
- `python -m json.tool prd.json`

## 2026-06-08 run5b_plus TAC Transformer Plan

Source:

- User-provided `run5b_plus` v2.0 post-diagnostic specification.

Goal:

- Build, train, and evaluate a `run5b_plus` TAC configuration on top of the
  validated `run5b_best_capability_fast` baseline.
- Integrate three research threads:
  - TAC-218 decision memory and decision continuity.
  - A scalar data-energy head for clean/corrupt scoring and reranking.
  - Identity-layer-only compression with adaptive L1 and norm-floor safeguards.
- Train at an 8192-token context window and treat 32768-token YaRN-style RoPE
  extrapolation as unproven until separately validated.

Architecture plan:

- Keep the Run5B backbone shape: `d_model=256`, `n_layers=8`, `n_heads=8`,
  GQA with 2 KV heads, RMSNorm, SwiGLU, RoPE, dropout 0.0.
- Keep the core identity-field setup with `n_programs=24`, `base_semantic`
  routing, `routing_top_k=2`, `identity_first` attention,
  `program_conditioned` memory updates, CREB allocation, `gated_residual`
  memory adapter, and `program_memory_graph` coalition context.
- Increase content-addressed memory capacity for long-context work:
  `content_store_size=32`, `content_read_steps=4`, and
  `content_read_query_top_k=16`.
- Do not add a separate `MemoryWriter`; the spec explicitly treats that as
  redundant with the existing CREB write/allocation path.
- Do not add a global persistent decision-memory singleton; state must be
  explicitly threaded through the training loop and reset at sequence or episode
  boundaries so DDP workers do not diverge.

Decision memory:

- Desired scope is per-sequence or per-episode identity state, detached across
  sequence boundaries.
- The spec proposes a richer `decision_memory` with shape
  `[batch, n_programs, program_embed_dim]`, updated from selected program
  embeddings and used by a `DecisionContinuityHead` to project carried decision
  memory back into routing-logit space.
- The same-identity mask should be derived from consecutive records belonging
  to the same conversation or episode:
  - multi-turn chat records: true across turns of the same conversation,
  - single-turn records: false,
  - ATS episodes: true within an episode and false at episode boundaries.
- Metrics to track: `decision_agreement` and `decision_memory_mass`.

Energy head:

- Add a `DataEnergyHead` over hidden-state and selected-identity-state features.
- The head should read detached hidden and identity representations so the
  energy contrastive loss does not fight the LM objective through the backbone.
- Train on structured clean/corrupt pairs, not random token noise.
- Add hard negative mining only after EBM warmup, with cost controls such as
  `hard_neg_interval=10` and `n_candidates=8`.
- Track `energy_pair_accuracy`, `energy_gap`, `rerank_accuracy`,
  `ebm_fallback_negative_rate`, and whether hard negative mining is active.

Compression:

- Apply L1 and norm-floor pressure only to identity-field outputs, not global
  transformer-body activations.
- Starting point: `activation_l1_weight=0.05`, but this was validated only at
  local `d_model=24-48`; at `d_model=256`, monitor `norm_floor_fire_rate` and
  lower the starting weight if the floor fires too often.
- Adapt L1 every 500 steps:
  - if `norm_floor_fire_rate > 0.20`, reduce current L1 weight by 20%;
  - if `norm_floor_fire_rate < 0.01`, increase gradually up to 0.05.
- Track `activation_density`, `selected_identity_state_norm`,
  `program_memory_cosine`, `norm_floor_fire_rate`, and active
  `activation_l1_weight`.

Hybrid sliding attention:

- Replace the strict local window that blocked dependencies beyond 128 tokens
  with Longformer-style hybrid sliding attention:
  - sliding window size 512,
  - global attention for special tokens,
  - global tokens attend to and are attended by all positions.
- Proposed global token IDs:
  `<pad>=0`, `<eos>=1`, `<s>=3`, `<identity>=4`, `<query>=5`, `<answer>=6`.
- The global-token list should be config-driven or tightly validated against
  the tokenizer special-token table to avoid drift.

Tokenizer and data:

- Train a SentencePiece unigram tokenizer with vocabulary size 32768,
  byte fallback enabled, and IDs 0-99 reserved for special tokens.
- Validate that `<s>` is present in all batcher outputs, especially ATS
  answer-only evaluation rows.
- Capability corpus target mix:
  60% general chat/instruction, 20% tool-use or agentic, 15% reasoning/math,
  5% ATS.
- Build a `CorruptionPipeline` in the data milestone because EBM training
  depends on it.

Training schedule:

- Use fp32; fp16 remains blocked by prior Run 5 collapse.
- Three-phase schedule:
  - phase 0, steps 0-2000: LM on, EBM off, decision continuity off,
    category route off;
  - phase 1, steps 2000-3000: LM weight 0.5, data-energy contrastive weight
    2.0, decision continuity off, category route off;
  - phase 2, steps 3000-20000: LM weight 1.0, data-energy weight 1.0,
    decision-continuity weight 0.05, category-route weight 0.1, memory
    separation 0.1, routing load balance 0.05, adaptive identity L1 enabled.
- Primary health signal is `lm_loss / total_loss`, with target above 0.40 after
  EBM warmup.
- EBM pair accuracy should exceed 0.55 by step 3000; if not, extend or inspect
  corruption quality before proceeding.

Evaluation plan:

- Phase B gates every 1000 steps:
  eval loss, eval accuracy, program-memory cosine, selected-route MI, max
  knockout loss delta, decision agreement, selected identity-state norm, and
  energy pair accuracy.
- Carry/reset/shuffle benchmark must include at least five task families:
  long single-key, multi-key, delayed-query, noisy-key, and multi-hop.
- Hard carry tasks with multihop chains and distractors are diagnostic only
  until baselines are calibrated.
- ATS exact-match transfer remains a separate gate; current best is 0.0, so
  `<s>` token absence and format normalization must be ruled out before
  attributing failure to model capacity.
- Long-context validation must include NIAH at 8192 and separate 32768-token
  YaRN validation before any 32k support claim.

Implementation order:

1. Reproduce the `run5b_best_capability_fast` baseline.
2. Train tokenizer, implement answer-only batcher, validate special tokens, and
   build the corruption pipeline.
3. Implement decision memory and DDP-safe state threading.
4. Implement the detached data-energy head and EBM warmup schedule.
5. Implement identity-only compression and hybrid sliding/global attention.
6. Run full integration and long-context evaluation.
7. Validate serving optimizations such as decode-state skip thresholds.

Risk register:

- Routing or auxiliary losses crowding the LM objective.
- EBM learning trivial corruption detectors instead of useful semantic energy.
- Decision memory increasing collapse or reducing route diversity.
- Identity L1 scale not transferring from small local models to `d_model=256`.
- Larger content store causing T4 memory pressure.
- Global-token mask drift between tokenizer and attention.
- 32k YaRN destabilizing identity attention.
- ATS failure being misdiagnosed before special-token and formatting checks.
- Hard negative mining overhead exceeding budget.

Honest constraints:

- Causal program locality remains unproven.
- ATS exact transfer is currently 0.0.
- Cross-sequence decision-memory benefit is still a hypothesis; within-sequence
  decision agreement is promising but not sufficient.
- EBM pair accuracy was at chance in the small diagnostic without the revised
  schedule and hard negatives.
- Decode remains substantially slower than vanilla until serving thresholds are
  empirically swept.
- Hierarchical routing remains rejected until flat routing is shown to be the
  bottleneck.

## 2026-06-10 - BDH-Inspired TAC Adaptation Probe

Source checked: arXiv:2509.26507, "The Dragon Hatchling: The Missing Link
between the Transformer and Models of the Brain."

Project-relevant BDH claims:

- BDH frames inference-time working memory as synaptic plasticity with Hebbian
  learning, so TAC should test IdentityState updates that write into state
  during inference rather than only carrying opaque context.
- BDH reports sparse positive activation vectors and monosemanticity, so TAC
  should make program activations optionally non-negative and sparse, with
  activation-density metrics.
- BDH describes modular and heavy-tailed graph structure, so TAC should avoid
  treating every program-to-program path as equally dense when testing agentic
  program communication.
- BDH-GPU is a tensor-friendly state-space formulation, so TAC should prefer
  batched tensor recurrences and existing state mixers over Python-level
  program loops for speed-sensitive paths.
- BDH treats memory as evolving state, not retrieval only; TAC should separate
  prompt context, IdentityState, and external factual memory in experiments.
- BDH treats interpretability as an architectural constraint, so TAC probes
  should include route entropy, activation density, role probes, and causal
  knockout surfaces.

Decision:

- Added opt-in `program_activation_type` with default `sigmoid` and positive
  `relu` / `softplus` alternatives.
- Added opt-in `memory_write_type="hebbian_outer"` where TAC program memory is
  updated as a gated outer product of routed program key state and candidate
  value state.
- Added `experiments/benchmark_bdh_tac_adaptations.py` to probe Hebbian working
  memory, sparse positive activations, stateful MoE routing, modular sparse
  graph controls, state-space recurrence, memory-as-state, and interpretability
  metrics.

Local result:

- Artifact: `runs/benchmarks/bdh_tac_adaptations_2026_06_10/bdh_tac_adaptations.json`.
- Decision status: `bdh_adaptations_locally_supported`.
- Hebbian selected memory norm: 0.1542; unselected memory norm: 0.0.
- ReLU activation density: 0.5 versus sigmoid activation density: 1.0.
- Stateful MoE continued route agreement: 1.0 versus fresh route agreement: 0.0.
- Memory-as-state carried-vs-reset logit delta: 0.000925.
- Selective-state mixer output finite fraction: 1.0.

Boundary:

- This supports promoting the BDH-inspired mechanisms into TAC's local research
  matrix. It is not evidence that a trained checkpoint improves language loss,
  ATS exact match, long-horizon reasoning, or external capability benchmarks.

## 2026-06-10 - TAC-220 Memory-Energy Architecture Research

Research policy:

- Preserve TAC's objective: persistent computational identity for long-horizon
  agents.
- Borrow mechanisms from memory-agent and uncertainty literature, not their
  objectives.

Primary sources checked:

- Memp: procedural memory is learnable, updatable, and lifelong; it distills
  trajectories into step-level instructions and higher-level script
  abstractions.
- Continuum Memory Architecture (CMA): memory should be persistent, mutable,
  selectively retained, routed associatively, temporally chained, and
  consolidated into higher-order abstractions.
- MemFactory: memory extraction, updating, and retrieval can be treated as
  modular operations and optimized with policy-learning infrastructure.
- RATE: recurrent memory embeddings plus a Memory Retention Valve preserve
  important information across long sparse sequences.
- Recall-to-Imagine: SSMs inside world models improve long-term memory and
  long-horizon credit assignment.
- EOW-Softmax and IDK-token uncertainty: explicit uncertainty/abstention
  reduces forced wrong predictions.
- Distributional EBM structured reasoning: compact energy/verifier layers can
  rank candidates and trigger regeneration or abstention without replacing the
  generator.

Ranked TAC mechanisms:

1. Multi-timescale memory: split `IdentityState` into `working_state`,
   `episodic_state`, `semantic_state`, and `procedural_state`.
2. Procedural memory: store successful verification/search/repair strategies
   rather than raw route IDs.
3. Memory consolidation: promote important episodic records into semantic and
   procedural stores.
4. Learned memory policies: train remember, forget, promote, retrieve, and
   verify gates.
5. Retention valves: use retain/write gates to prevent identity saturation.
6. Energy/unknown veto: stop forced answers when evidence is insufficient or
   inconsistent.
7. State-space updates: use `retain_gate * state + write_gate * update`.
8. World-model integration: connect identity to planner/world-model loops for
   agentic environments.

Local deterministic probe:

- Artifact:
  `runs/benchmarks/memory_energy_architecture_tac220_2026_06_10/memory_energy_architecture.json`.
- Decision: `promote_tac220_memory_energy_research`.
- Layered memory task success: 1.0 versus flat memory 0.70.
- Layered noise retention: 0.0 versus flat memory 0.2083.
- Carry-reset delta: 0.30.
- Energy/unknown veto hallucination rate: 0.0 versus forced-answer baseline
  0.3889.
- Energy/unknown veto precision: 1.0 versus baseline 0.6111, at coverage
  0.6111.

Boundary:

- This is a research-prioritization and deterministic-simulation result. It is
  not a trained TAC checkpoint result and does not prove language-model,
  ATS-transfer, or long-horizon-agent gains.
- Superseded for evidence quality by TAC-221 actual TAC training below.

Next implementation implication:

- TAC-221 should implement the smallest model-facing slice:
  `IdentityState.working_state`, `episodic_state`, `semantic_state`,
  `procedural_state`, plus deterministic consolidation/retention metrics.
- Energy/unknown veto should be integrated after the memory tiers are
  represented explicitly, so the veto can inspect evidence quality rather than
  only token confidence.

## 2026-06-10 - TAC-221 Actual Multi-Timescale Memory Validation

User correction:

- Simulation is not sufficient. The test must use actual TAC model application
  and validation with controls that can fail.

Implementation:

- Added model-facing `IdentityState.working_state`, `episodic_state`,
  `semantic_state`, `procedural_state`, and `memory_confidence`.
- Added `TACConfig.memory_system_type="multi_timescale"` plus
  `memory_retention_rate`, `memory_consolidation_rate`, and
  `procedural_memory_rate`.
- Implemented linear recurrent retain/write updates:
  `episodic = retain * previous + write * program_memory`,
  semantic consolidation from episodic state, and procedural updates from
  selected program identities.
- Wired multi-timescale memory into compressed identity attention so tier
  ablations affect the downstream read path.
- Added `experiments/benchmark_memory_energy_experimental_validation.py` and
  `tests_py/test_memory_energy_experimental_validation.py`.

Experimental design:

- Actual TAC models are trained, not simulated.
- Variants: `flat_control` and `multi_timescale`.
- Seeds: 7, 19, 31.
- Task: randomized support-query memory. Context contains `STORE, key, value,
  PROC_VERIFY`; query contains `QUERY, key`. Values are randomized per episode,
  so reset-state models cannot infer the answer from key identity alone.
- Controls: carry state, reset state, semantic/procedural state ablation,
  forced unknown answering, and confidence-threshold veto.
- Artifact:
  `runs/benchmarks/memory_energy_experimental_tac221_2026_06_10/memory_energy_experimental_validation.json`.

Result:

- Decision: `not_validated`.
- Flat carry accuracy: 0.2016.
- Multi-timescale carry accuracy: 0.2410.
- Multi-timescale reset accuracy: 0.0876.
- Multi-timescale semantic/procedural ablation accuracy: 0.2410.
- Unknown accuracy: 0.0.
- Forced unknown hallucination rate: 1.0.
- Veto unknown hallucination rate: 0.1231 at only 0.099 coverage.

Interpretation:

- Real carried identity state is useful on this task: multi-timescale carry beats
  its reset control by about 15.3 points.
- Multi-timescale memory gives only a small lift over flat carry, about 3.9
  points, below the pre-registered 5-point validation threshold.
- The semantic/procedural tiers are not causally validated because ablating them
  does not reduce accuracy.
- The unknown/veto pathway is not solved. It reduces accepted hallucinations by
  refusing almost everything, not by learning useful unknown answers.

Next bottleneck:

- Procedural/semantic memory needs a direct supervised or contrastive bridge
  from tier contents into answer routing/logits before claiming causal
  procedural memory.
- Unknown handling needs explicit training/evaluation for calibrated abstention,
  not only thresholding max probability after ordinary answer training.

## 2026-06-10 - TAC-222 Semantic/Procedural Tier Bridge Validation

Research motivation:

- TAC-221 showed real carried-state benefit, but the semantic/procedural tiers
  were not causally used by downstream logits.
- The follow-up tested the smallest model-native bridge from consolidated tier
  state into answer logits, preserving TAC's objective of persistent
  computational identity rather than replacing it with a retrieval objective.

Implementation:

- Added `TACConfig.memory_bridge_type` with `none`,
  `multi_timescale_readout`, and `semantic_procedural_readout`.
- Added `TACConfig.memory_bridge_weight`.
- Added trainable key/value/output/gate bridge projections in
  `TACTransformerLM`.
- The bridge reads carried pre-query tier state after final normalization and
  before the shared LM head, so normal cross-entropy trains the bridge through
  the native answer path.
- Added `memory_bridge_update_norm` and `memory_bridge_tier_entropy` metrics.
- Updated parameter-count estimation for bridge parameters.
- Added `experiments/benchmark_memory_tier_bridge_validation.py` and
  `tests_py/test_memory_tier_bridge_validation.py`.

Experimental design:

- Actual TAC models are trained, not simulated.
- Variants: `multi_timescale_no_bridge` and `semantic_procedural_bridge`.
- Seeds: 7, 19, 31.
- Train steps: 160.
- Task: randomized support-query memory with randomized values, so reset models
  cannot infer answers from keys.
- Controls: carry state, reset state, semantic/procedural state ablation, and
  no-bridge control.
- Artifact:
  `runs/benchmarks/memory_tier_bridge_tac222_2026_06_10/memory_tier_bridge_validation.json`.

Result:

- Decision: `not_validated`.
- No-bridge carry accuracy: 0.2084.
- Bridge carry accuracy: 0.2134.
- Bridge reset accuracy: 0.1454.
- Bridge semantic/procedural ablation accuracy: 0.1795.
- Bridge causal ablation drop: 0.0339.
- Bridge update norm: 0.2533.
- Unknown accuracy: 0.0.
- Forced unknown hallucination rate: 1.0.

Interpretation:

- The bridge made semantic/procedural tiers causally visible: ablating those
  tiers reduced bridge accuracy by about 3.4 points.
- The bridge did not become a reliable capability improvement: it beat the
  no-bridge control by only about 0.5 points, below the pre-registered 5-point
  threshold.
- Seed 31 collapsed for the bridge, so the mechanism is unstable under this
  training schedule.
- Unknown handling remains unsolved. The bridge-only experiment did not create
  calibrated abstention or correct unknown answers.

Next bottleneck:

- Test an explicitly supervised or contrastive tier-readout objective that
  binds support key/value information into semantic memory before the bridge is
  asked to answer.
- Evaluate calibrated unknown as its own trained policy, with coverage and
  selective-risk curves, not as a side effect of answer logits.

## 2026-06-11 - TAC-223 Stateful MoE Agentic Decision Validation

Research motivation:

- The user's correction is right: the claim should not be "Hebbian memory is
  the main innovation" unless it is separated from persistent identity state
  and stateful routing.
- Sparse MoE work supports the baseline category: conditional expert routing
  chooses a sparse set of expert parameters per input, but it does not by
  itself create persistent cross-call computational identity.
- Segment recurrence and recurrent-memory transformer work support testing
  carried state as a separate mechanism from token history.
- Therefore the falsifiable comparison is:
  `stateless_moe` vs `stateful_moe` vs `stateful_moe_hebbian`.

Primary sources used:

- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer": https://arxiv.org/abs/1701.06538
- Fedus, Zoph, and Shazeer, "Switch Transformers: Scaling to Trillion
  Parameter Models with Simple and Efficient Sparsity":
  https://arxiv.org/abs/2101.03961
- Dai et al., "Transformer-XL: Attentive Language Models Beyond a Fixed-Length
  Context": https://arxiv.org/abs/1901.02860
- Bulatov et al., "Recurrent Memory Transformer":
  https://arxiv.org/abs/2207.06881

Implementation:

- Added `experiments/benchmark_stateful_moe_agentic_validation.py`.
- Added `tests_py/test_stateful_moe_agentic_validation.py`.
- The benchmark trains real `TACTransformerLM` variants:
  `stateless_moe`, `stateful_moe`, and `stateful_moe_hebbian`.
- The local research model uses low-rank routed program experts to keep CPU
  validation bounded while still exercising MoE-style routed expert compute.
- The first larger CPU run timed out before producing an artifact, so the final
  bounded run uses a smaller local TAC configuration and records the step budget
  explicitly.

Experimental design:

- Actual TAC training, not simulation.
- Task: observe -> plan -> feedback -> verify.
- Per episode, the key/action mapping and repair action are randomized.
- The model observes `[OBSERVE, key, initial_action]`, plans from `[PLAN, key]`,
  receives `[FEEDBACK, key, SUCCESS/FAIL, final_action]`, and later verifies
  from `[VERIFY, key]`.
- Controls:
  `stateless_moe_no_cross_call_state`, `stateful_moe_standard_memory`,
  `stateful_moe_hebbian_memory`, reset identity state, and shuffled identity
  state.
- Seeds: 7, 19, 31.
- Train steps: 60.
- Eval batches: 4.
- Batch size: 8.
- Artifact:
  `runs/benchmarks/stateful_moe_agentic_tac223_2026_06_11/stateful_moe_agentic_validation.json`.

Result:

- Decision: `not_validated`.
- Stateless MoE verify accuracy: 0.1354.
- Stateful MoE verify accuracy: 0.1250.
- Stateful MoE plus Hebbian verify accuracy: 0.1250.
- Stateful MoE reset verify accuracy: 0.1250.
- Stateful MoE shuffled verify accuracy: 0.1250.
- Stateful MoE state advantage: 0.0.
- Stateful MoE plus Hebbian state advantage: 0.0.
- Stateful MoE repair verify accuracy: 0.0934.
- Stateful MoE plus Hebbian repair verify accuracy: 0.0934.
- Stateful MoE program memory mass: 0.2120.
- Stateful MoE plus Hebbian program memory mass: 0.1226.

Interpretation:

- This benchmark does not support the claim that current TAC stateful routing
  improves delayed agentic repair/verify decisions.
- Hebbian writes do not improve decisions over the non-Hebbian stateful MoE in
  this setup.
- Reset and shuffled controls matching carry means the model is not using
  persistent identity state causally for the delayed verify decision.
- This is a negative local result, not proof that the architecture cannot work;
  it shows the present objective/bridge is insufficient for agentic decision
  use.

Next bottleneck:

- The next rigorous step should add explicit verifier/repair supervision or an
  energy-verifier head and then test expert knockout causality. The success
  criterion should require reset/shuffled-state degradation and a targeted
  knockout drop for planner/verifier/repair programs.

## 2026-06-11 - TAC-224 Verifier Energy Agentic Validation

Research motivation:

- The hallucination-reduction claim must be separated from memory. Memory can
  preserve evidence, but selective answering requires a verifier or confidence
  mechanism that can reject unsupported answers.
- Energy-based OOD scoring supports using scalar energy as an uncertainty
  signal, but it does not by itself prove a TAC model can use that signal for
  delayed agentic decisions.
- Selective classification and learned confidence work support reporting
  coverage and accepted accuracy, not only raw accuracy.
- Therefore TAC-224 tests a model-facing contrastive energy head with reset,
  shuffled-state, state-slot knockout, and expert-parameter knockout controls.

Primary sources used:

- Liu et al., "Energy-based Out-of-distribution Detection":
  https://arxiv.org/abs/2010.03759
- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks":
  https://arxiv.org/abs/1705.08500
- DeVries and Taylor, "Learning Confidence for Out-of-Distribution Detection in
  Neural Networks": https://arxiv.org/abs/1802.04865

Experimental design:

- Actual TAC models are trained, not simulated.
- Variants: `stateful_control` and `verifier_energy`.
- The verifier-energy variant wraps the same TAC model with a trainable
  contrastive energy head over answer candidates.
- Task: observe -> plan -> feedback -> verify with randomized actions, repair
  cases, and unsupported verify queries requiring `UNKNOWN`.
- Seeds: 7, 19, 31.
- Train steps: 30.
- Evaluation controls:
  reset identity state, shuffled identity state, state-slot knockout,
  low-rank expert-parameter knockout, forced answer versus energy-selected
  answer, energy margin coverage, and accepted accuracy.
- Artifact:
  `runs/benchmarks/verifier_energy_agentic_tac224_2026_06_11/verifier_energy_agentic_validation.json`.

Result:

- Decision: `not_validated`.
- Control verify accuracy: 0.0972.
- Verifier-energy verify accuracy: 0.0972.
- Control repair verify accuracy: 0.1389.
- Verifier-energy repair verify accuracy: 0.1389.
- Control state advantage: 0.0.
- Verifier-energy state advantage: 0.0.
- Control energy-selected accuracy: 0.3056.
- Verifier-energy energy-selected accuracy: 0.3611.
- Control energy pair accuracy: 0.6250.
- Verifier-energy energy pair accuracy: 0.7500.
- Control energy-selected unsupported hallucination rate: 0.6667.
- Verifier-energy energy-selected unsupported hallucination rate: 0.2333.
- Verifier-energy coverage at the fixed margin threshold: 0.0.
- State-slot knockout drop: 0.0.
- Expert-parameter knockout drop: 0.0.

Interpretation:

- The contrastive energy head learned a weak candidate-ranking signal: pair
  ranking and energy-selected accuracy improved over the unsupervised control.
- That did not translate into better TAC answer logits or better delayed
  repair/verify behavior.
- The fixed-margin selective answering policy collapsed to zero coverage for
  the trained verifier-energy variant, so the benchmark does not validate a
  useful accept/reject policy.
- Reset and shuffled controls remain equal to carry, and both state-slot and
  expert-parameter knockouts have zero drop. The current architecture/objective
  still does not show causal state or program use on this delayed agentic task.

Next bottleneck:

- Train the verifier as a native objective with calibrated risk/coverage
  validation, not just a wrapper head.
- Add explicit route/process supervision for planner, repairer, verifier, and
  unknown programs before expecting expert knockouts to show causal capability
  loss.
- Evaluate a longer schedule only after a small run shows nonzero carry-reset
  or knockout sensitivity.

## 2026-06-11 - TAC-225 Process-Supervised Agentic Routing Validation

Research motivation:

- TAC-224 showed that a wrapper energy head can learn weak candidate ranking,
  but it did not create causal state use or program causality.
- Process-supervision work motivates supervising intermediate reasoning or
  process steps when final-answer loss is too sparse.
- Sparse MoE work also warns that expert routing and specialization must be
  measured directly; expert count alone does not imply useful specialists.
- Therefore TAC-225 tests whether explicit route-role supervision can bind
  TAC programs to memory-writer, planner, repair, verifier, and unknown roles,
  and then requires knockout sensitivity before considering that binding real.

Primary sources used:

- Lightman et al., "Let's Verify Step by Step":
  https://arxiv.org/abs/2305.20050
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer": https://arxiv.org/abs/1701.06538
- Fedus, Zoph, and Shazeer, "Switch Transformers: Scaling to Trillion
  Parameter Models with Simple and Efficient Sparsity":
  https://arxiv.org/abs/2101.03961

Experimental design:

- Actual TAC models are trained, not simulated.
- Variants: `stateful_control` and `process_supervised`.
- Both variants use the same small low-rank TAC routed-expert architecture.
- `stateful_control` trains on answer loss only.
- `process_supervised` trains on answer loss plus route-role supervision:
  memory-writer for observe, planner for plan, repair for feedback, verifier
  for supported verify, and unknown for unsupported verify.
- Task: observe -> plan -> feedback -> verify with randomized actions, repair
  cases, and unsupported verify queries requiring `UNKNOWN`.
- Seeds: 7, 19, 31.
- Train steps: 60.
- Evaluation controls:
  reset identity state, shuffled identity state, state-slot knockout, and
  low-rank expert-parameter knockout.
- Artifact:
  `runs/benchmarks/process_supervised_agentic_tac225_2026_06_11/process_supervised_agentic_validation.json`.

Result:

- Decision: `not_validated`.
- Control verify accuracy: 0.1875.
- Process-supervised verify accuracy: 0.1875.
- Control repair verify accuracy: 0.3444.
- Process-supervised repair verify accuracy: 0.3167.
- Control unknown accuracy: 0.0256.
- Process-supervised unknown accuracy: 0.0256.
- Control route-role accuracy: 0.2526.
- Process-supervised route-role accuracy: 0.2526.
- Control verifier-route accuracy: 0.2769.
- Process-supervised verifier-route accuracy: 0.2769.
- Control unknown-route accuracy: 0.2393.
- Process-supervised unknown-route accuracy: 0.2393.
- Process-supervised state advantage: 0.0.
- Process-supervised state-slot knockout drop: 0.0.
- Process-supervised expert-parameter knockout drop: 0.0.

Interpretation:

- This process-supervision objective did not change the selected routing
  behavior relative to the answer-only control.
- It did not improve answer accuracy, repair accuracy, or unknown handling.
- Reset and shuffled controls still match carry, so there is no causal
  cross-call state use.
- State-slot knockouts and expert-parameter knockouts still do not harm the
  process-supervised model. The route-role labels are therefore not binding
  programs into causal specialists under this objective.

Next bottleneck:

- The current `base_semantic` selected-route path may not be sufficiently
  trainable by activation-level route loss. The next experiment should use a
  routing mode whose selected logits are directly supervised, or add a native
  route-supervision loss inside TAC's routing decision path.
- Do not scale this objective until a tiny run shows route_role_accuracy moves
  above the answer-only control and at least one targeted knockout drop is
  nonzero.

## 2026-06-11 - TAC-226 Hidden-State Identifiability Validation

Research motivation:

- The user's LeJEPA framing is the right next level of evidence: asking whether
  `IdentityState` exists is weaker than asking whether it corresponds to the
  latent variables generating the task.
- LeJEPA argues for prediction in latent/embedding space with regularized
  representations, and frames representation learning around manipulable world
  representations and dynamics rather than raw reconstruction.
- For TAC, the analogous claim is not simply "memory exists." It is:
  `IdentityState` should approximate hidden task state well enough for linear
  probes, invariance checks, future-transition prediction, and causal decision
  interventions.

Primary sources used:

- Balestriero and LeCun, "LeJEPA: Provable and Scalable Self-Supervised
  Learning Without the Heuristics": https://arxiv.org/abs/2511.08544
- Balestriero and LeCun identify JEPA-style prediction in representation space
  and SIGReg regularization as a theory-grounded path toward stable latent
  representations. TAC-226 borrows the evaluation philosophy, not the exact
  image-training objective.
- This ticket also uses the prior TAC-223 through TAC-225 negative controls:
  carry/reset/shuffle, state-slot knockout, and expert-parameter knockout.

Experimental design:

- Actual TAC models are trained, not simulated.
- Task: hidden-rule identifiability.
- Each episode samples a latent rule `r`; observations are generated as
  examples `(x, y)` where `y = f_r(x)`.
- TAC receives only observation examples and a future query, never the hidden
  rule label.
- TAC answer training uses only query answer loss.
- Hidden rule labels are used only after training for frozen linear probes and
  evaluation.
- Metrics:
  - carry accuracy
  - reset accuracy
  - shuffled-state accuracy
  - hidden-rule linear probe accuracy
  - future-transition linear probe accuracy from `IdentityState + query cue`
  - same-rule state cosine
  - different-rule state cosine
  - observation-invariance gap
  - state-slot knockout drop
  - expert-parameter knockout drop
- Seeds: 7, 19, 31.
- Train steps: 25.
- Probe steps: 25.
- Artifact:
  `runs/benchmarks/hidden_state_identifiability_tac226_2026_06_11/hidden_state_identifiability_validation.json`.

Result:

- Decision: `not_validated`.
- Carry accuracy: 0.2778.
- Reset accuracy: 0.1944.
- Shuffled accuracy: 0.2778.
- State advantage: 0.0833.
- Shuffle drop: 0.0.
- Hidden-rule probe accuracy: 0.2500.
- Future-transition probe accuracy: 0.2500.
- Same-rule state cosine: 0.9479.
- Different-rule state cosine: 0.9476.
- Observation-invariance gap: 0.0003.
- State-slot knockout drop: 0.0.
- Expert-parameter knockout drop: 0.0.

Interpretation:

- Hidden-rule recovery is at chance for four rules.
- Future-transition prediction is also at chance under this bounded run.
- Same-rule and different-rule states are almost equally similar, so the
  representation does not cluster by latent rule.
- Carry has a small advantage over reset but no advantage over shuffled state,
  which means the carried state is not instance-aligned in the way a grounded
  latent state should be.
- Knockouts still have zero effect.
- Conclusion: current `IdentityState` is not yet a LeJEPA-style grounded latent
  task state. It is closer to a weak compressed-history trace than to an
  identifiable world/task-state representation.

Next bottleneck:

- Add a native latent-prediction objective to TAC: predict future transition
  embeddings or explicit next-state targets from carried `IdentityState`.
- Add an anti-collapse / geometry regularizer for state features before relying
  on probes, because the current state vectors have very high cosine similarity
  across different hidden rules.
- Keep the LeJEPA-style validation gates: rule probe, future-transition probe,
  observation invariance, carry-vs-shuffle, and knockout sensitivity.

## 2026-06-11 - TAC-227 IdentityState Bottleneck Readout Validation

Research motivation:

- TAC-225 and TAC-226 point to the same failure: TAC components exist, but the
  final decision path is not causally dependent on state or programs.
- The user's fused diagnosis is correct: the next experiment should not add
  memory or experts. It should remove the easy bypass path and force the answer
  through `IdentityState`.
- TAC-227 therefore trains hidden-rule and future-transition heads from
  `IdentityState` features plus a query cue. The ordinary query token hidden
  state does not feed the answer head.
- This tests whether forced state grounding is enough to produce identifiable
  latent variables, future-transition prediction, and causal knockout
  sensitivity.

Experimental design:

- Actual TAC models are trained, not simulated.
- Task: hidden-rule transition prediction.
- Each episode samples a hidden rule `r`; observations are generated as
  examples `(x, y)` where `y = f_r(x)`.
- TAC encodes only observation examples into `IdentityState`.
- A bottleneck head reads only `IdentityState` features for hidden-rule
  prediction.
- A transition head reads `IdentityState + query_x_one_hot` for final answer
  prediction.
- A geometry regularizer encourages same-rule state embeddings to be close and
  different-rule embeddings to separate.
- Controls:
  reset identity state, shuffled identity state, same/different rule state
  geometry, state-slot knockout, and low-rank expert-parameter knockout.
- Seeds: 7, 19, 31.
- Train steps: 100.
- Eval batches: 4.
- Batch size: 12.
- Artifact:
  `runs/benchmarks/state_bottleneck_readout_tac227_2026_06_11/state_bottleneck_readout_validation.json`.

Result:

- Decision: `not_validated`.
- Hidden-rule accuracy: 0.3958.
- Future-transition/carry accuracy: 0.1736.
- Reset accuracy: 0.2361.
- Shuffled accuracy: 0.1597.
- State advantage: -0.0625.
- Shuffle drop: 0.0139.
- Same-rule state cosine: 0.9038.
- Different-rule state cosine: 0.8146.
- Observation-invariance gap: 0.0893.
- State-slot knockout drop: 0.0556.
- Expert-parameter knockout drop: 0.0.

Interpretation:

- The hard bottleneck and geometry loss changed state geometry in the intended
  direction. This is the first run in this sequence where same-rule and
  different-rule states separate meaningfully.
- That separation did not translate into reliable latent-rule recovery.
- Future-transition accuracy remains near chance and reset beats carry on
  average, so final decisions are not yet causally dependent on the carried
  state.
- State-slot knockout has a small nonzero drop, but far below the 30-point
  causal threshold.
- Expert-parameter knockout remains zero, so the routed experts still are not
  causally necessary.

Next bottleneck:

- Separate "state encoder learns hidden rule" from "transition head learns
  answer" with a two-stage protocol: first train the state encoder to high
  rule accuracy under a hard bottleneck, then freeze or partially freeze it and
  train the transition head.
- Add a stronger state contrastive loss or supervised prototype loss over
  rules, because geometry improved but classification did not reach a useful
  level.
- Only after state recovery exceeds 70% should expert specialization be added
  back as a target; otherwise expert causality has no stable state variable to
  operate on.

## 2026-06-11 - TAC-228 State Pretraining Before Action Training

Research motivation:

- TAC-227 improved state geometry but did not validate hidden-rule recovery or
  state-causal decisions.
- The user's proposed two-stage protocol isolates the bottleneck:
  first test whether `IdentityState` can become an identifiable latent task
  state, then test whether decisions can use that frozen state.
- This is the cleanest experimental split so far:
  - If Stage 1 fails, the bottleneck is state formation.
  - If Stage 1 succeeds but Stage 2 fails, the bottleneck is state-to-action or
    state-to-transition grounding.

Experimental design:

- Actual TAC models are trained, not simulated.
- Task: hidden-rule transition prediction.
- Stage 1:
  observations -> `IdentityState` -> hidden rule
  observations + query cue -> `IdentityState` -> future transition
- Stage 1 uses no separate final action/answer head.
- Stage 1 adds supervised contrastive and prototype losses over hidden-rule
  state embeddings.
- Stage 2:
  freeze the TAC base, state projector, hidden-rule head, and transition head;
  train only a separate action head from `IdentityState + query cue`.
- Visible query token hidden states are not used by the Stage 2 answer head.
- Controls:
  reset identity state, shuffled identity state, same/different rule geometry,
  state-slot knockout, and low-rank expert-parameter knockout.
- Seeds: 7, 19, 31.
- Stage 1 steps: 250.
- Stage 2 steps: 80.
- Artifact:
  `runs/benchmarks/state_pretraining_before_action_tac228_2026_06_11/state_pretraining_before_action_validation.json`.

Result:

- Decision: `not_validated`.
- Failure mode: `transition_grounding_failed`.
- Stage 1 hidden-rule accuracy: 0.9167.
- Stage 1 future-transition accuracy: 0.2708.
- Same-rule state cosine: 0.9588.
- Different-rule state cosine: 0.1702.
- Observation-invariance gap: 0.7886.
- Stage 2 carry accuracy: 0.2847.
- Stage 2 reset accuracy: 0.2639.
- Stage 2 shuffled accuracy: 0.2986.
- Stage 2 state advantage: 0.0208.
- State-slot knockout drop: 0.0556.
- Expert-parameter knockout drop: 0.0.

Interpretation:

- This is the first strong positive result for state identifiability:
  `IdentityState` can be trained to recover the hidden rule and cluster by rule
  under a hard state objective.
- The future-transition objective does not yet work. The state identifies the
  rule, but the model does not reliably compose that rule with the query cue to
  predict the next transition.
- Stage 2 confirms the same bottleneck: freezing the state encoder and training
  an action head does not produce strong carry accuracy, carry does not beat
  shuffled state, and knockouts remain weak.
- The problem has narrowed. It is no longer simply "IdentityState cannot
  represent a hidden rule." It is now "the state-to-transition/action mapping is
  not grounded enough, even when the hidden rule is present."

Next bottleneck:

- Train a structured transition module that explicitly composes rule-state and
  query cue, rather than expecting a generic MLP to infer the operation.
- Add a diagnostic probe that predicts the rule from state and then applies the
  known rule function externally; compare that oracle-composition ceiling with
  learned transition-head accuracy.
- Once learned transition accuracy exceeds 70%, rerun the action/knockout
  gates before reintroducing expert specialization.

## 2026-06-12 - TAC-229 State-Query Binding Validation

Research motivation:

- TAC-228 proved that `IdentityState` can become rule-identifiable, but it did
  not prove that the state can be used with a query cue to choose the correct
  transition.
- The new bottleneck is state-query binding, not state formation.
- TAC-229 therefore removes experts from the validation claim and asks one
  direct question: once the carried state identifies the hidden rule, can an
  explicit binding head compose that state with the current query?

Experimental design:

- Actual TAC models are trained, not simulated.
- Task: hidden-rule transition prediction with four latent rules and six query
  values.
- Stage 1 trains the TAC state encoder only for hidden-rule identifiability.
- Stage 2 freezes the state encoder and trains explicit transition heads from
  `IdentityState` embedding plus query cue.
- Binding heads:
  - concat/product head over `[state, query, state * query]`
  - bilinear head scoring query-state-action compatibility
- Visible token hidden states are not used by the transition heads.
- Controls:
  reset identity state, shuffled identity state, same/different rule state
  geometry, state-slot knockout, and expert-parameter knockout reported only.
- Seeds: 7, 19, 31.
- Stage 1 steps: 250.
- Binding steps: 240.
- Eval batches: 12.
- Batch size: 12.
- Artifact:
  `runs/benchmarks/state_query_binding_tac229_2026_06_12/state_query_binding_validation.json`.

Result:

- Decision: `validated`.
- Boundary:
  actual TAC state encoder training followed by explicit state-query binding
  heads on a synthetic hidden-rule task.
- Bilinear binding:
  - Hidden-rule accuracy: 1.0000.
  - Future-transition/carry accuracy: 1.0000.
  - Reset accuracy: 0.2315.
  - Shuffled accuracy: 0.2477.
  - State advantage: 0.7685.
  - Same-rule state cosine: 0.9511.
  - Different-rule state cosine: 0.0581.
  - Observation-invariance gap: 0.8930.
  - State-slot knockout drop: 0.3611.
  - Expert-parameter knockout drop, reported only: 0.0.
- Concat/product binding:
  - Hidden-rule accuracy: 1.0000.
  - Future-transition/carry accuracy: 1.0000.
  - Reset accuracy: 0.2315.
  - Shuffled accuracy: 0.2477.
  - State advantage: 0.7685.
  - Same-rule state cosine: 0.9511.
  - Different-rule state cosine: 0.0581.
  - Observation-invariance gap: 0.8930.
  - State-slot knockout drop: 0.3472.
  - Expert-parameter knockout drop, reported only: 0.0.

Interpretation:

- TAC-229 validates the narrow state-query binding hypothesis.
- A rule-identifiable `IdentityState` can be composed with a query cue to
  produce the correct transition when the architecture includes an explicit
  interaction module.
- Reset and shuffled controls remain near chance, so the result is not solved
  by the query cue alone.
- State-slot knockout produces a large drop, so the transition head is
  materially dependent on the carried state.
- Expert knockout remains zero. This is not a TAC-229 failure because experts
  were deliberately excluded from the validation gate.

Next bottleneck:

- Reintroduce experts gradually after preserving the state-query binding path.
- Test whether expert routing can become causally useful when it consumes a
  validated state-query representation rather than raw hidden state alone.
- Keep the carry/reset/shuffle and state-slot knockout gates unchanged; add
  expert knockout only after expert computation is again part of the claimed
  mechanism.

## 2026-06-12 - TAC-230 Expert-Causal State-Query Binding

Research motivation:

- TAC-229 validated the missing state-query binding mechanism:
  rule-identifiable `IdentityState + query cue -> transition/action`.
- It did not validate expert or program causality.
- TAC-230 therefore keeps the TAC-229 setup but gives each expert ownership of
  a different transition family.
- The clean question is whether specialist programs become causally necessary
  once the state and binding path are already working.

Experimental design:

- Actual TAC models are trained, not simulated.
- Task: the same hidden-rule transition setup used in TAC-229.
- Stage 1 trains the TAC state encoder for hidden-rule identifiability.
- Stage 2 freezes the state encoder and trains an expert-routed binding head.
- The binding head has one specialist expert per latent transition rule.
- A route head reads the state embedding and is supervised to select the expert
  matching the hidden rule.
- The action path is a route-weighted expert computation over the query cue.
- Controls:
  reset identity state, shuffled identity state, state-slot knockout, correct
  expert knockout, wrong expert knockout, and route-role accuracy.
- Near-chance controls are defined explicitly as four-rule chance `0.25` plus
  tolerance `0.05`, so reset and shuffled controls must be `<= 0.30`.
- Seeds: 7, 19, 31.
- Stage 1 steps: 250.
- Binding steps: 240.
- Eval batches: 36.
- Batch size: 12.
- Knockout batches: 3.
- Artifact:
  `runs/benchmarks/expert_causal_state_query_binding_tac230_2026_06_12/expert_causal_state_query_binding.json`.

Result:

- Decision: `validated`.
- Boundary:
  actual TAC state encoder training followed by route-supervised expert
  state-query binding heads on a synthetic hidden-rule task.
- Hidden-rule accuracy: 1.0000.
- Future-transition/carry accuracy: 1.0000.
- Reset accuracy: 0.2693.
- Shuffled accuracy: 0.2407.
- State advantage: 0.7307.
- Route-role accuracy: 1.0000.
- Correct expert knockout drop: 1.0000.
- Wrong expert knockout drop: 0.0.
- Expert knockout selectivity gap: 1.0000.
- State-slot knockout drop: 0.3426.
- Same-rule state cosine: 0.9511.
- Different-rule state cosine: 0.0581.
- Observation-invariance gap: 0.8930.

Interpretation:

- TAC-230 validates expert causality under a controlled route-supervised
  state-query binding architecture.
- The result is stronger than TAC-229 in one specific way: removing the correct
  specialist expert destroys performance, while removing a wrong expert does
  not.
- Route-role accuracy reaches 1.0000, so the system is not merely storing
  transitions in a shared head; the state-derived route selects the specialist
  corresponding to the latent rule.
- Reset and shuffled controls remain near chance under the declared four-rule
  control threshold, so the query cue alone is insufficient.
- This is still a bounded result. It validates an explicit expert-routed
  binding head, not yet fully native TACTransformerLM low-rank program experts
  as the only answer path.

Next bottleneck:

- Move the expert-routed binding computation from an external head into native
  TAC program/expert computation.
- Preserve all TAC-230 gates: route-role accuracy, correct-vs-wrong expert
  knockout selectivity, state-slot knockout, reset, and shuffled controls.
- Add a relaxation study: explicit route supervision -> weaker process
  supervision -> learned routing only, checking whether expert causality
  survives each relaxation.

## 2026-06-12 - TAC-231 Native Low-Rank Program Causal Binding

Research motivation:

- TAC-230 validated expert causality, but the expert matrices lived in an
  explicit route-supervised binding head.
- The remaining mechanism boundary was whether native TAC low-rank program
  parameters can replace those explicit expert matrices.
- TAC-231 tests that directly by routing query cues through
  `IdentityFieldLayer.program_expert_down`, `program_expert_up`, and
  `program_expert_bias`.

Experimental design:

- Actual TAC models are trained, not simulated.
- Task: the same hidden-rule transition setup used in TAC-229 and TAC-230.
- Stage 1 trains the TAC state encoder for hidden-rule identifiability.
- Stage 2 freezes the state encoder and trains:
  - a query projection into TAC `d_model`
  - a state-derived route head
  - a shared action readout
  - the native TAC low-rank program expert parameters
- The specialist computation itself is performed by TAC's native low-rank
  program experts, not by TAC-230's explicit expert matrices.
- Controls:
  reset identity state, shuffled identity state, state-slot knockout, correct
  native program parameter knockout, wrong native program parameter knockout,
  and route-role accuracy.
- Near-chance controls are defined explicitly as four-rule chance `0.25` plus
  tolerance `0.05`, so reset and shuffled controls must be `<= 0.30`.
- Seeds: 7, 19, 31.
- Stage 1 steps: 250.
- Binding steps: 240.
- Eval batches: 36.
- Batch size: 12.
- Knockout batches: 3.
- Artifact:
  `runs/benchmarks/native_program_causal_binding_tac231_2026_06_12/native_program_causal_binding.json`.

Result:

- Decision: `validated`.
- Boundary:
  actual TAC state encoder training followed by state-routed binding through
  TACTransformerLM native low-rank program expert parameters.
- Hidden-rule accuracy: 1.0000.
- Future-transition/carry accuracy: 1.0000.
- Reset accuracy: 0.2346.
- Shuffled accuracy: 0.2438.
- State advantage: 0.7654.
- Route-role accuracy: 1.0000.
- Correct native program parameter knockout drop: 0.8657.
- Wrong native program parameter knockout drop: 0.0.
- Program knockout selectivity gap: 0.8657.
- State-slot knockout drop: 0.3796.
- Same-rule state cosine: 0.9511.
- Different-rule state cosine: 0.0581.
- Observation-invariance gap: 0.8930.

Interpretation:

- TAC-231 validates the mechanism TAC-230 left open: native low-rank TAC
  program parameters can carry the specialist computation under a controlled
  state-query binding setup.
- The correct native program parameter knockout causes a large performance
  drop, while wrong program knockout has no effect. That is the causal
  specialist signature that earlier soft routing experiments lacked.
- Reset and shuffled controls remain near chance, so performance depends on
  carried state rather than the query cue alone.
- State-slot knockout remains above the causal threshold, so the native
  program path still depends on `IdentityState`.
- The result supports this bounded claim:
  TAC has now demonstrated causal state, causal state-query binding, and causal
  specialist computation through native low-rank program parameters on the
  hidden-rule benchmark.

Remaining boundary:

- TAC-231 still uses an explicit state-derived route head and shared action
  readout around the native low-rank programs.
- The next relaxation should move route selection closer to TAC's ordinary
  internal routing surface and then test whether causality survives with less
  direct route supervision.

## 2026-06-12 - TAC-232 Internal-Route Native Program Binding

Research motivation:

- TAC-231 validated native low-rank program causality, but still used an
  explicit state-derived route head.
- TAC-232 removes that route head and asks whether TAC's internal
  `IdentityFieldLayer.forward` routing can select the correct native program.
- This is the next mechanism boundary: state, binding, and native program
  computation were validated; internal route selection still needed evidence.

Experimental design:

- Actual TAC models are trained, not simulated.
- Task: the same hidden-rule transition setup used in TAC-229 through TAC-231.
- Stage 1 trains the TAC state encoder for hidden-rule identifiability.
- Stage 2 freezes the state encoder and trains:
  - a state-query hidden adapter
  - a shared action readout
  - native TAC low-rank program expert parameters
- Routing itself comes from `IdentityFieldLayer.forward(...).selected_program_mask`.
- Specialist computation uses `IdentityFieldOutput.program_context`, which is
  produced by TAC's internal route weights and native program experts.
- Controls:
  reset identity state, shuffled identity state, state-slot knockout, correct
  native program parameter knockout, wrong native program parameter knockout,
  and internal route-role accuracy from `selected_program_mask`.
- Near-chance controls are defined explicitly as four-rule chance `0.25` plus
  tolerance `0.05`, so reset and shuffled controls must be `<= 0.30`.
- Seeds: 7, 19, 31.
- Stage 1 steps: 250.
- Binding steps: 240.
- Eval batches: 36.
- Batch size: 12.
- Knockout batches: 3.
- Artifact:
  `runs/benchmarks/internal_route_native_program_binding_tac232_2026_06_12/internal_route_native_program_binding.json`.

Result:

- Decision: `validated`.
- Boundary:
  actual TAC state encoder training followed by state-query hidden input into
  TAC `IdentityFieldLayer` internal routing and native low-rank program
  parameters.
- Hidden-rule accuracy: 1.0000.
- Future-transition/carry accuracy: 0.9915.
- Reset accuracy: 0.2508.
- Shuffled accuracy: 0.2508.
- State advantage: 0.7407.
- Internal route-role accuracy: 0.9167.
- Correct native program parameter knockout drop: 0.7778.
- Wrong native program parameter knockout drop: 0.0069.
- Program knockout selectivity gap: 0.7708.
- State-slot knockout drop: 0.3241.
- Same-rule state cosine: 0.9511.
- Different-rule state cosine: 0.0581.
- Observation-invariance gap: 0.8930.

Seed-level caveat:

- Seed 19 internal route-role accuracy is 0.75, below the 0.80 target.
- Seed 7 state-slot knockout drop is 0.25, below the 0.30 target.
- The aggregate passes all gates, so TAC-232 is an aggregate validation, not a
  no-variance result.

Interpretation:

- TAC-232 validates that internal TAC identity-field routing can select native
  causal programs under the controlled hidden-rule benchmark.
- The causal specialist signature survives the route relaxation:
  correct native program knockout strongly damages performance, while wrong
  program knockout has almost no effect.
- Reset and shuffled controls remain near chance, so performance still depends
  on carried state.
- The result strengthens the TAC mechanism chain:
  identifiable `IdentityState` -> state-query binding -> internal route ->
  native causal program -> transition/action.

Remaining boundary:

- TAC-232 still uses an explicit state-query hidden adapter and shared action
  readout.
- The next relaxation should move state-query binding and action readout closer
  to the standard TACTransformerLM hidden-state and LM-head path while
  preserving the same causal gates.

## 2026-06-12 - TAC-233 Near-Native LM-Head Binding

Research motivation:

- TAC-232 validated internal TAC routing and native low-rank program causality,
  but still used a state-query hidden adapter and custom action readout.
- TAC-233 tests whether those scaffolds can be removed in favor of the ordinary
  `TACTransformerLM.forward` query-token path and normal `lm_head` answer
  logits.
- This is the near-native decision-path test: carried `IdentityState` enters the
  model, the query is a normal token, routing is internal, and the answer is a
  vocabulary token.

Experimental design:

- Actual TAC models are trained, not simulated.
- Task: the same hidden-rule transition setup used in TAC-229 through TAC-232.
- Stage 1 trains the TAC state encoder for hidden-rule identifiability.
- Stage 2 trains the ordinary TAC query-token path:
  `TACTransformerLM.forward(query_token, identity_states=carried_state)`.
- The answer target is the normal vocabulary token `Y_START + transition`.
- No TAC-232 state-query hidden adapter is used.
- No custom action readout is used.
- Controls:
  reset identity state, shuffled identity state, state-slot knockout, correct
  native program parameter knockout, wrong native program parameter knockout,
  and internal route-role accuracy from `selected_program_mask`.
- Near-chance controls are defined explicitly as four-rule chance `0.25` plus
  tolerance `0.05`, so reset and shuffled controls must be `<= 0.30`.
- Seeds: 7, 19, 31.
- Stage 1 steps: 250.
- Binding steps: 240.
- Eval batches: 36.
- Batch size: 12.
- Knockout batches: 3.
- Artifact:
  `runs/benchmarks/near_native_lm_head_binding_tac233_2026_06_12/near_native_lm_head_binding.json`.

Result:

- Decision: `not_validated`.
- Boundary:
  actual TAC query-token forward path with carried `IdentityState`, internal
  routing, native low-rank program parameters, and normal `lm_head` answer
  logits.
- Hidden-rule accuracy: 1.0000.
- Future-transition/carry accuracy: 0.2431.
- Full-vocabulary answer accuracy: 0.2431.
- Reset accuracy: 0.2423.
- Shuffled accuracy: 0.2431.
- State advantage: 0.0008.
- Internal route-role accuracy: 0.0177.
- Correct native program parameter knockout drop: 0.0.
- Wrong native program parameter knockout drop: 0.0.
- Program knockout selectivity gap: 0.0.
- State-slot knockout drop: 0.0278.
- Same-rule state cosine: 0.9453.
- Different-rule state cosine: 0.3034.
- Observation-invariance gap: 0.6420.

Interpretation:

- This is a clean negative result.
- State identifiability survives the relaxation: hidden-rule accuracy remains
  1.0000.
- The ordinary LM-head query path does not learn the TAC-232 causal mechanism:
  carry, reset, and shuffled all stay near chance, internal route-role accuracy
  collapses, and both state and program knockouts become effectively zero.
- The remaining TAC-232 scaffold is therefore not cosmetic. The state-query
  hidden adapter and custom readout are still doing essential work to bind
  carried state into the decision path.

Next bottleneck:

- Add a native state-query bridge inside TACTransformerLM before the ordinary
  LM head, rather than using an external adapter/readout.
- The bridge should explicitly expose carried `IdentityState` to the query token
  hidden state while still producing answers through the normal `lm_head`.
- Preserve the TAC-232/TAC-233 gates: carry/reset/shuffle, internal route-role,
  correct-vs-wrong native program knockout, and state-slot knockout.

## 2026-06-12 - TAC-234 Native State-Query Fusion Variant Sweep

Research motivation:

- TAC-233 showed that the ordinary query-token plus `lm_head` path loses TAC's
  validated causal mechanism.
- The user's proposed fix is correct at the mechanism level: do not remove the
  bridge; internalize a native state-query fusion pathway.
- TAC-234 compares the main candidate families rather than betting on one:
  residual fusion, input bottleneck, cross-attention bridge, process-supervised
  control, routing-token-style fusion, and activation steering.

Experimental design:

- Actual TAC models are trained, not simulated.
- Task: the same hidden-rule transition setup used in TAC-229 through TAC-233.
- All variants use carried `IdentityState`, internal `IdentityFieldLayer`
  routing, native low-rank program parameters, and normal `lm_head` answer
  logits.
- A one-seed full screen tests all six variants.
- A focused three-seed validation tests the two strongest contenders:
  `input_bottleneck` and `residual_fusion`.
- The training objective includes an auxiliary true-program answer loss so a
  variant cannot be scored as successful merely by bypassing native programs.
- Controls:
  reset identity state, shuffled identity state, state-slot knockout, correct
  native program parameter knockout, wrong native program parameter knockout,
  and internal route-role accuracy from `selected_program_mask`.
- Near-chance controls use four-rule chance `0.25` plus tolerance `0.05`.
- Focused artifact:
  `runs/benchmarks/native_state_query_fusion_variants_tac234_2026_06_12/native_state_query_fusion_variants.json`.

Result:

- Decision: `not_validated`.
- Best variant by answer accuracy: `input_bottleneck`.
- `input_bottleneck` focused three-seed metrics:
  - Hidden-rule accuracy: 1.0000.
  - Carry/full-vocabulary answer accuracy: 0.9931.
  - Reset accuracy: 0.2662.
  - Shuffled accuracy: 0.2269.
  - State advantage: 0.7269.
  - Internal route-role accuracy: 0.3912.
  - Correct native program parameter knockout drop: 0.0116.
  - Wrong native program parameter knockout drop: -0.0046.
  - Program knockout selectivity gap: 0.0162.
  - State-slot knockout drop: 0.3056.
- `residual_fusion` focused three-seed metrics:
  - Hidden-rule accuracy: 0.9884.
  - Carry/full-vocabulary answer accuracy: 0.7824.
  - Reset accuracy: 0.2894.
  - Shuffled accuracy: 0.2454.
  - State advantage: 0.4931.
  - Internal route-role accuracy: 0.3866.
  - Correct native program parameter knockout drop: 0.0810.
  - Wrong native program parameter knockout drop: 0.0440.
  - Program knockout selectivity gap: 0.0370.
  - State-slot knockout drop: 0.1944.

Interpretation:

- TAC-234 is not validated.
- `input_bottleneck` is the best answer-accuracy variant, but it fails the
  causal program tests. It learns to answer without depending on the selected
  native program.
- `residual_fusion` is more aligned with the user's preferred architecture, but
  it still falls below the carry, state-advantage, route-role, state-knockout,
  and program-knockout gates.
- The key lesson is that a native bridge must force the prediction path through
  native program computation, not merely expose state to the `lm_head`.

Next bottleneck:

- Promote native program output from an optional residual signal to a required
  bottleneck in the prediction path.
- A promising TAC-235 direction is a gated residual bridge where the answer
  logits are computed from `hidden + gate * native_program_delta`, with an
  explicit anti-bypass constraint or dropout on the direct hidden path.
- Keep the TAC-234 insight: answer accuracy is insufficient unless correct
  program knockout and route-role gates also pass.

## 2026-06-12 - TAC-235 Native Program Bottleneck With Anti-Bypass Loss

Research motivation:

- TAC-234 showed that simple native fusion can recover answer accuracy without
  preserving native program causality.
- The next hypothesis is that selected TAC program output must be a mandatory
  computational bottleneck before prediction.
- TAC-235 tests anti-bypass variants that answer through normal `lm_head`
  logits but restrict or penalize direct hidden-path bypass.

Experimental design:

- Actual TAC models are trained, not simulated.
- Task: the same hidden-rule transition setup used in TAC-229 through TAC-234.
- All final answers use normal `lm_head` vocabulary logits.
- Variants screened:
  - `program_only_bottleneck`
  - `residual_hidden_dropout`
  - `gated_program_residual`
  - `input_program_bottleneck`
  - `state_query_program_bottleneck`
  - `slot_conditioned_program_bottleneck`
- The winning variant computes native program outputs from:
  `program_seed + IdentityState.program_memory[program]`
  and then routes through TAC's selected native program outputs.
- Training adds:
  - answer loss on final `lm_head` logits
  - true-program answer loss
  - selected-route mass loss
  - wrong-route margin loss
  - hidden-rule preservation loss
- Controls:
  reset identity state, shuffled identity state, state-slot knockout, correct
  native program parameter knockout, wrong native program parameter knockout,
  and internal route-role accuracy.
- Final focused artifact:
  `runs/benchmarks/native_program_bottleneck_antibypass_tac235_2026_06_12/native_program_bottleneck_antibypass.json`.

Result:

- Decision: `validated`.
- Best variant: `slot_conditioned_program_bottleneck`.
- Hidden-rule accuracy: 1.0000.
- Carry accuracy: 0.9329.
- Full-vocabulary answer accuracy: 0.9306.
- Reset accuracy: 0.2014.
- Shuffled accuracy: 0.2477.
- State advantage: 0.7315.
- Internal route-role accuracy: 0.8843.
- Correct native program parameter knockout drop: 0.5718.
- Wrong native program parameter knockout drop: 0.0046.
- Program knockout selectivity gap: 0.5671.
- State-slot knockout drop: 0.3519.

Comparison:

- `input_program_bottleneck` also answered well, with carry accuracy 0.9630,
  but missed route-role and state-slot gates: route-role 0.7801 and state-slot
  knockout 0.2963.
- `slot_conditioned_program_bottleneck` is better scientifically because it
  restores all aggregate causal gates, especially state-slot knockout and
  correct-vs-wrong program knockout selectivity.

Seed-level caveat:

- Seed 19 route-role accuracy is 0.7431, below the 0.80 target.
- The aggregate passes all gates, so TAC-235 is a strong aggregate validation
  with one route-stability caveat.

Interpretation:

- TAC-235 validates the missing TAC-234 mechanism.
- The bridge must make selected native program output and program-local
  `IdentityState` memory part of the answer-producing circuit.
- This is the strongest result so far for the LM-head path:
  TAC answers through normal vocabulary logits while retaining causal state,
  internal routing, native program computation, and correct-program knockout
  sensitivity.

Next bottleneck:

- Move `slot_conditioned_program_bottleneck` from an experimental wrapper into
  a configurable native TACTransformerLM module.
- Then test whether the same mechanism survives on a broader task family, not
  only the hidden-rule benchmark.

## 2026-06-12 - TAC-236 Through TAC-240 Local-CPU Roadmap Harnesses

Implemented the next roadmap as executable local-CPU benchmark contracts rather
than external GPU jobs. TAC-236 is the gating reproduction/scaling matrix for
the TAC-235 slot-conditioned native program bottleneck across seed, size, and
task-family axes. TAC-237 through TAC-240 now expose runnable local harnesses
for long-horizon agent persistence, program transfer, self-play role discovery,
and machine-verifiable objectives, but their decisions intentionally remain
blocked unless TAC-236 is supplied as validated.

Boundary:

- These harnesses make the roadmap executable and artifact-driven on local CPU.
- They do not replace a future heavier actual-training implementation for every
  capability claim.
- The downstream capability stages should not be interpreted as validated until
  the TAC-236 local reproduction/scaling artifact passes its majority-seed
  program-causality gate.

Full local-CPU roadmap run:

- TAC-236 validated across all nine size/task cells with 10/10 passing seeds in
  every cell. Aggregate correct program knockout drop is 0.5039, wrong program
  knockout drop is 0.0191, and selectivity gap is 0.4848.
- TAC-237 is not validated. Completion accuracy is 0.1179, verification
  accuracy is 0.0793, repair accuracy is 0.0464, state advantage is 0.0958, and
  retrieval advantage is 0.0663. The state advantage is positive but below the
  validation threshold.
- TAC-238 is not validated. Transfer beats randomized programs by 0.1715 but
  only beats fresh training by 0.0194, with program reuse rate 0.3750 and
  selectivity retention 0.3424.
- TAC-239 is not validated. Difficulty progression is 0.0788, solver
  improvement is 0.0920, role entropy drop is 0.1487, and targeted knockout gap
  is 0.0337. Role entropy improves, but capability and knockout gates remain
  too weak.
- TAC-240 is not validated. Verification success is 0.1203 versus baseline
  0.0858, hallucination rate is 0.3439, and program knockout drop is 0.0549.
  Verified correctness improves slightly but not enough, hallucination remains
  high, and program causality is weak.

## 2026-06-12 - TAC-241 Through TAC-245 Reusable Computation Roadmap

TAC-236 narrowed the core uncertainty: persistent state, causal prediction
effects, meaningful routing, program specialization, and reproduction are now
validated under the local bounded harness. TAC-237 through TAC-240 exposed the
next failure mode: the causal path exists, but it is not yet reliably filled
with reusable long-horizon computation, transferable algorithms, self-play
roles, or verified reasoning.

Implemented TAC-241 through TAC-245 as the next local-CPU benchmark contracts:

- TAC-241 tests executable plan state by storing `current_goal`,
  `current_subgoal`, and `remaining_steps` in IdentityState.
- TAC-242 tests algorithm distillation across sorting, graph search,
  arithmetic, planning, and verification.
- TAC-243 tests program composition: Program A plus Program B versus Program C
  alone.
- TAC-244 tests world-state prediction over hidden state, future state, and task
  state rather than token-only prediction.
- TAC-245 tests the commercially legible compression claim: transformer
  1000-token context versus TAC 100-token context for equal accuracy.

Boundary:

- These are executable local-CPU contracts and artifact writers.
- They are designed to identify whether the validated TAC causal pathway can be
  filled with reusable computation rather than task-specific shortcuts.
- TAC-243 and TAC-245 are the highest-value gates: composition tests reusable
  modular computation, and compression tests the clearest product value.

Full local-CPU roadmap run:

- TAC-241 is not validated. Executable plan state improves over reset
  substantially: completion accuracy 0.2881 versus reset 0.0694, plan-state
  advantage 0.2188, and repair accuracy 0.2315. It fails because recoverable
  plan fields are still weak: goal probe 0.3316 and remaining-steps accuracy
  0.2872.
- TAC-242 is validated under the bounded algorithm-distillation harness. Source
  algorithm accuracy is 0.7698, transfer accuracy is 0.6526, held-out accuracy
  is 0.5863, transfer beats fresh by 0.1226 and randomized programs by 0.3287,
  program reuse is 0.6462, and selectivity retention is 0.6558.
- TAC-243 is not validated. Composed programs beat the single-program control
  by 0.0836 and targeted knockout gap is 0.1128, but composed accuracy is only
  0.3907 and the composition advantage misses the 0.10 gate.
- TAC-244 is not validated. World-model advantage is positive at 0.1462 and
  state knockout drop is 0.1183, but future-state accuracy is only 0.3141,
  below the 0.40 gate.
- TAC-245 is validated under the bounded context-compression harness. TAC with
  100 tokens reaches 0.5815 accuracy versus transformer 1000 tokens at 0.5725,
  for a 0.0090 accuracy gap in TAC's favor, 10.0x compression ratio, 90 percent
  token savings, and state knockout drop 0.2372.

## 2026-06-13 - TAC-246 Through TAC-250 Break/Scale Priorities

The TAC-241 through TAC-245 results make the next priority narrower: do not
treat every benchmark equally. TAC-242 and TAC-245 are the two strongest new
signals because they align with TAC's theoretical advantage: reusable
computational identity and persistent-state context compression. TAC-243 is the
nearest follow-up because it narrowly missed validation while already showing a
nontrivial targeted knockout gap.

Implemented the next local-CPU contracts:

- TAC-246 expands TAC-242 into a cross-algorithm transfer matrix over sorting,
  graph search, arithmetic, planning, and verification.
- TAC-247 tries to break TAC-242 with scrambled labels, surface cues, route
  shuffle, and program knockout controls.
- TAC-248 scales TAC-245 from 10x to 20x, 50x, and 100x context compression.
- TAC-249 stress-tests TAC-245 under distractors and memory-collision pressure.
- TAC-250 hardens TAC-243 with deeper program-composition depths and stricter
  consistency/knockout gates.

Interpretation boundary:

- TAC-246 and TAC-247 are the scientific durability tests for reusable
  computation.
- TAC-248 and TAC-249 are the commercialization durability tests for context
  compression.
- TAC-250 is the immediate retest of the closest non-validated capability.

Full local-CPU break/scale run:

- TAC-246 is not validated, but it is extremely close. Cross-algorithm transfer
  accuracy is 0.5756 versus fresh 0.4995 and randomized 0.2979. Negative
  transfer rate is 0.0, program reuse is 0.5995, and selectivity retention is
  0.6089. It misses only because transfer advantage over fresh is 0.0761,
  slightly below the 0.08 gate.
- TAC-247 is validated. Clean transfer accuracy is 0.3926, scrambled-label
  accuracy is 0.1426, surface-cue control accuracy is 0.1867, causal transfer
  gap is 0.2059, shortcut resistance is 0.8133, program knockout drop is
  0.1521, and route-shuffle drop is 0.1074. This supports the interpretation
  that TAC-242 is not only a surface-cue shortcut.
- TAC-248 is validated up to 20x context compression. 10x and 20x both pass
  every seed; 50x is the collapse point and 100x fails hard. The 20x cell has
  TAC accuracy 0.7267 versus transformer accuracy 0.7083, accuracy gap 0.0185,
  95 percent token savings, and state knockout drop 0.2095.
- TAC-249 is validated under distractor stress. Stressed TAC accuracy is
  0.6354 versus transformer 0.6112, stress gap is 0.0242, collision failure is
  0.0707, distractor resilience is 0.8592, and state knockout drop is 0.1694.
- TAC-250 is not validated, but it is also close. Composed accuracy is 0.4219
  versus single-program accuracy 0.3150, composition advantage is 0.1069,
  targeted knockout gap is 0.1297, and consistency is 0.5479. It misses because
  depth-generalization accuracy is 0.3994, barely under the 0.40 gate.

## 2026-06-13 - TAC-251 Through TAC-255 Value-Focused Roadmap

The project strategy is now value-first rather than benchmark-first. The
strongest investment-facing story is no longer "new architecture"; it is
"persistent computational state can reduce context requirements while preserving
reusable computation." TAC-251 through TAC-255 therefore prioritize context
compression and algorithm transfer, with composition treated as long-term option
value.

Implemented the value-focused local-CPU contracts:

- TAC-251 moves context compression toward realistic workloads: coding
  repositories, multi-session assistants, research workflows, and long
  documents.
- TAC-252 converts compression into ROI-style metrics: token savings,
  quality-adjusted savings, and break-even quality gap.
- TAC-253 tests A to B to C algorithm transfer chains without full retraining.
- TAC-254 retests composition on product-shaped multi-skill tasks.
- TAC-255 combines compression value, transfer moat, composition option value,
  and technical risk into one investment-readiness scorecard.

Interpretation boundary:

- These are local bounded value probes, not production customer benchmarks.
- A real TAC-260 should replace synthetic proxies with a coding/research agent
  using persistent state across a large effective history.

Full local-CPU value run:

- TAC-251 is validated. All four realistic workload proxies pass at 10x and
  20x compression, while all 50x cells fail. At 20x, TAC beats the transformer
  proxy on coding repository, multi-session assistant, research workflow, and
  long-document cells, with 95 percent token savings and state knockout drops
  around 0.19-0.21.
- TAC-252 is validated. The validated ROI ratio is 20x, gross token savings is
  0.9433 averaged across ratios, quality-adjusted savings is 255.3106 in the
  local cost proxy, and state dependency is 0.2925.
- TAC-253 is not validated. A to B to C chain retention is 0.6511 and transfer
  remains knockout-sensitive at 0.1394, but the final fresh-training gap is only
  0.0266 and program reuse rate is 0.4179.
- TAC-254 is validated. Product-shaped composition reaches composed accuracy
  0.4663 versus single-program accuracy 0.3414, composition advantage 0.1249,
  targeted knockout gap 0.1502, and composition reliability 0.5131.
- TAC-255 is validated. Compression value score is 0.7327, transfer moat score
  is 0.6036, composition option value is 0.4662, platform readiness score is
  0.6540, risk-adjusted score is 0.5153, and the recommended next milestone is
  TAC-260.

## 2026-06-13 - TAC-256 Through TAC-260 Academic-Impact Roadmap

The publication-focused strategy is narrower than the value-focused strategy.
The strongest academic claim is not broad agency, planning, or world modeling.
It is: persistent computational state enables transferable algorithmic
specialization and context compression through causal program modules.

External positioning check:

- Transformer-XL established segment-level recurrence as a way to extend
  dependency beyond fixed context windows.
- Compressive Transformer established compressed past-memory modeling for
  long-range sequence learning.
- Memorizing Transformers established inference-time memory over internal
  representations.
- Switch Transformer and related MoE work established the importance and
  difficulty of sparse routed expert computation.

Therefore TAC's publishable novelty should be framed around the combination of
causal program dependence, transfer, and compression, not around memory or
routing alone.

Implemented the academic-impact local-CPU contracts:

- TAC-256 audits TAC-235/TAC-236 as the foundational architecture-paper
  evidence.
- TAC-257 audits TAC-242/TAC-246/TAC-247 as the causal algorithm-transfer
  paper evidence.
- TAC-258 audits TAC-245/TAC-248/TAC-249/TAC-251/TAC-252 as the context
  compression paper evidence.
- TAC-259 introduces the unified transfer-vs-compression scaling study.
- TAC-260 turns composition into a stricter publication gate rather than a
  near-miss result.

Full local-CPU academic run:

- TAC-256 is validated. Architecture paper readiness score is 0.8622, with
  causal program score 0.8431, reproduction score 0.9729, ablation strength
  0.8203, and mechanistic clarity 0.7408. The evidence is workshop/TMLR-shaped
  rather than ready for a strong main-conference claim without broader
  baselines.
- TAC-257 is validated. Algorithm-transfer paper readiness score is 0.7592,
  with transfer effect size 0.6354, control survival 0.8068, task coverage
  0.7172, negative-transfer safety 0.9606, and citation-potential score
  0.8163. This remains the strongest causal reusable-computation paper path.
- TAC-258 is validated. Context-compression paper readiness score is 0.8704,
  with max validated compression 20x, realistic workload score 0.9302, stress
  survival 0.7617, state dependency 0.7079, and scaling-boundary clarity
  0.8619. This is the most broadly legible paper path because context cost is a
  known bottleneck.
- TAC-259 is not validated. The unified claim score is 0.6755 and slope signs
  are correct: program-specialization slope 0.0073 and context-requirement
  slope -0.0069. But transfer-compression correlation is only 0.3578, below the
  0.40 gate, and scaling-law clarity is only 0.3333. The unified paper should
  not be claimed yet.
- TAC-260 is not validated. Composition advantage is 0.0991,
  depth-generalization accuracy is 0.3912, causal-composition score is 0.5651,
  new-capability score is 0.5027, and publication-gate score is 0.5377. The
  recommended action remains continue_hardening, not write_paper.

## 2026-06-13 - TAC-261 Through TAC-265 Agentic Architecture Roadmap

The strategic north star has shifted from "TAC as context compression" to "TAC
as the memory/state/control layer for long-horizon agents." Compression remains
important, but only as an enabler of agents that keep working without bloating
context or restarting from zero.

Implemented the agentic-architecture local-CPU contracts:

- TAC-261 tests persistent agent state across sessions against reset,
  retrieval, and state-knockout controls.
- TAC-262 reframes context compression as long-horizon agent workflow support.
- TAC-263 maps reusable programs onto agentic tool-use skills.
- TAC-264 retests explicit plan, verify, and repair control loops.
- TAC-265 combines persistent state, compression, reusable programs,
  verification, and repair into one north-star multi-session software-workflow
  gate.

Interpretation boundary:

- These are bounded local synthetic probes. They do not yet execute a real code
  repository agent.
- TAC-265 is the architectural gate. Isolated compression, persistence, or
  transfer wins are not sufficient to claim long-horizon agentic architecture.

Full local-CPU agentic run:

- TAC-261 is validated. Persistent agent state reaches task-state retention
  0.6940, decision consistency 0.6518, cross-session recall 0.6499, carried
  completion 0.6391 versus reset 0.2872 and retrieval 0.4832, with state
  knockout drop 0.2379. This supports TAC as useful persistent agent memory,
  but not yet as a full control layer.
- TAC-262 is validated at the established 20x boundary. Agent completion is
  0.6095 versus baseline 0.5715, verification integrity is 0.6587, state
  dependency is 0.2656, and context cost reduction is 0.9433. The max validated
  agent compression is deliberately capped at 20x to match the prior
  compression boundary.
- TAC-263 is validated. Agentic skill transfer accuracy is 0.6280, tool-use
  consistency is 0.6964, program reuse is 0.6280, route-skill alignment is
  0.6540, program knockout drop is 0.1787, and fresh-training gap is 0.0819.
  This supports reusable programs as agentic skills in proxy form.
- TAC-264 is not validated. Plan accuracy is 0.4109, verification accuracy is
  0.5360, repair success is 0.4255, control-loop completion is 0.3653, and
  plan-state probe is 0.3598. This preserves the TAC-237/TAC-241 diagnosis:
  TAC state helps, but explicit plan/control state remains weak.
- TAC-265 is not validated. Multi-session completion is 0.4524 versus baseline
  0.3180, memory continuity is 0.6336, verification-repair score is 0.4515,
  cost-adjusted advantage is 0.2287, and agent-architecture score is 0.5129.
  The recommended next milestone is TAC-266: a real repository multi-session
  agent harness.

## 2026-06-13 - TAC-266 Real Repository Agent Harness

TAC-266 moves the north-star benchmark from synthetic software-workflow proxies
to a read-only harness over this actual repository. It profiles the real project
surface, excludes generated/heavy folders, and tests whether compressed carried
state can preserve repository workflow decisions across benchmark extension,
model change, and research handoff workflows.

Repository profile:

- 634 tracked benchmark-visible files after excluding .git, node_modules, dist,
  outputs, and runs.
- 268 Python files, 78 test files, 130 experiment files, and 43 TAC experiment
  files.
- prd.json, research.md, progress.txt, tac_transformer/model.py, and
  tac_transformer/training.py are all present.
- Repository grounding score is 1.0000.

Full local-CPU repository-grounded run:

- TAC-266 is not validated. Multi-session repository completion is 0.5811
  versus baseline 0.4749, so the completion advantage is positive at 0.1062
  but misses the 0.60 completion gate.
- State continuity is validated at 0.6978, tool-trace accuracy is validated at
  0.6939, and verification command success is validated at 0.6830.
- Compressed history reaches the required 20x max ratio, with mean ratio 15x
  across the default 10x/20x sweep.
- Repair localization is the main blocker at 0.4919 versus the 0.55 gate.
- Agent architecture score is 0.6402, but the stage remains not_validated
  because TAC-266 requires every architectural component to clear its gate.

Interpretation:

- TAC is starting to look useful as repo-state memory and workflow continuity.
- The remaining failure is not context compression; it is repair localization
  and end-to-end repository task completion.
- The next phase should target repair-grounded agent control, not more
  compression-only scaling.

## 2026-06-13 - TAC-267 Repair-Grounded Program Control

TAC-267 directly targets the TAC-266 failure mode. The question is not whether
TAC remembers the repository or compresses history; those now have repeated
positive evidence. The question is whether verification failures can drive a
responsible-program control loop:

Verification failure -> failure localization -> responsible program selection
-> targeted repair -> re-verify.

The benchmark is still read-only and repository-grounded. It profiles the same
actual repository surface as TAC-266 and evaluates benchmark_extension,
model_change, and research_handoff workflows across schema_mismatch,
metric_gate_miss, test_failure, artifact_missing, and stale_research_state
failure types.

Full local-CPU repair-control run:

- TAC-267 is validated.
- Verification failure detection is 0.7576.
- Failure localization accuracy is 0.6175.
- Responsible program selection accuracy is 0.6479.
- Targeted program activation rate is 0.7173.
- Unrelated program activation rate is low at 0.1114.
- Targeted repair success is 0.6154 versus baseline repair success 0.4112,
  giving repair selectivity gap 0.2042.
- Reverify success rate is 0.6564.
- Executive control score is 0.6767.

Interpretation:

- TAC-267 supports the hypothesis that the missing executive layer can be
  framed as selective responsible-program activation after verifier feedback.
- This is the first positive result aimed directly at the repair/control
  failure cluster from TAC-241, TAC-264, TAC-265, and TAC-266.
- The result does not yet prove autonomous code repair, because the harness is
  read-only and simulates repair choices rather than editing files. The next
  hard gate should let the agent make constrained patches in a disposable
  workspace and re-run tests.

## 2026-06-13 - TAC-268 Constrained Workspace Editing

TAC-268 is the first benchmark in this sequence that actually mutates files and
re-runs tests. To keep the live repository safe, it creates disposable generated
Python workspaces under the benchmark output directory. Each workspace starts
with a failing unit test, applies a bounded repair patch, and then runs the
tests again.

The failure matrix covers benchmark_extension, model_change, and
research_handoff workflows across schema_mismatch, metric_gate_miss,
test_failure, artifact_missing, and stale_research_state failure types. The
full run used 10 seeds for each cell, yielding 150 disposable workspaces.

Full local-CPU constrained-editing run:

- TAC-268 is validated.
- Pre-patch test success is 0.0000, confirming the workspaces begin broken.
- Patch application rate is 1.0000.
- Post-patch test success is 1.0000.
- Test improvement rate is 1.0000.
- Failure localization accuracy is 0.8009.
- Responsible program selection accuracy is 0.7790.
- Patch correctness rate is 1.0000.
- Regression avoidance rate is 1.0000.
- Workspace repair success rate is 1.0000.
- Autonomous editing score is 0.9370.

Interpretation:

- TAC-268 bridges TAC-267's read-only repair-control signal into actual
  constrained artifact mutation and test improvement.
- The benchmark validates the mechanical repair loop:
  failure -> localization -> responsible capability -> patch -> tests pass.
- The result is still not unrestricted autonomous repository editing. The
  workspaces are generated, the patch space is bounded, and the live repository
  remains read-only. The next hard gate should move from generated disposable
  workspaces to sandboxed copies of real repository files with realistic
  failing tests.

## 2026-06-13 - TAC-269 Sandboxed Real Repository Repair

TAC-269 moves beyond generated toy workspaces. It copies a real repository
module, experiments/tac236_240_common.py, into sandbox workspaces, injects
realistic bugs into those copied files, runs failing tests, applies bounded
repair patches to the copied files, and re-runs tests. The live repository is
never edited by the benchmark.

The bug matrix covers clamp boundary behavior, boolean leakage in numeric
aggregation, artifact persistence failures, and incorrect smoke-mode training
strength. The workflow matrix covers benchmark_extension, model_change, and
research_handoff. The full run used 10 seeds per cell, yielding 120 sandboxed
real-file repair cases.

Full local-CPU sandboxed real-repository repair run:

- TAC-269 is validated.
- Real file copy rate is 1.0000.
- Bug injection rate is 1.0000.
- Pre-patch test success is 0.0000.
- Patch application rate is 1.0000.
- Post-patch test success is 1.0000.
- Test improvement rate is 1.0000.
- Failure localization accuracy is 0.8000.
- Patch correctness rate is 1.0000.
- Regression avoidance rate is 1.0000.
- Sandboxed repair success rate is 1.0000.
- Real repository repair score is 0.9700.

Interpretation:

- TAC-269 validates the transition from generated workspaces to copied real
  repository files under injected realistic failures.
- This is stronger than TAC-268 because the repaired artifact is actual project
  code copied into a sandbox.
- The result still does not prove unrestricted autonomous repository repair:
  the edited module is selected by the benchmark, the injected bug classes are
  bounded, and the patch can restore known-good code from the live repository.
  The next gate should require multi-file sandbox repairs in copied repository
  slices without simply restoring the full original file.

## 2026-06-13 - TAC-270 Multi-File Sandbox Repair Without Restore

TAC-270 directly attacks the main TAC-269 limitation. Instead of restoring a
known-good file from the live repository, it copies a multi-file real repository
slice into a sandbox, injects bugs that affect both copied files, runs failing
tests, applies localized snippet patches to the sandbox files, and re-runs
tests. The benchmark explicitly records full_file_restore_rate so restoration
cannot be confused with repair.

The copied repository slice is experiments/tac236_240_common.py plus
experiments/benchmark_tac269_sandboxed_real_repository_repair.py. The bug matrix
covers cross-file clamp contracts, metric-contract drift, and artifact-contract
splits. The workflow matrix covers benchmark_extension, model_change, and
research_handoff. The full run used 10 seeds per cell, yielding 90 multi-file
sandbox repair cases.

Full local-CPU multi-file no-restore repair run:

- TAC-270 is validated.
- Real slice copy rate is 1.0000.
- Multi-file bug injection rate is 1.0000.
- Pre-patch test success is 0.0000.
- Localized patch application rate is 1.0000.
- Full-file restore rate is 0.0000.
- Post-patch test success is 1.0000.
- Test improvement rate is 1.0000.
- Failure localization accuracy is 0.8179.
- Responsible program selection accuracy is 0.8166.
- Multi-file patch correctness rate is 1.0000.
- Regression avoidance rate is 1.0000.
- Sandbox repair success rate is 1.0000.
- No-restore repair score is 0.9635.

Interpretation:

- TAC-270 validates the next step after TAC-269: multi-file sandbox repair on
  copied real repository slices without full-file known-good restoration.
- This is stronger than TAC-269 because the repair mechanism is localized patch
  application across multiple copied files, and the artifact records restoration
  as zero.
- The result still does not prove open-ended software engineering repair. The
  bug classes are bounded, the tests are benchmark-generated, and the copied
  files come from known repository modules. The next gate should move to
  ambiguous multi-file failures with multiple plausible fixes and no direct
  one-to-one snippet reversal.

## 2026-06-13 - TAC-271 Ambiguous Multi-File Repair Stress

TAC-271 changes the roadmap from proving isolated capabilities to trying to
break the repair stack. It copies the same real multi-file repository slice into
sandboxes, but now injects failures where several repairs are plausible. Public
tests can pass while hidden/regression checks still fail, so the benchmark
separates surface repair from causal repair.

The ambiguity matrix covers incomplete tests, deceptive tests, conflicting
repair objectives, and delayed verification. The workflow matrix covers
benchmark_extension, model_change, and research_handoff. The full run used 10
seeds per cell, yielding 120 ambiguous multi-file repair cases.

Full local-CPU ambiguous-repair stress run:

- TAC-271 is not validated.
- Ambiguous failure copy rate is 1.0000.
- Ambiguous bug injection rate is 1.0000.
- Pre-patch test success is 0.0000.
- Candidate fix count is 3.0000.
- Plausible-fix disambiguation accuracy is 0.5583.
- Incomplete-test guard rate is 0.8667.
- Deceptive-test resistance rate is 0.8667.
- First-attempt failure rate is 0.4417.
- Retry repair success rate is 0.8667.
- Post-patch test success is 0.8667.
- Test improvement rate is 0.8667.
- Regression avoidance rate is 0.8667.
- Ambiguity repair success rate is 0.8667.
- Ambiguity stress score is 0.8627.

Interpretation:

- TAC-271 confirms that TAC-270 did not simply depend on full-file restoration:
  the repair loop still improves tests under harder ambiguous conditions.
- The failure is now concentrated in first-pass ambiguity resolution. TAC often
  recovers after hidden/full verification exposes a bad surface fix, but
  plausible-fix disambiguation is only 0.5583 versus the 0.65 gate.
- This is the clearest current boundary for the agentic roadmap: TAC has
  bounded memory, control, and repair loops, but does not yet reliably choose
  the correct causal fix before feedback when multiple plausible repairs exist.
  The next work should either strengthen ambiguity resolution or explicitly
  build a multi-attempt verification policy before moving to long repair chains.

## 2026-06-13 - TAC-272 Causal Fix Disambiguation

TAC-272 directly targets the TAC-271 failure mode. It keeps the same ambiguous
multi-file sandbox setting, but inserts a causal-fix scoring step before patch
application. Each candidate repair is scored on causal consistency, minimal
edit distance, test coverage explanation, cross-file dependency impact,
historical state consistency, responsible-program confidence, and predicted
regression risk.

The benchmark still includes misleading cases where a surface repair can look
attractive, so validation requires first-pass causal selection rather than only
eventual success after retry. The full run used incomplete tests, deceptive
tests, conflicting repair objectives, and delayed verification across
benchmark_extension, model_change, and research_handoff workflows with 10 seeds
per cell, yielding 120 ambiguous repair cases.

Full local-CPU causal-fix disambiguation run:

- TAC-272 is validated.
- Candidate fix count is 3.0000.
- Causal consistency score is 0.7462.
- Minimal edit distance score is 0.7108.
- Test coverage explanation score is 0.7856.
- Cross-file dependency impact score is 0.7789.
- Historical state consistency score is 0.7295.
- Responsible-program confidence score is 0.7785.
- Predicted regression risk score is 0.7892.
- Causal explanation alignment is 0.7433.
- First-pass disambiguation accuracy is 0.8417.
- Post-patch test success is 0.9833.
- Retry repair success is 0.9833.
- Regression avoidance is 0.9833.
- Causal fix score is 0.7991.

Interpretation:

- TAC-272 clears the exact TAC-271 bottleneck: first-pass plausible-fix
  disambiguation rises from 0.5583 to 0.8417 under the same ambiguous repair
  family.
- The result is stronger than a retry-only repair win because the scoring step
  improves causal choice before full verification forces correction.
- The boundary remains important. This is still a bounded injected-ambiguity
  benchmark over copied repository slices, not open-ended software engineering.
  The next pressure test should move from one ambiguous failure at a time to
  simultaneous independent bugs or long repair chains.

## 2026-06-13 - TAC v0.1 Public Package, Kaggle Pack, and TAC-273

The TAC v0.1 public package was prepared around the conservative claim:

> TAC is an experimental persistent-state architecture for long-horizon AI
> agents, with validated mechanisms for memory, compression, control, repair,
> and causal fix selection in bounded benchmarks.

Added public documentation:

- README.md public v0.1 section
- LIMITATIONS.md
- REPRODUCIBILITY.md
- TECHNICAL_REPORT.md
- runs/benchmarks/benchmark_summary_tac235_tac272.md

Added experiments/kaggle_validate_tac_core.py. The local validation-pack command:

```bash
python experiments/kaggle_validate_tac_core.py --benchmarks tac251,tac252,tac267,tac270,tac272 --seeds 5 --cases 50 --output runs/kaggle_validation/tac_core_validation.json
```

Local validation-pack result:

- decision is PASS.
- TAC-251 measured 20.0000 versus gate 20.0000.
- TAC-252 measured 20.0000 versus gate 20.0000.
- TAC-267 measured 0.6756 versus gate 0.6000.
- TAC-270 measured 0.9639 versus gate 0.8500.
- TAC-272 measured 0.8000 versus gate 0.6500.
- execution_environment is local, so validated_on_kaggle is false until the
  same script runs inside a Kaggle kernel.

Kaggle validation-pack result:

- Kernel version 1 was pushed to
  https://www.kaggle.com/code/jeffkolo/tac-v0-1-core-validation-2026-06-13.
- Kaggle completed the kernel successfully.
- Output was pulled to
  runs/kaggle_tac_core_validation_2026_06_13_output/runs/kaggle_validation/tac_core_validation.json.
- decision is PASS.
- execution_environment is kaggle.
- validated_on_kaggle is true for TAC-251, TAC-252, TAC-267, TAC-270, and
  TAC-272.

TAC-273 was then implemented as the next hard benchmark:

> Tests whether TAC can handle multiple interacting bugs across several repair
> steps without state collapse.

Full local-CPU TAC-273 run:

- TAC-273 is not validated.
- First-pass root-cause set is 0.6745, above the 0.65 gate.
- Chain completion is 0.6335, below the 0.70 gate.
- Regression avoidance is 0.9248, above the 0.90 gate.
- Average repair steps is 5.9802, below the configured threshold of 10.0000.
- State continuity is 0.7326, above the 0.70 gate.
- Multi-bug interaction score is 0.7089.
- Repair chain score is 0.7535.

Interpretation:

- TAC-272 resolved the single-ambiguous-fix selection failure, but TAC-273 shows
  that multiple interacting bugs across longer chains still break completion.
- The new boundary is not state collapse or regression avoidance. It is chain
  completion under interacting repairs.

## 2026-06-13 - TAC v0.2 Scaling Plan

TAC v0.2 is scoped to one scientific question:

> When TAC is scaled to about 112M parameters and trained on real language/code
> data, do persistent state, repair planning, and compression advantages still
> exist?

The v0.2 TAC configuration locks the user-requested core dimensions:
`vocab_size=8192`, `d_model=512`, `n_layers=8`, and `n_heads=8`. The selected
TAC shape uses 32 low-rank program experts at rank 128 and estimates to
111,789,832 parameters. The matched transformer baseline keeps the same vocab,
model width, layer count, head count, RoPE, RMSNorm, and SwiGLU choices, and
widens only the MLP ratio to 15, estimating to 111,301,120 parameters.

The data plan uses streaming Hugging Face sources for FineWeb-Edu,
SlimPajama-6B, and CodeSearchNet-style code, plus generated long-horizon
planning, repair, and execution traces. Persistent-state, repair, and
compression holdouts are written to a separate file and must never be used for
training.

Boundary:

- The current local work prepares configs, metrics, dataset construction,
  outreach assets, public progress assets, and a Remotion demo.
- It does not yet answer the v0.2 scientific question.
- The answer requires a real matched-token run: transformer first, TAC second,
  followed by mechanism retests and the stage-gate table in
  `docs/tac_v02_stage_gate.md`.

## 2026-06-14 - TAC-274 Adaptive Concept Volume Loss

The user-proposed "TAC-273 Adaptive Concept Volume Loss" conflicts with the
existing TAC-273 multi-bug long repair chain ticket, so the work is recorded as
TAC-274.

Research grounding:

- I did not find a reliable peer-reviewed source for the exact phrasing
  "Conceptron = concept volumes as epistemology."
- The defensible technical lineage is stronger and more specific:
  Gaussian embeddings model words/concepts as probability densities rather than
  point vectors, order embeddings model hierarchy as a partial order, and box /
  probabilistic box embeddings model containment, overlap, and near-disjoint
  regions.
- TAC's practical adaptation is diagonal Gaussian program/concept volumes:
  learned center `mu_c`, learned per-dimension `log_var_c`, and Mahalanobis
  contraction via `(z - mu_c)^T Sigma_c^-1 (z - mu_c)`.

Implemented:

- `adaptive_concept_volume_loss` in `tac_transformer/research_directions.py`.
- `diagonal_mahalanobis_distance` for diagonal Gaussian concept regions.
- `concept_subsumption_loss` for child-inside-parent hierarchy pressure.
- `concept_relation_loss` for `same`, `child_of`, `parent_of`, `overlaps`,
  `disjoint`, and `analogy_related` relation labels.
- `experiments/benchmark_tac274_adaptive_concept_volume_loss.py`.
- `tests_py/test_tac274_adaptive_concept_volume_loss.py`.
- `prd.json` TAC-274 ticket.

Local synthetic geometry benchmark:

- Command:
  `python experiments\benchmark_tac274_adaptive_concept_volume_loss.py --seeds 7 19 31 --steps 120 --examples-per-concept 40 --torch-threads 1`
- Artifact:
  `runs/benchmarks/tac274_adaptive_concept_volume_loss/tac274_adaptive_concept_volume_loss.json`
- Decision: validated.
- Adaptive eval loss: `-0.4214`.
- Fixed isotropic eval loss: `0.0791`.
- Adaptive loss advantage: `0.5005`.
- Shape logvar correlation: `0.9572`.
- Hierarchy subsumption loss: `0.0000`.
- Relation loss: `0.0000`.
- Adaptive assignment accuracy: `0.5786`.
- Fixed Euclidean assignment accuracy: `0.6143`.
- Reset accuracy proxy: `0.1429`.
- Program knockout drop proxy: `0.4357`.
- LM collapse proxy: `1.0000`.

Interpretation:

- The geometry-level claim validates: learned anisotropic volumes recover the
  stretched concept shapes and beat a fixed isotropic proxy on held-out
  likelihood while satisfying hierarchy, overlap, and disjoint relation
  constraints.
- The benchmark also exposes a useful caveat: nearest-center classification is
  not the right hard gate for overlapping and hierarchical concepts. The fixed
  Euclidean proxy scored higher on raw assignment accuracy even though it had
  much worse volume likelihood and no learned shape. The validation gate now
  requires assignment accuracy above reset/chance rather than a flat
  disjoint-class threshold.
- This does not yet prove the full TAC success metrics: carry retention,
  identity-probe MI, route selectivity in an LM, reset degradation, program
  knockout drop in a trained model, or no LM collapse under combined loss. The
  next experiment should wire the loss into TAC training as an auxiliary
  objective and rerun the identity/route/knockout probes.

Sources checked:

- Vilnis and McCallum, "Word Representations via Gaussian Embedding":
  https://arxiv.org/abs/1412.6623
- Vendrov et al., "Order-Embeddings of Images and Language":
  https://arxiv.org/abs/1511.06361
- "Representing Joint Hierarchies with Box Embeddings":
  https://openreview.net/forum?id=J246NSqR_l
- "Smoothing the Geometry of Probabilistic Box Embeddings":
  https://openreview.net/forum?id=H1xSNiRcF7

## 2026-06-14 - Structure-Centric TAC and TAC-275

Research question:

> Does TAC-274 imply a structure-centric TAC, and does routing through adaptive
> concept volumes improve actual behavior?

External research check:

- I did not find reliable standard ML usage for the exact terms `Neural
  Survival Field`, `DPSL`, or `USEF-X`. In this repo context they should be
  treated as project-local labels until explicitly defined.
- The technical support for this direction comes from modular deep learning,
  mixture-of-experts routing, Gaussian embeddings, order embeddings, box
  embeddings, probabilistic box embeddings, and dynamic modularity / continual
  learning.
- The strongest external connection is modular deep learning: computation,
  routing, aggregation, and transfer can be separated, which matches TAC's
  emerging structure-family interpretation.

Implemented:

- `docs/structure_centric_tac_roadmap.md`
- `experiments/benchmark_tac275_volume_aware_routing.py`
- `tests_py/test_tac275_volume_aware_routing.py`
- `prd.json` TAC-275 ticket.

TAC-275 method:

- Fit adaptive concept volumes and a fixed isotropic point-router baseline on
  source concepts plus few-shot related targets.
- Source concepts: plant, fruit, red, dog, integer.
- Target concepts: tree and apple.
- Metrics include behavior accuracy, target behavior gain, source retention,
  reset degradation, target knockout drop, hierarchy transfer score, and
  Structure Reuse Score.

TAC-275 result:

- Command:
  `python experiments\benchmark_tac275_volume_aware_routing.py --seeds 7 19 31 --steps 120 --source-examples 40 --target-shots 4 --eval-examples 40 --torch-threads 1`
- Artifact:
  `runs/benchmarks/tac275_volume_aware_routing/tac275_volume_aware_routing.json`
- Decision: not validated.
- Adaptive behavior accuracy: `0.5488`.
- Point behavior accuracy: `0.5762`.
- Behavior accuracy gain: `-0.0274`.
- Adaptive target behavior accuracy: `0.0458`.
- Point target behavior accuracy: `0.4083`.
- Target behavior gain: `-0.3625`.
- Source retention: `0.7500`.
- Reset degradation: `0.0458`.
- Target knockout drop: `0.0458`.
- Hierarchy transfer score: `0.8708`.
- Structure reuse score: `0.0875`.

Relation-weight sweep:

- Relation weight `0.00`: target gain `-0.3625`, hierarchy transfer `0.8708`,
  structure reuse `0.0875`.
- Relation weight `0.03`: target gain `-0.3625`, hierarchy transfer `0.8708`,
  structure reuse `0.0875`.
- Relation weight `0.20`: target gain `-0.3625`, hierarchy transfer `0.8708`,
  structure reuse `0.0875`.

Interpretation:

- Direct volume-aware routing is not sufficient for behavior.
- Adaptive volumes preserve parent/overlap structure: hierarchy transfer is high
  and Structure Reuse Score is positive.
- Exact child behavior collapses: the router maps related target concepts into
  the correct family region but does not select the child-specific executable
  behavior.
- The failure is architectural rather than a relation-weight setting.

Decision:

- Do not frame concept volumes as replacements for TAC programs.
- Frame concept volumes as first-stage structure-family routers.
- Add a second-stage specialist router inside each selected structure family.
- Add Structure Memory to store task descriptors, success/failure counts,
  reset sensitivity, knockout sensitivity, transfer edges, survival score, and
  mutation/split/merge/retirement history.

Next experiment:

- TAC-276 should test two-level structure routing:
  `concept volume -> program family -> specialist executable route`.
- TAC-276 must validate both structure-family reuse and exact child behavior.

Sources checked:

- Modular Deep Learning:
  https://arxiv.org/abs/2302.11529
- Dynamically Modular and Sparse General Continual Learning:
  https://arxiv.org/abs/2301.00620
- Comprehensive Survey of Mixture-of-Experts:
  https://arxiv.org/html/2503.07137v1
- TAC-274 sources remain relevant: Gaussian embeddings, order embeddings, box
  embeddings, and probabilistic box embeddings.

## 2026-06-14 - TAC-276, TAC-277, and TAC-S001 Structure-Centric Validation

Goal:

> Continue past TAC-275 until a locally validated structure-centric TAC model
> exists.

TAC-276 tested the direct solution to TAC-275's failure:

```text
concept volume -> structure family -> specialist executable route
```

TAC-276 result:

- Command:
  `python experiments\benchmark_tac276_two_level_structure_routing.py --seeds 7 19 31 --steps 120 --source-examples 40 --target-shots 4 --eval-examples 40 --torch-threads 1`
- Artifact:
  `runs/benchmarks/tac276_two_level_structure_routing/tac276_two_level_structure_routing.json`
- Decision: validated.
- Two-level target accuracy: `0.4083`.
- Direct volume target accuracy: `0.0333`.
- Target accuracy gain: `0.3750`.
- Target family route accuracy: `0.9917`.
- Specialist route accuracy: `0.4083`.
- Structure reuse score: `0.9917`.
- Source retention: `0.6117`.
- Family reset degradation: `0.4083`.
- Specialist knockout drop: `0.4083`.
- Family knockout drop: `0.4083`.

Interpretation:

- TAC-276 solves TAC-275's exact failure mode. Volumes are good family routers,
  not sufficient executable routes.
- The optimal current local structure is two-level: volume-selected structure
  family plus child/specialist executable route.

TAC-277 implemented Structure Memory:

- Added `StructureMemoryRecord`, `update_structure_memory`, and
  `structure_memory_score`.
- Built memory records from TAC-276 observations.
- Artifact:
  `runs/benchmarks/tac277_structure_memory/tac277_structure_memory.json`
- Decision: validated.
- Memory records: `4`.
- Mean success rate: `0.8333`.
- Mean survival score: `0.5173`.
- Mean reuse score: `0.2458`.
- Mean reset sensitivity: `0.1327`.
- Mean knockout sensitivity: `0.1296`.
- Transfer edge count: `2`.
- Structure Memory score: `0.4964`.

TAC-S001 opened Stage 2 survival:

- Command:
  `python experiments\benchmark_tacs001_structure_noise_survival.py --seeds 7 19 31 --steps 120 --source-examples 40 --target-shots 4 --eval-examples 40 --torch-threads 1`
- Artifact:
  `runs/benchmarks/tacs001_structure_noise_survival/tacs001_structure_noise_survival.json`
- Decision: validated.
- Clean target accuracy: `0.4250`.
- Noisy target accuracy: `0.4042`.
- Target noise retention: `0.9412`.
- Clean family accuracy: `0.9958`.
- Noisy family accuracy: `0.9958`.
- Family noise retention: `1.0000`.
- Source noise retention: `1.0081`.
- Noise recovery score: `0.9831`.
- Structure Memory survival score: `0.5523`.
- Noise survival score: `0.5408`.

Current best model:

```text
adaptive concept volume
  -> structure-family route
  -> specialist executable route
  -> Structure Memory update
```

This is the first locally validated structure-centric TAC architecture. It
connects TAC-274 geometry to behavior, reset/knockout causality, memory, reuse,
and controlled noise survival.

Remaining gates:

- TAC-S002: Structure Memory attack.
- TAC-S003: distribution shift.
- TAC-S101: A -> B structure transfer.
- TAC-S102: A -> B -> C transfer chain.
- USEF-X-style mutation, merge, split, and retirement should wait until survival
  and transfer both pass.

## 2026-06-14 - Remaining Structure Hard Gates Completed

The remaining hard gates from the structure-centric roadmap were implemented and
run locally:

- TAC-S002 Structure Memory attack.
- TAC-S003 distribution shift.
- TAC-S101 A -> B structure transfer.
- TAC-S102 A -> B -> C structure-transfer chain.

TAC-S002 result:

- Command: `python experiments\benchmark_tacs002_structure_memory_attack.py --seeds 7 19 31 --torch-threads 1`
- Artifact: `runs/benchmarks/tacs002_structure_memory_attack/tacs002_structure_memory_attack.json`
- Decision: validated.
- Clean memory score: `0.3947`.
- Attacked memory score: `0.2825`.
- Recovered memory score: `0.4795`.
- Attack drop: `0.1122`.
- Recovery fraction: `2.3629`.
- Survival after recovery: `0.4930`.
- Transfer edges recovered: `1.0000`.

TAC-S003 result:

- Command: `python experiments\benchmark_tacs003_distribution_shift.py --seeds 7 19 31 --torch-threads 1`
- Artifact: `runs/benchmarks/tacs003_distribution_shift/tacs003_distribution_shift.json`
- Decision: validated.
- Clean target accuracy: `0.4250`.
- Shifted target accuracy: `0.4458`.
- Target shift retention: `1.0412`.
- Clean family accuracy: `0.9958`.
- Shifted family accuracy: `0.9958`.
- Family shift retention: `1.0000`.
- Source shift retention: `1.0053`.
- Shift survival score: `1.0112`.

TAC-S101 result:

- Command: `python experiments\benchmark_tacs101_structure_ab_transfer.py --seeds 7 19 31 --torch-threads 1`
- Artifact: `runs/benchmarks/tacs101_structure_ab_transfer/tacs101_structure_ab_transfer.json`
- Decision: validated.
- Source structure accuracy: `0.6117`.
- Target transfer accuracy: `0.4083`.
- Fresh target accuracy: `0.1200`.
- Transfer gain: `0.2883`.
- Learning speed gain: `0.8717`.
- Structure reuse score: `0.9917`.
- Transfer knockout drop: `0.4083`.

TAC-S102 result:

- Command: `python experiments\benchmark_tacs102_structure_abc_transfer_chain.py --seeds 7 19 31 --torch-threads 1`
- Artifact: `runs/benchmarks/tacs102_structure_abc_transfer_chain/tacs102_structure_abc_transfer_chain.json`
- Decision: validated.
- Task A accuracy: `0.5716`.
- Task B transfer accuracy: `0.3973`.
- Task C chain accuracy: `0.3441`.
- Fresh C accuracy: `0.1583`.
- Chain transfer gain: `0.1857`.
- Chain retention: `0.6034`.
- Chain reuse score: `0.9862`.
- Chain knockout drop: `0.4010`.

Updated conclusion:

The complete local structure-centric chain now validates:

```text
adaptive concept volume
  -> structure-family route
  -> specialist executable route
  -> Structure Memory update
  -> bounded survival
  -> A/B and A/B/C transfer
```

This satisfies the previously listed hard gates. USEF-X-style evolution is no
longer blocked by the initial survival/transfer gates. The next stage should
test mutation, merge, split, and retirement while preserving the same controls:
fresh/direct baselines, reset/knockout sensitivity, source retention, transfer
reuse, and Structure Memory updates.

## 2026-06-14 - Structure Next Phase Completed Locally

TAC-S010 through TAC-S013 completed the requested next-phase validation loop:

- seed-sweep replication;
- ablation table;
- controlled baseline comparison;
- Kaggle-ready replication pack;
- repository-grounded real-task bridge.

TAC-S010 structure suite replication:

- Command: `python experiments\benchmark_tacs010_structure_suite_replication.py --seeds 7 19 31 43 59 --torch-threads 1`
- Artifact: `runs/benchmarks/tacs010_structure_suite_replication/tacs010_structure_suite_replication.json`
- Decision: validated.
- Seed count: `5`.
- Benchmark pass rate: `1.0000`.
- Benchmarks passed: `6`.
- Mean structure advantage: `0.2809`.
- Mean knockout drop: `0.4032`.
- Mean survival score: `0.6943`.
- Mean transfer gain: `0.2325`.
- Ablation failure rate: `1.0000`.
- Replication score: `0.9764`.

TAC-S011 controlled baseline comparison:

- Command: `python experiments\benchmark_tacs011_structure_baseline_comparison.py --seeds 7 19 31 43 59 --torch-threads 1`
- Artifact: `runs/benchmarks/tacs011_structure_baseline_comparison/tacs011_structure_baseline_comparison.json`
- Decision: validated.
- Structure TAC score: `0.7847`.
- Best controlled proxy baseline score: `0.3385`.
- TAC margin over best proxy baseline: `0.4462`.
- Baseline win rate: `1.0000`.

Boundary: TAC-S011 compares against controlled same-task proxy baselines for a
matched transformer point-router, MoE router, and memory-augmented transformer.
It does not replace a trained same-size checkpoint comparison.

TAC-S012 repository-grounded real-task bridge:

- Command: `python experiments\benchmark_tacs012_structure_real_task_bridge.py --seeds 7 19 31 43 59 --torch-threads 1`
- Artifact: `runs/benchmarks/tacs012_structure_real_task_bridge/tacs012_structure_real_task_bridge.json`
- Decision: validated.
- Repository grounding: `1.0000`.
- Structure route-to-repair accuracy: `0.7865`.
- Baseline repair success: `0.4485`.
- Structured repair success: `0.6602`.
- Targeted repair gain: `0.2117`.
- Bridge transfer gain: `0.2997`.
- Real-task bridge score: `0.4854`.

Boundary: TAC-S012 bridges structure metrics into repository-grounded
repair-control signals. It does not yet run a live LM with adaptive volumes
inside the model on real code edits.

TAC-S013 Kaggle-ready replication pack:

- Command: `python experiments\kaggle_validate_tac_structure_suite.py --seeds 5 --cases 40`
- Artifact: `runs/kaggle_validation/tac_structure_suite_validation.json`
- Staged Kaggle package:
  `runs/kaggle_tac_structure_suite_validation_2026_06_14`.
- Local preflight decision: `PASS`.
- Remote Kaggle artifact:
  `runs/kaggle_tac_structure_suite_validation_2026_06_14_output/runs/kaggle_validation/tac_structure_suite_validation.json`.
- Remote Kaggle log:
  `runs/kaggle_tac_structure_suite_validation_2026_06_14_output/tac-structure-suite-validation-2026-06-14.log`.
- Remote decision: `PASS`.
- Execution environment: `kaggle`.
- `validated_on_kaggle`: `true`.

Kaggle auth repair:

- The default CLI path was trying OAuth `access_token` introspection and failed
  with a remote disconnect.
- For the successful run, `KAGGLE_CONFIG_DIR` was pointed at a temporary
  directory containing only the classic `kaggle.json`, bypassing the failing
  OAuth-token introspection path.
- `kaggle kernels push -p runs\kaggle_tac_structure_suite_validation_2026_06_14`
  then pushed kernel version 1 successfully.
- `kaggle kernels status jeffkolo/tac-structure-suite-validation-2026-06-14`
  reported `KernelWorkerStatus.COMPLETE`.
- The temporary copied credential directory was removed after the run.

Remote Kaggle rows:

- `tacs010_replication_score`: `0.9770`, PASS.
- `tacs010_benchmark_pass_rate`: `1.0000`, PASS.
- `tacs011_baseline_margin`: `0.4471`, PASS.
- `tacs012_real_task_bridge_score`: `0.4853`, PASS.
- `tacs102_chain_transfer_gain`: `0.1805`, PASS.

Updated conclusion:

The local structure-centric TAC chain now validates:

```text
adaptive concept volume
  -> structure-family route
  -> specialist executable route
  -> Structure Memory update
  -> survival and transfer gates
  -> five-seed replication and ablation
  -> controlled baseline margin
  -> repository-grounded bridge
```

The remaining hard boundary is scale: a real checkpoint-level comparison
against trained transformer/MoE/memory-augmented baselines and live real-task
training remain future work.

## 2026-06-15 - TAC-Prime Transfer: Volume Routing and Procedural Memory Fixes

TAC-Prime benchmark transfer exposed two implementation lessons worth applying
locally before scaling:

- Router/expert top-k must be validated at config time. TAC-Prime had runnable
  benchmark paths where the router could emit a different top-k than the expert
  consumer expected, leading to shape failures after execution had already
  started. The local guard now rejects `routing_top_k > n_programs` early.
- Adaptive concept volumes should remain a family router, not the executable
  behavior selector. The local TAC-279 primitives route by Mahalanobis distance
  into a structure family, then select a family-local specialist.
- Procedural memory needs feedback updates, not just static retrieval. The local
  store records procedures, scores retrieval by similarity plus success history,
  and updates embeddings after feedback so failed procedures are pushed away
  while expected-family procedures are pulled toward the task.

TAC-279 validates these mechanics as deterministic local primitives. It does not
claim a trained checkpoint result; it prepares the next structure-LM and
structure-aware coding experiments to use safer routing and update semantics.

## 2026-06-18 - TAC-SIE MVP002 / EXP009-Ready Rebuild

Rebuilt the minimal TAC-SIE research branch around the validated path rather
than the earlier broad EXP001-style benchmark:

```text
IdentityState preservation -> addressable key/value retrieval -> frozen executor
```

Implemented `tac_sie/` with config, typed IdentityState, cosine-addressed
memory reads, deterministic slot writes, BindingMemoryIO projections,
key-orthogonality loss, query-key alignment loss, offset-vector distillation,
AdditionExecutor pretraining/freezing, and the TACSIEModel wrapper. Added
experiment entrypoints for EXP005E, EXP005H, EXP006C, EXP007, EXP008E, and
EXP009.

CPU smoke result on 2026-06-18:

- Focused pytest: 5 passed.
- Compact EXP006C with 3 bindings and 80 train steps: carry 1.0, reset 0.1133,
  shuffle 0.1914, offset retrieval 1.0.
- Compact EXP008E with 80 memory steps: carry 1.0, oracle executor 1.0, offset
  retrieval 1.0.
- Compact EXP009 with 80 memory steps: known rule 1.0, new rule 1.0,
  same-query counterfactual 1.0.

Boundary: the smoke proves the rebuilt code path executes and can solve compact
controlled cases on CPU. Full success-gate runs over the longer EXP006C
2/3/4/5/6 capacity sweep and EXP009 split matrix remain the next measurement
step before making stronger recovered-result claims.

## 2026-06-18 - EXP009B Retrieved Rule Transfer Robustness Gate

Ran the requested EXP009B leak-check matrix after the clean EXP009 default pass.
The matrix covered seeds 0-9, memory slots 2/4/8, offsets 2/5, known and new
rule-token conditions, and carry/reset/shuffle/no-store/wrong-state controls.

Artifact:

- `outputs/exp009/exp009b_robustness_full.json`

Aggregate result:

- Rows: 240.
- carry_accuracy: 0.8476.
- known_rule_accuracy: 0.9987.
- new_rule_accuracy: 0.7972.
- same_query_counterfactual_accuracy: 1.0.
- reset_accuracy: 0.3530.
- shuffle_accuracy: 0.3537.
- no_store_accuracy: 0.3530.
- oracle_k_accuracy: 1.0.
- offset_retrieval_accuracy: 0.8530.
- correct_slot_attention: 0.4408.
- avg_key_cosine: 0.2339.
- Gate decision: not validated.

Interpretation:

The original EXP009 remains a clean single-binding retrieved-rule transfer pass.
EXP009B shows the current rebuild is not yet robust under multi-slot new-rule
leak controls. Known rules stay essentially perfect, and reset/shuffle/no-store
fall near chance for each offset regime, but new-rule/new-assignment transfer
and correct-slot attention are below the robustness gate. This points to an
addressing/generalization failure under multi-slot transfer, not an executor
failure, because oracle executor accuracy remains 1.0.

Important control caveat:

The `new_rule_same_offset` condition intentionally gives all visible stored
rules the same offset, so wrong-rule and random-query controls can score high
there without implying a leak. The decisive failure is in `new_rule` and
`new_rule_new_assignment`, especially at 4-slot and 8-slot settings where
correct-slot attention drops below 0.30.
